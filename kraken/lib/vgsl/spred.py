#
# Copyright 2025 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
VGSL segmentation inference (BLLA)
"""
import PIL
import uuid
import torch
import logging
import numpy as np
import shapely.geometry as geom
import torch.nn.functional as F
import torchvision.transforms as tf

from scipy.ndimage import gaussian_filter
from skimage.filters import sobel

from torchvision.transforms import v2
from typing import Any, TYPE_CHECKING, Optional

from kraken.lib.util import is_bitonal
from kraken.containers import BaselineLine, Segmentation, Region
from kraken.lib.dataset import ImageInputTransforms
from kraken.lib.segmentation import (calculate_polygonal_environment,
                                     is_in_region, scale_polygonal_lines,
                                     scale_regions, vectorize_lines,
                                     vectorize_regions)

__all__ = ['VGSLSegmentationInference']

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image


def vec_regions(heatmap: torch.Tensor, cls_map: dict, scale: float, **kwargs) -> dict[str, list[Region]]:
    """
    Computes regions from a stack of heatmaps, a class mapping, and scaling
    factor.

    Args:
        heatmap: A stack of heatmaps of shape `NxHxW` output from the network.
        cls_map: dictionary mapping string identifiers to indices on the stack
                 of heatmaps.
        scale: Scaling factor between heatmap and unscaled input image.

    Returns:
        A dictionary containing a key for each region type with a list of
        regions inside.
    """
    logger.info('Vectorizing regions')
    regions = {}
    for region_type, idx in cls_map['regions'].items():
        logger.debug(f'Vectorizing regions of type {region_type}')
        regions[region_type] = vectorize_regions(heatmap[idx])
    for reg_type, regs in regions.items():
        regions[reg_type] = [Region(id=f'_{uuid.uuid4()}', boundary=x, tags={'type': [{'type': reg_type}]}) for x in scale_regions(regs, scale)]
    return regions


def vec_lines(heatmap: torch.Tensor,
              cls_map: dict[str, dict[str, int]],
              scale: float,
              text_direction: str = 'horizontal-lr',
              regions: list[np.ndarray] = None,
              scal_im: np.ndarray = None,
              suppl_obj: list[np.ndarray] = None,
              topline: Optional[bool] = False,
              raise_on_error: bool = False,
              **kwargs) -> list[dict[str, Any]]:
    r"""
    Computes lines from a stack of heatmaps, a class mapping, and scaling
    factor.

    Args:
        heatmap: A stack of heatmaps of shape `NxHxW` output from the network.
        cls_map: dictionary mapping string identifiers to indices on the stack
                 of heatmaps.
        scale: Scaling factor between heatmap and unscaled input image.
        text_direction: Text directions used as hints in the reading order
                        algorithm.
        regions: Regions to be used as boundaries during polygonization and
                 atomic blocks during reading order determination for lines
                 contained within.
        scal_im: A numpy array containing the scaled input image.
        suppl_obj: Supplementary objects which are used as boundaries during
                   polygonization.
        topline: True for a topline, False for baseline, or None for a
                 centerline.
        raise_on_error: Raises error instead of logging them when they are
                        not-blocking

    Returns:
        A list of dictionaries containing the baselines, bounding polygons, and
        line type in reading order:

        .. code-block::
           :force:

            [{'script': '$baseline_type', baseline': [[x0, y0], [x1, y1], ..., [x_n, y_n]], 'boundary': [[x0, y0, x1, y1], ... [x_m, y_m]]},
             {'script': '$baseline_type', baseline': [[x0, ...]], 'boundary': [[x0, ...]]},
             {'script': '$baseline_type', baseline': [[x0, ...]], 'boundary': [[x0, ...]]},
             ...
            ]
    """

    st_sep = cls_map['aux']['_start_separator']
    end_sep = cls_map['aux']['_end_separator']

    logger.info('Vectorizing baselines')
    baselines = []
    for bl_type, idx in cls_map['baselines'].items():
        logger.debug(f'Vectorizing lines of type {bl_type}')
        baselines.extend([(bl_type, x) for x in vectorize_lines(heatmap[(st_sep, end_sep, idx), :, :], text_direction=text_direction[:-3])])
    logger.debug('Polygonizing lines')

    im_feats = gaussian_filter(sobel(scal_im), 0.5)

    lines = []
    reg_pols = [geom.Polygon(x) for x in regions]
    for bl_idx in range(len(baselines)):
        bl = baselines[bl_idx]
        bl_ls = geom.LineString(bl[1])
        suppl_obj = [x[1] for x in baselines[:bl_idx] + baselines[bl_idx + 1:]]
        for reg_idx, reg_pol in enumerate(reg_pols):
            if is_in_region(bl_ls, reg_pol):
                suppl_obj.append(regions[reg_idx])
        pol = calculate_polygonal_environment(baselines=[bl[1]],
                                              im_feats=im_feats,
                                              suppl_obj=suppl_obj,
                                              topline=topline,
                                              raise_on_error=raise_on_error)
        if pol[0] is not None:
            lines.append((bl[0], bl[1], pol[0]))

    logger.debug('Scaling vectorized lines')
    sc = scale_polygonal_lines([x[1:] for x in lines], scale)

    lines = list(zip([x[0] for x in lines], [x[0] for x in sc], [x[1] for x in sc]))
    return [{'tags': {'type': [{'type': bl_type}]}, 'baseline': bl, 'boundary': pl} for bl_type, bl, pl in lines]


class VGSLSegmentationInference:
    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def _segmentation_pred(self, im: 'Image.Image') -> Segmentation:
        """
        Segmentation inference.
        """
        if self.input[1] == 1 and self.one_channel_mode == '1' and not is_bitonal(im):
            logger.warning('Running binary model on non-binary input image '
                           '(mode {}). This will result in severely degraded '
                           'performance'.format(im.mode))

        if 'topline' in self.user_metadata:
            loc = {None: 'center',
                   True: 'top',
                   False: 'bottom'}[self.user_metadata['topline']]
            logger.debug(f'Baseline location: {loc}')

        lines = []
        regions = {}

        with self._fabric.init_tensor():
            rets = self._compute_segmentation_map(im)
        _regions = vec_regions(**rets)
        for reg_key, reg_val in _regions.items():
            regions.setdefault(reg_key, [])
            regions[reg_key].extend(reg_val)

        # flatten regions for line ordering/fetch bounding regions
        line_regs = []
        suppl_obj = []
        for cls, regs in _regions.items():
            line_regs.extend(regs)
            if rets['bounding_regions'] is not None and cls in rets['bounding_regions']:
                suppl_obj.extend(regs)

        # convert back to net scale
        suppl_obj = scale_regions([x.boundary for x in suppl_obj], 1 / rets['scale'])
        line_regs = scale_regions([x.boundary for x in line_regs], 1 / rets['scale'])

        lines = vec_lines(**rets,
                          regions=line_regs,
                          text_direction=self._inf_config.text_direction,
                          suppl_obj=suppl_obj,
                          topline=bool(self.user_metadata.get('topline', False)),
                          raise_on_error=self._inf_config.raise_on_error)

        script_detection = len(rets['cls_map']['baselines']) > 1

        # create objects and assign IDs
        blls = []
        _shp_regs = {}
        for reg_type, rgs in regions.items():
            for reg in rgs:
                _shp_regs[reg.id] = geom.Polygon(reg.boundary)

        for line in lines:
            line_regs = []
            for reg_id, reg in _shp_regs.items():
                line_ls = geom.LineString(line['baseline'])
                if is_in_region(line_ls, reg):
                    line_regs.append(reg_id)
            blls.append(BaselineLine(id=f'_{uuid.uuid4()}', baseline=line['baseline'], boundary=line['boundary'], tags=line['tags'], regions=line_regs))

        if blls:
            all_regions = [reg for rgs in regions.values() for reg in rgs]
            ro = self._inf_config.baseline_ro_fn(lines=blls,
                                                 regions=all_regions,
                                                 text_direction=self._inf_config.text_direction[-2:])
            blls = [blls[idx] for idx in ro]

        return Segmentation(text_direction=self._inf_config.text_direction,
                            imagename=getattr(im, 'filename', None),
                            type='baselines',
                            lines=blls,
                            regions=regions,
                            script_detection=script_detection,
                            line_orders=[])

    def _compute_segmentation_map(self, im: PIL.Image.Image) -> dict[str, Any]:
        """
        Args:
            im: Input image
            model: A TorchVGSLModel containing a segmentation model.
            device: The target device to run the neural network on.
            autocast: Runs the model with automatic mixed precision

        Returns:
            A dictionary containing the heatmaps ('heatmap', torch.Tensor), class
            map ('cls_map', dict[str, dict[str, int]]), the bounding regions for
            polygonization purposes ('bounding_regions', list[str]), the scale
            between the input image and the network output ('scale', float), and
            the scaled input image to the network ('scal_im', PIL.Image.Image).
        """
        batch, channels, height, width = self.input
        padding = self._inf_config.input_padding
        # expand padding to 4-tuple (left, right, top, bottom)
        if isinstance(padding, int):
            padding = (padding,) * 4
        elif len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])

        transforms = ImageInputTransforms(batch, height, width, channels, padding, valid_norm=False, dtype=self._m_dtype)
        tf_idx, _ = next(filter(lambda x: isinstance(x[1], v2.PILToTensor), enumerate(transforms.transforms)))
        res_tf = tf.Compose(transforms.transforms[:tf_idx])
        scal_im = np.array(res_tf(im).convert('L'))

        tensor_im = transforms(im)

        logger.debug('Running network forward pass')
        o, _ = self.nn(tensor_im.unsqueeze(0))

        logger.debug('Upsampling network output')
        o = F.interpolate(o, size=scal_im.shape)
        o = torch.sigmoid(o)
        # remove padding
        padding = [pad if pad else None for pad in padding]
        padding[1] = -padding[1] if padding[1] else None
        padding[3] = -padding[3] if padding[3] else None
        o = o[:, :, padding[2]:padding[3], padding[0]:padding[1]]
        scal_im = scal_im[padding[2]:padding[3], padding[0]:padding[1]]

        o = o.squeeze().cpu().float().numpy()
        scale = np.divide(im.size, o.shape[:0:-1])

        return {'heatmap': o,
                'cls_map': self.user_metadata['class_mapping'],
                'bounding_regions': self.user_metadata.get('bounding_regions', None),
                'scale': scale,
                'scal_im': scal_im}
