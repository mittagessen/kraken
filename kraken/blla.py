#
# Copyright 2019 Benjamin Kiessling
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
kraken.blla
~~~~~~~~~~~

Trainable layout analysis tools for kraken for line and region detection. The
line recognizer uses the baseline paradigm.
"""

import logging
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import importlib_resources
import numpy as np
import PIL
import shapely.geometry as geom
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel

from kraken.containers import BaselineLine, Region, Segmentation
from kraken.lib import dataset, vgsl
from kraken.lib.exceptions import (KrakenInputException,
                                   KrakenInvalidModelException)
from kraken.lib.segmentation import (calculate_polygonal_environment,
                                     is_in_region, neural_reading_order,
                                     polygonal_reading_order,
                                     scale_polygonal_lines, scale_regions,
                                     vectorize_lines, vectorize_regions)
from kraken.lib.util import get_im_str, is_bitonal

__all__ = ['segment']

logger = logging.getLogger(__name__)


def compute_segmentation_map(im: PIL.Image.Image,
                             mask: Optional[np.ndarray] = None,
                             model: vgsl.TorchVGSLModel = None,
                             device: str = 'cpu',
                             autocast: bool = False) -> Dict[str, Any]:
    """
    Args:
        im: Input image
        mask: A bi-level mask array of the same size as `im` where 0-valued
              regions are ignored for segmentation purposes. Disables column
              detection.
        model: A TorchVGSLModel containing a segmentation model.
        device: The target device to run the neural network on.
        autocast: Runs the model with automatic mixed precision

    Returns:
        A dictionary containing the heatmaps ('heatmap', torch.Tensor), class
        map ('cls_map', Dict[str, Dict[str, int]]), the bounding regions for
        polygonization purposes ('bounding_regions', List[str]), the scale
        between the input image and the network output ('scale', float), and
        the scaled input image to the network ('scal_im', PIL.Image.Image).

    Raises:
        KrakenInputException: When given an invalid mask.
    """

    if model.input[1] == 1 and model.one_channel_mode == '1' and not is_bitonal(im):
        logger.warning('Running binary model on non-binary input image '
                       '(mode {}). This will result in severely degraded '
                       'performance'.format(im.mode))

    model.eval()
    model.to(device)

    batch, channels, height, width = model.input
    padding = model.user_metadata['hyper_params']['padding'] if 'padding' in model.user_metadata['hyper_params'] else (0, 0)
    # expand padding to 4-tuple (left, right, top, bottom)
    if isinstance(padding, int):
        padding = (padding,) * 4
    elif len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])

    transforms = dataset.ImageInputTransforms(batch, height, width, channels, padding, valid_norm=False)
    tf_idx, _ = next(filter(lambda x: isinstance(x[1], tf.ToTensor), enumerate(transforms.transforms)))
    res_tf = tf.Compose(transforms.transforms[:tf_idx])
    scal_im = np.array(res_tf(im).convert('L'))

    tensor_im = transforms(im)
    if mask:
        if mask.mode != '1' and not is_bitonal(mask):
            logger.error('Mask is not bitonal')
            raise KrakenInputException('Mask is not bitonal')
        mask = mask.convert('1')
        if mask.size != im.size:
            logger.error('Mask size {mask.size} doesn\'t match image size {im.size}')
            raise KrakenInputException('Mask size {mask.size} doesn\'t match image size {im.size}')
        logger.info('Masking enabled in segmenter.')
        tensor_im[~transforms(mask).bool()] = 0

    with torch.autocast(device_type=device.split(":")[0], enabled=autocast):
        with torch.no_grad():
            logger.debug('Running network forward pass')
            o, _ = model.nn(tensor_im.unsqueeze(0).to(device))

    logger.debug('Upsampling network output')
    o = F.interpolate(o, size=scal_im.shape)
    # remove padding
    padding = [pad if pad else None for pad in padding]
    padding[1] = -padding[1] if padding[1] else None
    padding[3] = -padding[3] if padding[3] else None
    o = o[:, :, padding[2]:padding[3], padding[0]:padding[1]]
    scal_im = scal_im[padding[2]:padding[3], padding[0]:padding[1]]

    o = o.squeeze().cpu().float().numpy()
    scale = np.divide(im.size, o.shape[:0:-1])

    bounding_regions = model.user_metadata['bounding_regions'] if 'bounding_regions' in model.user_metadata else None
    return {'heatmap': o,
            'cls_map': model.user_metadata['class_mapping'],
            'bounding_regions': bounding_regions,
            'scale': scale,
            'scal_im': scal_im}


def vec_regions(heatmap: torch.Tensor, cls_map: Dict, scale: float, **kwargs) -> Dict[str, List[Region]]:
    """
    Computes regions from a stack of heatmaps, a class mapping, and scaling
    factor.

    Args:
        heatmap: A stack of heatmaps of shape `NxHxW` output from the network.
        cls_map: Dictionary mapping string identifiers to indices on the stack
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
        regions[reg_type] = [Region(id=str(uuid.uuid4()), boundary=x, tags={'type': reg_type}) for x in scale_regions(regs, scale)]
    return regions


def vec_lines(heatmap: torch.Tensor,
              cls_map: Dict[str, Dict[str, int]],
              scale: float,
              text_direction: str = 'horizontal-lr',
              regions: List[np.ndarray] = None,
              scal_im: np.ndarray = None,
              suppl_obj: List[np.ndarray] = None,
              topline: Optional[bool] = False,
              raise_on_error: bool = False,
              fixed_offset_polygons: bool = False,
              **kwargs) -> List[Dict[str, Any]]:
    r"""
    Computes lines from a stack of heatmaps, a class mapping, and scaling
    factor.

    Args:
        heatmap: A stack of heatmaps of shape `NxHxW` output from the network.
        cls_map: Dictionary mapping string identifiers to indices on the stack
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
        suppl_obj = [x[1] for x in baselines[:bl_idx] + baselines[bl_idx+1:]]
        for reg_idx, reg_pol in enumerate(reg_pols):
            if is_in_region(bl_ls, reg_pol):
                suppl_obj.append(regions[reg_idx])
        pol = calculate_polygonal_environment(baselines=[bl[1]],
                                              im_feats=im_feats,
                                              suppl_obj=suppl_obj,
                                              topline=topline,
                                              raise_on_error=raise_on_error,
                                              fixed_offset_polygons=fixed_offset_polygons)
        if pol[0] is not None:
            lines.append((bl[0], bl[1], pol[0]))

    logger.debug('Scaling vectorized lines')
    sc = scale_polygonal_lines([x[1:] for x in lines], scale)

    lines = list(zip([x[0] for x in lines], [x[0] for x in sc], [x[1] for x in sc]))
    return [{'tags': {'type': bl_type}, 'baseline': bl, 'boundary': pl} for bl_type, bl, pl in lines]


def segment(im: PIL.Image.Image,
            text_direction: Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'] = 'horizontal-lr',
            mask: Optional[np.ndarray] = None,
            reading_order_fn: Callable = polygonal_reading_order,
            model: Union[List[vgsl.TorchVGSLModel], vgsl.TorchVGSLModel] = None,
            device: str = 'cpu',
            raise_on_error: bool = False,
            autocast: bool = False,
            fixed_offset_polygons: bool = False) -> Segmentation:
    r"""
    Segments a page into text lines using the baseline segmenter.

    Segments a page into text lines and returns the polyline formed by each
    baseline and their estimated environment.

    Args:
        im: Input image. The mode can generally be anything but it is possible
            to supply a binarized-input-only model which requires accordingly
            treated images.
        text_direction: Passed-through value for serialization.serialize.
        mask: A bi-level mask image of the same size as `im` where 0-valued
              regions are ignored for segmentation purposes. Disables column
              detection.
        reading_order_fn: Function to determine the reading order.  Has to
                          accept a list of tuples (baselines, polygon) and a
                          text direction (`lr` or `rl`).
        model: One or more TorchVGSLModel containing a segmentation model. If
               none is given a default model will be loaded.
        device: The target device to run the neural network on.
        raise_on_error: Raises error instead of logging them when they are
                        not-blocking
        autocast: Runs the model with automatic mixed precision

    Returns:
        A :class:`kraken.containers.Segmentation` class containing reading
        order sorted baselines (polylines) and their respective polygonal
        boundaries as :class:`kraken.containers.BaselineLine` records. The
        last and first point of each boundary polygon are connected.

    Raises:
        KrakenInvalidModelException: if the given model is not a valid
                                     segmentation model.
        KrakenInputException: if the mask is not bitonal or does not match the
                              image size.

    Notes:
        Multi-model operation is most useful for combining one or more region
        detection models and one text line model. Detected lines from all
        models are simply combined without any merging or duplicate detection
        so the chance of the same line appearing multiple times in the output
        are high. In addition, neural reading order determination is disabled
        when more than one model outputs lines.
    """
    if model is None:
        logger.info('No segmentation model given. Loading default model.')
        model = vgsl.TorchVGSLModel.load_model(importlib_resources.files(__name__).joinpath('blla.mlmodel'))

    if isinstance(model, vgsl.TorchVGSLModel):
        model = [model]

    for nn in model:
        if nn.model_type != 'segmentation':
            raise KrakenInvalidModelException(f'Invalid model type {nn.model_type} for {nn}')
        if 'class_mapping' not in nn.user_metadata:
            raise KrakenInvalidModelException(f'Segmentation model {nn} does not contain valid class mapping')

    im_str = get_im_str(im)
    logger.info(f'Segmenting {im_str}')

    lines = []
    order = None
    regions = {}
    multi_lines = False
    # flag to indicate that multiple models produced line output -> disable
    # neural reading order
    for net in model:
        if 'topline' in net.user_metadata:
            loc = {None: 'center',
                   True: 'top',
                   False: 'bottom'}[net.user_metadata['topline']]
            logger.debug(f'Baseline location: {loc}')
        rets = compute_segmentation_map(im, mask, net, device, autocast=autocast)
        _regions = vec_regions(**rets)
        for reg_key, reg_val in vec_regions(**rets).items():
            if reg_key not in regions:
                regions[reg_key] = []
            regions[reg_key].extend(reg_val)

        # flatten regions for line ordering/fetch bounding regions
        line_regs = []
        suppl_obj = []
        for cls, regs in _regions.items():
            line_regs.extend(regs)
            if rets['bounding_regions'] is not None and cls in rets['bounding_regions']:
                suppl_obj.extend(regs)
        # convert back to net scale
        suppl_obj = scale_regions([x.boundary for x in suppl_obj], 1/rets['scale'])
        line_regs = scale_regions([x.boundary for x in line_regs], 1/rets['scale'])

        _lines = vec_lines(**rets,
                           regions=line_regs,
                           text_direction=text_direction,
                           suppl_obj=suppl_obj,
                           topline=net.user_metadata['topline'] if 'topline' in net.user_metadata else False,
                           raise_on_error=raise_on_error,
                           fixed_offset_polygons=fixed_offset_polygons)

        if 'ro_model' in net.aux_layers:
            logger.info(f'Using reading order model found in segmentation model {net}.')
            _order = neural_reading_order(lines=_lines,
                                          regions=_regions,
                                          text_direction=text_direction[-2:],
                                          model=net.aux_layers['ro_model'],
                                          im_size=im.size,
                                          class_mapping=net.user_metadata['ro_class_mapping'])
        else:
            _order = None

        if _lines and lines or multi_lines:
            multi_lines = True
            order = None
            logger.warning('Multiple models produced line output. This is '
                           'likely unintended. Suppressing neural reading '
                           'order.')
        else:
            order = _order

        lines.extend(_lines)

    if len(rets['cls_map']['baselines']) > 1:
        script_detection = True
    else:
        script_detection = False

    # create objects and assign IDs
    blls = []
    _shp_regs = {}
    for reg_type, rgs in regions.items():
        for reg in rgs:
            _shp_regs[reg.id] = geom.Polygon(reg.boundary)

    # reorder lines
    logger.debug(f'Reordering baselines with main RO function {reading_order_fn}.')
    basic_lo = reading_order_fn(lines=lines, regions=_shp_regs.values(), text_direction=text_direction[-2:])
    lines = [lines[idx] for idx in basic_lo]

    for line in lines:
        line_regs = []
        for reg_id, reg in _shp_regs.items():
            line_ls = geom.LineString(line['baseline'])
            if is_in_region(line_ls, reg):
                line_regs.append(reg_id)
        blls.append(BaselineLine(id=str(uuid.uuid4()), baseline=line['baseline'], boundary=line['boundary'], tags=line['tags'], regions=line_regs))

    return Segmentation(text_direction=text_direction,
                        imagename=getattr(im, 'filename', None),
                        type='baselines',
                        lines=blls,
                        regions=regions,
                        script_detection=script_detection,
                        line_orders=[order] if order else [])
