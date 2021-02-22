# -*- coding: utf-8 -*-
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
~~~~~~~~~~~~~~

Trainable baseline layout analysis tools for kraken
"""

import torch
import logging
import numpy as np
import pkg_resources
import shapely.geometry as geom
import torch.nn.functional as F
import torchvision.transforms as tf

from functools import partial
from typing import Optional, Dict, Callable, Union, List

from scipy.ndimage.filters import gaussian_filter
from skimage.filters import sobel

from kraken.lib import vgsl, dataset
from kraken.lib.util import pil2array, is_bitonal, get_im_str
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.segmentation import (polygonal_reading_order,
                                     vectorize_lines, vectorize_regions,
                                     scale_polygonal_lines,
                                     calculate_polygonal_environment,
                                     scale_regions)

__all__ = ['segment']

logger = logging.getLogger(__name__)

def compute_segmentation_map(im,
                             mask: Optional[np.array] = None,
                             model=None,
                             device: str = 'cpu'):
    """

    """
    im_str = get_im_str(im)
    logger.info(f'Segmenting {im_str}')

    if model.input[1] == 1 and model.one_channel_mode == '1' and not is_bitonal(im):
        logger.warning('Running binary model on non-binary input image '
                       '(mode {}). This will result in severely degraded '
                       'performance'.format(im.mode))

    model.eval()
    model.to(device)

    if mask:
        if mask.mode != '1' and not is_bitonal(mask):
            logger.error('Mask is not bitonal')
            raise KrakenInputException('Mask is not bitonal')
        mask = mask.convert('1')
        if mask.size != im.size:
            logger.error('Mask size {mask.size} doesn\'t match image size {im.size}')
            raise KrakenInputException('Mask size {mask.size} doesn\'t match image size {im.size}')
        logger.info('Masking enabled in segmenter.')
        mask = pil2array(mask)

    batch, channels, height, width = model.input
    transforms = dataset.generate_input_transforms(batch, height, width, channels, 0, valid_norm=False)
    res_tf = tf.Compose(transforms.transforms[:3])
    scal_im = res_tf(im).convert('L')

    with torch.no_grad():
        logger.debug('Running network forward pass')
        o, _ = model.nn(transforms(im).unsqueeze(0).to(device))
    logger.debug('Upsampling network output')
    o = F.interpolate(o, size=scal_im.size[::-1])
    o = o.squeeze().cpu().numpy()
    scale = np.divide(im.size, o.shape[:0:-1])
    bounding_regions = model.user_metadata['bounding_regions'] if 'bounding_regions' in model.user_metadata else None
    return {'heatmap': o,
            'cls_map': model.user_metadata['class_mapping'],
            'bounding_regions': bounding_regions,
            'scale': scale,
            'scal_im': scal_im}


def vec_regions(heatmap: torch.Tensor, cls_map: Dict, scale: float, **kwargs):
    """
    Computes regions from a stack of heatmaps, a class mapping, and scaling
    factor.
    """
    logger.info('Vectorizing regions')
    regions = {}
    for region_type, idx in cls_map['regions'].items():
        logger.debug(f'Vectorizing regions of type {region_type}')
        regions[region_type] = vectorize_regions(heatmap[idx])
    for reg_id, regs in regions.items():
        regions[reg_id] = scale_regions(regs, scale)
    return regions


def vec_lines(heatmap: torch.Tensor,
              cls_map: Dict,
              scale: float,
              text_direction: str = 'horizontal-lr',
              reading_order_fn: Callable = polygonal_reading_order,
              regions: Dict = None,
              scal_im = None,
              suppl_obj = None,
              **kwargs):
    """
    Computes lines from a stack of heatmaps, a class mapping, and scaling
    factor.
    """
    st_sep = cls_map['aux']['_start_separator']
    end_sep = cls_map['aux']['_end_separator']

    logger.info('Vectorizing baselines')
    baselines = []
    for bl_type, idx in cls_map['baselines'].items():
        logger.debug(f'Vectorizing lines of type {bl_type}')
        baselines.extend([(bl_type,x) for x in vectorize_lines(heatmap[(st_sep, end_sep, idx), :, :])])
    logger.debug('Polygonizing lines')

    im_feats = gaussian_filter(sobel(scal_im), 0.5)

    lines = []
    reg_pols = [geom.Polygon(x) for x in regions]
    for bl_idx in range(len(baselines)):
        bl = baselines[bl_idx]
        mid_point = geom.LineString(bl[1]).interpolate(0.5, normalized=True)

        suppl_obj = [x[1] for x in baselines[:bl_idx] + baselines[bl_idx+1:]]
        for reg_idx, reg_pol in enumerate(reg_pols):
            if reg_pol.contains(mid_point):
                suppl_obj.append(regions[reg_idx])

        pol = calculate_polygonal_environment(baselines=[bl[1]], im_feats=im_feats, suppl_obj=suppl_obj)
        if pol[0] is not None:
            lines.append((bl[0], bl[1], pol[0]))

    logger.debug('Scaling vectorized lines')
    sc = scale_polygonal_lines([x[1:] for x in lines], scale)
    lines = list(zip([x[0] for x in lines], [x[0] for x in sc], [x[1] for x in sc]))
    logger.debug('Reordering baselines')
    lines = reading_order_fn(lines=lines, regions=regions, text_direction=text_direction[-2:])
    return [{'script': bl_type, 'baseline': bl, 'boundary': pl} for bl_type, bl, pl in lines]


def segment(im,
            text_direction: str = 'horizontal-lr',
            mask: Optional[np.array] = None,
            reading_order_fn: Callable = polygonal_reading_order,
            model: Union[List[vgsl.TorchVGSLModel], vgsl.TorchVGSLModel] = None,
            device: str = 'cpu'):
    """
    Segments a page into text lines using the baseline segmenter.

    Segments a page into text lines and returns the polyline formed by each
    baseline and their estimated environment.

    Args:
        im (PIL.Image): An RGB image.
        text_direction (str): Ignored by the segmenter but kept for
                              serialization.
        mask (PIL.Image): A bi-level mask image of the same size as `im` where
                          0-valued regions are ignored for segmentation
                          purposes. Disables column detection.
        reading_order_fn (function): Function to determine the reading order.
                                     Has to accept a list of tuples (baselines,
                                     polygon) and a text direction (`lr` or
                                     `rl`).
        model (vgsl.TorchVGSLModel or list): One or more TorchVGSLModel
                                             containing a segmentation model.
                                             If none is given a default model
                                             will be loaded.
        device (str or torch.Device): The target device to run the neural
                                      network on.

    Returns:
        {'text_direction': '$dir',
         'type': 'baseline',
         'lines': [
            {'baseline': [[x0, y0], [x1, y1], ..., [x_n, y_n]], 'boundary': [[x0, y0, x1, y1], ... [x_m, y_m]]},
            {'baseline': [[x0, ...]], 'boundary': [[x0, ...]]}
          ]
          'regions': [
            {'region': [[x0, y0], [x1, y1], ..., [x_n, y_n]], 'type': 'image'},
            {'region': [[x0, ...]], 'type': 'text'}
          ]
        }: A dictionary containing the text direction and under the key 'lines'
        a list of reading order sorted baselines (polylines) and their
        respective polygonal boundaries. The last and first point of each
        boundary polygon is connected.

    Raises:
        KrakenInputException if the input image is not binarized or the text
        direction is invalid.
    """
    if model is None:
        logger.info('No segmentation model given. Loading default model.')
        model = vgsl.TorchVGSLModel.load_model(pkg_resources.resource_filename(__name__, 'blla.mlmodel'))

    if isinstance(model, vgsl.TorchVGSLModel):
        model = [model]

    im_str = get_im_str(im)
    logger.info(f'Segmenting {im_str}')

    for net in model:
        rets = compute_segmentation_map(im, mask, net, device)
        regions = vec_regions(**rets)
        # flatten regions for line ordering/fetch bounding regions
        line_regs = []
        suppl_obj = []
        for cls, regs in regions.items():
            line_regs.extend(regs)
            if rets['bounding_regions'] is not None and cls in rets['bounding_regions']:
                suppl_obj.extend(regs)
        lines = vec_lines(**rets,
                          regions=line_regs,
                          reading_order_fn=reading_order_fn,
                          text_direction=text_direction,
                          suppl_obj=suppl_obj)

    if len(rets['cls_map']['baselines']) > 1:
        script_detection = True
    else:
        script_detection = False

    return {'text_direction': text_direction,
            'type': 'baselines',
            'lines': lines,
            'regions': regions,
            'script_detection': script_detection}
