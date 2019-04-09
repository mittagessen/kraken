# -*- coding: utf-8 -*-
#
# Copyright 2015 Benjamin Kiessling
#           2014 Thomas M. Breuel
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
kraken.binarization
~~~~~~~~~~~~~~~~~~~

An adaptive binarization algorithm.
"""
import warnings
import logging
import numpy as np

from PIL import Image
from kraken.lib.util import pil2array, array2pil, is_bitonal, get_im_str
from scipy.ndimage import filters, interpolation, morphology, affine_transform

from kraken.lib.exceptions import KrakenInputException

__all__ = ['nlbin']

logger = logging.getLogger(__name__)


def nlbin(im: Image.Image,
          threshold: float = 0.5,
          zoom: float = 0.5,
          escale: float = 1.0,
          border: float = 0.1,
          perc: int = 80,
          range: int = 20,
          low: int = 5,
          high: int = 90) -> Image:
    """
    Performs binarization using non-linear processing.

    Args:
        im (PIL.Image.Image):
        threshold (float):
        zoom (float): Zoom for background page estimation
        escale (float): Scale for estimating a mask over the text region
        border (float): Ignore this much of the border
        perc (int): Percentage for filters
        range (int): Range for filters
        low (int): Percentile for black estimation
        high (int): Percentile for white estimation

    Returns:
        PIL.Image containing the binarized image

    Raises:
        KrakenInputException when trying to binarize an empty image.
    """
    im_str = get_im_str(im)
    logger.info('Binarizing {}'.format(im_str))
    if is_bitonal(im):
        logger.info('Skipping binarization because {} is bitonal.'.format(im_str))
        return im
    # convert to grayscale first
    logger.debug('Converting {} to grayscale'.format(im_str))
    im = im.convert('L')
    raw = pil2array(im)
    logger.debug('Scaling and normalizing')
    # rescale image to between -1 or 0 and 1
    raw = raw/np.float(np.iinfo(raw.dtype).max)
    # perform image normalization
    if np.amax(raw) == np.amin(raw):
        logger.warning('Trying to binarize empty image {}'.format(im_str))
        raise KrakenInputException('Image is empty')
    image = raw-np.amin(raw)
    image /= np.amax(image)

    logger.debug('Interpolation and percentile filtering')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        m = interpolation.zoom(image, zoom)
        m = filters.percentile_filter(m, perc, size=(range, 2))
        m = filters.percentile_filter(m, perc, size=(2, range))
        mh, mw = m.shape
        oh, ow = image.shape
        scale = np.diag([mh * 1.0/oh, mw * 1.0/ow])
        m = affine_transform(m, scale, output_shape=image.shape)
    w, h = np.minimum(np.array(image.shape), np.array(m.shape))
    flat = np.clip(image[:w, :h]-m[:w, :h]+1, 0, 1)

    # estimate low and high thresholds
    d0, d1 = flat.shape
    o0, o1 = int(border*d0), int(border*d1)
    est = flat[o0:d0-o0, o1:d1-o1]
    logger.debug('Threshold estimates {}'.format(est))
    # by default, we use only regions that contain
    # significant variance; this makes the percentile
    # based low and high estimates more reliable
    logger.debug('Refine estimates')
    v = est-filters.gaussian_filter(est, escale*20.0)
    v = filters.gaussian_filter(v**2, escale*20.0)**0.5
    v = (v > 0.3*np.amax(v))
    v = morphology.binary_dilation(v, structure=np.ones((int(escale * 50), 1)))
    v = morphology.binary_dilation(v, structure=np.ones((1, int(escale * 50))))
    est = est[v]
    lo = np.percentile(est.ravel(), low)
    hi = np.percentile(est.ravel(), high)
    flat -= lo
    flat /= (hi-lo)
    flat = np.clip(flat, 0, 1)
    logger.debug('Thresholding at {}'.format(threshold))
    bin = np.array(255*(flat > threshold), 'B')
    return array2pil(bin)
