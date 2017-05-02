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

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import warnings
import numpy as np

from kraken.lib.util import pil2array, array2pil
from kraken.lib.exceptions import KrakenInputException
from scipy.ndimage import filters, interpolation, morphology


def is_bitonal(im):
    """
    Tests a PIL.Image for bitonality.

    Args:
        im (PIL.Image): Image to test

    Returns:
        True if the image contains only two different color values. False
        otherwise.
    """
    if im.getcolors(2):
        return True
    else:
        return False


def nlbin(im, threshold=0.5, zoom=0.5, escale=1.0, border=0.1, perc=80,
          range=20, low=5, high=90):
    """
    Performs binarization using non-linear processing.

    Args:
        im (PIL.Image):
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
    """
    if im.mode == '1':
        return im
    raw = pil2array(im)
    # rescale image to between -1 or 0 and 1
    raw = raw/np.float(np.iinfo(raw.dtype).max)
    if raw.ndim == 3:
        raw = np.mean(raw, 2)
    # perform image normalization
    if np.amax(raw) == np.amin(raw):
        raise KrakenInputException('Image is empty')
    image = raw-np.amin(raw)
    image /= np.amax(image)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        m = interpolation.zoom(image, zoom)
        m = filters.percentile_filter(m, perc, size=(range, 2))
        m = filters.percentile_filter(m, perc, size=(2, range))
        m = interpolation.zoom(m, 1.0/zoom)
    w, h = np.minimum(np.array(image.shape), np.array(m.shape))
    flat = np.clip(image[:w, :h]-m[:w, :h]+1, 0, 1)

    # estimate low and high thresholds
    d0, d1 = flat.shape
    o0, o1 = int(border*d0), int(border*d1)
    est = flat[o0:d0-o0, o1:d1-o1]
    # by default, we use only regions that contain
    # significant variance; this makes the percentile
    # based low and high estimates more reliable
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
    bin = np.array(255*(flat > threshold), 'B')
    return array2pil(bin)
