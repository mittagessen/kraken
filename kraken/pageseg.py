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
kraken.pageseg
~~~~~~~~~~~~~~

Layout analysis and script detection methods.
"""
from itertools import groupby

import json
import logging
import numpy as np
import pkg_resources

from typing import Tuple, Sequence, List
from scipy.ndimage.filters import (gaussian_filter, uniform_filter,
                                   maximum_filter)

from kraken.lib import models
from kraken.lib import morph, sl
from kraken.lib.util import pil2array, is_bitonal, get_im_str
from kraken.lib.exceptions import KrakenInputException

from kraken.rpred import rpred
from kraken.serialization import max_bbox

__all__ = ['segment', 'detect_scripts']

logger = logging.getLogger(__name__)


class record(object):
    """
    Simple dict-like object.
    """
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.label = 0  # type: int
        self.bounds = []  # type: List
        self.mask = None  # type: np.array


def find(condition):
    "Return the indices where ravel(condition) is true"
    res, = np.nonzero(np.ravel(condition))
    return res


def binary_objects(binary: np.array) -> np.array:
    """
    Labels features in an array and segments them into objects.
    """
    labels, _ = morph.label(binary)
    objects = morph.find_objects(labels)
    return objects


def estimate_scale(binary: np.array) -> float:
    """
    Estimates image scale based on number of connected components.
    """
    objects = binary_objects(binary)
    bysize = sorted(objects, key=sl.area)
    scalemap = np.zeros(binary.shape)
    for o in bysize:
        if np.amax(scalemap[o]) > 0:
            continue
        scalemap[o] = sl.area(o)**0.5
    scale = np.median(scalemap[(scalemap > 3) & (scalemap < 100)])
    return scale


def compute_boxmap(binary: np.array, scale: float, threshold: Tuple[float, int] = (.5, 4), dtype: str = 'i') -> np.array:
    """
    Returns grapheme cluster-like boxes based on connected components.
    """
    objects = binary_objects(binary)
    bysize = sorted(objects, key=sl.area)
    boxmap = np.zeros(binary.shape, dtype)
    for o in bysize:
        if sl.area(o)**.5 < threshold[0]*scale:
            continue
        if sl.area(o)**.5 > threshold[1]*scale:
            continue
        boxmap[o] = 1
    return boxmap


def compute_lines(segmentation, scale):
    """Given a line segmentation map, computes a list
    of tuples consisting of 2D slices and masked images."""
    logger.debug('Convert segmentation to lines')
    lobjects = morph.find_objects(segmentation)
    lines = []
    for i, o in enumerate(lobjects):
        if o is None:
            continue
        if sl.dim1(o) < 2*scale or sl.dim0(o) < scale:
            continue
        mask = (segmentation[o] == i+1)
        if np.amax(mask) == 0:
            continue
        result = record()
        result.label = i+1
        result.bounds = o
        result.mask = mask
        lines.append(result)
    return lines


def reading_order(lines: Sequence, text_direction: str = 'lr') -> List:
    """Given the list of lines (a list of 2D slices), computes
    the partial reading order.  The output is a binary 2D array
    such that order[i,j] is true if line i comes before line j
    in reading order."""

    logger.info('Compute reading order on {} lines in {} direction'.format(len(lines), text_direction))

    order = np.zeros((len(lines), len(lines)), 'B')

    def _x_overlaps(u, v):
        return u[1].start < v[1].stop and u[1].stop > v[1].start

    def _above(u, v):
        return u[0].start < v[0].start

    def _left_of(u, v):
        return u[1].stop < v[1].start

    def _separates(w, u, v):
        if w[0].stop < min(u[0].start, v[0].start):
            return 0
        if w[0].start > max(u[0].stop, v[0].stop):
            return 0
        if w[1].start < u[1].stop and w[1].stop > v[1].start:
            return 1
        return 0

    if text_direction == 'rl':
        def horizontal_order(u, v):
            return not _left_of(u, v)
    else:
        horizontal_order = _left_of

    for i, u in enumerate(lines):
        for j, v in enumerate(lines):
            if _x_overlaps(u, v):
                if _above(u, v):
                    order[i, j] = 1
            else:
                if [w for w in lines if _separates(w, u, v)] == []:
                    if horizontal_order(u, v):
                        order[i, j] = 1
    return order


def topsort(order: np.array) -> np.array:
    """Given a binary array defining a partial order (o[i,j]==True means i<j),
    compute a topological sort.  This is a quick and dirty implementation
    that works for up to a few thousand elements."""
    logger.info(u'Perform topological sort on partially ordered lines')
    n = len(order)
    visited = np.zeros(n)
    L = []

    def _visit(k):
        if visited[k]:
            return
        visited[k] = 1
        a, = np.nonzero(np.ravel(order[:, k]))
        for l in a:
            _visit(l)
        L.append(k)

    for k in range(n):
        _visit(k)
    return L


def compute_separators_morph(binary: np.array, scale: float, sepwiden: int = 10, maxcolseps: int = 2) -> np.array:
    """Finds vertical black lines corresponding to column separators."""
    logger.debug(u'Finding vertical black column lines')
    d0 = int(max(5, scale/4))
    d1 = int(max(5, scale)) + sepwiden
    thick = morph.r_dilation(binary, (d0, d1))
    vert = morph.rb_opening(thick, (10*scale, 1))
    vert = morph.r_erosion(vert, (d0//2, sepwiden))
    vert = morph.select_regions(vert, sl.dim1, min=3, nbest=2*maxcolseps)
    vert = morph.select_regions(vert, sl.dim0, min=20*scale, nbest=maxcolseps)
    return vert


def compute_colseps_conv(binary: np.array, scale: float = 1.0, minheight: int = 10, maxcolseps: int = 2) -> np.array:
    """Find column separators by convolution and thresholding.

    Args:
        binary (numpy.array):
        scale (float):
        minheight (int):
        maxcolseps (int):

    Returns:
        Separators
    """
    logger.debug('Finding column separators')
    # find vertical whitespace by thresholding
    smoothed = gaussian_filter(1.0*binary, (scale, scale*0.5))
    smoothed = uniform_filter(smoothed, (5.0*scale, 1))
    thresh = (smoothed < np.amax(smoothed)*0.1)
    # find column edges by filtering
    grad = gaussian_filter(1.0*binary, (scale, scale*0.5), order=(0, 1))
    grad = uniform_filter(grad, (10.0*scale, 1))
    grad = (grad > 0.5*np.amax(grad))
    # combine edges and whitespace
    seps = np.minimum(thresh, maximum_filter(grad, (int(scale), int(5*scale))))
    seps = maximum_filter(seps, (int(2*scale), 1))
    # select only the biggest column separators
    seps = morph.select_regions(seps, sl.dim0, min=minheight*scale,
                                nbest=maxcolseps)
    return seps


def compute_black_colseps(binary: np.array, scale: float, maxcolseps: int) -> Tuple[np.array, np.array]:
    """
    Computes column separators from vertical black lines.

    Args:
        binary (numpy.array): Numpy array of the binary image
        scale (float):

    Returns:
        (colseps, binary):
    """
    logger.debug('Extract vertical black column separators from lines')
    seps = compute_separators_morph(binary, scale, maxcolseps)
    colseps = np.maximum(compute_colseps_conv(binary, scale, maxcolseps), seps)
    binary = np.minimum(binary, 1-seps)
    return colseps, binary


def compute_white_colseps(binary, scale, maxcolseps):
    """
    Computes column separators either from vertical black lines or whitespace.

    Args:
        binary (numpy.array): Numpy array of the binary image
        scale (float):

    Returns:
        colseps:
    """
    return compute_colseps_conv(binary, scale, maxcolseps)


def norm_max(v):
    """
    Normalizes the input array by maximum value.
    """
    return v/np.amax(v)


def compute_gradmaps(binary, scale, gauss=False):
    """
    Use gradient filtering to find baselines

    Args:
        binary (numpy.array):
        scale (float):
        gauss (bool): Use gaussian instead of uniform filtering

    Returns:
        (bottom, top, boxmap)
    """
    # use gradient filtering to find baselines
    logger.debug('Computing gradient maps')
    boxmap = compute_boxmap(binary, scale)
    cleaned = boxmap*binary
    if gauss:
        grad = gaussian_filter(1.0*cleaned, (0.3*scale, 6*scale), order=(1, 0))
    else:
        grad = gaussian_filter(1.0*cleaned, (max(4, 0.3*scale),
                                             scale), order=(1, 0))
        grad = uniform_filter(grad, (1, 6*scale))
    bottom = norm_max((grad < 0)*(-grad))
    top = norm_max((grad > 0)*grad)
    return bottom, top, boxmap


def compute_line_seeds(binary, bottom, top, colseps, scale, threshold=0.2):
    """
    Base on gradient maps, computes candidates for baselines and xheights.
    Then, it marks the regions between the two as a line seed.
    """
    logger.debug(u'Finding line seeds')
    vrange = int(scale)
    bmarked = maximum_filter(bottom == maximum_filter(bottom, (vrange, 0)),
                             (2, 2))
    bmarked = bmarked * (bottom > threshold*np.amax(bottom)*threshold)*(1-colseps)
    tmarked = maximum_filter(top == maximum_filter(top, (vrange, 0)), (2, 2))
    tmarked = tmarked * (top > threshold*np.amax(top)*threshold/2)*(1-colseps)
    tmarked = maximum_filter(tmarked, (1, 20))
    seeds = np.zeros(binary.shape, 'i')
    delta = max(3, int(scale/2))
    for x in range(bmarked.shape[1]):
        transitions = sorted([(y, 1) for y in find(bmarked[:, x])] +
                             [(y, 0) for y in find(tmarked[:, x])])[::-1]
        transitions += [(0, 0)]
        for l in range(len(transitions)-1):
            y0, s0 = transitions[l]
            if s0 == 0:
                continue
            seeds[y0-delta:y0, x] = 1
            y1, s1 = transitions[l+1]
            if s1 == 0 and (y0-y1) < 5*scale:
                seeds[y1:y0, x] = 1
    seeds = maximum_filter(seeds, (1, int(1+scale)))
    seeds = seeds * (1-colseps)
    seeds, _ = morph.label(seeds)
    return seeds


def remove_hlines(binary, scale, maxsize=10):
    """
    Removes horizontal black lines that only interfere with page segmentation.

        Args:
            binary (numpy.array):
            scale (float):
            maxsize (int): maximum size of removed lines

        Returns:
            numpy.array containing the filtered image.

    """
    logger.debug(u'Filtering horizontal lines')
    labels, _ = morph.label(binary)
    objects = morph.find_objects(labels)
    for i, b in enumerate(objects):
        if sl.width(b) > maxsize*scale:
            labels[b][labels[b] == i+1] = 0
    return np.array(labels != 0, 'B')


def rotate_lines(lines, angle, offset):
    """
    Rotates line bounding boxes around the origin and adding and offset.
    """
    logger.debug(u'Rotate line coordinates by {} with offset {}'.format(angle, offset))
    angle = np.radians(angle)
    r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    p = np.array(lines).reshape((-1, 2))
    offset = np.array([2*offset])
    p = p.dot(r).reshape((-1, 4)).astype(int) + offset
    x = np.sort(p[:, [0, 2]])
    y = np.sort(p[:, [1, 3]])
    return np.column_stack((x.flatten(), y.flatten())).reshape(-1, 4)


def segment(im, text_direction='horizontal-lr', scale=None, maxcolseps=2,
            black_colseps=False, no_hlines=True, pad=0, mask=None):
    """
    Segments a page into text lines.

    Segments a page into text lines and returns the absolute coordinates of
    each line in reading order.

    Args:
        im (PIL.Image): A bi-level page of mode '1' or 'L'
        text_direction (str): Principal direction of the text
                              (horizontal-lr/rl/vertical-lr/rl)
        scale (float): Scale of the image
        maxcolseps (int): Maximum number of whitespace column separators
        black_colseps (bool): Whether column separators are assumed to be
                              vertical black lines or not
        no_hlines (bool): Switch for horizontal line removal
        pad (int or tuple): Padding to add to line bounding boxes. If int the
                            same padding is used both left and right. If a
                            2-tuple, uses (padding_left, padding_right).
        mask (PIL.Image): A bi-level mask image of the same size as `im` where
                          0-valued regions are ignored for segmentation
                          purposes. Disables column detection.

    Returns:
        {'text_direction': '$dir', 'boxes': [(x1, y1, x2, y2),...]}: A
        dictionary containing the text direction and a list of reading order
        sorted bounding boxes under the key 'boxes'.

    Raises:
        KrakenInputException if the input image is not binarized or the text
        direction is invalid.
    """
    im_str = get_im_str(im)
    logger.info('Segmenting {}'.format(im_str))

    if im.mode != '1' and not is_bitonal(im):
        logger.error('Image {} is not bi-level'.format(im_str))
        raise KrakenInputException('Image {} is not bi-level'.format(im_str))

    # rotate input image for vertical lines
    if text_direction.startswith('horizontal'):
        angle = 0
        offset = (0, 0)
    elif text_direction == 'vertical-lr':
        angle = 270
        offset = (0, im.size[1])
    elif text_direction == 'vertical-rl':
        angle = 90
        offset = (im.size[0], 0)
    else:
        logger.error('Invalid text direction \'{}\''.format(text_direction))
        raise KrakenInputException('Invalid text direction {}'.format(text_direction))

    logger.debug('Rotating input image by {} degrees'.format(angle))
    im = im.rotate(angle, expand=True)

    # honestly I've got no idea what's going on here. In theory a simple
    # np.array(im, 'i') should suffice here but for some reason the
    # tostring/fromstring magic in pil2array alters the array in a way that is
    # needed for the algorithm to work correctly.
    a = pil2array(im)
    binary = np.array(a > 0.5*(np.amin(a) + np.amax(a)), 'i')
    binary = 1 - binary

    if not scale:
        scale = estimate_scale(binary)

    if no_hlines:
        binary = remove_hlines(binary, scale)
    # emptyish images wll cause exceptions here.

    try:
        if mask:
            if mask.mode != '1' and not is_bitonal(mask):
                logger.error('Mask is not bitonal')
                raise KrakenInputException('Mask is not bitonal')
            mask = mask.convert('1')
            if mask.size != im.size:
                logger.error('Mask size {} doesn\'t match image size {}'.format(mask.size, im.size))
                raise KrakenInputException('Mask size {} doesn\'t match image size {}'.format(mask.size, im.size))
            logger.info('Masking enabled in segmenter. Disabling column detection.')
            mask = mask.rotate(angle, expand=True)
            colseps = pil2array(mask)
        elif black_colseps:
            colseps, binary = compute_black_colseps(binary, scale, maxcolseps)
        else:
            colseps = compute_white_colseps(binary, scale, maxcolseps)
    except ValueError:
        logger.warning('Exception in column finder (probably empty image) for {}.'.format(im_str))
        return {'text_direction': text_direction, 'boxes':  []}

    bottom, top, boxmap = compute_gradmaps(binary, scale)
    seeds = compute_line_seeds(binary, bottom, top, colseps, scale)
    llabels = morph.propagate_labels(boxmap, seeds, conflict=0)
    spread = morph.spread_labels(seeds, maxdist=scale)
    llabels = np.where(llabels > 0, llabels, spread*binary)
    segmentation = llabels*binary

    lines = compute_lines(segmentation, scale)
    order = reading_order([l.bounds for l in lines], text_direction[-2:])
    lsort = topsort(order)
    lines = [lines[i].bounds for i in lsort]
    lines = [(s2.start, s1.start, s2.stop, s1.stop) for s1, s2 in lines]

    if isinstance(pad, int):
        pad = (pad, pad)
    lines = [(max(x[0]-pad[0], 0), x[1], min(x[2]+pad[1], im.size[0]), x[3]) for x in lines]

    return {'text_direction': text_direction, 'boxes':  rotate_lines(lines, 360-angle, offset).tolist(), 'script_detection': False}


def detect_scripts(im, bounds, model=pkg_resources.resource_filename(__name__, 'script.mlmodel'), valid_scripts=None):
    """
    Detects scripts in a segmented page.

    Classifies lines returned by the page segmenter into runs of scripts/writing systems.

    Args:
        im (PIL.Image): A bi-level page of mode '1' or 'L'
        bounds (dict): A dictionary containing a 'boxes' entry with a list of
                       coordinates (x0, y0, x1, y1) of a text line in the image
                       and an entry 'text_direction' containing
                       'horizontal-lr/rl/vertical-lr/rl'.
        model (str): Location of the script classification model or None for default.
        valid_scripts (list): List of valid scripts.

    Returns:
        {'script_detection': True, 'text_direction': '$dir', 'boxes':
        [[(script, (x1, y1, x2, y2)),...]]}: A dictionary containing the text
        direction and a list of lists of reading order sorted bounding boxes
        under the key 'boxes' with each list containing the script segmentation
        of a single line. Script is a ISO15924 4 character identifier.

    Raises:
        KrakenInvalidModelException if no clstm module is available.
    """
    raise NotImplementedError('Temporarily unavailable. Please open a github ticket if you want this fixed sooner.')
    im_str = get_im_str(im)
    logger.info(u'Detecting scripts with {} in {} lines on {}'.format(model, len(bounds['boxes']), im_str))
    logger.debug(u'Loading detection model {}'.format(model))
    rnn = models.load_any(model)
    # load numerical to 4 char identifier map
    logger.debug(u'Loading label to identifier map')
    with pkg_resources.resource_stream(__name__, 'iso15924.json') as fp:
        n2s = json.load(fp)
    # convert allowed scripts to labels
    val_scripts = []
    if valid_scripts:
        logger.debug(u'Converting allowed scripts list {}'.format(valid_scripts))
        for k, v in n2s.items():
            if v in valid_scripts:
                val_scripts.append(chr(int(k) + 0xF0000))
    else:
        valid_scripts = []
    it = rpred(rnn, im, bounds, bidi_reordering=False)
    preds = []
    logger.debug(u'Running detection')
    for pred, bbox in zip(it, bounds['boxes']):
        # substitute inherited scripts with neighboring runs
        def _subs(m, s, r=False):
            p = u''
            for c in s:
                if c in m and p and not r:
                    p += p[-1]
                elif c not in m and p and r:
                    p += p[-1]
                else:
                    p += c
            return p

        logger.debug(u'Substituting scripts')
        p = _subs([u'\U000f03e2', u'\U000f03e6'], pred.prediction)
        # do a reverse run to fix leading inherited scripts
        pred.prediction = ''.join(reversed(_subs([u'\U000f03e2', u'\U000f03e6'], reversed(p))))
        # group by valid scripts. two steps: 1. substitute common confusions
        # (Latin->Fraktur and Syriac->Arabic) if given in script list.
        if 'Arab' in valid_scripts and 'Syrc' not in valid_scripts:
            pred.prediction = pred.prediction.replace(u'\U000f0087', u'\U000f00a0')
        if 'Latn' in valid_scripts and 'Latf' not in valid_scripts:
            pred.prediction = pred.prediction.replace(u'\U000f00d9', u'\U000f00d7')
        # next merge adjacent scripts
        if val_scripts:
            pred.prediction = _subs(val_scripts, pred.prediction, r=True)

        # group by grapheme
        t = []
        logger.debug(u'Merging detections')
        # if line contains only a single script return whole line bounding box
        if len(set(pred.prediction)) == 1:
            logger.debug('Only one script on line. Emitting whole line bbox')
            k = ord(pred.prediction[0]) - 0xF0000
            t.append((n2s[str(k)], bbox))
        else:
            for k, g in groupby(pred, key=lambda x: x[0]):
                # convert to ISO15924 numerical identifier
                k = ord(k) - 0xF0000
                b = max_bbox(x[1] for x in g)
                t.append((n2s[str(k)], b))
        preds.append(t)
    return {'boxes': preds, 'text_direction': bounds['text_direction'], 'script_detection': True}
