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
Processing for baseline segmenter output
"""
import math
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.signal import convolve2d
from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter

from skimage.draw import line
from skimage.graph import MCP_Connect
from skimage.filters import apply_hysteresis_threshold, threshold_sauvola
from skimage.measure import approximate_polygon
from skimage.morphology import skeletonize_3d

from itertools import combinations
from collections import defaultdict

def denoising_hysteresis_thresh(im, low, high, sigma):
    im = gaussian_filter(im, sigma)
    return apply_hysteresis_threshold(im, low, high)

def vectorize_lines(im: np.ndarray, error: int = 3):
    """
    Vectorizes lines from a binarized array.

    Args:
        im (np.ndarray): Boolean array of baseline candidates
        error (int): Maximum error in polyline vectorization

    Returns:
        [[x0, y0, ... xn, yn], [xm, ym, ..., xk, yk], ... ]
        A list of lists containing the points of all baseline polylines.
    """

    line_skel = skeletonize_3d(im)
    # find extremities by convolving with 3x3 filter (value == 2 on the line because of
    # 8-connected skeleton)
    line_skel = line_skel > 0
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    line_extrema = np.transpose(np.where((convolve2d(line_skel, kernel, mode='same') == 11) * line_skel))

    # this is the ugly hack from dhSegment. Instead calculating the graph
    # diameter to find the centerline of the skeleton (which is unbearably
    # slow) just take the two points with the largest euclidian distance as
    # endpoints. This breaks down in case of folded or spiral lines as the true
    # end points are closer closer than random branches on the skeleton.
    candidates = defaultdict(list)
    label_im, _ = label(line_skel, structure=np.ones((3, 3)))
    for pt in line_extrema:
        candidates[label_im[tuple(pt)]].append(pt)
    cc_extrema = []
    for pts in candidates.values():
        distance = squareform(pdist(np.stack(pts), 'euclidean'))
        i, j = np.unravel_index(distance.argmax(), distance.shape)
        cc_extrema.append(pts[i])
        cc_extrema.append(pts[j])

    class LineMCP(MCP_Connect):
        def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.connections = dict()
           self.scores = defaultdict(lambda: np.inf)

        def create_connection(self, id1, id2, pos1, pos2, cost1, cost2):
            k = (min(id1, id2), max(id1, id2))
            s = cost1 + cost2
            if self.scores[k] > s:
                self.connections[k] = (pos1, pos2, s)
                self.scores[k] = s

        def get_connections(self):
            results = []
            for k, (pos1, pos2, s) in self.connections.items():
                results.append(np.concatenate([self.traceback(pos1), self.traceback(pos2)[::-1]]))
            return results

        def goal_reached(self, int_index, float_cumcost):
            return 2 if float_cumcost else 0

    mcp = LineMCP(~line_skel)
    try:
        mcp.find_costs(cc_extrema)
    except ValueError as e:
        return []
    return [line[:,::-1].tolist() if line[0][0] > line[-1][0] else line.tolist() for line in mcp.get_connections()]


def calculate_polygonal_environment(im, baselines, error=3):
    """
    Given a list of baselines and an input image, calculates a polygonal
    environment around each baseline.

    Args:
        im (PIL.Image): Input image
        baselines (sequence): List of lists containing a single baseline per
                              entry.
        error (int): Maximum boundary polygon approximation error

    Returns:
        List of tuples (baseline, polygonization) where each is a list of coordinates.
    """
    siz = im.size
    context = 10
    low_context = 50

    from shapely.geometry import LineString, Polygon

    def _clamp(x, bound):
        return min(max(0, x), bound)

    blpl = []
    for baseline in baselines:
        bl = approximate_polygon(np.array(baseline), error)
        if bl[0][0] > bl[-1][0]:
            bl = list(reversed(bl))
        bl = LineString(bl)
        upper = list(bl.parallel_offset(low_context, 'right').coords)
        if upper[0][0] > upper[-1][0]:
            upper = list(reversed(upper))
        lower = list(bl.parallel_offset(context, 'left').coords)
        if lower[0][0] < lower[-1][0]:
            lower = list(reversed(lower))
        blpl.append((baseline, approximate_polygon(np.array(upper + lower), error).astype('int').tolist()))
    return blpl
