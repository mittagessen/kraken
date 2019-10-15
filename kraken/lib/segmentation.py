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
import logging
import warnings
import numpy as np
import shapely.geometry as geom

from PIL import Image, ImageDraw

from numpy.polynomial import Polynomial

from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.ndimage.morphology import grey_dilation, binary_dilation
from scipy.spatial import distance_matrix, ConvexHull, Delaunay
from scipy.spatial.distance import cdist, pdist, squareform

from skimage.draw import line
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import approximate_polygon
from skimage.morphology import skeletonize
from skimage.transform import PiecewiseAffineTransform, warp

from itertools import combinations
from collections import defaultdict, OrderedDict

from typing import List, Tuple, Optional, Generator, Union, Dict, Any, Sequence

from kraken.lib import morph, util
from kraken.binarization import nlbin

logger = logging.getLogger('kraken')


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
    logger.info('Perform topological sort on partially ordered lines')
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

def denoising_hysteresis_thresh(im, low, high, sigma):
    im = gaussian_filter(im, sigma)
    return apply_hysteresis_threshold(im, low, high)

def vectorize_lines(im: np.ndarray, threshold: float = 0.2, min_sp_dist: int = 3, radii: Sequence[int] = [32, 64, 128, 256]):
    """
    Vectorizes lines from a binarized array.

    Args:
        im (np.ndarray): Array of shape (3, H, W) with the first dimension
                         being a probability distribution over (background,
                         baseline, separators).
        error (int): Maximum error in polyline vectorization

    Returns:
        [[x0, y0, ... xn, yn], [xm, ym, ..., xk, yk], ... ]
        A list of lists containing the points of all baseline polylines.
    """
    # split into baseline and separator map
    bl_map = im[1]
    sep_map = im[2]
    # binarize
    bin = im > threshold
    # skeletonize 
    logger.debug('Finding superpixels')
    skel = skeletonize(bin[1])
    conf_map = bl_map * skel
    sp_idx = np.unravel_index(np.argsort(1.-conf_map, axis=None), conf_map.shape)
    if not sp_idx[0].any():
        logger.info('No superpixel candidates found for line vectorizer. Likely empty page.')
        return []
    zeroes_idx = conf_map[sp_idx].argmin()
    sp_idx = sp_idx[0][:zeroes_idx], sp_idx[1][:zeroes_idx]
    sp_can = [(sp_idx[0][0], sp_idx[1][0])]
    for x in range(len(sp_idx[0])):
        loc = np.array([[sp_idx[0][x], sp_idx[1][x]]])
        if min(cdist(sp_can, loc)) > min_sp_dist:
            sp_can.extend(loc.tolist())
    sp_can = np.array(sp_can)
    logger.debug('Triangulating superpixels')
    tri = Delaunay(sp_can)
    indices, indptr = tri.vertex_neighbor_vertices
    # dict mapping each edge to its intensity. Needed for subsequent clustering step.
    intensities = {}
    states = {}
    dists = squareform(pdist(sp_can))
    # radius of circular environment around SP for ILD estimation
    nb_indices = tuple(np.nonzero(dists < proj_env) for proj_env in radii)
    logger.debug('Computing superpixel state information')
    for vertex in range(len(sp_can)):
        # look up neighboring indices
        neighbors = tri.points[indptr[indices[vertex]:indices[vertex+1]]]
        # calculate intensity of line segments to neighbors in both bl map and separator map
        intensity = []
        for nb in neighbors.astype('int'):
            key = [tuple(sp_can[vertex]), tuple(nb)]
            key.sort()
            key = tuple(key)
            line_locs = line(*(key[0] + key[1]))
            intensities[key] = (bl_map[line_locs].mean(), sep_map[line_locs].mean(), sep_map[line_locs].max())
            intensity.append(intensities[key][0])
        slope_pts = neighbors[np.argsort(1-np.array(intensity))[:2]]
        # orientation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theta = np.arctan((slope_pts[1,0]-slope_pts[0,0])/(slope_pts[1,1]-slope_pts[0,1]))
        # calculate projection profiles
        data_energy = []
        for nbs in nb_indices:
            env_nbs = nbs[1][nbs[0] == vertex]
            vecs = np.abs(tri.points[env_nbs] - sp_can[vertex])
            vals = np.abs(np.cross(vecs, np.array((np.sin(theta), np.cos(theta)))).astype('int'))
            frqs = np.fft.fft(np.bincount(vals))
            frqs = np.abs(frqs)
            de = (np.abs(frqs)**2)/(np.linalg.norm(frqs, 2)**2)
            data_energy.extend(de[3:6].tolist())
        data_energy = np.array(data_energy)
        if not data_energy.any():
            ild = 0
        else:
            ild = radii[data_energy.argmax() // 3] * 2 / (data_energy.argmax() % 3 + 3)
        states[tuple(sp_can[vertex])] = (theta, ild)
    for k, v in list(intensities.items()):
        if v[0] < 0.4:
            del intensities[k]
            continue
#        # filter edges with high separator affinity
#        if v[1] > 0.125 or v[2] > 0.25 or v[0] < 0.5:
#            del intensities[k]
#            continue
#        # filter edges of different local orientations
#        if np.abs(states[k[0]][0] - states[k[1]][0]) % np.pi > np.pi/4:
#           del intensities[k]

    # sort edges by (1 - off_orientation) * intensity
    def _off_orientation(p, q):
        theta = np.mean((states[p][0], states[q][0]))
        return np.abs((p[1]-q[1])*np.sin(theta) - (p[0]-q[0])*np.cos(theta))

    def _oow_intensity(edge):
        p = edge[0]
        q = edge[1]
        return (1 - (_off_orientation(p, q)/np.linalg.norm(np.array(p)-np.array(q)))) * intensities[edge][0]

    edge_list = list(intensities.keys())
    edge_list.sort(key=lambda x:_oow_intensity(x), reverse=True)

    def _point_in_cluster(p):
        for idx, cluster in enumerate(clusters[1:]):
            if p in [point for edge in cluster for point in edge]:
                return idx+1
        return 0

    # cluster 
    logger.debug('Computing clusters')
    n = 0
    clusters = [edge_list]
    while len(edge_list) != n:
        n = len(edge_list)
        for edge in edge_list:
            cl_p0 = _point_in_cluster(edge[0])
            cl_p1 = _point_in_cluster(edge[1])
            # new cluster casea
            if not cl_p0 and not cl_p1:
                edge_list.remove(edge)
                clusters.append([edge])
            # extend case
            elif cl_p0 and not cl_p1:
                edge_list.remove(edge)
                clusters[cl_p0].append(edge)
            elif cl_p1 and not cl_p0:
                edge_list.remove(edge)
                clusters[cl_p1].append(edge)
            # merge case
            elif cl_p0 != cl_p1 and cl_p0 and cl_p1:
                edge_list.remove(edge)
                clusters[min(cl_p0, cl_p1)].extend(clusters.pop(max(cl_p0, cl_p1)))
                clusters[min(cl_p0, cl_p1)].append(edge)

    logger.debug('Reticulating splines')
    lines = []
    for cluster in clusters[1:]:
        points = sorted(set(point for edge in cluster for point in edge), key=lambda x: x[1])
        x = [x[1] for x in points]
        y = [x[0] for x in points]
        # very short lines might not have enough superpixels to ensure a well-conditioned regression
        deg = min(len(x)-1, 3)
        poly = Polynomial.fit(x, y, deg=deg)
        deriv = poly.deriv()
        xp, yp = poly.linspace(max(np.diff(poly.domain)//deg, 2))
        xp = xp.astype('int')
        yp = yp.astype('int')
        ls = geom.LineString(list(zip(yp, xp)))
        ilds = []
        proj_points = []
        for point in points:
            ilds.append(states[point][1])
            proj_points.append(np.array(ls.interpolate(ls.project(geom.Point(point))).coords)[0].astype('int').tolist())
        ilds = gaussian_filter1d(ilds, 0.5).tolist()
        low = []
        up = []
        for idx, pt in enumerate(proj_points):
            theta = np.pi/2 - np.arctan(deriv(pt[1]))
            low.insert(0, (pt[::-1] + np.array((np.cos(theta), np.sin(theta))) * ilds[idx] * 1/3).astype('int').tolist())
            up.append((pt[::-1] - np.array((np.cos(theta), np.sin(theta))) * ilds[idx] * 2/3).astype('int').tolist())
        lines.append(([point[::-1] for point in proj_points], up + low))
    return lines


def polygonal_reading_order(lines: Sequence[Tuple[List, List]], text_direction: str = 'lr') -> Sequence[Tuple[List, List]]:
    """
    Given a list of baselines, calculates the correct reading order and applies
    it to the input.

    Args:
        lines (Sequence): List of tuples containing the baseline and it's
                          polygonization.
        text_direction (str): Set principal text direction for column ordering.
                              Can be 'lr' or 'rl'

    Returns:
        A reordered input.
    """
    bounds = []
    for line in lines:
        l = geom.LineString(line[0]).bounds
        bounds.append((slice(l[0], l[1]), slice(l[2], l[3])))
    order = reading_order(bounds, text_direction)
    lsort = topsort(order)
    return [lines[i] for i in lsort]


def scale_polygonal_lines(lines: Sequence[Tuple[List, List]], scale: Union[float, Tuple[float, float]]) -> Sequence[Tuple[List, List]]:
    """
    Scales baselines/polygon coordinates by a certain factor.

    Args:
        lines (Sequence): List of tuples containing the baseline and it's
                          polygonization.
        scale (float or tuple of floats): Scaling factor
    """
    if isinstance(scale, float):
        scale = (scale, scale)
    scaled_lines = []
    for line in lines:
        bl, pl = line
        scaled_lines.append(((np.array(bl) * scale).astype('int').tolist(),
                             (np.array(pl) * scale).astype('int').tolist()))
    return scaled_lines


def extract_polygons(im: Image.Image, bounds: Dict[str, Any]) -> Image:
    """
    Yields the subimages of image im defined in the list of bounding polygons
    with baselines preserving order.

    Args:
        im (PIL.Image.Image): Input image
        bounds (list): A list of tuples (x1, y1, x2, y2)

    Yields:
        (PIL.Image) the extracted subimage
    """
    if 'type' in bounds and bounds['type'] == 'baselines':
        old_settings = np.seterr(all='ignore')

        siz = im.size
        white = Image.new(im.mode, siz)
        for line in bounds['lines']:
            mask = Image.new('1', siz, 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon([tuple(x) for x in line['boundary']], outline=1, fill=1)
            masked_line = Image.composite(im, white, mask)
            bl = np.array(line['baseline'])
            ls = np.dstack((bl[:-1:], bl[1::]))
            bisect_points = np.mean(ls, 2)
            norm_vec = (ls[...,1] - ls[...,0])[:,::-1]
            norm_vec_len = np.sqrt(np.sum(norm_vec**2, axis=1))
            unit_vec = norm_vec / np.tile(norm_vec_len, (2, 1)).T # without
                                                                  # multiplication
                                                                  # with (1,-1)-upper/
                                                                  # (-1, 1)-lower
            bounds = np.array(line['boundary'])
            src_points = np.stack([_test_intersect(bp, uv, bounds) for bp, uv in zip(bisect_points, unit_vec)])
            upper_dist = np.diag(distance_matrix(src_points[:,:2], bisect_points))
            upper_dist = np.dstack((np.zeros_like(upper_dist), upper_dist)).squeeze(0)
            lower_dist = np.diag(distance_matrix(src_points[:,2:], bisect_points))
            lower_dist = np.dstack((np.zeros_like(lower_dist), lower_dist)).squeeze(0)
            # map baseline points to straight baseline
            bl_dists = np.cumsum(np.diag(np.roll(squareform(pdist(bl)), 1)))
            bl_dst_pts = bl[0] + np.dstack((bl_dists, np.zeros_like(bl_dists))).squeeze(0)
            rect_bisect_pts = np.mean(np.dstack((bl_dst_pts[:-1:], bl_dst_pts[1::])), 2)
            upper_dst_pts = rect_bisect_pts - upper_dist
            lower_dst_pts = rect_bisect_pts + lower_dist
            src_points = np.concatenate((bl, src_points[:,:2], src_points[:,2:]))
            dst_points = np.concatenate((bl_dst_pts, upper_dst_pts, lower_dst_pts))
            tform = PiecewiseAffineTransform()
            tform.estimate(src_points, dst_points)
            i = Image.fromarray((warp(masked_line, tform) * 255).astype('uint8'))
            yield i.crop(i.getbbox()), line
    else:
        if bounds['text_direction'].startswith('vertical'):
            angle = 90
        else:
            angle = 0
        for box in bounds['boxes']:
            if isinstance(box, tuple):
                box = list(box)
            if (box < [0, 0, 0, 0] or box[::2] > [im.size[0], im.size[0]] or
                    box[1::2] > [im.size[1], im.size[1]]):
                logger.error('bbox {} is outside of image bounds {}'.format(box, im.size))
                raise KrakenInputException('Line outside of image bounds')
            yield im.crop(box).rotate(angle, expand=True), box
