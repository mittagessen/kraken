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
import PIL
import logging
import warnings
import numpy as np
import shapely.geometry as geom

from PIL import Image, ImageDraw

from numpy.polynomial import Polynomial

from scipy.ndimage import black_tophat
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import grey_dilation
from scipy.spatial import distance_matrix, Delaunay
from scipy.spatial.distance import cdist, pdist, squareform

from shapely.ops import nearest_points, unary_union

from skimage import draw
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import skeletonize, watershed
from skimage.transform import PiecewiseAffineTransform, warp

from typing import List, Tuple, Union, Dict, Any, Sequence


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


def _find_superpixels(skeleton, heatmap, min_sp_dist):
    logger.debug('Finding superpixels')
    conf_map = heatmap * skeleton
    sp_idx = np.unravel_index(np.argsort(1.-conf_map, axis=None), conf_map.shape)
    if not sp_idx[0].any():
        logger.info('No superpixel candidates found for line vectorizer. Likely empty page.')
        return np.empty(0)
    zeroes_idx = conf_map[sp_idx].argmin()
    if not zeroes_idx:
        logger.info('No superpixel candidates found for line vectorizer. Likely empty page.')
        return np.empty(0)
    sp_idx = sp_idx[0][:zeroes_idx], sp_idx[1][:zeroes_idx]
    sp_can = [(sp_idx[0][0], sp_idx[1][0])]
    for x in range(len(sp_idx[0])):
        loc = np.array([[sp_idx[0][x], sp_idx[1][x]]])
        if min(cdist(sp_can, loc)) > min_sp_dist:
            sp_can.extend(loc.tolist())
    return np.array(sp_can)


def _compute_sp_states(sp_can, bl_map, sep_map):
    """
    Estimates the superpixel state information.
    """
    logger.debug('Triangulating superpixels')
    tri = Delaunay(sp_can, qhull_options="QJ Pp")
    indices, indptr = tri.vertex_neighbor_vertices
    # dict mapping each edge to its intensity. Needed for subsequent clustering step.
    intensities = {}
    # radius of circular environment around SP for ILD estimation
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
            line_locs = draw.line(*(key[0] + key[1]))
            intensities[key] = (bl_map[line_locs].mean(), bl_map[line_locs].var(), sep_map[line_locs].mean(), sep_map[line_locs].max())
            intensity.append(intensities[key][0])

    logger.debug('Filtering triangulation')
    # filter edges in triangulation
    for k, v in list(intensities.items()):
        if v[0] < 0.4:
            del intensities[k]
            continue
        if v[1] > 5e-02:
            del intensities[k]
            continue
        # filter edges with high separator affinity
        if v[2] > 0.125 or v[3] > 0.25 or v[0] < 0.5:
            del intensities[k]
            continue

    return intensities


def _cluster_lines(intensities):
    """
    Clusters lines according to their intensities.
    """
    edge_list = list(intensities.keys())

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
    return clusters


def _interpolate_lines(clusters):
    """
    Interpolates the baseline clusters.
    """
    logger.debug('Reticulating splines')
    lines = []
    for cluster in clusters[1:]:
        points = sorted(set(point for edge in cluster for point in edge), key=lambda x: x[1])
        x = [x[1] for x in points]
        y = [x[0] for x in points]
        # very short lines might not have enough superpixels to ensure a well-conditioned regression
        deg = min(len(x)-1, 3)
        poly = Polynomial.fit(x, y, deg=deg)
        xp, yp = poly.linspace(max(np.diff(poly.domain)//deg, 2))
        xp = xp.astype('int')
        yp = yp.astype('int')
        line = np.array(list(zip(xp, yp)))
        line = approximate_polygon(line, 3)
        line = line.astype('uint').tolist()
        lines.append(line)
    return lines


def vectorize_lines(im: np.ndarray, threshold: float = 0.2, min_sp_dist: int = 10):
    """
    Vectorizes lines from a binarized array.

    Args:
        im (np.ndarray): Array of shape (3, H, W) with the first dimension
                         being a probability distribution over (background,
                         baseline, separators).

    Returns:
        [[x0, y0, ... xn, yn], [xm, ym, ..., xk, yk], ... ]
        A list of lists containing the points of all baseline polylines.
    """
    # split into baseline and separator map
    bl_map = im[1]
    sep_map = im[2]
    # binarize
    bin = im > threshold
    skel = skeletonize(bin[1])
    sp_can = _find_superpixels(skel, heatmap=bl_map, min_sp_dist=min_sp_dist)
    if not sp_can.size:
        logger.warning('No superpixel candidates found in network output. Likely empty page.')
        return []
    intensities = _compute_sp_states(sp_can, bl_map, sep_map)
    clusters = _cluster_lines(intensities)
    lines = _interpolate_lines(clusters)
    return lines


def calculate_polygonal_environment(im: PIL.Image.Image, baselines: Sequence[Tuple[int, int]]):
    """
    Given a list of baselines and an input image, calculates a polygonal
    environment around each baseline.

    Args:
        im (PIL.Image): grayscale input image (mode 'L')
        baselines (sequence): List of lists containing a single baseline per
                              entry.
        bl_mask (numpy.array): Optional raw baselines output maps from the
                               recognition net.

    Returns:
        List of tuples (polygonization, baseline) where each is a list of coordinates.
    """
    bounds = np.array(im.size, dtype=np.float)
    im = np.array(im)

    # compute tophat features of input image
    im_feats = black_tophat(im, 3)

    def _ray_intersect_boundaries(ray, direction, aabb):
        """
        Simplified version of [0] for 2d and AABB anchored at (0,0).

        [0] http://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
        """
        dir_fraction = np.empty(2, dtype=ray.dtype)
        dir_fraction[direction == 0.0] = np.inf
        dir_fraction[direction != 0.0] = np.divide(1.0, direction[direction != 0.0])

        t1 = (-ray[0]) * dir_fraction[0]
        t2 = (aabb[0] - ray[0]) * dir_fraction[0]
        t3 = (-ray[1]) * dir_fraction[1]
        t4 = (aabb[1] - ray[1]) * dir_fraction[1]

        tmin = max(min(t1, t2), min(t3, t4))
        tmax = min(max(t1, t2), max(t3, t4))

        t = min(x for x in [tmin, tmax] if x >= 0)
        return ray + (direction * t)


    def _extract_patch(env_up, env_bottom, baseline):
        """
        Calculate a line image patch from a ROI and the original baseline
        """
        markers = np.zeros(bounds.astype('int')[::-1], dtype=np.int)
        for l in zip(baseline[:-1], baseline[1:]):
            line_locs = draw.line(l[0][1], l[0][0], l[1][1], l[1][0])
            markers[line_locs] = 2
        for l in zip(env_up[:-1], env_up[1:]):
            line_locs = draw.line(l[0][1], l[0][0], l[1][1], l[1][0])
            markers[line_locs] = 1
        for l in zip(env_bottom[:-1], env_bottom[1:]):
            line_locs = draw.line(l[0][1], l[0][0], l[1][1], l[1][0])
            markers[line_locs] = 1
        markers = grey_dilation(markers, size=3)
        full_polygon = np.concatenate(([baseline[0]], env_up, [baseline[-1]], env_bottom[::-1]))
        r, c = draw.polygon(full_polygon[:,0], full_polygon[:,1])
        mask = np.zeros(bounds.astype('int')[::-1], dtype=np.bool)
        mask[c, r] = True
        patch = im_feats.copy()
        patch[mask != True] = 0
        coords = np.argwhere(mask)
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        patch = patch[r_min:r_max+1, c_min:c_max+1]
        markers = markers[r_min:r_max+1, c_min:c_max+1]
        mask = mask[r_min:r_max+1, c_min:c_max+1]
        # run watershed
        ws = watershed(patch, markers, 8, mask=mask)
        ws = grey_dilation(ws, size=3)
        # pad output to ensure contour is closed
        ws = np.pad(ws, 1)
        # find contour of central basin
        contours = find_contours(ws, 1.5, fully_connected='high')
        contour = np.array(unary_union([geom.Polygon(contour.tolist()) for contour in contours]).boundary, dtype='uint')
        ## approximate + remove offsets + transpose
        contour = (approximate_polygon(contour, 5)-1+(r_min, c_min)).astype('uint')
        return contour.tolist()

    polygons = []
    for idx, line in enumerate(baselines):
        # find intercepts with image bounds on each side of baseline
        lr = np.array(line[:2], dtype=np.float)
        lr_dir = lr[1] - lr[0]
        lr_dir = (lr_dir.T  / np.sqrt(np.sum(lr_dir**2,axis=-1)))
        lr_up_intersect = _ray_intersect_boundaries(lr[0], (lr_dir*(-1,1))[::-1], bounds-1).astype('int')
        lr_bottom_intersect = _ray_intersect_boundaries(lr[0], (lr_dir*(1,-1))[::-1], bounds-1).astype('int')
        rr = np.array(line[-2:], dtype=np.float)
        rr_dir = rr[1] - rr[0]
        rr_dir = (rr_dir.T  / np.sqrt(np.sum(rr_dir**2,axis=-1)))
        rr_up_intersect = _ray_intersect_boundaries(rr[1], (rr_dir*(-1,1))[::-1], bounds-1).astype('int')
        rr_bottom_intersect = _ray_intersect_boundaries(rr[1], (rr_dir*(1,-1))[::-1], bounds-1).astype('int')
        # build polygon between baseline and bbox intersects
        upper_polygon = geom.Polygon([lr_up_intersect.tolist()] + line + [rr_up_intersect.tolist()])
        bottom_polygon = geom.Polygon([lr_bottom_intersect.tolist()] + line + [rr_bottom_intersect.tolist()])
        # select baselines at least partially in each polygon
        side_a = [geom.LineString([lr_up_intersect.tolist(), rr_up_intersect.tolist()])]
        side_b = [geom.LineString([lr_bottom_intersect.tolist(), rr_bottom_intersect.tolist()])]
        for adj_line in baselines[:idx] + baselines[idx+1:]:
            adj_line = geom.LineString(adj_line)
            if upper_polygon.intersects(adj_line):
                side_a.append(adj_line)
            elif bottom_polygon.intersects(adj_line):
                side_b.append(adj_line)
        side_a = unary_union(side_a)
        side_b = unary_union(side_b)
        env_up = []
        env_bottom = []
        # find nearest points from baseline to previously selected baselines
        for point in line:
            _, upper_limit = nearest_points(geom.Point(point), side_a)
            _, bottom_limit = nearest_points(geom.Point(point), side_b)
            env_up.extend(list(upper_limit.coords))
            env_bottom.extend(list(bottom_limit.coords))
        env_up = np.array(env_up, dtype='uint')
        env_bottom = np.array(env_bottom, dtype='uint')
        polygons.append(_extract_patch(env_up, env_bottom, line))
    return polygons


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


def _test_intersect(bp, uv, bs):
    """
    Returns the intersection points of a ray with direction `uv` from
    `bp` with a polygon `bs`.
    """
    u = bp - np.roll(bs, 2)
    v = bs - np.roll(bs, 2)
    points = []
    for dir in ((1,-1), (-1,1)):
        w = (uv * dir * (1,-1))[::-1]
        z = np.dot(v, w)
        t1 = np.cross(v, u) / z
        t2 = np.dot(u, w) / z
        t1 = t1[np.logical_and(t2 >= 0.0, t2 <= 1.0)]
        points.extend(bp + (t1[np.where(t1 >= 0)[0].min()] * (uv * dir)))
    return np.array(points)


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
            pl = np.array(line['boundary'])
            pl = measure.subdivide_polygon(pl, preserve_ends=True)
            pl = geom.MultiPoint(pl)
            bl = np.array(line['baseline'])
            bl = np.dstack((bl[:-1:], bl[1::]))
            bl = [geom.LineString(x) for x in bl]
            # distance of intercept from start point 
            for point in pl.geoms:
                line.interpolate(line.project(point) for line in bl:
                    line.interpolate(line.project(point))
            root_dists = [bl.project(point) for point in pl.geoms]
            # actual intercept 
            root_points = [bl.interpolate(dist) for dist in zip(pl.geoms, root_dists)]
            

            r, c = draw.polygon(full_polygon[:,0], full_polygon[:,1])
            mask = np.zeros(bounds.astype('int')[::-1], dtype=np.bool)
            mask[c, r] = True
            patch = im_feats.copy()
            patch[mask != True] = 0
            coords = np.argwhere(mask)
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            patch = patch[r_min:r_max+1, c_min:c_max+1]
            markers = markers[r_min:r_max+1, c_min:c_max+1]
            mask = mask[r_min:r_max+1, c_min:c_max+1]


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
