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
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.ndimage.morphology import grey_dilation
from scipy.spatial import distance_matrix, Delaunay
from scipy.spatial.distance import cdist, pdist, squareform

from shapely.ops import nearest_points, unary_union

from skimage import draw, measure
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import skeletonize, watershed
from skimage.transform import PiecewiseAffineTransform, warp

from typing import List, Tuple, Union, Dict, Any, Sequence


logger = logging.getLogger('kraken')

__all__ = ['reading_order',
           'denoising_hysteresis_thresh',
           'vectorize_lines',
           'calculate_polygonal_environment',
           'polygonal_reading_order',
           'scale_polygonal_lines',
           'compute_polygon_section',
           'extract_polygons']

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
    # some pages might not contain
    if len(sp_can) < 2:
        logger.warning('Less than 2 superpixels in image. Nothing to vectorize.')
        return {}
    elif len(sp_can) < 3:
        logger.warning('Less than 3 superpixels in image. Skipping triangulation')
        key = tuple([tuple(sp_can[0]), tuple(sp_can[1])])
        line_locs = draw.line(*(key[0] + key[1]))
        intensities = {key: (bl_map[line_locs].mean(), bl_map[line_locs].var(), sep_map[line_locs].mean(), sep_map[line_locs].max())}
        return intensities
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
        ws = np.pad(ws, 1, mode='constant')
        # find contour of central basin
        contours = find_contours(ws, 1.5, fully_connected='high')
        contour = np.array(unary_union([geom.Polygon(contour.tolist()) for contour in contours]).boundary, dtype='int')
        ## approximate + remove offsets + transpose
        contour_y = gaussian_filter1d(contour[:, 0], 3)
        contour_x = gaussian_filter1d(contour[:, 1], 3)
        contour = np.dstack((contour_x, contour_y))[0]
        contour = (approximate_polygon(contour, 1)-1+(c_min, r_min)).astype('uint')
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


def compute_polygon_section(baseline, boundary, dist1, dist2):
    """
    Given a baseline, polygonal boundary, and two points on the baseline return
    the rectangle formed by the orthogonal cuts on that baseline segment. The
    resulting polygon is not garantueed to have a non-zero area.

    Args:
        baseline (list): A polyline ((x1, y1), ..., (xn, yn))
        boundary (list): A bounding polygon around the baseline (same format as
                         baseline).
        dist1 (int): Absolute distance along the baseline of the first point.
        dist2 (int): Absolute distance along the baseline of the second point.

    Returns:
        A sequence of polygon points.
    """
    # find baseline segments the points are in
    bl = np.array(baseline)
    dists = np.cumsum(np.diag(np.roll(squareform(pdist(bl)), 1)))
    segs_idx = np.searchsorted(dists, [dist1, dist2])
    segs = np.dstack((bl[segs_idx-1], bl[segs_idx]))
    # compute unit vector of segments (NOT orthogonal)
    norm_vec = (segs[...,1] - segs[...,0])
    norm_vec_len = np.sqrt(np.sum(norm_vec**2, axis=1))
    unit_vec = norm_vec / np.tile(norm_vec_len, (2, 1)).T
    # find point start/end point on segments
    seg_dists = (dist1, dist2) - dists[segs_idx-1]
    seg_points = segs[...,0] + (seg_dists * unit_vec.T).T
    # get intersects
    bounds = np.array(boundary)
    try:
        points = [_test_intersect(point, uv[::-1], bounds).round() for point, uv in zip(seg_points, unit_vec)]
    except ValueError:
        logger.warning('No intercepts with polygon (possibly misshaped polygon)')
        return seg_points.astype('int') 
    o = np.int_(points[0]).reshape(-1, 2).tolist()
    o.extend(np.int_(np.roll(points[1], 2)).reshape(-1, 2).tolist())
    return o


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

        siz = np.array(im.size, dtype=np.float)
        # select proper interpolation scheme depending on shape
        if im.mode == '1':
            order = 0
        else:
            order = 1
        im = np.array(im)

        for line in bounds['lines']:
            pl = np.array(line['boundary'])
            full_polygon = measure.subdivide_polygon(pl, preserve_ends=True)
            pl = geom.MultiPoint(full_polygon)
            baseline = np.array(line['baseline'])
            bl = zip(baseline[:-1:], baseline[1::])
            bl = [geom.LineString(x) for x in bl]
            cum_lens = np.cumsum([0] + [l.length for l in bl])
            # distance of intercept from start point and number of line segment
            control_pts = []
            for point in pl.geoms:
                npoint = np.array(point)
                line_idx, dist, intercept = min(((idx, line.project(point), np.array(line.interpolate(line.project(point)))) for idx, line in enumerate(bl)), key=lambda x: np.linalg.norm(npoint-x[2]))
                # absolute distance from start of line
                line_dist = cum_lens[line_idx] + dist
                intercept = np.array(intercept)
                # side of line the point is at
                side = np.linalg.det(np.array([[baseline[line_idx+1][0]-baseline[line_idx][0],
                                                npoint[0]-baseline[line_idx][0]],
                                               [baseline[line_idx+1][1]-baseline[line_idx][1],
                                                npoint[1]-baseline[line_idx][1]]]))
                side = np.sign(side)
                # signed perpendicular distance from the rectified distance
                per_dist = side * np.linalg.norm(npoint-intercept)
                control_pts.append((line_dist, per_dist))
            # calculate baseline destination points
            bl_dst_pts = baseline[0] + np.dstack((cum_lens, np.zeros_like(cum_lens)))[0]
            # calculate bounding polygon destination points
            pol_dst_pts = np.array([baseline[0] + (line_dist, per_dist) for line_dist, per_dist in control_pts])
            # extract bounding box patch
            c_min, c_max = int(full_polygon[:,0].min()), int(full_polygon[:,0].max())
            r_min, r_max = int(full_polygon[:,1].min()), int(full_polygon[:,1].max())
            c_dst_min, c_dst_max = int(pol_dst_pts[:,0].min()), int(pol_dst_pts[:,0].max())
            r_dst_min, r_dst_max = int(pol_dst_pts[:,1].min()), int(pol_dst_pts[:,1].max())
            output_shape = np.around((r_dst_max - r_dst_min + 1, c_dst_max - c_dst_min + 1))
            patch = im[r_min:r_max+1,c_min:c_max+1].copy()
            # offset src points by patch shape
            offset_polygon = full_polygon - (c_min, r_min)
            offset_baseline = baseline - (c_min, r_min)
            # offset dst point by dst polygon shape
            offset_bl_dst_pts = bl_dst_pts - (c_dst_min, r_dst_min)
            offset_pol_dst_pts = pol_dst_pts - (c_dst_min, r_dst_min)
            # mask out points outside bounding polygon
            mask = np.zeros(patch.shape[:2], dtype=np.bool)
            r, c = draw.polygon(offset_polygon[:,1], offset_polygon[:,0])
            mask[r, c] = True
            patch[mask != True] = 0
            # estimate piecewise transform
            src_points = np.concatenate((offset_baseline, offset_polygon))
            dst_points = np.concatenate((offset_bl_dst_pts, offset_pol_dst_pts))
            tform = PiecewiseAffineTransform()
            tform.estimate(src_points, dst_points)
            o = warp(patch, tform.inverse, output_shape=output_shape, preserve_range=True, order=order)
            i = Image.fromarray(o.astype('uint8'))
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
