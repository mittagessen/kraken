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
from collections import defaultdict
from typing import (TYPE_CHECKING, Dict, List, Literal, Optional, Sequence,
                    Tuple, Union, TypeVar, Any, Generator)

import numpy as np
import shapely.geometry as geom
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from PIL.Image import Resampling, Transform
from scipy.ndimage import (binary_erosion, distance_transform_cdt,
                           gaussian_filter, maximum_filter, affine_transform)
from scipy.signal import convolve2d
from scipy.spatial.distance import pdist, squareform
from shapely.ops import nearest_points, unary_union
from shapely.validation import explain_validity
from skimage import draw, filters
from skimage.filters import sobel
from skimage.graph import MCP_Connect
from skimage.measure import (approximate_polygon, label, regionprops,
                             subdivide_polygon)
from skimage.morphology import skeletonize
from skimage.transform import (AffineTransform, PiecewiseAffineTransform, warp)

from kraken.lib import default_specs
from kraken.lib.exceptions import KrakenInputException

if TYPE_CHECKING:
    from kraken.containers import Segmentation, BBoxLine, BaselineLine
    from kraken.lib.vgsl import TorchVGSLModel


_T_pil_or_np = TypeVar('_T_pil_or_np', Image.Image, np.ndarray)

logger = logging.getLogger('kraken')

__all__ = ['reading_order',
           'neural_reading_order',
           'vectorize_lines',
           'calculate_polygonal_environment',
           'polygonal_reading_order',
           'scale_polygonal_lines',
           'scale_regions',
           'compute_polygon_section',
           'extract_polygons']


# faster implementation of PiecewiseAffineTransform - see
# https://github.com/scikit-image/scikit-image/issues/6864 and
# https://github.com/scikit-image/scikit-image/pull/6963
class FastPiecewiseAffineTransform(PiecewiseAffineTransform):
    def __call__(self, coords):
        coords = np.asarray(coords)

        simplex = self._tesselation.find_simplex(coords)

        affines = np.array(
            [self.affines[i].params for i in range(len(self._tesselation.simplices))]
        )[simplex]

        pts = np.c_[coords, np.ones((coords.shape[0], 1))]

        result = np.einsum("ij,ikj->ik", pts, affines)
        result[simplex == -1, :] = -1

        return result


def reading_order(lines: Sequence[Tuple[slice, slice]], text_direction: Literal['lr', 'rl'] = 'lr') -> np.ndarray:
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
        if w == u or w == v:
            return 0
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


def topsort(order: np.ndarray) -> List[int]:
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
        for line in a:
            _visit(line)
        L.append(k)

    for k in range(n):
        _visit(k)
    return L


def moore_neighborhood(current, backtrack):
    operations = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1],
                           [0, -1], [-1, -1]])
    neighbors = (current + operations).astype(int)

    for i, point in enumerate(neighbors):
        if np.all(point == backtrack):
            # we return the sorted neighborhood
            return np.concatenate((neighbors[i:], neighbors[:i]))
    return 0


def boundary_tracing(region):
    """
    Find coordinates of the region's boundary. The region must not have isolated
    points.

    Code copied from
    https://github.com/machine-shop/deepwings/blob/master/deepwings/method_features_extraction/image_processing.py#L185

    Args:
        region: object obtained with skimage.measure.regionprops().

    Returns:
        List of coordinates of pixels in the boundary.
    """

    # creating the binary image
    coords = region.coords
    maxs = np.amax(coords, axis=0)
    binary = np.zeros((maxs[0] + 2, maxs[1] + 2))
    x = coords[:, 1]
    y = coords[:, 0]
    binary[tuple([y, x])] = 1

    # initialization
    # starting point is the most upper left point
    idx_start = 0
    while True:  # asserting that the starting point is not isolated
        start = [y[idx_start], x[idx_start]]
        focus_start = binary[start[0]-1:start[0]+2, start[1]-1:start[1]+2]
        if np.sum(focus_start) > 1:
            break
        idx_start += 1

    # Determining backtrack pixel for the first element
    if (binary[start[0] + 1, start[1]] == 0 and
            binary[start[0]+1, start[1]-1] == 0):
        backtrack_start = [start[0]+1, start[1]]
    else:
        backtrack_start = [start[0], start[1] - 1]

    current = start
    backtrack = backtrack_start
    boundary = []

    while True:
        neighbors_current = moore_neighborhood(current, backtrack)
        y = neighbors_current[:, 0]
        x = neighbors_current[:, 1]
        idx = np.argmax(binary[tuple([y, x])])
        boundary.append(current)
        backtrack = neighbors_current[idx-1]
        current = neighbors_current[idx]

        if (np.all(current == start) and np.all(backtrack == backtrack_start)):
            break

    return np.array(boundary)


def _extend_boundaries(baselines, bin_bl_map):
    # find baseline blob boundaries
    labelled = label(bin_bl_map)
    boundaries = []
    for x in regionprops(labelled):
        try:
            # skip lines with very small bounding polygons
            if x.area < 6:
                logger.info(f'Skipping baseline extension for very small blob of area {x.area}')
                continue
            b = boundary_tracing(x)
            if len(b) > 3:
                boundaries.append(geom.Polygon(b).simplify(0.01).buffer(0))
        except Exception as e:
            logger.warning(f'Boundary tracing failed in baseline elongation: {e}')
            continue

    # extend lines to polygon boundary
    for bl in baselines:
        ls = geom.LineString(bl)
        try:
            boundary_pol = next(filter(lambda x: x.contains(ls), boundaries))
        except Exception:
            continue
        # 'left' side
        if boundary_pol.contains(geom.Point(bl[0])):
            l_point = boundary_pol.boundary.intersection(geom.LineString([(bl[0][0]-10*(bl[1][0]-bl[0][0]),
                                                                           bl[0][1]-10*(bl[1][1]-bl[0][1])), bl[0]]))
            if l_point.geom_type != 'Point':
                bl[0] = np.array(nearest_points(geom.Point(bl[0]), boundary_pol)[1].coords[0], 'int').tolist()
            else:
                bl[0] = np.array(l_point.coords[0], 'int').tolist()
        # 'right' side
        if boundary_pol.contains(geom.Point(bl[-1])):
            r_point = boundary_pol.boundary.intersection(geom.LineString([(bl[-1][0]-10*(bl[-2][0]-bl[-1][0]),
                                                                           bl[-1][1]-10*(bl[-2][1]-bl[-1][1])), bl[-1]]))
            if r_point.geom_type != 'Point':
                bl[-1] = np.array(nearest_points(geom.Point(bl[-1]), boundary_pol)[1].coords[0], 'int').tolist()
            else:
                bl[-1] = np.array(r_point.coords[0], 'int').tolist()
    return baselines


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


def vectorize_lines(im: np.ndarray, threshold: float = 0.17, min_length=5,
                    text_direction: str = 'horizontal'):
    """
    Vectorizes lines from a binarized array.

    Args:
        im (np.ndarray): Array of shape (3, H, W) with the first dimension
                         being probabilities for (start_separators,
                         end_separators, baseline).
        threshold (float): Threshold for baseline blob detection.
        min_length (int): Minimal length of output baselines.
        text_direction (str): Base orientation of the text line (horizontal or
                              vertical).

    Returns:
        [[x0, y0, ... xn, yn], [xm, ym, ..., xk, yk], ... ]
        A list of lists containing the points of all baseline polylines.
    """
    if text_direction not in ['horizontal', 'vertical']:
        raise ValueError(f'Invalid text direction "{text_direction}"')

    # split into baseline and separator map
    st_map = im[0]
    end_map = im[1]
    bl_map = im[2]
    bl_map = filters.sato(bl_map, black_ridges=False, mode='constant')
    bin_bl_map = bl_map > threshold
    # skeletonize
    line_skel = skeletonize(bin_bl_map)
    # find end points
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    line_extrema = np.transpose(np.where((convolve2d(line_skel, kernel, mode='same') == 11) * line_skel))

    mcp = LineMCP(~line_skel)
    try:
        mcp.find_costs(line_extrema)
    except ValueError:
        return []

    lines = [approximate_polygon(line, 3).tolist() for line in mcp.get_connections()]
    # extend baselines to blob boundary
    lines = _extend_boundaries(lines, bin_bl_map)

    # orient lines
    f_st_map = maximum_filter(st_map, size=20)
    f_end_map = maximum_filter(end_map, size=20)

    oriented_lines = []
    for bl in lines:
        l_end = tuple(bl[0])
        r_end = tuple(bl[-1])
        if f_st_map[l_end] - f_end_map[l_end] > 0.2 and f_st_map[r_end] - f_end_map[r_end] < -0.2:
            pass
        elif f_st_map[l_end] - f_end_map[l_end] < -0.2 and f_st_map[r_end] - f_end_map[r_end] > 0.2:
            bl = bl[::-1]
        else:
            if text_direction == 'horizontal':
                logger.debug('Insufficient marker confidences in output. Defaulting to horizontal upright line.')
                if bl[0][1] > bl[-1][1]:
                    bl = bl[::-1]
            else:
                logger.debug('Insufficient marker confidences in output. Defaulting to top-to-bottom line.')
                if bl[0][0] > bl[-1][0]:
                    bl = bl[::-1]
        if geom.LineString(bl).length >= min_length:
            oriented_lines.append([x[::-1] for x in bl])
    return oriented_lines


def vectorize_regions(im: np.ndarray, threshold: float = 0.5):
    """
    Vectorizes lines from a binarized array.

    Args:
        im (np.ndarray): Array of shape (H, W) with the first dimension
                         being a probability distribution over the region.
        threshold (float): Threshold for binarization

    Returns:
        [[x0, y0, ... xn, yn], [xm, ym, ..., xk, yk], ... ]
        A list of lists containing the region polygons.
    """
    bin = im > threshold
    labelled = label(bin)
    boundaries = []
    for x in regionprops(labelled):
        boundary = boundary_tracing(x)
        if len(boundary) > 2:
            boundaries.append(geom.Polygon(boundary))
    # merge regions that overlap
    boundaries = unary_union(boundaries)
    # simplify them afterwards
    if boundaries.geom_type == 'Polygon':
        boundaries = [boundaries.boundary.simplify(10)]
    else:
        boundaries = [x.boundary.simplify(10) for x in boundaries.geoms]
    return [np.array(x.coords, dtype=np.uint)[:, [1, 0]].tolist() for x in boundaries]


def _rotate(image: _T_pil_or_np,
            angle: float,
            center: Any,
            scale: float,
            cval: int = 0,
            order: int = 0,
            use_skimage_warp: bool = False) -> Tuple[AffineTransform, _T_pil_or_np]:
    """
    Rotate an image at an angle with optional scaling

    Args:
        image (PIL.Image.Image or (H, W, C) np.ndarray): Input image
        angle (float): Angle in radians
        center (tuple): unused
        scale (float): x-Axis scaling factor
        cval (int): Padding value
        order (int): Interpolation order
    Returns:
        A tuple containing the transformation matrix and the rotated image.
    Note: this function is much faster applied on PIL images than on numpy ndarrays.
    """
    if isinstance(image, Image.Image):
        rows, cols = image.height, image.width
    else:
        rows, cols = image.shape[:2]
        assert len(image.shape) == 3 or len(image.shape) == 2, 'Image must be 2D or 3D'

    tform = AffineTransform(rotation=angle, scale=(1/scale, 1))
    corners = np.array([
        [0, 0],
        [0, rows - 1],
        [cols - 1, rows - 1],
        [cols - 1, 0]
    ])
    corners = tform.inverse(corners)
    minc = corners[:, 0].min()
    minr = corners[:, 1].min()
    maxc = corners[:, 0].max()
    maxr = corners[:, 1].max()
    out_rows = maxr - minr + 1
    out_cols = maxc - minc + 1
    output_shape = tuple(int(o) for o in np.around((out_rows, out_cols)))
    # fit output image in new shape
    translation = tform([[minc, minr]])
    tform = AffineTransform(rotation=angle, scale=(1/scale, 1), translation=[f for f in translation.flatten()])

    if isinstance(image, Image.Image):
        # PIL is much faster than scipy
        pdata = tform.params.flatten().tolist()[:6]
        resample = {0: Resampling.NEAREST, 1: Resampling.BILINEAR, 2: Resampling.BICUBIC, 3: Resampling.BICUBIC}.get(order, Resampling.NEAREST)
        return tform, image.transform(output_shape[::-1], Transform.AFFINE, data=pdata, resample=resample, fillcolor=cval)

    # params for scipy
    # swap X and Y axis for scipy
    pdata = tform.params.copy()[[1, 0, 2], :][:, [1, 0, 2]]
    # we copy the translation vector
    offset = pdata[:2, 2].copy()
    # scipy expects a 3x3 *linear* matrix (to include channel axis), we don't want the channel axis to be modified
    pdata[:2, 2] = 0
    if use_skimage_warp:
        return tform, warp(image, tform, output_shape=output_shape, order=order, cval=cval, clip=False, preserve_range=True)
    else:
        return tform, affine_transform(image, pdata, offset=(*offset, 0), output_shape=(*output_shape, *image.shape[2:]), cval=cval, order=order)


def line_regions(line, regions):
    """
    Filters a list of regions by line association.

    Args:
        line (list): Polyline representing the line.
        regions (list): List of region polygons

    Returns:
        A list of regions that contain the line mid-point.
    """
    mid_point = geom.LineString(line).interpolate(0.5, normalized=True)

    reg_pols = [geom.Polygon(x) for x in regions]
    regs = []
    for reg_idx, reg_pol in enumerate(reg_pols):
        if reg_pol.contains(mid_point):
            regs.append(regions[reg_idx])
    return regs


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


def _calc_seam(baseline, polygon, angle, im_feats, bias=150):
    """
    Calculates seam between baseline and ROI boundary on one side.

    Adds a baseline-distance-weighted bias to the feature map, masks
    out the bounding polygon and rotates the line so it is roughly
    level.
    """
    MASK_VAL = 99999
    c_min, c_max = int(polygon[:, 0].min()), int(polygon[:, 0].max())
    r_min, r_max = int(polygon[:, 1].min()), int(polygon[:, 1].max())
    patch = im_feats[r_min:r_max+2, c_min:c_max+2].copy()
    # bias feature matrix by distance from baseline
    mask = np.ones_like(patch)
    for line_seg in zip(baseline[:-1] - (c_min, r_min), baseline[1:] - (c_min, r_min)):
        line_locs = draw.line(line_seg[0][1],
                              line_seg[0][0],
                              line_seg[1][1],
                              line_seg[1][0])
        mask[line_locs] = 0
    dist_bias = distance_transform_cdt(mask)
    # absolute mask
    mask = np.array(make_polygonal_mask(polygon-(c_min, r_min), patch.shape[::-1])) <= 128
    # dilate mask to compensate for aliasing during rotation
    mask = binary_erosion(mask, border_value=True, iterations=2)
    # combine weights with features
    patch[mask] = MASK_VAL
    patch += (dist_bias*(np.mean(patch[patch != MASK_VAL])/bias))
    extrema = baseline[(0, -1), :] - (c_min, r_min)
    # scale line image to max 600 pixel width
    scale = min(1.0, 600/(c_max-c_min))
    tform, rotated_patch = _rotate(patch,
                                   angle,
                                   center=extrema[0],
                                   scale=scale,
                                   cval=MASK_VAL,
                                   use_skimage_warp=True)
    # ensure to cut off padding after rotation
    x_offsets = np.sort(np.around(tform.inverse(extrema)[:, 0]).astype('int'))
    rotated_patch = rotated_patch[:, x_offsets[0]:x_offsets[1]+1]
    # infinity pad for seamcarve
    rotated_patch = np.pad(rotated_patch, ((1, 1), (0, 0)),  mode='constant', constant_values=np.inf)
    r, c = rotated_patch.shape
    # fold into shape (c, r-2 3)
    A = np.lib.stride_tricks.as_strided(rotated_patch, (c, r-2, 3), (rotated_patch.strides[1],
                                                                     rotated_patch.strides[0],
                                                                     rotated_patch.strides[0]))
    B = rotated_patch[1:-1, 1:].swapaxes(0, 1)
    backtrack = np.zeros_like(B, dtype='int')
    T = np.empty((B.shape[1]), 'f')
    R = np.arange(-1, len(T)-1)
    for i in np.arange(c-1):
        A[i].min(1, T)
        backtrack[i] = A[i].argmin(1) + R
        B[i] += T
    # backtrack
    seam = []
    j = np.argmin(rotated_patch[1:-1, -1])
    for i in range(c-2, -2, -1):
        seam.append((i+x_offsets[0]+1, j))
        j = backtrack[i, j]
    seam = np.array(seam)[::-1]
    seam_mean = seam[:, 1].mean()
    seam_std = seam[:, 1].std()
    seam[:, 1] = np.clip(seam[:, 1], seam_mean-seam_std, seam_mean+seam_std)
    # rotate back
    seam = tform(seam).astype('int')
    # filter out seam points in masked area of original patch/in padding
    seam = seam[seam.min(axis=1) >= 0, :]
    m = (seam < mask.shape[::-1]).T
    seam = seam[np.logical_and(m[0], m[1]), :]
    seam = seam[np.invert(mask[seam.T[1], seam.T[0]])]
    seam += (c_min, r_min)
    return seam


def _extract_patch(env_up, env_bottom, baseline, offset_baseline, end_points, dir_vec, topline, offset, im_feats, bounds):
    """
    Calculate a line image patch from a ROI and the original baseline.
    """
    upper_polygon = np.concatenate((baseline, env_up[::-1]))
    bottom_polygon = np.concatenate((baseline, env_bottom[::-1]))
    upper_offset_polygon = np.concatenate((offset_baseline, env_up[::-1]))
    bottom_offset_polygon = np.concatenate((offset_baseline, env_bottom[::-1]))

    angle = np.arctan2(dir_vec[1], dir_vec[0])
    roi_polygon = unary_union([geom.Polygon(upper_polygon), geom.Polygon(bottom_polygon)])

    if topline:
        upper_seam = _calc_seam(baseline, upper_polygon, angle, im_feats)
        bottom_seam = _calc_seam(offset_baseline, bottom_offset_polygon, angle, im_feats)
    else:
        upper_seam = _calc_seam(offset_baseline, upper_offset_polygon, angle, im_feats)
        bottom_seam = _calc_seam(baseline, bottom_polygon, angle, im_feats)

    upper_seam = geom.LineString(upper_seam).simplify(5)
    bottom_seam = geom.LineString(bottom_seam).simplify(5)

    # ugly workaround against GEOM parallel_offset bug creating a
    # MultiLineString out of offset LineString
    if upper_seam.parallel_offset(offset//2, side='right').geom_type == 'MultiLineString' or offset == 0:
        upper_seam = np.array(upper_seam.coords, dtype=int)
    else:
        upper_seam = np.array(upper_seam.parallel_offset(offset//2, side='right').coords, dtype=int)[::-1]
    if bottom_seam.parallel_offset(offset//2, side='left').geom_type == 'MultiLineString' or offset == 0:
        bottom_seam = np.array(bottom_seam.coords, dtype=int)
    else:
        bottom_seam = np.array(bottom_seam.parallel_offset(offset//2, side='left').coords, dtype=int)

    # offsetting might produce bounds outside the image. Clip it to the image bounds.
    polygon = np.concatenate(([end_points[0]], upper_seam, [end_points[-1]], bottom_seam[::-1]))
    polygon = geom.Polygon(polygon)
    if not polygon.is_valid:
        polygon = np.concatenate(([end_points[-1]], upper_seam, [end_points[0]], bottom_seam))
        polygon = geom.Polygon(polygon)
    if not polygon.is_valid:
        raise Exception(f'Invalid bounding polygon computed: {explain_validity(polygon)}')
    polygon = np.array(roi_polygon.intersection(polygon).boundary.coords, dtype=int)
    return polygon


def _calc_roi(line, bounds, baselines, suppl_obj, p_dir):
    # interpolate baseline
    ls = geom.LineString(line)
    ip_line = [line[0]]
    dist = 10
    while dist < ls.length:
        ip_line.append(np.array(ls.interpolate(dist).coords[0]))
        dist += 10
    ip_line.append(line[-1])
    ip_line = np.array(ip_line)
    upper_bounds_intersects = []
    bottom_bounds_intersects = []
    for point in ip_line:
        upper_bounds_intersects.append(_ray_intersect_boundaries(point, (p_dir*(-1, 1))[::-1], bounds+1).astype('int'))
        bottom_bounds_intersects.append(_ray_intersect_boundaries(point, (p_dir*(1, -1))[::-1], bounds+1).astype('int'))
    # build polygon between baseline and bbox intersects
    upper_polygon = geom.Polygon(ip_line.tolist() + upper_bounds_intersects)
    bottom_polygon = geom.Polygon(ip_line.tolist() + bottom_bounds_intersects)

    # select baselines at least partially in each polygon
    side_a = [geom.LineString(upper_bounds_intersects)]
    side_b = [geom.LineString(bottom_bounds_intersects)]

    for adj_line in baselines + suppl_obj:
        adj_line = geom.LineString(adj_line)
        if upper_polygon.intersects(adj_line):
            side_a.append(adj_line)
        elif bottom_polygon.intersects(adj_line):
            side_b.append(adj_line)
    side_a = unary_union(side_a).buffer(1).boundary
    side_b = unary_union(side_b).buffer(1).boundary

    def _find_closest_point(pt, intersects):
        spt = geom.Point(pt)
        if intersects.is_empty:
            raise Exception(f'No intersection with boundaries. Shapely intersection object: {intersects.wkt}')
        if intersects.geom_type == 'MultiPoint':
            return min([p for p in intersects.geoms], key=lambda x: spt.distance(x))
        elif intersects.geom_type == 'Point':
            return intersects
        elif intersects.geom_type == 'GeometryCollection' and len(intersects.geoms) > 0:
            t = min([p for p in intersects.geoms], key=lambda x: spt.distance(x))
            if t == 'Point':
                return t
            else:
                return nearest_points(spt, t)[1]
        else:
            raise Exception(f'No intersection with boundaries. Shapely intersection object: {intersects.wkt}')

    env_up = []
    env_bottom = []
    # find orthogonal (to linear regression) intersects with adjacent objects to complete roi
    for point, upper_bounds_intersect, bottom_bounds_intersect in zip(ip_line, upper_bounds_intersects, bottom_bounds_intersects):
        upper_limit = _find_closest_point(point, geom.LineString(
            [point, upper_bounds_intersect]).intersection(side_a))
        bottom_limit = _find_closest_point(point, geom.LineString(
            [point, bottom_bounds_intersect]).intersection(side_b))
        env_up.append(upper_limit.coords[0])
        env_bottom.append(bottom_limit.coords[0])
    env_up = np.array(env_up, dtype='uint')
    env_bottom = np.array(env_bottom, dtype='uint')
    return env_up, env_bottom


def calculate_polygonal_environment(im: Image.Image = None,
                                    baselines: Sequence[Sequence[Tuple[int, int]]] = None,
                                    suppl_obj: Sequence[Sequence[Tuple[int, int]]] = None,
                                    im_feats: np.ndarray = None,
                                    scale: Tuple[int, int] = None,
                                    topline: bool = False,
                                    raise_on_error: bool = False):
    """
    Given a list of baselines and an input image, calculates a polygonal
    environment around each baseline.

    Args:
        im: grayscale input image (mode 'L')
        baselines: List of lists containing a single baseline per entry.
        suppl_obj: List of lists containing additional polylines that should be
                   considered hard boundaries for polygonizaton purposes. Can
                   be used to prevent polygonization into non-text areas such
                   as illustrations or to compute the polygonization of a
                   subset of the lines in an image.
        im_feats: An optional precomputed seamcarve energy map. Overrides data
                  in `im`. The default map is `gaussian_filter(sobel(im), 2)`.
        scale: A 2-tuple (h, w) containing optional scale factors of the input.
               Values of 0 are used for aspect-preserving scaling. `None` skips
               input scaling.
        topline: Switch to change default baseline location for offset
                 calculation purposes. If set to False, baselines are assumed
                 to be on the bottom of the text line and will be offset
                 upwards, if set to True, baselines are on the top and will be
                 offset downwards. If set to None, no offset will be applied.
        raise_on_error: Raises error instead of logging them when they are
                        not-blocking
    Returns:
        List of lists of coordinates. If no polygonization could be compute for
        a baseline `None` is returned instead.
    """
    if scale is not None and (scale[0] > 0 or scale[1] > 0):
        w, h = im.size
        oh, ow = scale
        if oh == 0:
            oh = int(h * ow/w)
        elif ow == 0:
            ow = int(w * oh/h)
        im = im.resize((ow, oh))
        scale = np.array((ow/w, oh/h))
        # rescale baselines
        baselines = [(np.array(bl) * scale).astype('int').tolist() for bl in baselines]
        # rescale suppl_obj
        if suppl_obj is not None:
            suppl_obj = [(np.array(bl) * scale).astype('int').tolist() for bl in suppl_obj]

    if im_feats is None:
        bounds = np.array(im.size, dtype=float) - 1
        im = np.array(im.convert('L'))
        # compute image gradient
        im_feats = gaussian_filter(sobel(im), 0.5)
    else:
        bounds = np.array(im_feats.shape[::-1], dtype=float) - 1

    polygons = []
    if suppl_obj is None:
        suppl_obj = []

    for idx, line in enumerate(baselines):
        try:
            end_points = (line[0], line[-1])
            line = geom.LineString(line)
            offset = default_specs.SEGMENTATION_HYPER_PARAMS['line_width'] if topline is not None else 0
            offset_line = line.parallel_offset(offset, side='left' if topline else 'right')
            line = np.array(line.coords, dtype=float)
            offset_line = np.array(offset_line.coords, dtype=float)

            # calculate magnitude-weighted average direction vector
            lengths = np.linalg.norm(np.diff(line.T), axis=0)
            p_dir = np.mean(np.diff(line.T) * lengths/lengths.sum(), axis=1)
            p_dir = (p_dir.T / np.sqrt(np.sum(p_dir**2, axis=-1)))
            env_up, env_bottom = _calc_roi(line, bounds, baselines[:idx] + baselines[idx+1:], suppl_obj, p_dir)

            polygons.append(_extract_patch(env_up,
                                           env_bottom,
                                           line.astype('int'),
                                           offset_line.astype('int'),
                                           end_points,
                                           p_dir,
                                           topline,
                                           offset,
                                           im_feats,
                                           bounds))
        except Exception as e:
            if raise_on_error:
                raise
            logger.warning(f'Polygonizer failed on line {idx}: {e}')
            polygons.append(None)

    if scale is not None:
        polygons = [(np.array(pol)/scale).astype('uint').tolist() if pol is not None else None for pol in polygons]
    return polygons


def polygonal_reading_order(lines: Sequence[Dict],
                            text_direction: Literal['lr', 'rl'] = 'lr',
                            regions: Optional[Sequence[geom.Polygon]] = None) -> Sequence[int]:
    """
    Given a list of baselines and regions, calculates the correct reading order
    and applies it to the input.

    Args:
        lines: List of tuples containing the baseline and its polygonization.
        regions: List of region polygons.
        text_direction: Set principal text direction for column ordering. Can
                        be 'lr' or 'rl'

    Returns:
        The indices of the ordered input.
    """
    lines = [(line['tags']['type'], line['baseline'], line['boundary']) for line in lines]

    bounds = []
    if regions is None:
        regions = []
    region_lines = [[] for _ in range(len(regions))]
    indizes = {}
    for line_idx, line in enumerate(lines):
        s_line = geom.LineString(line[1])
        in_region = False
        for idx, reg in enumerate(regions):
            if is_in_region(s_line, reg):
                region_lines[idx].append((line_idx, (slice(s_line.bounds[1], s_line.bounds[3]),
                                                     slice(s_line.bounds[0], s_line.bounds[2]))))
                in_region = True
                break
        if not in_region:
            bounds.append((slice(s_line.bounds[1], s_line.bounds[3]),
                           slice(s_line.bounds[0], s_line.bounds[2])))
            indizes[line_idx] = ('line', line)
    # order everything in regions
    intra_region_order = [[] for _ in range(len(regions))]
    for idx, reg in enumerate(regions):
        if len(region_lines[idx]) > 0:
            order = reading_order([x[1] for x in region_lines[idx]], text_direction)
            lsort = topsort(order)
            intra_region_order[idx] = [region_lines[idx][i][0] for i in lsort]
            reg = reg.bounds
            bounds.append((slice(reg[1], reg[3]), slice(reg[0], reg[2])))
            indizes[line_idx+idx+1] = ('region', idx)
    # order unassigned lines and regions
    order = reading_order(bounds, text_direction)
    lsort = topsort(order)
    sidz = sorted(indizes.keys())
    lsort = [sidz[i] for i in lsort]
    ordered_idxs = []
    for i in lsort:
        if indizes[i][0] == 'line':
            ordered_idxs.append(i)
        else:
            ordered_idxs.extend(intra_region_order[indizes[i][1]])
    return ordered_idxs


def is_in_region(line: geom.LineString, region: geom.Polygon) -> bool:
    """
    Tests if a line is inside a region, i.e. if the mid point of the baseline
    is inside the region.

    Args:
        line: line to test
        region: region to test against

    Returns:
        False if line is not inside region, True otherwise.
    """
    l_obj = line.interpolate(0.5, normalized=True)
    return region.contains(l_obj)


def neural_reading_order(lines: Sequence[Dict],
                         text_direction: str = 'lr',
                         regions: Optional[Sequence[geom.Polygon]] = None,
                         im_size: Tuple[int, int] = None,
                         model: 'TorchVGSLModel' = None,
                         class_mapping: Dict[str, int] = None) -> Sequence[int]:
    """
    Given a list of baselines and regions, calculates the correct reading order
    and applies it to the input.

    Args:
        lines: List of tuples containing the baseline and its polygonization.
        model: torch Module for

    Returns:
        The indices of the ordered input.
    """
    if len(lines) == 0:
        return None
    elif len(lines) == 1:
        return torch.tensor([0])

    lines = [(line['tags']['type'], line['baseline'], line['boundary']) for line in lines]
    # construct all possible pairs
    h, w = im_size
    features = []
    for i in lines:
        for j in lines:
            if i == j and len(lines) != 1:
                continue
            num_classes = len(class_mapping) + 1
            cl_i = torch.zeros(num_classes, dtype=torch.float)
            cl_j = torch.zeros(num_classes, dtype=torch.float)
            cl_i[class_mapping.get(i[0], 0)] = 1
            cl_j[class_mapping.get(j[0], 0)] = 1
            line_coords_i = np.array(i[1]) / (w, h)
            line_center_i = np.mean(line_coords_i, axis=0)
            line_coords_j = np.array(j[1]) / (w, h)
            line_center_j = np.mean(line_coords_j, axis=0)
            features.append(torch.cat((cl_i,
                                       torch.tensor(line_center_i, dtype=torch.float),  # lin
                                       torch.tensor(line_coords_i[0, :], dtype=torch.float),
                                       torch.tensor(line_coords_i[-1, :], dtype=torch.float),
                                       cl_j,
                                       torch.tensor(line_center_j, dtype=torch.float),  # lin
                                       torch.tensor(line_coords_j[0, :], dtype=torch.float),
                                       torch.tensor(line_coords_j[-1, :], dtype=torch.float))))
    features = torch.stack(features)
    output = F.sigmoid(model(features))

    order = torch.zeros((len(lines), len(lines)))
    idx = 0
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j and len(lines) != 1:
                continue
            order[i, j] = output[idx]
            idx += 1
    # decode order relation matrix
    path = _greedy_order_decoder(order)
    return path


def _greedy_order_decoder(P):
    """
    A greedy decoder of order-relation matrix. For each position in the
    reading order we select the most probable one, then move to the next
    position. Most probable for position:

    .. math::
        z^{\\star}_t = \\argmax_{(s,\\nu) \\ni z^{\\star}}
        \\prod_{(s',\\nu') \\in z^\\star}{\\tilde{P}(Y=1\\mid s',s)}
        \\times \\prod_{\\substack{(s'',\\nu'') \\ni z^\\star\\
         s'' \\ne s}}{\\tilde{P}(r=0\\mid s'',s)}, 1\\le t \\le n
    """
    A = P + torch.finfo(torch.float).eps
    N = P.shape[0]
    A = (A + (1-A).T)/2
    for i in range(A.shape[0]):
        A[i, i] = torch.finfo(torch.float).eps
    best_path = []
    # use log(p(R\mid s',s)) to shift multiplication to sum
    lP = torch.log(A)
    for i in range(N):
        lP[i, i] = 0
    for t in range(N):
        for i in range(N):
            idx = torch.argmax(lP.sum(axis=1))
            if idx not in best_path:
                best_path.append(idx)
                lP[idx, :] = lP[:, idx]
                lP[:, idx] = 0
                break
    return torch.tensor(best_path)


def scale_regions(regions: Sequence[Tuple[List[int], List[int]]],
                  scale: Union[float, Tuple[float, float]]) -> Sequence[Tuple[List, List]]:
    """
    Scales baselines/polygon coordinates by a certain factor.

    Args:
        lines: List of tuples containing the baseline and its polygonization.
        scale: Scaling factor
    """
    if isinstance(scale, float):
        scale = (scale, scale)
    scaled_regions = []
    for region in regions:
        scaled_regions.append((np.array(region) * scale).astype('uint').tolist())
    return scaled_regions


def scale_polygonal_lines(lines: Sequence[Tuple[List, List]], scale: Union[float, Tuple[float, float]]) -> Sequence[Tuple[List, List]]:
    """
    Scales baselines/polygon coordinates by a certain factor.

    Args:
        lines: List of tuples containing the baseline and its polygonization.
        scale: Scaling factor
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
    for dir in ((1, -1), (-1, 1)):
        w = (uv * dir * (1, -1))[::-1]
        z = np.dot(v, w)
        t1 = np.cross(v, u) / (z + np.finfo(float).eps)
        t2 = np.dot(u, w) / (z + np.finfo(float).eps)
        t1 = t1[np.logical_and(t2 >= 0.0, t2 <= 1.0)]
        points.extend(bp + (t1[np.where(t1 >= 0)[0].min()] * (uv * dir)))
    return np.array(points)


def compute_polygon_section(baseline: Sequence[Tuple[int, int]],
                            boundary: Sequence[Tuple[int, int]],
                            dist1: int,
                            dist2: int) -> Tuple[Tuple[int, int]]:
    """
    Given a baseline, polygonal boundary, and two points on the baseline return
    the rectangle formed by the orthogonal cuts on that baseline segment. The
    resulting polygon is not garantueed to have a non-zero area.

    The distance can be larger than the actual length of the baseline if the
    baseline endpoints are inside the bounding polygon. In that case the
    baseline will be extrapolated to the polygon edge.

    Args:
        baseline: A polyline ((x1, y1), ..., (xn, yn))
        boundary: A bounding polygon around the baseline (same format as
                  baseline). Last and first point are automatically connected.
        dist1: Absolute distance along the baseline of the first point.
        dist2: Absolute distance along the baseline of the second point.

    Returns:
        A sequence of polygon points.
    """
    # find baseline segments the points are in
    if dist1 == 0:
        dist1 = np.finfo(float).eps
    if dist2 == 0:
        dist2 = np.finfo(float).eps
    boundary_pol = geom.Polygon(boundary)
    bl = np.array(baseline)
    # extend first/last segment of baseline if not on polygon boundary
    if boundary_pol.contains(geom.Point(bl[0])):
        logger.debug(f'Extending leftmost end of baseline {bl} to polygon boundary')
        l_point = boundary_pol.boundary.intersection(geom.LineString(
            [(bl[0][0]-10*(bl[1][0]-bl[0][0]), bl[0][1]-10*(bl[1][1]-bl[0][1])), bl[0]]))
        # intersection is incidental with boundary so take closest point instead
        if l_point.geom_type != 'Point':
            bl[0] = np.array(nearest_points(geom.Point(bl[0]), boundary_pol)[1].coords[0], 'int')
        else:
            bl[0] = np.array(l_point.coords[0], 'int')
    if boundary_pol.contains(geom.Point(bl[-1])):
        logger.debug(f'Extending rightmost end of baseline {bl} to polygon boundary')
        r_point = boundary_pol.boundary.intersection(geom.LineString(
            [(bl[-1][0]-10*(bl[-2][0]-bl[-1][0]), bl[-1][1]-10*(bl[-2][1]-bl[-1][1])), bl[-1]]))
        if r_point.geom_type != 'Point':
            bl[-1] = np.array(nearest_points(geom.Point(bl[-1]), boundary_pol)[1].coords[0], 'int')
        else:
            bl[-1] = np.array(r_point.coords[0], 'int')
    dist1 = min(geom.LineString(bl).length - np.finfo(float).eps, dist1)
    dist2 = min(geom.LineString(bl).length - np.finfo(float).eps, dist2)
    dists = np.cumsum(np.diag(np.roll(squareform(pdist(bl)), 1)))
    segs_idx = np.searchsorted(dists, [dist1, dist2])
    segs = np.dstack((bl[segs_idx-1], bl[segs_idx]))
    # compute unit vector of segments (NOT orthogonal)
    norm_vec = (segs[..., 1] - segs[..., 0])
    norm_vec_len = np.sqrt(np.sum(norm_vec**2, axis=1))
    unit_vec = norm_vec / np.tile(norm_vec_len, (2, 1)).T
    # find point start/end point on segments
    seg_dists = (dist1, dist2) - dists[segs_idx-1]
    seg_points = segs[..., 0] + (seg_dists * unit_vec.T).T
    # get intersects
    bounds = np.array(boundary)
    try:
        points = [_test_intersect(point, uv[::-1], bounds).round() for point, uv in zip(seg_points, unit_vec)]
    except ValueError:
        logger.debug('No intercepts with polygon (possibly misshaped polygon)')
        return seg_points.astype('int').tolist()
    o = np.int_(points[0]).reshape(-1, 2).tolist()
    o.extend(np.int_(np.roll(points[1], 2)).reshape(-1, 2).tolist())
    return tuple(o)


def _bevelled_warping_envelope(baseline: np.ndarray,
                               output_bl_start: Tuple[float, float],
                               output_shape: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Calculates the source and target envelope for a piecewise affine transform
    """
    def _as_int_tuple(x):
        return tuple(int(i) for i in x)

    envelope_dy = [-output_bl_start[1], output_shape[0] - output_bl_start[1]]
    diff_bl = np.diff(baseline, axis=0)
    diff_bl_normed = diff_bl / np.linalg.norm(diff_bl, axis=1)[:, None]
    l_bl = len(baseline)
    cum_lens = np.cumsum([0] + np.linalg.norm(diff_bl, axis=1).tolist())

    bl_seg_normals = np.array([-diff_bl_normed[:, 1], diff_bl_normed[:, 0]]).T
    ini_point = baseline[0] - diff_bl_normed[0] * output_bl_start[0]
    source_envelope = [
        _as_int_tuple(ini_point + envelope_dy[0]*bl_seg_normals[0]),
        _as_int_tuple(ini_point + envelope_dy[1]*bl_seg_normals[0]),
    ]
    target_envelope = [
        (0, 0),
        (0, output_shape[0])
    ]
    MAX_BEVEL_WIDTH = output_shape[0] / 3
    BEVEL_STEP_WIDTH = MAX_BEVEL_WIDTH / 2

    for k in range(l_bl-2):
        pt = baseline[k+1]
        seg_prev = baseline[k] - pt
        seg_next = baseline[k+2] - pt
        bevel_prev = seg_prev / max(2., np.linalg.norm(seg_prev) / MAX_BEVEL_WIDTH)
        bevel_next = seg_next / max(2., np.linalg.norm(seg_next) / MAX_BEVEL_WIDTH)
        bevel_nsteps = max(1, np.round((np.linalg.norm(bevel_prev) + np.linalg.norm(bevel_next)) / BEVEL_STEP_WIDTH))
        l_prev = np.linalg.norm(bevel_prev)
        l_next = np.linalg.norm(bevel_next)
        for i in range(int(bevel_nsteps)+1):
            # bezier interp
            t = i / bevel_nsteps
            tpt = pt + (1-t)**2 * bevel_prev + t**2 * bevel_next
            tx = output_bl_start[0] + cum_lens[k+1] - (1-t)**2 * l_prev + t**2 * l_next
            tnormal = (1-t) * bl_seg_normals[k] + t * bl_seg_normals[k+1]
            tnormal /= np.linalg.norm(tnormal)
            source_points = [_as_int_tuple(tpt + envelope_dy[0]*tnormal), _as_int_tuple(tpt + envelope_dy[1]*tnormal)]
            target_points = [(int(tx), 0), (int(tx), output_shape[0])]
            # avoid duplicate points leading to singularities
            if source_points[0] == source_envelope[-2] or source_points[1] == source_envelope[-1] or target_points[0] == target_envelope[-2]:
                continue
            source_envelope += source_points
            target_envelope += target_points

    end_point = baseline[-1] + diff_bl_normed[-1]*(output_shape[1]-cum_lens[-1]-output_bl_start[0])
    source_envelope += [
        end_point + envelope_dy[0]*bl_seg_normals[-1],
        end_point + envelope_dy[1]*bl_seg_normals[-1],
    ]
    target_envelope += [
        (output_shape[1], 0),
        (output_shape[1], output_shape[0])
    ]
    return source_envelope, target_envelope


def make_polygonal_mask(polygon: np.ndarray, shape: Tuple[int, int]) -> Image.Image:
    """
    Creates a mask from a polygon.

    Args:
        polygon: A polygon as a list of points.
        shape: The shape of the mask to create.

    Returns:
        A PIL.Image.Image instance containing the mask.
    """
    mask = Image.new('L', shape, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon.astype(int).tolist()], fill=255, width=2)
    return mask


def apply_polygonal_mask(img: Image.Image, polygon: np.ndarray, cval: int = 0) -> Image.Image:
    """
    Extract the polygonal mask of an image.
    """
    mask = make_polygonal_mask(polygon, img.size)
    out = Image.new(img.mode, (img.width, img.height), cval)
    out.paste(img, mask=mask)
    return out


def extract_polygons(im: Image.Image,
                     bounds: "Segmentation",
                     legacy: bool = False) -> Generator[Tuple[Image.Image, Union["BBoxLine", "BaselineLine"],], None, None]:
    """
    Yields the subimages of image im defined in the list of bounding polygons
    with baselines preserving order.

    Args:
        im: Input image
        bounds: A Segmentation class containing a bounding box or baseline
                segmentation.
        legacy: Use the old, slow, and deprecated path

    Yields:
        The extracted subimage, and the corresponding bounding box or baseline
    """
    if bounds.type == 'baselines':
        # select proper interpolation scheme depending on shape
        if im.mode == '1':
            order = 0
            im = im.convert('L')
        else:
            order = 1

        for line in bounds.lines:
            if line.boundary is None:
                raise KrakenInputException('No boundary given for line')
            if len(line.baseline) < 2 or geom.LineString(line.baseline).length < 5:
                raise KrakenInputException('Baseline length below minimum 5px')
            pl = np.array(line.boundary)
            baseline = np.array(line.baseline)
            c_min, c_max = int(pl[:, 0].min()), int(pl[:, 0].max())
            r_min, r_max = int(pl[:, 1].min()), int(pl[:, 1].max())

            imshape = np.array([im.height, im.width])

            if (pl < 0).any() or (pl.max(axis=0)[::-1] >= imshape).any():
                raise KrakenInputException('Line polygon outside of image bounds')
            if (baseline < 0).any() or (baseline.max(axis=0)[::-1] >= imshape).any():
                raise KrakenInputException('Baseline outside of image bounds')

            if legacy:
                im = np.asarray(im)
                # Old, slow, and deprecated path
                # fast path for straight baselines requiring only rotation
                if len(baseline) == 2:
                    baseline = baseline.astype(float)
                    # calculate direction vector
                    lengths = np.linalg.norm(np.diff(baseline.T), axis=0)
                    p_dir = np.mean(np.diff(baseline.T) * lengths/lengths.sum(), axis=1)
                    p_dir = (p_dir.T / np.sqrt(np.sum(p_dir**2, axis=-1)))
                    angle = np.arctan2(p_dir[1], p_dir[0])
                    patch = im[r_min:r_max+1, c_min:c_max+1].copy()
                    offset_polygon = pl - (c_min, r_min)
                    offset_polygon2 = offset_polygon.flatten().tolist()
                    img = Image.new('L', patch.shape[:2][::-1], 0)
                    ImageDraw.Draw(img).polygon(offset_polygon2, outline=1, fill=1)
                    mask = np.asarray(img, dtype=bool)
                    patch[np.invert(mask)] = 0
                    extrema = offset_polygon[(0, -1), :]
                    # scale line image to max 600 pixel width
                    tform, rotated_patch = _rotate(patch, angle, center=extrema[0], scale=1.0, cval=0)
                    i = Image.fromarray(rotated_patch.astype('uint8'))
                # normal slow path with piecewise affine transformation
                else:
                    if len(pl) > 50:
                        pl = approximate_polygon(pl, 2)
                    full_polygon = subdivide_polygon(pl, preserve_ends=True)
                    pl = geom.MultiPoint(full_polygon)

                    bl = zip(baseline[:-1:], baseline[1::])
                    bl = [geom.LineString(x) for x in bl]
                    cum_lens = np.cumsum([0] + [line.length for line in bl])
                    # distance of intercept from start point and number of line segment
                    control_pts = []
                    for point in pl.geoms:
                        npoint = np.array(point.coords)[0]
                        line_idx, dist, intercept = min(((idx, line.project(point),
                                                        np.array(line.interpolate(line.project(point)).coords)) for idx, line in enumerate(bl)),
                                                        key=lambda x: np.linalg.norm(npoint-x[2]))
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
                    c_dst_min, c_dst_max = int(pol_dst_pts[:, 0].min()), int(pol_dst_pts[:, 0].max())
                    r_dst_min, r_dst_max = int(pol_dst_pts[:, 1].min()), int(pol_dst_pts[:, 1].max())
                    output_shape = np.around((r_dst_max - r_dst_min + 1, c_dst_max - c_dst_min + 1))
                    patch = im[r_min:r_max+1, c_min:c_max+1].copy()
                    # offset src points by patch shape
                    offset_polygon = full_polygon - (c_min, r_min)
                    offset_baseline = baseline - (c_min, r_min)
                    # offset dst point by dst polygon shape
                    offset_bl_dst_pts = bl_dst_pts - (c_dst_min, r_dst_min)
                    offset_pol_dst_pts = pol_dst_pts - (c_dst_min, r_dst_min)
                    # mask out points outside bounding polygon
                    offset_polygon2 = offset_polygon.flatten().tolist()
                    img = Image.new('L', patch.shape[:2][::-1], 0)
                    ImageDraw.Draw(img).polygon(offset_polygon2, outline=1, fill=1)
                    mask = np.asarray(img, dtype=bool)
                    patch[np.invert(mask)] = 0
                    # estimate piecewise transform
                    src_points = np.concatenate((offset_baseline, offset_polygon))
                    dst_points = np.concatenate((offset_bl_dst_pts, offset_pol_dst_pts))
                    tform = FastPiecewiseAffineTransform()
                    tform.estimate(src_points, dst_points)
                    o = warp(patch, tform.inverse, output_shape=output_shape, preserve_range=True, order=order)
                    i = Image.fromarray(o.astype('uint8'))

            else:  # if not legacy
                # new, fast, and efficient path
                # fast path for straight baselines requiring only rotation
                if len(baseline) == 2:
                    baseline = baseline.astype(float)
                    # calculate direction vector
                    lengths = np.linalg.norm(np.diff(baseline.T), axis=0)
                    p_dir = np.mean(np.diff(baseline.T) * lengths/lengths.sum(), axis=1)
                    p_dir = (p_dir.T / np.sqrt(np.sum(p_dir**2, axis=-1)))
                    angle = np.arctan2(p_dir[1], p_dir[0])
                    # crop out bounding box
                    patch = im.crop((c_min, r_min, c_max+1, r_max+1))
                    offset_polygon = pl - (c_min, r_min)
                    patch = apply_polygonal_mask(patch, offset_polygon, cval=0)
                    extrema = offset_polygon[(0, -1), :]
                    tform, i = _rotate(patch, angle, center=extrema[0], scale=1.0, cval=0, order=order)
                # normal slow path with piecewise affine transformation
                else:
                    if len(pl) > 50:
                        pl = approximate_polygon(pl, 2)
                    full_polygon = subdivide_polygon(pl, preserve_ends=True)

                    # baseline segment vectors
                    diff_bl = np.diff(baseline, axis=0)
                    diff_bl_norms = np.linalg.norm(diff_bl, axis=1)
                    diff_bl_normed = diff_bl / diff_bl_norms[:, None]

                    l_poly = len(full_polygon)
                    cum_lens = np.cumsum([0] + np.linalg.norm(diff_bl, axis=1).tolist())

                    # calculate baseline destination points :
                    bl_dst_pts = baseline[0] + np.dstack((cum_lens, np.zeros_like(cum_lens)))[0]

                    # calculate bounding polygon destination points :
                    # diff[k, p] = baseline[k] - polygon[p]
                    poly_bl_diff = full_polygon[None, :] - baseline[:-1, None]
                    # local x coordinates of polygon points on baseline segments
                    # x[k, p] = (baseline[k] - polygon[p]) . (baseline[k+1] - baseline[k]) / |baseline[k+1] - baseline[k]|
                    poly_bl_x = np.einsum('kpm,km->kp', poly_bl_diff, diff_bl_normed)
                    # distance to baseline segments
                    poly_bl_segdist = np.maximum(-poly_bl_x, poly_bl_x - diff_bl_norms[:, None])
                    # closest baseline segment index
                    poly_closest_bl = np.argmin((poly_bl_segdist), axis=0)
                    poly_bl_x = poly_bl_x[poly_closest_bl, np.arange(l_poly)]
                    poly_bl_diff = poly_bl_diff[poly_closest_bl, np.arange(l_poly)]
                    # signed distance between polygon points and baseline segments (to get y coordinates)
                    poly_bl_y = np.cross(diff_bl_normed[poly_closest_bl], poly_bl_diff)
                    # final destination points
                    pol_dst_pts = np.array(
                        [cum_lens[poly_closest_bl] + poly_bl_x, poly_bl_y]
                    ).T + baseline[:1]

                    # extract bounding box patch
                    c_dst_min, c_dst_max = int(pol_dst_pts[:, 0].min()), int(pol_dst_pts[:, 0].max())
                    r_dst_min, r_dst_max = int(pol_dst_pts[:, 1].min()), int(pol_dst_pts[:, 1].max())
                    output_shape = np.around((r_dst_max - r_dst_min + 1, c_dst_max - c_dst_min + 1))
                    patch = im.crop((c_min, r_min, c_max+1, r_max+1))
                    # offset src points by patch shape
                    offset_polygon = full_polygon - (c_min, r_min)
                    offset_baseline = baseline - (c_min, r_min)
                    # offset dst point by dst polygon shape
                    offset_bl_dst_pts = bl_dst_pts - (c_dst_min, r_dst_min)
                    # mask out points outside bounding polygon
                    patch = apply_polygonal_mask(patch, offset_polygon, cval=0)

                    # estimate piecewise transform by beveling angles
                    source_envelope, target_envelope = _bevelled_warping_envelope(offset_baseline, offset_bl_dst_pts[0], output_shape)
                    # mesh for PIL, as (box, quad) tuples : box is (NW, SE) and quad is (NW, SW, SE, NE)
                    deform_mesh = [
                        (
                            (*target_envelope[i], *target_envelope[i+3]),
                            (*source_envelope[i], *source_envelope[i+1], *source_envelope[i+3], *source_envelope[i+2])
                        )
                        for i in range(0, len(source_envelope)-3, 2)
                    ]
                    # warp
                    resample = {0: Resampling.NEAREST, 1: Resampling.BILINEAR, 2: Resampling.BICUBIC, 3: Resampling.BICUBIC}.get(order, Resampling.NEAREST)
                    i = patch.transform((output_shape[1], output_shape[0]), Image.MESH, data=deform_mesh, resample=resample)

            yield i.crop(i.getbbox()), line
    else:
        if bounds.text_direction.startswith('vertical'):
            angle = 90
        else:
            angle = 0
        for line in bounds.lines:
            box = line.bbox
            if isinstance(box, tuple):
                box = list(box)
            if (box < [0, 0, 0, 0] or box[::2] >= [im.size[0], im.size[0]] or
                    box[1::2] >= [im.size[1], im.size[1]]):
                logger.error('bbox {} is outside of image bounds {}'.format(box, im.size))
                raise KrakenInputException('Line outside of image bounds')
            yield im.crop(box).rotate(angle, expand=True), box
