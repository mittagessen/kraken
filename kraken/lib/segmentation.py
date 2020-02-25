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

from scipy.stats import linregress
from scipy.signal import savgol_filter
from scipy.spatial import distance_matrix, Delaunay
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.ndimage.morphology import distance_transform_cdt

from shapely.ops import nearest_points, unary_union, linemerge

from skimage import draw
from skimage.filters import apply_hysteresis_threshold, sobel
from skimage.measure import approximate_polygon, subdivide_polygon, find_contours
from skimage.morphology import medial_axis
from skimage.transform import PiecewiseAffineTransform, SimilarityTransform, AffineTransform, warp

from typing import List, Tuple, Union, Dict, Any, Sequence, Optional

logger = logging.getLogger('kraken')

__all__ = ['reading_order',
           'denoising_hysteresis_thresh',
           'vectorize_lines',
           'calculate_polygonal_environment',
           'polygonal_reading_order',
           'scale_polygonal_lines',
           'scale_regions',
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


def _interpolate_lines(clusters, elongation_offset, extent):
    """
    Interpolates the baseline clusters.
    """
    logger.debug('Reticulating splines')
    lines = []
    extent = geom.Polygon([(0, 0), (extent[1]-1, 0), (extent[1]-1, extent[0]-1), (0, extent[0]-1), (0, 0)])
    for cluster in clusters[1:]:
        # find start-end point
        points = [point for edge in cluster for point in edge]
        dists = squareform(pdist(points))
        i, j = np.unravel_index(dists.argmax(), dists.shape)
        if points[i][1] > points[j][1]:
            i, j = j, i
        # build adjacency matrix for shortest path algo
        adj_mat = np.full_like(dists, np.inf)
        for l, r in cluster:
            idx_l = points.index(l)
            idx_r = points.index(r)
            adj_mat[idx_l, idx_r] = dists[idx_l, idx_r]
        # shortest path
        _, pr = shortest_path(adj_mat, directed=False, return_predecessors=True, indices=i)
        k = j
        line = [points[j]]
        while pr[k] != -9999:
            k = pr[k]
            line.append(points[k])
        # smooth line
        line = np.array(line[::-1])
        filter_len = min(len(line)//2*2-1, 11)
        poly_order = min(filter_len-1, 3)
        y = savgol_filter(line[:,0], filter_len, polyorder=poly_order)
        x = savgol_filter(line[:,1], filter_len, polyorder=poly_order)
        line = np.around(np.dstack((x, y)))[0]
        line = approximate_polygon(line, 1)
        lr_dir = line[0] - line[1]
        lr_dir = (lr_dir.T  / np.sqrt(np.sum(lr_dir**2,axis=-1))) * elongation_offset
        line[0] = line[0] + lr_dir
        rr_dir = line[-1] - line[-2]
        rr_dir = (rr_dir.T  / np.sqrt(np.sum(rr_dir**2,axis=-1))) * elongation_offset
        line[-1] = line[-1] + rr_dir
        ins = geom.LineString(line).intersection(extent)
        if ins.type == 'MultiLineString':
            ins = linemerge(ins)
            # skip lines that don't merge cleanly
            if ins.type != 'LineString':
                continue
        line = np.array(ins, dtype='uint')
        lines.append(line.tolist())
    return lines


def vectorize_lines(im: np.ndarray, threshold: float = 0.2, min_sp_dist: int = 10):
    """
    Vectorizes lines from a binarized array.

    Args:
        im (np.ndarray): Array of shape (3, H, W) with the first dimension
                         being probabilities for (start_separators,
                         end_separators, baseline).

    Returns:
        [[x0, y0, ... xn, yn], [xm, ym, ..., xk, yk], ... ]
        A list of lists containing the points of all baseline polylines.
    """
    # split into baseline and separator map
    st_map = im[0]
    end_map = im[1]
    sep_map = st_map + end_map
    bl_map = im[2]
    # binarize
    bin = im > threshold
    skel, skel_dist_map = medial_axis(bin[2], return_distance=True)
    elongation_offset = np.max(skel_dist_map)
    sp_can = _find_superpixels(skel, heatmap=bl_map, min_sp_dist=min_sp_dist)
    if not sp_can.size:
        logger.warning('No superpixel candidates found in network output. Likely empty page.')
        return []
    intensities = _compute_sp_states(sp_can, bl_map, sep_map)
    clusters = _cluster_lines(intensities)
    lines = _interpolate_lines(clusters, elongation_offset, bl_map.shape)
    return lines


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
    contours = find_contours(bin, 0.5, fully_connected='high', positive_orientation='high')
    if len(contours) == 0:
        return contours
    approx_contours = []
    for contour in contours:
        approx_contours.append(approximate_polygon(contour, 1).astype('uint').tolist())
    return approx_contours


def _rotate(image, angle, center, scale, cval=0):
    """
    Rotate function taken mostly from scikit image. Main difference is that
    this one allows dimensional scaling and records the final translation
    to ensure no image content is lost. This is needed to rotate the seam
    back into the original image.
    """
    rows, cols = image.shape[0], image.shape[1]
    tform1 = SimilarityTransform(translation=center)
    tform2 = SimilarityTransform(rotation=angle)
    tform3 = SimilarityTransform(translation=-center)
    tform4 = AffineTransform(scale=(1/scale, 1))
    tform = tform4 + tform3 + tform2 + tform1
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
    output_shape = np.around((out_rows, out_cols))
    # fit output image in new shape
    translation = (minc, minr)
    tform5 = SimilarityTransform(translation=translation)
    tform = tform5 + tform
    tform.params[2] = (0, 0, 1)
    return tform, warp(image, tform, output_shape=output_shape, order=0, cval=cval, clip=False, preserve_range=True)


def calculate_polygonal_environment(im: PIL.Image.Image,
                                    baselines: Sequence[Tuple[int, int]] = None,
                                    suppl_obj: Sequence[Tuple[int, int]] = None,
                                    im_feats: np.array = None,
                                    scale: Tuple[int, int] = None):
    """
    Given a list of baselines and an input image, calculates a polygonal
    environment around each baseline.

    Args:
        im (PIL.Image): grayscale input image (mode 'L')
        baselines (sequence): List of lists containing a single baseline per
                              entry.
        suppl_obj (sequence): List of lists containing additional polylines
                              that should be considered hard boundaries for
                              polygonizaton purposes. Can be used to prevent
                              polygonization into non-text areas such as
                              illustrations or to compute the polygonization of
                              a subset of the lines in an image.
        im_feats (numpy.array): An optional precomputed seamcarve energy map.
                                Overrides data in `im`. The default map is
                                `gaussian_filter(sobel(im), 2)`.
        scale (tuple): A 2-tuple (h, w) containing optional scale factors of
                       the input. Values of 0 are used for aspect-preserving
                       scaling. `None` skips input scaling.
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

    bounds = np.array(im.size, dtype=np.float)
    im = np.array(im)
    if im_feats is None:
         # compute image gradient
        im_feats = gaussian_filter(sobel(im), 2)

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

    def _extract_patch(env_up, env_bottom, baseline, dir_vec):
        """
        Calculate a line image patch from a ROI and the original baseline.
        """
        upper_polygon = np.concatenate((baseline, env_up[::-1]))
        bottom_polygon = np.concatenate((baseline, env_bottom[::-1]))
        angle = np.arctan2(dir_vec[1], dir_vec[0])

        def _calc_seam(polygon, bias=100):
            """
            Calculates seam between baseline and ROI boundary on one side.

            Adds a baseline-distance-weighted bias to the feature map, masks
            out the bounding polygon and rotates the line so it is roughly
            level.
            """
            MASK_VAL = 99999
            r, c = draw.polygon(polygon[:,1], polygon[:,0])
            c_min, c_max = int(polygon[:,0].min()), int(polygon[:,0].max())
            r_min, r_max = int(polygon[:,1].min()), int(polygon[:,1].max())
            patch = im_feats[r_min:r_max+2, c_min:c_max+2].copy()
            # bias feature matrix by distance from baseline
            mask = np.ones_like(patch)
            for l in zip(baseline[:-1] - (c_min, r_min), baseline[1:] - (c_min, r_min)):
                line_locs = draw.line(l[0][1], l[0][0], l[1][1], l[1][0])
                mask[line_locs] = 0
            dist_bias = distance_transform_cdt(mask)
            # absolute mask
            mask = np.ones_like(patch, dtype=np.bool)
            mask[r-r_min, c-c_min] = False
            # combine weights with features
            patch[mask] = MASK_VAL
            patch += (dist_bias*(np.mean(patch[patch != MASK_VAL])/bias))
            extrema = baseline[(0,-1),:] - (c_min, r_min)
            # scale line image to max 600 pixel width
            scale = min(1.0, 600/(c_max-c_min))
            tform, rotated_patch = _rotate(patch, angle, center=extrema[0], scale=scale, cval=MASK_VAL)
            # ensure to cut off padding after rotation
            x_offsets = np.sort(np.around(tform.inverse(extrema)[:,0]).astype('int'))
            rotated_patch = rotated_patch[:,x_offsets[0]+1:x_offsets[1]]
            # infinity pad for seamcarve
            rotated_patch = np.pad(rotated_patch, ((1, 1), (0, 0)),  mode='constant', constant_values=np.inf)
            r, c = rotated_patch.shape
            # fold into shape (c, r-2 3)
            A = np.lib.stride_tricks.as_strided(rotated_patch, (c, r-2, 3), (rotated_patch.strides[1],
                                                                             rotated_patch.strides[0],
                                                                             rotated_patch.strides[0]))
            B = rotated_patch[1:-1,1:].swapaxes(0, 1)
            backtrack = np.zeros_like(B, dtype='int')
            T = np.empty((B.shape[1]), 'f')
            R = np.arange(-1, len(T)-1)
            for i in np.arange(c-1):
                A[i].min(1, T)
                backtrack[i] = A[i].argmin(1) + R
                B[i] += T
            # backtrack
            seam = []
            j = np.argmin(rotated_patch[1:-1,-1])
            for i in range(c-2, 0, -1):
                seam.append((i+x_offsets[0]+1, j))
                j = backtrack[i, j]
            seam = np.array(seam)[::-1]
            # rotate back
            seam = tform(seam).astype('int')
            # filter out seam points in masked area of original patch/in padding
            seam = seam[seam.min(axis=1)>=0,:]
            m = (seam < mask.shape[::-1]).T
            seam = seam[np.logical_and(m[0], m[1]), :]
            seam = seam[np.invert(mask[seam.T[1], seam.T[0]])]
            seam += (c_min, r_min)
            return seam

        upper_seam = _calc_seam(upper_polygon).astype('int')
        bottom_seam = _calc_seam(bottom_polygon).astype('int')[::-1]
        polygon = np.concatenate(([baseline[0]], upper_seam, [baseline[-1]], bottom_seam))
        return approximate_polygon(polygon, 3).tolist()

    polygons = []
    if suppl_obj is None:
        suppl_obj = []
    for idx, line in enumerate(baselines):
        try:
            # find intercepts with image bounds on each side of baseline
            line = np.array(line, dtype=np.float)
            # calculate direction vector
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                slope, _, _, _, _ = linregress(line[:, 0], line[:, 1])
            if np.isnan(slope):
                p_dir = np.array([0., np.sign(np.diff(line[(0, -1),1])).item()*1.])
            else:
                p_dir = np.array([1, np.sign(np.diff(line[(0, -1),0])).item()*slope])
                p_dir = (p_dir.T / np.sqrt(np.sum(p_dir**2,axis=-1)))
            # interpolate baseline
            ls = geom.LineString(line)
            ip_line = [line[0]]
            dist = 10
            while dist < ls.length:
                ip_line.append(np.array(ls.interpolate(dist)))
                dist += 10
            ip_line.append(line[-1])
            ip_line = np.array(ip_line)
            upper_bounds_intersects = []
            bottom_bounds_intersects = []
            for point in ip_line:
                upper_bounds_intersects.append(_ray_intersect_boundaries(point, (p_dir*(-1,1))[::-1], bounds+1).astype('int'))
                bottom_bounds_intersects.append(_ray_intersect_boundaries(point, (p_dir*(1,-1))[::-1], bounds+1).astype('int'))
            # build polygon between baseline and bbox intersects
            upper_polygon = geom.Polygon(ip_line.tolist() + upper_bounds_intersects)
            bottom_polygon = geom.Polygon(ip_line.tolist() + bottom_bounds_intersects)

            # select baselines at least partially in each polygon
            side_a = [geom.LineString(upper_bounds_intersects)]
            side_b = [geom.LineString(bottom_bounds_intersects)]
            for adj_line in baselines[:idx] + baselines[idx+1:] + suppl_obj:
                adj_line = geom.LineString(adj_line)
                if upper_polygon.intersects(adj_line):
                    side_a.append(adj_line)
                elif bottom_polygon.intersects(adj_line):
                    side_b.append(adj_line)
            side_a = unary_union(side_a).buffer(1).boundary
            side_b = unary_union(side_b).buffer(1).boundary
            def _find_closest_point(pt, intersects):
                spt = geom.Point(pt)
                if intersects.type == 'MultiPoint':
                    return min([p for p in intersects], key=lambda x: spt.distance(x))
                elif intersects.type == 'Point':
                    return intersects
                elif intersects.type == 'GeometryCollection' and len(intersects) > 0:
                    t = min([p for p in intersects], key=lambda x: spt.distance(x))
                    if t == 'Point':
                        return t
                    else:
                        return nearest_points(spt, t)[1]
                else:
                    raise Exception('No intersection with boundaries. Shapely intersection object: {}'.format(intersects.wkt))
            # interpolate baseline
            env_up = []
            env_bottom = []
            # find orthogonal (to linear regression) intersects with adjacent objects to complete roi
            for point, upper_bounds_intersect, bottom_bounds_intersect in zip(ip_line, upper_bounds_intersects, bottom_bounds_intersects):
                upper_limit = _find_closest_point(point, geom.LineString([point, upper_bounds_intersect]).intersection(side_a))
                bottom_limit = _find_closest_point(point, geom.LineString([point, bottom_bounds_intersect]).intersection(side_b))
                env_up.append(upper_limit.coords[0])
                env_bottom.append(bottom_limit.coords[0])
            env_up = np.array(env_up, dtype='uint')
            env_bottom = np.array(env_bottom, dtype='uint')
            polygons.append(_extract_patch(env_up, env_bottom, line.astype('int'), p_dir))
        except Exception as e:
            polygons.append(None)

    if scale is not None:
        polygons = [(np.array(pol)/scale).astype('uint').tolist() for pol in polygons if pol is not None]
    return polygons


def polygonal_reading_order(lines: Sequence[Tuple[List, List]],
                            text_direction: str = 'lr',
                            regions: Optional[Sequence[List[Tuple[int, int]]]] = None) -> Sequence[Tuple[List, List]]:
    """
    Given a list of baselines and regions, calculates the correct reading order
    and applies it to the input.

    Args:
        lines (Sequence): List of tuples containing the baseline and its
                          polygonization.
        regions (Sequence): List of region polygons.
        text_direction (str): Set principal text direction for column ordering.
                              Can be 'lr' or 'rl'

    Returns:
        A reordered input.
    """
    bounds = []
    if regions is not None:
        r = [geom.Polygon(reg) for reg in regions]
    else:
        r = []
    region_lines = [[] for _ in range(len(r))]
    indizes = {}
    for line_idx, line in enumerate(lines):
        l = geom.LineString(line[1])
        is_in_region = False
        for idx, reg in enumerate(r):
            if reg.contains(l):
                region_lines[idx].append((line_idx, (slice(l.bounds[1], l.bounds[0]), slice(l.bounds[3], l.bounds[2]))))
                is_in_region = True
                break
        if not is_in_region:
            bounds.append((slice(l.bounds[1], l.bounds[0]), slice(l.bounds[3], l.bounds[2])))
            indizes[line_idx] = ('line', line)
    # order everything in regions
    intra_region_order = [[] for _ in range(len(r))]
    for idx, reg in enumerate(r):
        if len(region_lines[idx]) > 0:
            order = reading_order([x[1] for x in region_lines[idx]], text_direction)
            lsort = topsort(order)
            intra_region_order[idx] = [region_lines[idx][i][0] for i in lsort]
            reg = reg.bounds
            bounds.append((slice(reg[1], reg[0]), slice(reg[3], reg[2])))
            indizes[line_idx+idx] = ('region', idx)
    # order unassigned lines and regions
    order = reading_order(bounds, text_direction)
    lsort = topsort(order)
    sidz = sorted(indizes.keys())
    lsort = [sidz[i] for i in lsort]
    ordered_lines = []
    for i in lsort:
        if indizes[i][0] == 'line':
            ordered_lines.append(indizes[i][1])
        else:
            ordered_lines.extend(lines[x] for x in intra_region_order[indizes[i][1]])
    return ordered_lines


def scale_regions(regions: Sequence[Tuple[List, List]],
                  scale: Union[float, Tuple[float, float]]) -> Sequence[Tuple[List, List]]:
    """
    Scales baselines/polygon coordinates by a certain factor.

    Args:
        lines (Sequence): List of tuples containing the baseline and it's
                          polygonization.
        scale (float or tuple of floats): Scaling factor
    """
    if isinstance(scale, float):
        scale = (scale, scale)
    scaled_regions = []
    for region in regions:
        scaled_regions.append((np.array(region) * scale).astype('int').tolist())
    return scaled_regions


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
        return seg_points.astype('int').tolist()
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
            baseline = np.array(line['baseline'])
            c_min, c_max = int(pl[:,0].min()), int(pl[:,0].max())
            r_min, r_max = int(pl[:,1].min()), int(pl[:,1].max())

            # fast path for straight baselines requiring only rotation
            if len(baseline) == 2:
                baseline = baseline.astype(np.float)
                # calculate direction vector
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    slope, _, _, _, _ = linregress(baseline[:, 0], baseline[:, 1])
                if np.isnan(slope):
                    p_dir = np.array([0., np.sign(np.diff(baseline[(0, -1),1])).item()*1.])
                else:
                    p_dir = np.array([1, np.sign(np.diff(baseline[(0, -1),0])).item()*slope])
                    p_dir = (p_dir.T / np.sqrt(np.sum(p_dir**2,axis=-1)))
                angle = np.arctan2(p_dir[1], p_dir[0])
                patch = im[r_min:r_max+1, c_min:c_max+1].copy()
                offset_polygon = pl - (c_min, r_min)
                r, c = draw.polygon(offset_polygon[:,1], offset_polygon[:,0])
                mask = np.zeros(patch.shape[:2], dtype=np.bool)
                mask[r, c] = True
                patch[mask != True] = 0
                extrema = offset_polygon[(0,-1),:]
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
