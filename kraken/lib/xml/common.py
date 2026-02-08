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
Shared constants and utilities for ALTO/PageXML parsing.
"""
import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from itertools import groupby

from typing import Optional

logger = logging.getLogger(__name__)

# fallback mapping between PAGE region types and tags
page_regions = {'TextRegion': 'text',
                'ImageRegion': 'image',
                'LineDrawingRegion': 'line drawing',
                'GraphicRegion': 'graphic',
                'TableRegion': 'table',
                'ChartRegion': 'chart',
                'MapRegion': 'map',
                'SeparatorRegion': 'separator',
                'MathsRegion': 'maths',
                'ChemRegion': 'chem',
                'MusicRegion': 'music',
                'AdvertRegion': 'advert',
                'NoiseRegion': 'noise',
                'UnknownRegion': 'unknown',
                'CustomRegion': 'custom'}

# same for ALTO
alto_regions = {'TextBlock': 'text',
                'Illustration': 'illustration',
                'GraphicalElement': 'graphic',
                'ComposedBlock': 'composed'}


def parse_alto_pointstype(coords: str) -> Sequence[tuple[float, float]]:
    """
    ALTO's PointsType is underspecified so a variety of serializations are valid:

        x0, y0 x1, y1 ...
        x0 y0 x1 y1 ...
        (x0, y0) (x1, y1) ...
        (x0 y0) (x1 y1) ...

    Returns:
        A list of tuples [(x0, y0), (x1, y1), ...]
    """
    float_re = re.compile(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?')
    try:
        points = [int(float(point.group())) for point in float_re.finditer(coords)]
    except (ValueError, TypeError):
        raise ValueError(f'Invalid points sequence string: {coords}')
    if len(points) % 2:
        raise ValueError(f'Odd number of points in points sequence: {points}')
    pts = zip(points[::2], points[1::2])
    return [k for k, g in groupby(pts)]


def parse_page_coords(coords):
    points = [x for x in coords.split(' ')]
    points = [int(c) for point in points for c in point.split(',')]
    pts = zip(points[::2], points[1::2])
    return [k for k, g in groupby(pts)]


def parse_page_custom(s):
    o = defaultdict(list)
    s = s.strip()
    l_chunks = [l_chunk for l_chunk in s.split('}') if l_chunk.strip()]
    if l_chunks:
        for chunk in l_chunks:
            tag, vals = chunk.split('{')
            tag_vals = {}
            vals = [val.strip() for val in vals.split(';') if val.strip()]
            for val in vals:
                key, *val = val.split(':')
                tag_vals[key] = ":".join(val).strip()
            o[tag.strip()].append(tag_vals)
    return dict(o)


def flatten_order_to_lines(raw_order: list[str],
                           lines_dict: dict,
                           region_ids: set[str],
                           line_implicit_order: list[str],
                           string_to_line_map: Optional[dict[str, str]] = None) -> list[str]:
    """
    Flatten a raw reading order (list of IDs) to line-level.

    For each ID:
    - Line ID: append directly
    - Region ID: expand to contained lines using implicit order
    - String ID (ALTO only): map to parent TextLine, deduplicate consecutive
    - Unknown ID: log warning, skip
    """
    result = []
    for ref_id in raw_order:
        if ref_id in lines_dict:
            result.append(ref_id)
        elif ref_id in region_ids:
            # expand region to its lines in implicit order
            for lid in line_implicit_order:
                if lines_dict[lid].regions and lines_dict[lid].regions[0] == ref_id:
                    result.append(lid)
        elif string_to_line_map is not None and ref_id in string_to_line_map:
            parent_line = string_to_line_map[ref_id]
            # deduplicate consecutive same-line refs
            if not result or result[-1] != parent_line:
                result.append(parent_line)
        else:
            logger.info(f'Unknown element ID {ref_id} in reading order, skipping.')
    return result


def flatten_order_to_regions(raw_order: list[str],
                             lines_dict: dict,
                             region_ids: set[str],
                             string_to_line_map: Optional[dict[str, str]] = None) -> list[str]:
    """
    Flatten a raw reading order (list of IDs) to region-level.

    For each ID:
    - Region ID: append directly
    - Line ID: resolve to parent region, deduplicate consecutive
    - String ID (ALTO only): resolve to parent line then region, deduplicate
    - Unknown ID: log warning, skip
    """
    result = []
    for ref_id in raw_order:
        if ref_id in region_ids:
            if not result or result[-1] != ref_id:
                result.append(ref_id)
        elif ref_id in lines_dict:
            parent_region = lines_dict[ref_id].regions[0] if lines_dict[ref_id].regions else None
            if parent_region and (not result or result[-1] != parent_region):
                result.append(parent_region)
        elif string_to_line_map is not None and ref_id in string_to_line_map:
            parent_line = string_to_line_map[ref_id]
            if parent_line in lines_dict:
                parent_region = lines_dict[parent_line].regions[0] if lines_dict[parent_line].regions else None
                if parent_region and (not result or result[-1] != parent_region):
                    result.append(parent_region)
        else:
            logger.info(f'Unknown element ID {ref_id} in reading order, skipping.')
    return result


def validate_and_clean_order(flat_order: list[str],
                             valid_ids: set[str]) -> tuple[list[str], bool]:
    """
    Validate a flattened order.

    Checks:
    - All IDs exist in valid_ids
    - No duplicate IDs (indicates circular reference)

    Returns:
        (cleaned_order, is_valid)
    """
    cleaned = []
    seen = set()
    is_valid = True
    for ref_id in flat_order:
        if ref_id not in valid_ids:
            logger.info(f'ID {ref_id} in reading order not found in document, removing.')
            is_valid = False
            continue
        if ref_id in seen:
            logger.info(f'Duplicate ID {ref_id} in reading order, removing duplicate.')
            is_valid = False
            continue
        seen.add(ref_id)
        cleaned.append(ref_id)
    return cleaned, is_valid
