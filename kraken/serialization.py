# -*- coding: utf-8 -*-
#
# Copyright 2015 Benjamin Kiessling
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
from jinja2 import Environment, PackageLoader

import regex
import logging
import datetime
import numpy as np
import shapely.geometry as geom

from shapely.ops import unary_union
from collections import Counter

from scipy.spatial import ConvexHull

from kraken.rpred import ocr_record
from kraken.lib.util import make_printable

from typing import List, Tuple, Iterable, Optional, Sequence, Dict

logger = logging.getLogger(__name__)

__all__ = ['serialize', 'render_report']


def _rescale(val: Sequence[float], low: float, high: float) -> List[float]:
    """
    Rescales a list of confidence value between 0 and 1 to an interval [low,
    high].

    Args:
        val (float): List of values in interval (0,1)
        low (float): Lower bound of rescaling interval
        high (float): Upper bound of rescaling interval

    Returns:
        Rescaled value (float).
    """
    return [(high - low) * x + low for x in val]


def max_bbox(boxes: Iterable[Sequence[int]]) -> Tuple[int, int, int, int]:
    """
    Calculates the minimal bounding box containing all contained in an
    iterator.

    Args:
        boxes (iterator): An iterator returning tuples of the format ((x0, y0),
        (x1, y1), ... (xn, yn)).
    Returns:
        A box (x0, y0, x1, y1) covering all bounding boxes in the input
        argument.
    """
    flat_box = [point for pol in boxes for point in pol]
    flat_box = [x for point in flat_box for x in point]
    xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
    ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
    o = xmin, ymin, xmax, ymax  # type: ignore
    return o


def serialize(records: Sequence[ocr_record],
              image_name: str = None,
              image_size: Tuple[int, int] = (0, 0),
              writing_mode: str = 'horizontal-tb',
              scripts: Optional[Iterable[str]] = None,
              regions: Optional[Dict[str, List[List[Tuple[int, int]]]]] = None,
              template: str = 'hocr') -> str:
    """
    Serializes a list of ocr_records into an output document.

    Serializes a list of predictions and their corresponding positions by doing
    some hOCR-specific preprocessing and then renders them through one of
    several jinja2 templates.

    Note: Empty records are ignored for serialization purposes.

    Args:
        records (iterable): List of kraken.rpred.ocr_record
        image_name (str): Name of the source image
        image_size (tuple): Dimensions of the source image
        writing_mode (str): Sets the principal layout of lines and the
                            direction in which blocks progress. Valid values
                            are horizontal-tb, vertical-rl, and
                            vertical-lr.
        scripts (list): List of scripts contained in the OCR records
        regions (list): Dictionary mapping region types to a list of region
                        polygons.
        template (str): Selector for the serialization format. May be
                        'hocr' or 'alto'.

    Returns:
            (str) rendered template.
    """
    logger.info(f'Serialize {len(records)} records from {image_name} with template {template}.')
    page = {'entities': [],
            'size': image_size,
            'name': image_name,
            'writing_mode': writing_mode,
            'scripts': scripts,
            'date': datetime.datetime.now().isoformat()}  # type: dict
    seg_idx = 0
    char_idx = 0
    region_map = {}
    idx = 0
    if regions is not None:
        for id, regs in regions.items():
            for reg in regs:
                region_map[idx] = (id, geom.Polygon(reg), reg)
                idx += 1

    # build region and line type dict
    page['types'] = list(set(line.script for line in records if line.script is not None))
    if regions is not None:
        page['types'].extend(list(regions.keys()))

    is_in_region = -1
    for idx, record in enumerate(records):
        if record.type == 'baselines':
            l_obj = geom.LineString(record.baseline).interpolate(0.5, normalized=True)
        else:
            l_obj = geom.LineString(record.line).interpolate(0.5, normalized=True)
        reg = list(filter(lambda x: x[1][1].contains(l_obj), region_map.items()))
        if len(reg) == 0:
            cur_ent = page['entities']
        elif reg[0][0] != is_in_region:
            reg = reg[0]
            is_in_region = reg[0]
            region = {'index': reg[0],
                      'bbox': [int(x) for x in reg[1][1].bounds],
                      'boundary': [list(x) for x in reg[1][2]],
                      'region_type': reg[1][0],
                      'lines': [],
                      'type': 'region'
                     }
            page['entities'].append(region)
            cur_ent = region['lines']

        # set field to indicate the availability of baseline segmentation in
        # addition to bounding boxes
        if record.type == 'baselines':
            page['seg_type'] = 'baselines'
        line = {'index': idx,
                'bbox': max_bbox([record.line]),
                'cuts': record.cuts,
                'confidences': record.confidences,
                'recognition': [],
                'boundary': [list(x) for x in record.line],
                'type': 'line'
                }
        if record.script is not None:
            line['script'] = record.script
        if record.type == 'baselines':
            line['baseline'] = [list(x) for x in record.baseline]
        splits = regex.split(r'(\s+)', record.prediction)
        line_offset = 0
        logger.debug(f'Record contains {len(splits)} segments')
        for segment in splits:
            if len(segment) == 0:
                continue
            seg_bbox = max_bbox(record.cuts[line_offset:line_offset + len(segment)])
            seg_struct = {'bbox': seg_bbox,
                          'confidences': record.confidences[line_offset:line_offset + len(segment)],
                          'cuts': record.cuts[line_offset:line_offset + len(segment)],
                          'text': segment,
                          'recognition': [{'bbox': max_bbox([cut]), 'boundary': cut, 'confidence': conf, 'text': char, 'index': cid}
                                          for conf, cut, char, cid in
                                          zip(record.confidences[line_offset:line_offset + len(segment)],
                                              record.cuts[line_offset:line_offset + len(segment)],
                                              segment,
                                              range(char_idx, char_idx + len(segment)))],
                          'index': seg_idx}
            # compute convex hull of all characters in segment
            if record.type == 'baselines':
                pols = []
                for x in record.cuts[line_offset:line_offset + len(segment)]:
                    try:
                        pol = geom.Polygon(x)
                    except ValueError:
                        pol = geom.LineString(x).buffer(0.5, cap_style=2)
                    if pol.area == 0.0:
                        pol = pol.buffer(0.5)
                    pols.append(pol)
                pols = unary_union(pols)
                coords = np.array(pols.convex_hull.exterior.coords, dtype=np.uint).tolist()
                seg_struct['boundary'] = coords
            line['recognition'].append(seg_struct)
            char_idx += len(segment)
            seg_idx += 1
            line_offset += len(segment)
        cur_ent.append(line)
    logger.debug('Initializing jinja environment.')
    env = Environment(loader=PackageLoader('kraken', 'templates'),
                      trim_blocks=True,
                      lstrip_blocks=False,
                      autoescape=True)
    env.tests['whitespace'] = str.isspace
    env.filters['rescale'] = _rescale
    logger.debug('Retrieving template.')
    tmpl = env.get_template(template)
    logger.debug('Rendering data.')
    return tmpl.render(page=page)


def render_report(model: str,
                  chars: int,
                  errors: int,
                  char_confusions: Counter,
                  scripts: Counter,
                  insertions: Counter,
                  deletions: int,
                  substitutions: Counter) -> str:
    """
    Renders an accuracy report.

    Args:
        model (str): Model name.
        errors (int): Number of errors on test set.
        char_confusions (dict): Dictionary mapping a tuple (gt, pred) to a
                                number of occurrences.
        scripts (dict): Dictionary counting character per script.
        insertions (dict): Dictionary counting insertion operations per Unicode
                           script
        deletions (int): Number of deletions
        substitutions (dict): Dictionary counting substitution operations per
                              Unicode script.

    Returns:
        A string containing the rendered report.
    """
    logger.info(f'Serializing report for {model}.')

    report = {'model': model,
              'chars': chars,
              'errors': errors,
              'accuracy': (chars-errors)/chars * 100,
              'insertions': sum(insertions.values()),
              'deletions': deletions,
              'substitutions': sum(substitutions.values()),
              'scripts': sorted([{'script': k,
                                  'count': v,
                                  'errors': insertions[k] + substitutions[k],
                                  'accuracy': 100 * (v-(insertions[k] + substitutions[k]))/v} for k, v in scripts.items()],
                                key=lambda x: x['accuracy'],
                                reverse=True),
              'counts': sorted([{'correct': make_printable(k[0]),
                                 'generated': make_printable(k[1]),
                                 'errors': v} for k, v in char_confusions.items() if k[0] != k[1]],
                               key=lambda x: x['errors'],
                               reverse=True)}
    logger.debug('Initializing jinja environment.')
    env = Environment(loader=PackageLoader('kraken', 'templates'),
                      trim_blocks=True,
                      lstrip_blocks=True,
                      autoescape=True)
    logger.debug('Retrieving template.')
    tmpl = env.get_template('report')
    logger.debug('Rendering data.')
    return tmpl.render(report=report)
