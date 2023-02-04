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
from jinja2 import Environment, PackageLoader, FunctionLoader

import regex
import logging
import datetime
import shapely.geometry as geom

from os import PathLike
from pkg_resources import get_distribution
from collections import Counter

from kraken.rpred import BaselineOCRRecord, BBoxOCRRecord, ocr_record
from kraken.lib.util import make_printable
from kraken.lib.segmentation import is_in_region

from typing import Union, List, Tuple, Iterable, Optional, Sequence, Dict, Any, Literal

logger = logging.getLogger(__name__)

__all__ = ['serialize', 'serialize_segmentation', 'render_report']


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
              image_name: Union[PathLike, str] = None,
              image_size: Tuple[int, int] = (0, 0),
              writing_mode: Literal['horizontal-tb', 'vertical-lr', 'vertical-rl'] = 'horizontal-tb',
              scripts: Optional[Iterable[str]] = None,
              regions: Optional[Dict[str, List[List[Tuple[int, int]]]]] = None,
              template: [PathLike, str] = 'alto',
              template_source: Literal['native', 'custom'] = 'native',
              processing_steps: Optional[List[Dict[str, Union[Dict, str, float, int, bool]]]] = None) -> str:
    """
    Serializes a list of ocr_records into an output document.

    Serializes a list of predictions and their corresponding positions by doing
    some hOCR-specific preprocessing and then renders them through one of
    several jinja2 templates.

    Note: Empty records are ignored for serialization purposes.

    Args:
        records: List of kraken.rpred.ocr_record
        image_name: Name of the source image
        image_size: Dimensions of the source image
        writing_mode: Sets the principal layout of lines and the
                      direction in which blocks progress. Valid values are
                      horizontal-tb, vertical-rl, and vertical-lr.
        scripts: List of scripts contained in the OCR records
        regions: Dictionary mapping region types to a list of region polygons.
        template: Selector for the serialization format. May be 'hocr',
                  'alto', 'page' or any template found in the template
                  directory. If template_source is set to `custom` a path to a
                  template is expected.
        template_source: Switch to enable loading of custom templates from
                         outside the kraken package.
        processing_steps: A list of dictionaries describing the processing kraken performed on the inputs::

                          {'category': 'preprocessing',
                           'description': 'natural language description of process',
                           'settings': {'arg0': 'foo', 'argX': 'bar'}
                          }

    Returns:
        The rendered template
    """
    logger.info(f'Serialize {len(records)} records from {image_name} with template {template}.')
    page = {'entities': [],
            'size': image_size,
            'name': image_name,
            'writing_mode': writing_mode,
            'scripts': scripts,
            'date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'base_dir': [rec.base_dir for rec in records][0] if len(records) else None}  # type: dict
    metadata = {'processing_steps': processing_steps,
                'version': get_distribution('kraken').version}

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
    types = []
    for line in records:
        if hasattr(line, 'tags') and line.tags is not None:
            types.extend(line.tags.values())
    page['types'] = list(set(types))
    if regions is not None:
        page['types'].extend(list(regions.keys()))

    is_in_reg = -1
    for idx, record in enumerate(records):
        if record.type == 'baselines':
            l_obj = geom.LineString(record.baseline)
        else:
            l_obj = geom.LineString(record.line)
        reg = list(filter(lambda x: is_in_region(l_obj, x[1][1]), region_map.items()))
        if len(reg) == 0:
            cur_ent = page['entities']
        elif reg[0][0] != is_in_reg:
            reg = reg[0]
            is_in_reg = reg[0]
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
        if hasattr(record, 'tags') and record.tags is not None:
            line['tags'] = record.tags
        if record.type == 'baselines':
            line['baseline'] = [list(x) for x in record.baseline]
        splits = regex.split(r'(\s+)', record.prediction)
        line_offset = 0
        logger.debug(f'Record contains {len(splits)} segments')
        for segment in splits:
            if len(segment) == 0:
                continue
            seg_cuts = record.cuts[line_offset:line_offset + len(segment)]
            seg_bbox = max_bbox(seg_cuts)
            seg_struct = {'bbox': seg_bbox,
                          'confidences': record.confidences[line_offset:line_offset + len(segment)],
                          'cuts': seg_cuts,
                          'text': segment,
                          'recognition': [{'bbox': max_bbox([cut]),
                                           'boundary': cut,
                                           'confidence': conf,
                                           'text': char,
                                           'index': cid}
                                          for conf, cut, char, cid in
                                          zip(record.confidences[line_offset:line_offset + len(segment)],
                                              seg_cuts,
                                              segment,
                                              range(char_idx, char_idx + len(segment)))],
                          'index': seg_idx}
            # compute convex hull of all characters in segment
            if record.type == 'baselines':
                seg_struct['boundary'] = record[line_offset:line_offset + len(segment)][1]
            line['recognition'].append(seg_struct)
            char_idx += len(segment)
            seg_idx += 1
            line_offset += len(segment)
        cur_ent.append(line)

    # No records but there are regions -> serialize all regions
    if not records and regions:
        logger.debug(f'No lines given but {len(region_map)}. Serialize all regions.')
        for reg in region_map.items():
            region = {'index': reg[0],
                      'bbox': [int(x) for x in reg[1][1].bounds],
                      'boundary': [list(x) for x in reg[1][2]],
                      'region_type': reg[1][0],
                      'lines': [],
                      'type': 'region'
                      }
            page['entities'].append(region)

    if template_source == 'native':
        logger.debug('Initializing native jinja environment.')
        loader = PackageLoader('kraken', 'templates')
    elif template_source == 'custom':
        def _load_template(name):
            return open(template, 'r').read(), name, lambda: True
        loader = FunctionLoader(_load_template)

    env = Environment(loader=loader,
                      trim_blocks=True,
                      lstrip_blocks=True,
                      autoescape=True)
    env.tests['whitespace'] = str.isspace
    env.filters['rescale'] = _rescale
    logger.debug('Retrieving template.')
    tmpl = env.get_template(template)
    logger.debug('Rendering data.')
    return tmpl.render(page=page, metadata=metadata)


def serialize_segmentation(segresult: Dict[str, Any],
                           image_name: Union[PathLike, str] = None,
                           image_size: Tuple[int, int] = (0, 0),
                           template: Union[PathLike, str] = 'alto',
                           template_source: Literal['native', 'custom'] = 'native',
                           processing_steps: Optional[List[Dict[str, Union[Dict, str, float, int, bool]]]] = None) -> str:
    """
    Serializes a segmentation result into an output document.

    Args:
        segresult: Result of blla.segment
        image_name: Name of the source image
        image_size: Dimensions of the source image
        template: Selector for the serialization format. Any value accepted by
                  `serialize` is valid.
        template_source: Enables/disables loading of external templates.

    Returns:
            (str) rendered template.
    """
    if 'type' in segresult and segresult['type'] == 'baselines':
        records = [BaselineOCRRecord('', (), (), bl) for bl in segresult['lines']]
    else:
        records = []
        for line in segresult['boxes']:
            xmin, xmax = min(line[::2]), max(line[::2])
            ymin, ymax = min(line[1::2]), max(line[1::2])
            records.append(BBoxOCRRecord('', (), (), ((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin))))
    return serialize(records,
                     image_name=image_name,
                     image_size=image_size,
                     regions=segresult['regions'] if 'regions' in segresult else None,
                     template=template,
                     template_source=template_source,
                     processing_steps=processing_steps)


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
