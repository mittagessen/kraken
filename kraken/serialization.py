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
import datetime
import importlib.metadata
import logging
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Literal,
                    Optional, Sequence, Tuple)

import regex
from jinja2 import Environment, FunctionLoader, PackageLoader

from kraken.lib.util import make_printable

if TYPE_CHECKING:
    from collections import Counter
    from os import PathLike

    from kraken.containers import ProcessingStep, Segmentation

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


def serialize(results: 'Segmentation',
              image_size: Tuple[int, int] = (0, 0),
              writing_mode: Literal['horizontal-tb', 'vertical-lr', 'vertical-rl'] = 'horizontal-tb',
              scripts: Optional[Iterable[str]] = None,
              template: ['PathLike', str] = 'alto',
              template_source: Literal['native', 'custom'] = 'native',
              processing_steps: Optional[List['ProcessingStep']] = None) -> str:
    """
    Serializes recognition and segmentation results into an output document.

    Serializes a Segmentation container object containing either segmentation
    or recognition results into an output document. The rendering is performed
    with jinja2 templates that can either be shipped with kraken
    (`template_source` == 'native') or custom (`template_source` == 'custom').

    Note: Empty records are ignored for serialization purposes.

    Args:
        segmentation: Segmentation container object
        image_size: Dimensions of the source image
        writing_mode: Sets the principal layout of lines and the
                      direction in which blocks progress. Valid values are
                      horizontal-tb, vertical-rl, and vertical-lr.
        scripts: List of scripts contained in the OCR records
        template: Selector for the serialization format. May be 'hocr',
                  'alto', 'page' or any template found in the template
                  directory. If template_source is set to `custom` a path to a
                  template is expected.
        template_source: Switch to enable loading of custom templates from
                         outside the kraken package.
        processing_steps: A list of ProcessingStep container classes describing
                          the processing kraken performed on the inputs.

    Returns:
        The rendered template
    """
    logger.info(f'Serialize {len(results.lines)} records from {results.imagename} with template {template}.')
    page: Dict[str, Any] = {'entities': [],
                            'size': image_size,
                            'name': results.imagename,
                            'writing_mode': writing_mode,
                            'scripts': scripts,
                            'date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            'base_dir': [rec.base_dir for rec in results.lines][0] if len(results.lines) else None,
                            'seg_type': results.type}
    metadata = {'processing_steps': processing_steps,
                'version': importlib.metadata.version('kraken')}

    seg_idx = 0
    char_idx = 0

    # build region and line type dict
    types = []
    for line in results.lines:
        if line.tags is not None:
            types.extend((k, v) for k, v in line.tags.items())
    page['line_types'] = list(set(types))
    page['region_types'] = list(results.regions.keys())

    # map reading orders indices to line IDs
    ros = []
    for ro in results.line_orders:
        ros.append([results.lines[idx].id for idx in ro])
    page['line_orders'] = ros
    # build region ID to region dict
    reg_dict = {}
    for key, regs in results.regions.items():
        for reg in regs:
            reg_dict[reg.id] = reg

    regs_with_lines = set()
    prev_reg = None
    for idx, record in enumerate(results.lines):
        # line not in region
        if not record.regions or len(record.regions) == 0:
            cur_ent = page['entities']
        # line not in same region as previous line
        elif prev_reg != record.regions[0]:
            prev_reg = record.regions[0]
            reg = reg_dict[record.regions[0]]
            regs_with_lines.add(reg.id)
            region = {'id': reg.id,
                      'bbox': max_bbox([reg.boundary]) if reg.boundary else [],
                      'boundary': [list(x) for x in reg.boundary] if reg.boundary else [],
                      'tags': reg.tags,
                      'lines': [],
                      'type': 'region'
                      }
            page['entities'].append(region)
            cur_ent = region['lines']

        # set field to indicate the availability of baseline segmentation in
        # addition to bounding boxes
        line = {'id': record.id,
                'bbox': max_bbox([record.boundary]) if record.type == 'baselines' else record.bbox,
                'cuts': [list(x) for x in getattr(record, 'cuts', [])],
                'confidences': getattr(record, 'confidences', []),
                'recognition': [],
                'boundary': [list(x) for x in record.boundary] if record.type == 'baselines' else [[record.bbox[0], record.bbox[1]],
                                                                                                   [record.bbox[2], record.bbox[1]],
                                                                                                   [record.bbox[2], record.bbox[3]],
                                                                                                   [record.bbox[0], record.bbox[3]]],
                'type': 'line'
                }
        if record.tags is not None:
            line['tags'] = record.tags
        if record.type == 'baselines':
            line['baseline'] = [list(x) for x in record.baseline]

        splits = regex.split(r'(\s+)', getattr(record, 'prediction', ''))
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
            if record.type == 'baselines':
                seg_struct['boundary'] = record[line_offset:line_offset + len(segment)][1]
            line['recognition'].append(seg_struct)
            char_idx += len(segment)
            seg_idx += 1
            line_offset += len(segment)
        cur_ent.append(line)

    # serialize all remaining (line-less) regions
    for reg_id in regs_with_lines:
        reg_dict.pop(reg_id)
    logger.debug(f'No lines given but {len(results.regions)}. Serialize all regions.')
    for reg in reg_dict.values():
        region = {'id': reg.id,
                  'bbox': max_bbox([reg.boundary]),
                  'boundary': [list(x) for x in reg.boundary],
                  'tags': reg.tags,
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


def render_report(model: str,
                  chars: int,
                  errors: int,
                  char_accuracy: float,
                  word_accuracy: float,
                  char_confusions: 'Counter',
                  scripts: 'Counter',
                  insertions: 'Counter',
                  deletions: int,
                  substitutions: 'Counter') -> str:
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
              'character_accuracy': char_accuracy * 100,
              'word_accuracy': word_accuracy * 100,
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
