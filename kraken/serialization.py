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
import unicodedata

from collections import Counter

from kraken.rpred import ocr_record
from kraken.lib.util import make_printable

from typing import List, Tuple, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ['serialize']


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


def max_bbox(boxes: Iterable[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """
    Calculates the minimal bounding box containing all boxes contained in an
    iterator.

    Args:
        boxes (iterator): An iterator returning tuples of the format (x0, y0,
                          x1, y1)
    Returns:
        A box covering all bounding boxes in the input argument
    """
    # XXX: fix type hinting
    sbox = list(map(sorted, list(zip(*boxes))))
    return (sbox[0][0], sbox[1][0], sbox[2][-1], sbox[3][-1])  # type: ignore


def serialize(records: Sequence[ocr_record],
              image_name: str = None,
              image_size: Tuple[int, int] = (0, 0),
              writing_mode: str = 'horizontal-tb',
              scripts: Optional[Iterable[str]] = None,
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
        template (str): Selector for the serialization format. May be
                        'hocr' or 'alto'.

    Returns:
            (str) rendered template.
    """
    logger.info('Serialize {} records from {} with template {}.'.format(len(records), image_name, template))
    page = {'lines': [], 'size': image_size, 'name': image_name, 'writing_mode': writing_mode, 'scripts': scripts}  # type: dict
    seg_idx = 0
    char_idx = 0
    for idx, record in enumerate(records):
        # skip empty records
        if not record.prediction:
            logger.debug('Empty record. Skipping')
            continue
        line = {'index': idx,
                'bbox': max_bbox(record.cuts),
                'cuts': record.cuts,
                'confidences': record.confidences,
                'recognition': []
                }
        splits = regex.split(r'(\s+)', record.prediction)
        line_offset = 0
        logger.debug('Record contains {} segments'.format(len(splits)))
        for segment in splits:
            if len(segment) == 0:
                continue
            seg_bbox = max_bbox(record.cuts[line_offset:line_offset + len(segment)])

            line['recognition'].extend([{'bbox': seg_bbox,
                                         'confidences': record.confidences[line_offset:line_offset + len(segment)],
                                         'cuts': record.cuts[line_offset:line_offset + len(segment)],
                                         'text': segment,
                                         'recognition': [{'bbox': cut, 'confidence': conf, 'text': char, 'index': cid}
                                                         for conf, cut, char, cid in
                                                         zip(record.confidences[line_offset:line_offset + len(segment)],
                                                             record.cuts[line_offset:line_offset + len(segment)],
                                                             segment,
                                                             range(char_idx, char_idx + len(segment)))],
                                         'index': seg_idx}])
            char_idx += len(segment)
            seg_idx += 1
            line_offset += len(segment)
        page['lines'].append(line)
    logger.debug('Initializing jinja environment.')
    env = Environment(loader=PackageLoader('kraken', 'templates'),
                      trim_blocks=True,
                      lstrip_blocks=True,
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
    logger.info('Serializing report for {}'.format(model))

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

