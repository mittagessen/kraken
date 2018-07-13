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

logger = logging.getLogger(__name__)

__all__ = ['serialize']


def _rescale(val, low, high):
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


def max_bbox(boxes):
    """
    Calculates the minimal bounding box containing all boxes contained in an
    iterator.

    Args:
        boxes (iterator): An iterator returning tuples of the format (x0, y0,
                          x1, y1)
    Returns:
        A box covering all bounding boxes in the input argument
    """
    sbox = list(map(sorted, list(zip(*boxes))))
    return (sbox[0][0], sbox[1][0], sbox[2][-1], sbox[3][-1])


def serialize(records, image_name=u'', image_size=(0, 0), writing_mode='horizontal-tb', scripts=None, template='hocr'):
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
    """
    logger.info(u'Serialize {} records from {} with template {}.'.format(len(records), image_name, template))
    page = {'lines': [], 'size': image_size, 'name': image_name, 'writing_mode': writing_mode, 'scripts': scripts}
    seg_idx = 0
    char_idx = 0
    for idx, record in enumerate(records):
        # skip empty records
        if not record.prediction:
            logger.debug(u'Empty record. Skipping')
            continue
        line = {'index': idx,
                'bbox': max_bbox(record.cuts),
                'cuts': record.cuts,
                'confidences': record.confidences,
                'recognition': []
                }
        splits = regex.split(r'(\s+)', record.prediction)
        line_offset = 0
        logger.debug(u'Record contains {} segments'.format(len(splits)))
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
    logger.debug(u'Initializing jinja environment.')
    env = Environment(loader=PackageLoader('kraken', 'templates'), trim_blocks=True, lstrip_blocks=True)
    env.tests['whitespace'] = str.isspace
    env.filters['rescale'] = _rescale
    logger.debug(u'Retrieving template.')
    tmpl = env.get_template(template)
    logger.debug(u'Rendering data.')
    return tmpl.render(page=page)
