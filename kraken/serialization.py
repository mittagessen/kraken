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


from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from builtins import map
from builtins import zip
from builtins import str
from builtins import object

from jinja2 import Environment, PackageLoader

import regex

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


def delta(root=(0, 0, 0, 0), coordinates=None):
    """Calculates the running delta from a root coordinate according to the
    hOCR standard.

    It uses a root bounding box (x0, y0, x1, y1) and calculates the delta from
    the points (min(x0, x1), min(y0, y1)) and (min(x0, x1), max(y0, y1)) for
    the first and second pair of values in a delta (dx0, dy0, dx1, dy1)
    respectively.

    Args:
        coordinates (list): List of tuples of length 4 containing absolute
                            coordinates for character bounding boxes.

    Returns:
        A tuple dx0, dy0, dx1, dy1
    """
    for box in coordinates:
        yield (min(box[0], box[2]) - min(root[0], root[2]),
               min(box[1], box[3]) - min(root[1], root[3]),
               max(box[0], box[2]) - min(root[0], root[2]),
               max(box[1], box[3]) - max(root[1], root[3]))
        root = box

def serialize(records, image_name=u'', image_size=(0, 0), template='hocr'):
    """
    Serializes a list of ocr_records into an output document.

    Serializes a list of predictions and their corresponding positions by doing
    some hOCR-specific preprocessing and then renders them through one of
    several jinja2 templates.

    Args:
        records (iterable): List of kraken.rpred.ocr_record
        image_name (unicode): Name of the source image
        image_size (tuple): Dimensions of the source image
        template (unicode): Selector for the serialization format. May be
                            'hocr' or 'alto'. 
    """
    page = {'lines': [], 'size': image_size, 'name': image_name}
    seg_idx = 0
    for idx, record in enumerate(records):
        line = {'index': idx,
                'bbox': max_bbox(record.cuts),
                'deltas': ' '.join(['{},{},{},{}'.format(*x) for x in delta(max_bbox(record.cuts), record.cuts)]),
                'recognition': []
                }

        splits = regex.split(u'(\s+)', record.prediction)
        bbox = max_bbox(record.cuts)
        line_offset = 0
        for segment in splits:
            if len(segment) == 0:
                continue
            seg_bbox = max_bbox(record.cuts[line_offset:line_offset + len(segment)])
            line['recognition'].append({'bbox': seg_bbox,
                                        'confidences': record.confidences[line_offset:line_offset + len(segment)],
                                        'text': segment,
                                        'index': seg_idx})
            seg_idx += 1
            line_offset += len(segment)
        page['lines'].append(line)
    env = Environment(loader=PackageLoader('kraken', 'templates'), trim_blocks=True, lstrip_blocks=True)
    env.tests['whitespace'] = str.isspace
    env.filters['rescale'] = _rescale
    tmpl = env.get_template(template)
    return tmpl.render(page=page)
