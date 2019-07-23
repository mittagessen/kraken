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
ALTO/Page data loaders for segmentation training
"""

import logging

logger = logging.getLogger(__name__)

from PIL import Image
from lxml import etree

from kraken.lib.exceptions import KrakenInputException

def parse_alto(filename):
    """
    Parses an ALTO file, returns the baselines defined in it, and loads the
    referenced image.

    Args:
        filename (str): path to an ALTO file.

    Returns:
        A dict {'image': impath, lines: [{'boundary': [[x0, y0], ...], 'baseline':
        [[x0, y0], ...]}, {...], 'text': 'apdjfqpf'}
    """
    with open(filename, 'rb') as fp:
        doc = etree.parse(fp)
        image = doc.find('.//{*}fileName')
        if image is None or not image.text:
            raise KrakenInputException('No valid filename found in ALTO file')
        lines = doc.findall('.//{*}TextLine')
        data = {'image': image.text, 'lines': []}
        for line in lines:
            if not line.find('./{*}Shape') or not line.get('BASELINE'):
                raise KrakenInputException('ALTO file {} contains no baseline information'.format(filename))
            pol = line.find('./{*}Shape/{*}Polygon')
            points = [int(x) for x in pol.get('POINTS').split(' ')]
            boundary = list(zip(points[::2], points[1::2]))
            points = [int(x) for x in line.get('BASELINE').split(' ')]
            baseline = list(zip(points[::2], points[1::2]))
            text = ''
            for el in line.xpath(".//*[local-name() = 'String'] | .//*[local-name() = 'SP']"):
                text += el.get('CONTENT') if el.get('CONTENT') else ' '
            data['lines'].append({'baseline': baseline, 'boundary': boundary, 'text': text})
        return data
