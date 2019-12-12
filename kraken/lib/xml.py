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

import os.path
import logging

from lxml import etree
from os.path import dirname

from kraken.lib.exceptions import KrakenInputException

logger = logging.getLogger(__name__)

__all__ = ['parse_page', 'parse_alto']

def parse_page(filename):
    """
    Parses a PageXML file, returns the baselines defined in it, and loads the
    referenced image.

    Args:
        filename (str): path to a PageXML file.

    Returns:
        A dict {'image': impath, lines: [{'boundary': [[x0, y0], ...], 'baseline':
        [[x0, y0], ...]}, {...], 'text': 'apdjfqpf'}
    """
    with open(filename, 'rb') as fp:
        base_dir = dirname(filename)
        try:
            doc = etree.parse(fp)
        except etree.XMLSyntaxError as e:
            raise KrakenInputException('Parsing {} failed: {}'.format(filename, e))
        image = doc.find('.//{*}Page')
        if image is None or image.get('imageFilename') is None:
            raise KrakenInputException('No valid image filename found in PageXML file {}'.format(filename))
        lines = doc.findall('.//{*}TextLine')
        data = {'image': os.path.join(base_dir, image.get('imageFilename')), 'lines': [], 'type': 'baselines'}
        for line in lines:
            pol = line.find('./{*}Coords')
            boundary = None
            if pol is not None and not pol.get('points').isspace() and len(pol.get('points')):
                points = [x for x in pol.get('points').split(' ')]
                points = [int(c) for point in points for c in point.split(',')]
                boundary = list(zip(points[::2], points[1::2]))
            else:
                logger.info('TextLine {} without polygon'.format(line.get('id')))
            base = line.find('./{*}Baseline')
            baseline = None
            if base is not None and not base.get('points').isspace() and len(base.get('points')):
                points = [x for x in base.get('points').split(' ')]
                points = [int(c) for point in points for c in point.split(',')]
                baseline = list(zip(points[::2], points[1::2]))
            else:
                logger.warning('TextLine {} without baseline'.format(line.get('id')))
            text = ''
            manual_transcription = line.find('./{*}TextEquiv')
            if manual_transcription is not None:
                line = manual_transcription
            for el in line.findall('.//{*}Unicode'):
                if el.text:
                    text += el.text
            data['lines'].append({'baseline': baseline, 'boundary': boundary, 'text': text})
        return data


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
        base_dir = dirname(filename)
        try:
            doc = etree.parse(fp)
        except etree.XMLSyntaxError as e:
            raise KrakenInputException('Parsing {} failed: {}'.format(filename, e))
        image = doc.find('.//{*}fileName')
        if image is None or not image.text:
            raise KrakenInputException('No valid filename found in ALTO file')
        lines = doc.findall('.//{*}TextLine')
        data = {'image': os.path.join(base_dir, image.text), 'lines': [], 'type': 'baselines'}
        for line in lines:
            if line.get('BASELINE') is None:
                raise KrakenInputException('ALTO file {} contains no baseline information'.format(filename))
            pol = line.find('./{*}Shape/{*}Polygon')
            boundary = None
            if pol is not None:
                points = [int(x) for x in pol.get('POINTS').split(' ')]
                boundary = list(zip(points[::2], points[1::2]))
            points = [int(x) for x in line.get('BASELINE').split(' ')]
            baseline = list(zip(points[::2], points[1::2]))
            text = ''
            for el in line.xpath(".//*[local-name() = 'String'] | .//*[local-name() = 'SP']"):
                text += el.get('CONTENT') if el.get('CONTENT') else ' '
            data['lines'].append({'baseline': baseline, 'boundary': boundary, 'text': text})
        return data
