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

from collections import defaultdict
from kraken.lib.exceptions import KrakenInputException

logger = logging.getLogger(__name__)

__all__ = ['parse_xml', 'parse_page', 'parse_alto']

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
                'CustomRegion': 'custom'
               }

def parse_xml(filename):
    """
    Parses either a PageXML or ALTO file with autodetermination of the file
    format.

    Args:
        filename (str): path to an XML file.

    Returns:
        A dict {'image': impath, lines: [{'boundary': [[x0, y0], ...],
        'baseline': [[x0, y0], ...]}, {...], 'text': 'apdjfqpf', 'script':
        'script_type'}, regions: {'region_type_0': [[[x0, y0], ...], ...],
        ...}}
    """
    with open(filename, 'rb') as fp:
        try:
            doc = etree.parse(fp)
        except etree.XMLSyntaxError as e:
            raise KrakenInputException(f'Parsing {filename} failed: {e}')
    if doc.getroot().tag.endswith('alto'):
        return parse_alto(filename)
    elif doc.getroot().tag.endswith('PcGts'):
        return parse_page(filename)
    else:
        raise KrakenInputException(f'Unknown XML format in {filename}')


def parse_page(filename):
    """
    Parses a PageXML file, returns the baselines defined in it, and loads the
    referenced image.

    Args:
        filename (str): path to a PageXML file.

    Returns:
        A dict {'image': impath, lines: [{'boundary': [[x0, y0], ...],
        'baseline': [[x0, y0], ...]}, {...], 'text': 'apdjfqpf', 'script':
        'script_type'}, regions: {'region_type_0': [[[x0, y0], ...], ...],
        ...}}
    """
    def _parse_page_custom(s):
        o = {}
        s = s.strip()
        l_chunks = [l_chunk for l_chunk in s.split('}') if l_chunk.strip()]
        if l_chunks:
            for chunk in l_chunks:
                tag, vals = chunk.split('{')
                tag_vals = {}
                vals = [val.strip() for val in vals.split(';') if val.strip()]
                for val in vals:
                    key, val = val.split(':')
                    tag_vals[key] = val
                o[tag.strip()] = tag_vals
        return o

    def _parse_coords(coords):
        points = [x for x in coords.split(' ')]
        points = [int(c) for point in points for c in point.split(',')]
        return list(zip(points[::2], points[1::2]))


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
        data = {'image': os.path.join(base_dir, image.get('imageFilename')), 'lines': [], 'type': 'baselines', 'regions': {}}
        # find all image regions
        regions = []
        for x in page_regions.keys():
            regions.extend(doc.findall('.//{{*}}{}'.format(x)))
        # parse region type and coords
        region_data = defaultdict(list)
        for region in regions:
            coords = region.find('{*}Coords')
            if coords is not None and not coords.get('points').isspace() and len(coords.get('points')):
                try:
                    coords = _parse_coords(coords.get('points'))
                except:
                    logger.warning('Region {} without coordinates'.format(region.get('id')))
                    continue
            else:
                logger.warning('Region {} without coordinates'.format(region.get('id')))
                continue
            rtype = region.get('type')
            # parse transkribus-style custom field if possible
            custom_str = region.get('custom')
            if not rtype and custom_str:
                cs = _parse_page_custom(custom_str)
                if 'structure' in cs and 'type' in cs['structure']:
                    rtype = cs['structure']['type']
            # fall back to default region type if nothing is given
            if not rtype:
                rtype = page_regions[region.tag.split('}')[-1]]
            region_data[rtype].append(coords)

        data['regions'] = region_data

        # parse line information
        scripts = set()
        for line in lines:
            pol = line.find('./{*}Coords')
            boundary = None
            if pol is not None and not pol.get('points').isspace() and len(pol.get('points')):
                try:
                    boundary = _parse_coords(pol.get('points'))
                except:
                    logger.info('TextLine {} without polygon'.format(line.get('id')))
                    pass
            else:
                logger.info('TextLine {} without polygon'.format(line.get('id')))
            base = line.find('./{*}Baseline')
            baseline = None
            if base is not None and not base.get('points').isspace() and len(base.get('points')):
                try:
                    baseline = _parse_coords(pol.get('points'))
                except:
                    logger.info('TextLine {} without baseline'.format(line.get('id')))
                    continue
            else:
                logger.warning('TextLine {} without baseline'.format(line.get('id')))
                continue
            text = ''
            manual_transcription = line.find('./{*}TextEquiv')
            if manual_transcription is not None:
                line = manual_transcription
            for el in line.findall('.//{*}Unicode'):
                if el.text:
                    text += el.text
            # retrieve line tag if custom string is set and contains
            l_type = 'default'
            custom_str = line.get('custom')
            if custom_str:
                cs = _parse_page_custom(custom_str)
                if 'structure' in cs and 'type' in cs['structure']:
                    l_type = cs['structure']['type']
            scripts.add(l_type)
            data['lines'].append({'baseline': baseline, 'boundary': boundary, 'text': text, 'script': l_type})
        if len(scripts) > 1:
            data['script_detection'] = True
        else:
            data['script_detection'] = False
        return data


def parse_alto(filename):
    """
    Parses an ALTO file, returns the baselines defined in it, and loads the
    referenced image.

    Args:
        filename (str): path to an ALTO file.

    Returns:
        A dict {'image': impath, lines: [{'boundary': [[x0, y0], ...],
        'baseline': [[x0, y0], ...]}, {...], 'text': 'apdjfqpf', 'script':
        'script_type'}, regions: {'region_type_0': [[[x0, y0], ...], ...],
        ...}}
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
        data = {'image': os.path.join(base_dir, image.text), 'lines': [], 'type': 'baselines', 'regions': {}}
        for line in lines:
            if line.get('BASELINE') is None:
                raise KrakenInputException('ALTO file {} contains no baseline information'.format(filename))
            pol = line.find('./{*}Shape/{*}Polygon')
            boundary = None
            if pol is not None:
                points = [int(float(x)) for x in pol.get('POINTS').split(' ')]
                boundary = list(zip(points[::2], points[1::2]))
            points = [int(float(x)) for x in line.get('BASELINE').split(' ')]
            baseline = list(zip(points[::2], points[1::2]))
            text = ''
            for el in line.xpath(".//*[local-name() = 'String'] | .//*[local-name() = 'SP']"):
                text += el.get('CONTENT') if el.get('CONTENT') else ' '
            data['lines'].append({'baseline': baseline, 'boundary': boundary, 'text': text, 'script': 'default'})
        data['script_detection'] = False
        return data
