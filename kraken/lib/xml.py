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

from itertools import groupby
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

# same for ALTO
alto_regions = {'TextBlock': 'text',
                'IllustrationType': 'illustration',
                'GraphicalElementType': 'graphic',
                'ComposedBlock': 'composed'}

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
        pts = zip(points[::2], points[1::2])
        return [k for k, g in groupby(pts)]


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
        scripts = set(('default',))
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
                    baseline = _parse_coords(base.get('points'))
                except:
                    logger.info('TextLine {} without baseline'.format(line.get('id')))
                    continue
            else:
                logger.info('TextLine {} without baseline'.format(line.get('id')))
                continue
            text = ''
            manual_transcription = line.find('./{*}TextEquiv')
            if manual_transcription is not None:
                transcription = manual_transcription
            else:
                transcription = line
            for el in transcription.findall('.//{*}Unicode'):
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
        # find all image regions
        regions = []
        for x in alto_regions.keys():
            regions.extend(doc.findall('./{{*}}Layout/{{*}}Page/{{*}}PrintSpace/{{*}}{}'.format(x)))
        # find overall dimensions to filter out dummy TextBlocks
        ps = doc.find('./{*}Layout/{*}Page/{*}PrintSpace')
        x_min = int(float(ps.get('HPOS')))
        y_min = int(float(ps.get('VPOS')))
        width = int(float(ps.get('WIDTH')))
        height = int(float(ps.get('HEIGHT')))
        page_boundary = [(x_min, y_min),
                         (x_min, y_min + height),
                         (x_min + width, y_min + height),
                         (x_min + width, y_min)]

        # parse tagrefs
        cls_map = {}
        tags = doc.find('.//{*}Tags')
        if tags is not None:
            for x in ['StructureTag', 'LayoutTag', 'OtherTag']:
                for tag in tags.findall('./{{*}}{}'.format(x)):
                    cls_map[tag.get('ID')] = tag.get('LABEL')
        # parse region type and coords
        region_data = defaultdict(list)
        for region in regions:
            # try to find shape object
            coords = region.find('./{*}Shape/{*}Polygon')
            if coords is not None:
                points = [int(float(x)) for x in coords.get('POINTS').split(' ')]
                boundary = zip(points[::2], points[1::2])
                boundary = [k for k, g in groupby(boundary)]
            elif region.get('HPOS') is not None and region.get('VPOS') is not None and region.get('WIDTH') is not None and region.get('HEIGHT') is not None:
                # use rectangular definition
                x_min = int(float(region.get('HPOS')))
                y_min = int(float(region.get('VPOS')))
                width = int(float(region.get('WIDTH')))
                height = int(float(region.get('HEIGHT')))
                boundary = [(x_min, y_min),
                            (x_min, y_min + height),
                            (x_min + width, y_min + height),
                            (x_min + width, y_min)]
            else:
                continue
            rtype = region.get('TYPE')
            # fall back to default region type if nothing is given
            tagrefs = region.get('TAGREFS')
            if tagrefs is not None and rtype is None:
                for tagref in tagrefs.split():
                    rtype = cls_map.get(tagref, None)
                    if rtype is not None:
                        break
            if rtype is None:
                rtype = alto_regions[region.tag.split('}')[-1]]
            if boundary == page_boundary and rtype == 'text':
                    logger.info('Skipping TextBlock with same size as page image.')
                    continue
            region_data[rtype].append(boundary)
        data['regions'] = region_data

        scripts = set(('default',))
        for line in lines:
            if line.get('BASELINE') is None:
                logger.info('TextLine {} without baseline'.format(line.get('ID')))
                continue
            pol = line.find('./{*}Shape/{*}Polygon')
            boundary = None
            if pol is not None:
                try:
                    points = [int(float(x)) for x in pol.get('POINTS').split(' ')]
                    boundary = zip(points[::2], points[1::2])
                    boundary = [k for k, g in groupby(boundary)]
                except ValueError:
                    logger.info('TextLine {} without polygon'.format(line.get('ID')))
            else:
                logger.info('TextLine {} without polygon'.format(line.get('ID')))

            baseline = None
            try:
                points = [int(float(x)) for x in line.get('BASELINE').split(' ')]
                baseline = list(zip(points[::2], points[1::2]))
                baseline =  [k for k, g in groupby(baseline)]
            except ValueError:
                logger.info('TextLine {} without baseline'.format(line.get('ID')))

            text = ''
            for el in line.xpath(".//*[local-name() = 'String'] | .//*[local-name() = 'SP']"):
                text += el.get('CONTENT') if el.get('CONTENT') else ' '
            # find line type
            ltype = None
            tagrefs = line.get('TAGREFS')
            if tagrefs is not None:
                for tagref in tagrefs.split():
                    ltype = cls_map.get(tagref, None)
                    if ltype is not None:
                        scripts.add(ltype)
                        break
            data['lines'].append({'baseline': baseline, 'boundary': boundary, 'text': text, 'script': ltype if ltype is not None else 'default'})

        if len(scripts) > 1:
            data['script_detection'] = True
        else:
            data['script_detection'] = False
        return data
