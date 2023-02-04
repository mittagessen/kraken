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
import re
import logging
from pathlib import Path

from itertools import groupby
from lxml import etree
from PIL import Image
from typing import Union, Dict, Any, Sequence, Tuple

from os import PathLike
from collections import defaultdict
from kraken.lib.segmentation import calculate_polygonal_environment
from kraken.lib.exceptions import KrakenInputException

logger = logging.getLogger(__name__)

__all__ = ['parse_xml', 'parse_page', 'parse_alto', 'preparse_xml_data']

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
                'CustomRegion': 'custom'}

# same for ALTO
alto_regions = {'TextBlock': 'text',
                'IllustrationType': 'illustration',
                'GraphicalElementType': 'graphic',
                'ComposedBlock': 'composed'}


def preparse_xml_data(filenames: Sequence[Union[str, PathLike]],
                      format_type: str = 'xml',
                      repolygonize: bool = False) -> Dict[str, Any]:
    """
    Loads training data from a set of xml files.

    Extracts line information from Page/ALTO xml files for training of
    recognition models.

    Args:
        filenames: List of XML files.
        format_type: Either `page`, `alto` or `xml` for autodetermination.
        repolygonize: (Re-)calculates polygon information using the kraken
                      algorithm.

    Returns:
        A list of dicts {'text': text, 'baseline': [[x0, y0], ...], 'boundary':
        [[x0, y0], ...], 'image': PIL.Image}.
    """
    training_pairs = []
    if format_type == 'xml':
        parse_fn = parse_xml
    elif format_type == 'alto':
        parse_fn = parse_alto
    elif format_type == 'page':
        parse_fn = parse_page
    else:
        raise ValueError(f'invalid format {format_type} for preparse_xml_data')

    for fn in filenames:
        try:
            data = parse_fn(fn)
        except KrakenInputException as e:
            logger.warning(e)
            continue
        try:
            with open(data['image'], 'rb') as fp:
                Image.open(fp)
        except FileNotFoundError as e:
            logger.warning(f'Could not open file {e.filename} in {fn}')
            continue
        if repolygonize:
            logger.info('repolygonizing {} lines in {}'.format(len(data['lines']), data['image']))
            data['lines'] = _repolygonize(data['image'], data['lines'])
        for line in data['lines']:
            training_pairs.append({'image': data['image'], **line})
    return training_pairs


def _repolygonize(im: Image.Image, lines: Sequence[Dict[str, Any]]):
    """
    Helper function taking an output of the lib.xml parse_* functions and
    recalculating the contained polygonization.

    Args:
        im (Image.Image): Input image
        lines (list): List of dicts [{'boundary': [[x0, y0], ...], 'baseline': [[x0, y0], ...], 'text': 'abcvsd'}, {...]

    Returns:
        A data structure `lines` with a changed polygonization.
    """
    im = Image.open(im).convert('L')
    polygons = calculate_polygonal_environment(im, [x['baseline'] for x in lines])
    return [{'boundary': polygon,
             'baseline': orig['baseline'],
             'text': orig['text'],
             'script': orig['script']} for orig, polygon in zip(lines, polygons)]


def parse_xml(filename: Union[str, PathLike]) -> Dict[str, Any]:
    """
    Parses either a PageXML or ALTO file with autodetermination of the file
    format.

    Args:
        filename: path to an XML file.

    Returns:
        A dict::

            {'image': impath,
             'lines': [{'boundary': [[x0, y0], ...],
                        'baseline': [[x0, y0], ...],
                        'text': apdjfqpf',
                        'tags': {'type': 'default', ...}},
                       ...
                       {...}],
             'regions': {'region_type_0': [[[x0, y0], ...], ...], ...}}
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


def parse_page(filename: Union[str, PathLike]) -> Dict[str, Any]:
    """
    Parses a PageXML file, returns the baselines defined in it, and loads the
    referenced image.

    Args:
        filename: path to a PageXML file.

    Returns:
        A dict::

            {'image': impath,
             'lines': [{'boundary': [[x0, y0], ...],
                        'baseline': [[x0, y0], ...],
                        'text': apdjfqpf',
                        'tags': {'type': 'default', ...}},
                       ...
                       {...}],
             'regions': {'region_type_0': [[[x0, y0], ...], ...], ...}}
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
                    key, *val = val.split(':')
                    tag_vals[key] = ":".join(val)
                o[tag.strip()] = tag_vals
        return o

    def _parse_coords(coords):
        points = [x for x in coords.split(' ')]
        points = [int(c) for point in points for c in point.split(',')]
        pts = zip(points[::2], points[1::2])
        return [k for k, g in groupby(pts)]

    with open(filename, 'rb') as fp:
        base_dir = Path(filename).parent
        try:
            doc = etree.parse(fp)
        except etree.XMLSyntaxError as e:
            raise KrakenInputException('Parsing {} failed: {}'.format(filename, e))
        image = doc.find('.//{*}Page')
        if image is None or image.get('imageFilename') is None:
            raise KrakenInputException('No valid image filename found in PageXML file {}'.format(filename))
        try:
            base_direction = {'left-to-right': 'L',
                              'right-to-left': 'R',
                              'top-to-bottom': 'L',
                              'bottom-to-top': 'R',
                              None: None}[image.get('readingDirection')]
        except KeyError:
            logger.warning(f'Invalid value {image.get("readingDirection")} encountered in page-level reading direction.')
            base_direction = None
        lines = doc.findall('.//{*}TextLine')
        data = {'image': base_dir.joinpath(image.get('imageFilename')),
                'lines': [],
                'type': 'baselines',
                'base_dir': base_direction,
                'regions': {}}
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
                except Exception:
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
        tag_set = set(('default',))
        for line in lines:
            pol = line.find('./{*}Coords')
            boundary = None
            if pol is not None and not pol.get('points').isspace() and len(pol.get('points')):
                try:
                    boundary = _parse_coords(pol.get('points'))
                except Exception:
                    logger.info('TextLine {} without polygon'.format(line.get('id')))
            else:
                logger.info('TextLine {} without polygon'.format(line.get('id')))
            base = line.find('./{*}Baseline')
            baseline = None
            if base is not None and not base.get('points').isspace() and len(base.get('points')):
                try:
                    baseline = _parse_coords(base.get('points'))
                except Exception:
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
            # retrieve line tags if custom string is set and contains
            tags = {'type': 'default'}
            split_type = None
            custom_str = line.get('custom')
            if custom_str:
                cs = _parse_page_custom(custom_str)
                if 'structure' in cs and 'type' in cs['structure']:
                    tags['type'] = cs['structure']['type']
                    tag_set.add(tags['type'])
                # retrieve data split if encoded in custom string.
                if 'split' in cs and 'type' in cs['split'] and cs['split']['type'] in ['train', 'validation', 'test']:
                    split_type = cs['split']['type']
                    tags['split'] = split_type
                    tag_set.add(split_type)

            data['lines'].append({'baseline': baseline,
                                  'boundary': boundary,
                                  'text': text,
                                  'split': split_type,
                                  'tags': tags})
        if len(tag_set) > 1:
            data['script_detection'] = True
        else:
            data['script_detection'] = False
        return data


def parse_alto(filename: Union[str, PathLike]) -> Dict[str, Any]:
    """
    Parses an ALTO file, returns the baselines defined in it, and loads the
    referenced image.

    Args:
        filename: path to an ALTO file.

    Returns:
        A dict::

            {'image': impath,
             'lines': [{'boundary': [[x0, y0], ...],
                        'baseline': [[x0, y0], ...],
                        'text': apdjfqpf',
                        'tags': {'type': 'default', ...}},
                       ...
                       {...}],
             'regions': {'region_type_0': [[[x0, y0], ...], ...], ...}}
    """
    def _parse_pointstype(coords: str) -> Sequence[Tuple[float, float]]:
        """
        ALTO's PointsType is underspecified so a variety of serializations are valid:

            x0, y0 x1, y1 ...
            x0 y0 x1 y1 ...
            (x0, y0) (x1, y1) ...
            (x0 y0) (x1 y1) ...

        Returns:
            A list of tuples [(x0, y0), (x1, y1), ...]
        """
        float_re = re.compile(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?')
        points = [float(point.group()) for point in float_re.finditer(coords)]
        if len(points) % 2:
            raise ValueError(f'Odd number of points in points sequence: {points}')
        pts = zip(points[::2], points[1::2])
        return [k for k, g in groupby(pts)]

    with open(filename, 'rb') as fp:
        base_dir = Path(filename).parent
        try:
            doc = etree.parse(fp)
        except etree.XMLSyntaxError as e:
            raise KrakenInputException('Parsing {} failed: {}'.format(filename, e))
        image = doc.find('.//{*}fileName')
        if image is None or not image.text:
            raise KrakenInputException('No valid filename found in ALTO file')
        lines = doc.findall('.//{*}TextLine')
        data = {'image': base_dir.joinpath(image.text),
                'lines': [],
                'type': 'baselines',
                'base_dir': None,
                'regions': {}}
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
                    cls_map[tag.get('ID')] = (x[:-3].lower(), tag.get('LABEL'))
        # parse region type and coords
        region_data = defaultdict(list)
        for region in regions:
            # try to find shape object
            coords = region.find('./{*}Shape/{*}Polygon')
            if coords is not None:
                boundary = _parse_pointstype(coords.get('POINTS'))
            elif (region.get('HPOS') is not None and region.get('VPOS') is not None and
                  region.get('WIDTH') is not None and region.get('HEIGHT') is not None):
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
                    ttype, rtype = cls_map.get(tagref, (None, None))
                    if rtype is not None and ttype:
                        break
            if rtype is None:
                rtype = alto_regions[region.tag.split('}')[-1]]
            if boundary == page_boundary and rtype == 'text':
                logger.info('Skipping TextBlock with same size as page image.')
                continue
            region_data[rtype].append(boundary)
        data['regions'] = region_data

        tag_set = set(('default',))
        for line in lines:
            if line.get('BASELINE') is None:
                logger.info('TextLine {} without baseline'.format(line.get('ID')))
                continue
            pol = line.find('./{*}Shape/{*}Polygon')
            boundary = None
            if pol is not None:
                try:
                    boundary = _parse_pointstype(pol.get('POINTS'))
                except ValueError:
                    logger.info('TextLine {} without polygon'.format(line.get('ID')))
            else:
                logger.info('TextLine {} without polygon'.format(line.get('ID')))

            baseline = None
            try:
                baseline = _parse_pointstype(line.get('BASELINE'))
            except ValueError:
                logger.info('TextLine {} without baseline'.format(line.get('ID')))

            text = ''
            for el in line.xpath(".//*[local-name() = 'String'] | .//*[local-name() = 'SP']"):
                text += el.get('CONTENT') if el.get('CONTENT') else ' '
            # find line type
            tags = {'type': 'default'}
            split_type = None
            tagrefs = line.get('TAGREFS')
            if tagrefs is not None:
                for tagref in tagrefs.split():
                    ttype, ltype = cls_map.get(tagref, (None, None))
                    if ltype is not None:
                        tag_set.add(ltype)
                        if ttype == 'other':
                            tags['type'] = ltype
                        else:
                            tags[ttype] = ltype
                    if ltype in ['train', 'validation', 'test']:
                        split_type = ltype
            data['lines'].append({'baseline': baseline,
                                  'boundary': boundary,
                                  'text': text,
                                  'tags': tags,
                                  'split': split_type})

        if len(tag_set) > 1:
            data['tags'] = True
        else:
            data['tags'] = False
        return data
