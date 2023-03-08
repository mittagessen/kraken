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
from typing import Union, Dict, Any, Sequence, Tuple, Literal, Optional, List

from os import PathLike
from collections import defaultdict
from kraken.lib.segmentation import calculate_polygonal_environment

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
                'Illustration': 'illustration',
                'GraphicalElement': 'graphic',
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
        except ValueError as e:
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
            region_data[rtype].append({'id': region.get('ID'), 'boundary': boundary})
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

class XMLPage(object):

    type: Literal['baselines', 'bbox'] == 'baselines'
    base_dir: Optional[Literal['L', 'R']] = None
    imagename: pathlib.Path = None
    _orders: Dict[str, Dict[str, Any]] = None
    has_tags: bool = False
    _tag_set: Optional[Dict] = None
    has_splits: bool = False
    _split_set: Optional[List] = None

    def __init__(self,
                 filename: Union[str, pathlib.Path],
                 filetype: Literal['xml', 'alto', 'page'] = 'xml'):
        super().__init__()
        self.filename = Path(filename)
        self.filetype = filetype

        self._regions = {}
        self._baselines = {}
        self._orders = {'line_implicit': {'order': [], 'is_total': True, 'description': 'Implicit line order derived from element sequence'},
                        'region_implicit': {'order': [], 'is_total': True, 'description': 'Implicit region order derived from element sequence'}}

        if filetype == 'xml':
            self._parse_xml()
        elif filetype == 'alto':
            self._parse_alto()
        elif filetype == 'page':
            self._parse_page()

    def _parse_xml(self):
        with open(self.filename, 'rb') as fp:
            try:
                doc = etree.parse(fp)
            except etree.XMLSyntaxError as e:
                raise ValueError(f'Parsing {self.filename} failed: {e}')
        if doc.getroot().tag.endswith('alto'):
            return self._parse_alto()
        elif doc.getroot().tag.endswith('PcGts'):
            return self._parse_page()
        else:
            raise ValueError(f'Unknown XML format in {self.filename}')

    def _parse_alto(self):
        with open(self.filename, 'rb') as fp:
            base_directory = self.filename.parent
            try:
                doc = etree.parse(fp)
            except etree.XMLSyntaxError as e:
                raise ValueError('Parsing {} failed: {}'.format(self.filename, e))
            image = doc.find('.//{*}fileName')
            if image is None or not image.text:
                raise ValueError('No valid image filename found in ALTO file {self.filename}')
            self.imagename = base_directory.joinpath(image.text)

            lines = doc.findall('.//{*}TextLine')
            # find all image regions in order
            regions = []
            for el in doc.iterfind('./{*}Layout/{*}Page/{*}PrintSpace/{*}*'):
                for block_type in alto_regions.keys():
                    if el.tag.endswith(block_type):
                        regions.append(el)
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
                    boundary = self._parse_alto_pointstype(coords.get('POINTS'))
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
                region_data[rtype].append({'id': region.get('ID'), 'boundary': boundary})
                # register implicit reading order
                self._orders['region_implicit']['order'].append(region.get('ID'))
            self._regions = region_data

            self._tag_set = set(('default',))
            self.lines = []
            for line in lines:
                if line.get('BASELINE') is None:
                    logger.info('TextLine {} without baseline'.format(line.get('ID')))
                    continue
                pol = line.find('./{*}Shape/{*}Polygon')
                boundary = None
                if pol is not None:
                    try:
                        boundary = self._parse_alto_pointstype(pol.get('POINTS'))
                    except ValueError:
                        logger.info('TextLine {} without polygon'.format(line.get('ID')))
                else:
                    logger.info('TextLine {} without polygon'.format(line.get('ID')))

                baseline = None
                try:
                    baseline = self._parse_alto_pointstype(line.get('BASELINE'))
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
                            self._tag_set.add(ltype)
                            if ttype == 'other':
                                tags['type'] = ltype
                            else:
                                tags[ttype] = ltype
                        if ltype in ['train', 'validation', 'test']:
                            split_type = ltype
                self.lines.append({'id': line.get('ID'),
                                   'baseline': baseline,
                                   'boundary': boundary,
                                   'text': text,
                                   'tags': tags,
                                   'split': split_type})
                # register implicit reading order
                self._orders['line_implicit']['order'].append(line.get('ID'))

            if len(self._tag_set) > 1:
                self.has_tags = True
            else:
                self.has_tags = False

            # parse explicit reading orders if they exist
            ro_el = doc.find('.//{*}ReadingOrder')
            if ro_el is not None:
                reading_orders = ro_el.getchildren()
                # UnorderedGroup at top-level => treated as multiple reading orders
                if len(reading_orders) == 1 and reading_orders[0].tag.endswith('UnorderedGroup'):
                     reading_orders = reading_orders[0].getchildren()
                else:
                    reading_orders = [reading_orders]
                def _parse_group(el):
                    _ro = []
                    if el.tag.endswith('UnorderedGroup'):
                        _ro.append([_parse_group(x) for x in el.iterchildren()])
                        is_total = False
                    elif el.tag.endswith('OrderedGroup'):
                        _ro.extend(_parse_group(x) for x in el.iterchildren())
                    else:
                        ref = el.get('REF')
                        res = doc.find(f'.//{{*}}*[@ID="{ref}"]')
                        if res is None:
                            logger.warning(f'Nonexistant element with ID {ref} in reading order. Skipping RO {ro.get("ID")}.')
                            is_valid = False
                            return _ro
                        tag = res.tag.split('}')[-1]
                        if tag not in alto_regions.keys() and tag != 'TextLine':
                            logger.warning(f'Sub-line element with ID {ref} in reading order. Skipping RO {ro.get("ID")}.')
                            is_valid = False
                    return _ro

                for ro in reading_orders:
                    is_total = True
                    is_valid = True
                    joint_order = _parse_group(ro)
                    if is_valid:
                        tag = ro.get('TAGREFS')
                        self._orders[ro.get('ID')] = {'order': joint_order,
                                                      'is_total': is_total,
                                                      'description': cls_map[tag] if tag and tag in cls_map else ''}
        self.filetype = 'alto'

    def _parse_page(self):
        with open(self.filename, 'rb') as fp:
            base_directory = self.filename.parent

            try:
                doc = etree.parse(fp)
            except etree.XMLSyntaxError as e:
                raise ValueError(f'Parsing {self.filename} failed: {e}')
            image = doc.find('.//{*}Page')
            if image is None or image.get('imageFilename') is None:
                raise ValueError(f'No valid image filename found in PageXML file {self.filename}')
            try:
                self.base_dir = {'left-to-right': 'L',
                                 'right-to-left': 'R',
                                 'top-to-bottom': 'L',
                                 'bottom-to-top': 'R',
                                 None: None}[image.get('readingDirection')]
            except KeyError:
                logger.warning(f'Invalid value {image.get("readingDirection")} encountered in page-level reading direction.')
            lines = doc.findall('.//{*}TextLine')
            self.imagename = base_directory.joinpath(image.get('imageFilename'))
            # find all image regions
            regions = [reg for reg in image.iterfind('./{*}*')]
            # parse region type and coords
            region_data = defaultdict(list)
            tr_region_order = []
            for region in regions:
                coords = region.find('./{*}Coords')
                if coords is not None and not coords.get('points').isspace() and len(coords.get('points')):
                    try:
                        coords = self._parse_page_coords(coords.get('points'))
                    except Exception:
                        logger.warning('Region {} without coordinates'.format(region.get('id')))
                        continue
                else:
                    logger.warning('Region {} without coordinates'.format(region.get('id')))
                    continue
                rtype = region.get('type')
                # parse transkribus-style custom field if possible
                custom_str = region.get('custom')
                if custom_str:
                    cs = self._parse_page_custom(custom_str)
                    if not rtype and 'structure' in cs and 'type' in cs['structure']:
                        rtype = cs['structure']['type']
                    # transkribus-style reading order
                    if 'readingOrder' in cs and 'index'in cs['readingOrder']:
                        tr_region_order.append((region.get('id'), int(cs['readingOrder']['index'])))
                # fall back to default region type if nothing is given
                if not rtype:
                    rtype = page_regions[region.tag.split('}')[-1]]
                region_data[rtype].append({'id': region.get('id'), 'boundary': coords})
                # register implicit reading order
                self._orders['region_implicit']['order'].append(region.get('id'))
            # add transkribus-style region order
            self._orders['region_transkribus'] = {'order': [x[0] for x in sorted(tr_region_order, key=lambda k: k[1])],
                                                  'is_total': True if len(set(map(lambda x: x[0], tr_region_order))) == len(tr_region_order) else False,
                                                  'description': 'Explicit region order from `custom` attribute'}

            self._regions = region_data

            # parse line information
            self._tag_set = set(('default',))
            tmp_transkribus_line_order = defaultdict(list)
            valid_tr_lo = True
            for line in lines:
                pol = line.find('./{*}Coords')
                boundary = None
                if pol is not None and not pol.get('points').isspace() and len(pol.get('points')):
                    try:
                        boundary = self._parse_coords(pol.get('points'))
                    except Exception:
                        logger.info('TextLine {} without polygon'.format(line.get('id')))
                else:
                    logger.info('TextLine {} without polygon'.format(line.get('id')))
                base = line.find('./{*}Baseline')
                baseline = None
                if base is not None and not base.get('points').isspace() and len(base.get('points')):
                    try:
                        baseline = self._parse_page_coords(base.get('points'))
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
                    cs = self._parse_page_custom(custom_str)
                    if 'structure' in cs and 'type' in cs['structure']:
                        tags['type'] = cs['structure']['type']
                        self._tag_set.add(tags['type'])
                    # retrieve data split if encoded in custom string.
                    if 'split' in cs and 'type' in cs['split'] and cs['split']['type'] in ['train', 'validation', 'test']:
                        split_type = cs['split']['type']
                        tags['split'] = split_type
                        self._tag_set.add(split_type)
                    if 'readingOrder' in cs and 'index' in cs['readingOrder']:
                        # look up region index from parent
                        reg_cus = self._parse_page_custom(line.getparent().get('custom'))
                        if 'readingOrder' not in reg_cus or 'index' not in reg_cus['readingOrder']:
                            logger.warning('Incomplete `custom` attribute reading order found.')
                            valid_tr_lo = False
                        else:
                            tmp_transkribus_line_order[int(reg_cus['readingOrder']['index'])].append((int(cs['readingOrder']['index']), line.get('id')))

                self._baselines[line.get('id')] = {'baseline': baseline,
                                                   'boundary': boundary,
                                                   'text': text,
                                                   'split': split_type,
                                                   'tags': tags}

                # register implicit reading order
                self._orders['line_implicit']['order'].append(line.get('id'))
            if tmp_transkribus_line_order:
                # sort by regions
                tmp_reg_order = sorted(((k, v) for k, v in tmp_transkribus_line_order.items()), key=lambda k: k[0])
                # flatten
                tr_line_order = []
                for _, lines in tmp_reg_order:
                    tr_line_order.extend([x[1] for x in sorted(lines, key=lambda k: k[0])])
                self._orders['line_transkribus'] = {'order': tr_line_order,
                                                    'is_total': True,
                                                    'description': 'Explicit line order from `custom` attribute'}

            # parse explicit reading orders if they exist
            ro_el = doc.find('.//{*}ReadingOrder')
            if ro_el is not None:
               reading_orders = ro_el.getchildren()
               # UnorderedGroup at top-level => treated as multiple reading orders
               if len(reading_orders) == 1 and reading_orders[0].tag.endswith('UnorderedGroup'):
                    reading_orders = reading_orders.getchildren()
               else:
                   reading_orders = [reading_orders]
               def _parse_group(el):
                   _ro = []
                   if el.tag.endswith('UnorderedGroup'):
                       _ro.append([_parse_group(x) for x in el.iterchildren()])
                       is_total = False
                   elif el.tag.endswith('OrderedGroup'):
                       _ro.extend(_parse_group(x) for x in el.iterchildren())
                   else:
                       return el.get('regionRef')
                   return _ro

               for ro in reading_orders:
                   is_total = True
                   self._orders[ro.get('id')] = {'order': _parse_group(ro),
                                                 'is_total': is_total,
                                                 'description': ro.get('caption') if ro.get('caption') else ''}

        if len(self._tag_set) > 1:
            self.has_tags = True
        else:
            self.has_tags = False

        self.filetype = 'page'

    @property
    def regions(self):
        return self._regions

    @property
    def baselines(self):
        return self._baselines

    @property
    def reading_orders(self):
        return self._orders

    def get_baselines_by_region(self, region):
        pass

    def get_baselines_by_tag(self, key, value):
        return {k: v for k, v in self._baselines.items() if v['tags'].get(key) == value}

    def get_baselines_by_split(self, split: Literal['train', 'validation', 'test']):
        return {k: v for k, v in self._baselines.items() if v['tags'].get(key) == split}

    @property
    def tags(self):
        return self._tag_set

    @property
    def splits(self):
        return self._split_set

    @staticmethod
    def _parse_alto_pointstype(coords: str) -> Sequence[Tuple[float, float]]:
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

    @staticmethod
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

    @staticmethod
    def _parse_page_coords(coords):
        points = [x for x in coords.split(' ')]
        points = [int(c) for point in points for c in point.split(',')]
        pts = zip(points[::2], points[1::2])
        return [k for k, g in groupby(pts)]

    def __str__(self):
        return f'XMLPage {self.filename} (format: {self.filetype}, image: {self.imagename})'

    def __repr__(self):
        return f'XMLPage(filename={self.filename}, filetype={self.filetype})'
