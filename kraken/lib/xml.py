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
import re
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Optional,
                    Sequence, Tuple, Union)

from lxml import etree

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

from kraken.containers import BBoxLine, BaselineLine, Region, Segmentation

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['XMLPage']


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


class XMLPage(object):
    """
    Parses XML facsimiles in ALTO or PageXML format.

    The parser is able to deal with most (but not all) features supported by
    those standards. In particular, any data below the line level is discarded.

    Args:
        filename: Path to the XML file
        filetype: Selector for explicit subparser choice.
        linetype: Parse line data as baselines or bounding box type.
    """
    type: Literal['baselines', 'bbox'] = 'baselines'
    base_dir: Optional[Literal['L', 'R']] = None
    imagename: 'PathLike' = None
    image_size: Tuple[int, int] = None
    _orders: Dict[str, Dict[str, Any]] = None
    has_tags: bool = False
    _tag_set: Optional[Dict] = None
    has_splits: bool = False
    _split_set: Optional[List] = None

    def __init__(self,
                 filename: Union[str, 'PathLike'],
                 filetype: Literal['xml', 'alto', 'page'] = 'xml',
                 linetype: Literal['baselines', 'bbox'] = 'baselines'):
        super().__init__()
        self.filename = Path(filename)
        self.filetype = filetype
        self.type = linetype

        self._regions = {}
        self._lines = {}
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
                raise ValueError(f'Parsing {self.filename} failed: {e}')

        if (mu := doc.find('.//{*}MeasurementUnit')) is not None and mu.text.strip() != 'pixel':
            raise ValueError(f'Measurement unit in ALTO file {self.filename} is "{mu.text.strip()} not "pixel".')

        if (image := doc.find('.//{*}fileName')) is None or not image.text:
            raise ValueError(f'No valid image filename found in ALTO file {self.filename}')

        self.imagename = base_directory.joinpath(image.text)

        if (page := doc.find('.//{*}Page')) is None:
            raise ValueError(f'No "Page" element found in ALTO file {self.filename}')

        try:
            self.image_size = int(page.get('WIDTH')), int(page.get('HEIGHT'))
        except (ValueError, TypeError) as e:
            raise ValueError(f'Invalid image dimensions in {self.filename}: {e}')

        page_default_lang = page.get('LANG')

        # find all image regions in order
        regions = []
        for el in doc.iterfind('./{*}Layout/{*}Page/{*}PrintSpace/{*}*'):
            for block_type in alto_regions.keys():
                if el.tag.endswith(block_type):
                    regions.append(el)

        # parse tagrefs
        cls_map = {}
        if (tags := doc.find('.//{*}Tags')) is not None:
            for x in ['StructureTag', 'LayoutTag', 'OtherTag']:
                for tag in tags.findall(f'./{{*}}{x}'):
                    cls_map[tag.get('ID')] = (x[:-3].lower(), tag.get('TYPE'), tag.get('LABEL'))

        self._tag_set = set(('default',))

        # parse region type and coords
        region_data = defaultdict(list)
        for region in regions:
            region_id = region.get('ID')
            region_default_direction = {'ltr': 'L',
                                        'rtl': 'R',
                                        'ttb': 'L',
                                        'btt': 'R'}.get(region.get('BASEDIRECTION'), None)

            # try to find shape object
            boundary = None
            if (coords := region.find('./{*}Shape/{*}Polygon')) is not None:
                boundary = self._parse_alto_pointstype(coords.get('POINTS'))
            else:
                # use rectangular definition
                reg_pos = region.get('HPOS'), region.get('VPOS'), region.get('WIDTH'), region.get('HEIGHT')
                try:
                    x_min, y_min, width, height = map(int, map(float, reg_pos))
                    boundary = [(x_min, y_min),
                                (x_min, y_min + height),
                                (x_min + width, y_min + height),
                                (x_min + width, y_min)]
                except (ValueError, TypeError):
                    pass
            # parse region tags
            tags = self._parse_alto_tagrefs(cls_map, region.get('TAGREFS'))
            tag_type = tags.pop('region') if 'region' in tags else tags.pop('type', None)
            if (rtype := region.get('TYPE')) is None:
                rtype = tag_type
            else:
                rtype = [{'type': rtype}]
            if rtype is None:
                rtype = [{'type': alto_regions[region.tag.split('}')[-1]]}]
            tags['type'] = rtype

            region_default_lang = self._parse_alto_langs(region,
                                                         cls_map,
                                                         [page_default_lang] if page_default_lang is not None else None)

            region_data[rtype[0]['type']].append(Region(id=region_id,
                                                        boundary=boundary,
                                                        tags=tags,
                                                        language=region_default_lang))
            # register implicit reading order
            self._orders['region_implicit']['order'].append(region_id)

            # parse lines in region
            for line in region.iterfind('./{*}TextLine'):
                line_id = line.get('ID')
                if self.type == 'baselines':
                    try:
                        baseline = self._parse_alto_pointstype(line.get('BASELINE'))
                    except ValueError:
                        logger.info(f'TextLine {line_id} without baseline')
                        continue

                    boundary = None
                    try:
                        pol = line.find('./{*}Shape/{*}Polygon')
                        boundary = self._parse_alto_pointstype(pol.get('POINTS'))
                    except (ValueError, AttributeError):
                        logger.info(f'TextLine {line_id} without polygon')

                elif self.type == 'bbox':
                    line_pos = line.get('HPOS'), line.get('VPOS'), line.get('WIDTH'), line.get('HEIGHT')
                    try:
                        x_min, y_min, width, height = map(int, map(float, line_pos))
                        bbox = (x_min, y_min, x_min+width, y_min+height)
                    except (ValueError, TypeError):
                        logger.info(f'TextLine {line_id} without complete bounding box data.')
                        continue

                text = ''
                for el in line.xpath(".//*[local-name() = 'String'] | .//*[local-name() = 'SP']"):
                    text += el.get('CONTENT') if el.get('CONTENT') else ' '
                # parse tags
                tags = self._parse_alto_tagrefs(cls_map, line.get('TAGREFS'))
                line_langs = self._parse_alto_langs(line, cls_map, region_default_lang)
                line_split = None
                if (split := tags.get('split', None)) is not None and len(split) == 1:
                    line_split = split[0]['type']
                    tags.pop('split')

                # get base text direction
                line_dir = {'ltr': 'L',
                            'rtl': 'R',
                            'ttb': 'L',
                            'btt': 'R'}.get(line.get('BASEDIRECTION'), None)
                if region_default_direction and line_dir is None:
                    line_dir = region_default_direction

                if self.type == 'baselines':
                    line_obj = BaselineLine(id=line_id,
                                            baseline=baseline,
                                            boundary=boundary,
                                            text=text,
                                            tags=tags if tags else None,
                                            language=line_langs,
                                            split=line_split,
                                            base_dir=line_dir,
                                            regions=[region_id])
                elif self.type == 'bbox':
                    line_obj = BBoxLine(id=line_id,
                                        bbox=bbox,
                                        text=text,
                                        tags=tags if tags else None,
                                        language=line_langs,
                                        split=line_split,
                                        base_dir=line_dir,
                                        regions=[region_id])

                self._lines[line_id] = line_obj
                # register implicit reading order
                self._orders['line_implicit']['order'].append(line_id)

        self._regions = region_data

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

            def _parse_group(el):
                nonlocal is_valid

                _ro = []
                if el.tag.endswith('UnorderedGroup'):
                    _ro = [_parse_group(x) for x in el.iterchildren()]
                    is_total = False  # NOQA
                elif el.tag.endswith('OrderedGroup'):
                    _ro.extend(_parse_group(x) for x in el.iterchildren())
                else:
                    ref = el.get('REF')
                    res = doc.find(f'.//{{*}}*[@ID="{ref}"]')
                    if res is None:
                        logger.info(f'Nonexistent element with ID {ref} in reading order. Skipping RO {ro.get("ID")}.')
                        is_valid = False
                        return _ro
                    tag = res.tag.split('}')[-1]
                    if tag not in alto_regions.keys() and tag != 'TextLine':
                        logger.info(f'Sub-line element with ID {ref} in reading order. Skipping RO {ro.get("ID")}.')
                        is_valid = False
                        return _ro
                    return ref
                return _ro

            for ro in reading_orders:
                is_total = True
                is_valid = True
                joint_order = _parse_group(ro)
                if is_valid:
                    tags = self._parse_alto_tagrefs(cls_map, ro.get('TAGREFS'))
                    self._orders[ro.get('ID')] = {'order': joint_order,
                                                  'is_total': is_total,
                                                  'description': tags.get('type', '')}
        self.filetype = 'alto'

    def _parse_page(self):
        with open(self.filename, 'rb') as fp:
            base_directory = self.filename.parent

            try:
                doc = etree.parse(fp)
            except etree.XMLSyntaxError as e:
                raise ValueError(f'Parsing {self.filename} failed: {e}')

        if (image := doc.find('.//{*}Page')) is None or image.get('imageFilename') is None:
            raise ValueError(f'No valid image filename found in PageXML file {self.filename}')
        page_default_direction = {'left-to-right': 'L',
                                  'right-to-left': 'R',
                                  'top-to-bottom': 'L',
                                  'bottom-to-top': 'R'}.get(image.get('readingDirection'), None)

        page_default_lang = self._parse_page_langs(image)

        self.imagename = base_directory.joinpath(image.get('imageFilename'))
        self.image_size = int(image.get('imageWidth')), int(image.get('imageHeight'))

        # parse region type and coords
        region_data = defaultdict(list)
        tr_region_order = []

        self._tag_set = set(('default',))
        tmp_transkribus_line_order = defaultdict(list)

        for region in image.iterfind('./{*}*'):
            if not any([True if region.tag.endswith(k) else False for k in page_regions.keys()]):
                continue
            region_id = region.get('id')
            coords = region.find('./{*}Coords')
            try:
                coords = self._parse_page_coords(coords.get('points'))
            except Exception:
                logger.info(f'Region {region_id} without coordinates')
                coords = None
            tags = {}
            rtype = region.get('type')
            # parse transkribus-style custom field if possible
            region_default_lang = self._parse_page_langs(region, page_default_lang)
            if (custom_str := region.get('custom')) is not None:
                cs = self._parse_page_custom(custom_str)
                if not rtype and 'structure' in cs and 'type' in cs['structure'][0]:
                    rtype = cs['structure'][0]['type']
                # transkribus-style reading order
                if (reg_ro := cs.get('readingOrder')) is not None and (reg_ro_idx := reg_ro[0].get('index')) is not None:
                    tr_region_order.append((region_id, int(reg_ro_idx)))
                tags.update(cs)

            if region_default_lang is None:
                region_default_lang = page_default_lang

            # fall back to default region type if nothing is given
            if not rtype:
                rtype = page_regions[region.tag.split('}')[-1]]

            tags['type'] = [{'type': rtype}]
            region_data[rtype].append(Region(id=region_id, boundary=coords, tags=tags, language=region_default_lang))

            region_default_direction = {'left-to-right': 'L',
                                        'right-to-left': 'R',
                                        'top-to-bottom': 'L',
                                        'bottom-to-top': 'R'}.get(region.get('readingDirection'))

            # register implicit reading order
            self._orders['region_implicit']['order'].append(region_id)

            # parse line information
            for line in region.iterfind('./{*}TextLine'):
                line_id = line.get('id')
                base = line.find('./{*}Baseline')
                baseline = None
                try:
                    baseline = self._parse_page_coords(base.get('points'))
                except Exception:
                    logger.info(f'TextLine {line_id} without baseline')
                    if self.type == 'baselines':
                        continue

                pol = line.find('./{*}Coords')
                boundary = None
                try:
                    boundary = self._parse_page_coords(pol.get('points'))
                except Exception:
                    logger.info(f'TextLine {line_id} without polygon')
                    if self.type == 'bbox':
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
                tags = {}
                custom_str = line.get('custom')
                if custom_str:
                    cs = self._parse_page_custom(custom_str)
                    if (structure := cs.get('structure')) is not None and (ltype := structure[0].get('type')):
                        tags['type'] = [{'type': ltype}]
                    if (line_ro := cs.get('readingOrder')) is not None and (line_ro_idx := line_ro[0].get('index')) is not None:
                        # look up region index from parent
                        reg_cus = self._parse_page_custom(line.getparent().get('custom'))
                        if 'readingOrder' not in reg_cus or 'index' not in reg_cus['readingOrder']:
                            logger.info('Incomplete `custom` attribute reading order found.')
                        else:
                            tmp_transkribus_line_order[int(reg_cus['readingOrder'][0]['index'])].append((int(line_ro_idx), line_id))
                    tags.update(cs)

                # get base text direction
                line_dir = {'left-to-right': 'L',
                            'right-to-left': 'R',
                            'top-to-bottom': 'L',
                            'bottom-to-top': 'R'}.get(line.get('readingDirection'), None)
                if region_default_direction and line_dir is None:
                    line_dir = region_default_direction
                elif page_default_direction and line_dir is None:
                    line_dir = page_default_direction

                line_langs = self._parse_page_langs(line, region_default_lang)
                line_split = None
                if (split := tags.get('split', None)) is not None and len(split) == 1:
                    line_split = split[0]['type']
                    tags.pop('split')

                if self.type == 'baselines':
                    line_obj = BaselineLine(id=line_id,
                                            baseline=baseline,
                                            boundary=boundary,
                                            text=text,
                                            tags=tags,
                                            language=line_langs,
                                            split=line_split,
                                            base_dir=line_dir,
                                            regions=[region_id])
                elif self.type == 'bbox':
                    flat_box = [point for pol in boundary for point in pol]
                    xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
                    ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
                    line_obj = BBoxLine(id=line_id,
                                        bbox=(xmin, ymin, xmax, ymax),
                                        text=text,
                                        tags=tags,
                                        language=line_langs,
                                        split=line_split,
                                        base_dir=line_dir,
                                        regions=[region_id])

                self._lines[line_id] = line_obj
                # register implicit reading order
                self._orders['line_implicit']['order'].append(line_id)

        # add transkribus-style region order
        self._orders['region_transkribus'] = {'order': [x[0] for x in sorted(tr_region_order, key=lambda k: k[1])],
                                              'is_total': True if len(set(map(lambda x: x[0], tr_region_order))) == len(tr_region_order) else False,
                                              'description': 'Explicit region order from `custom` attribute'}

        self._regions = region_data

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

            def _parse_group(el):

                _ro = []
                if el.tag.endswith('UnorderedGroup'):
                    _ro = [_parse_group(x) for x in el.iterchildren()]
                    is_total = False  # NOQA
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
    def lines(self):
        return self._lines

    @property
    def reading_orders(self):
        return self._orders

    def get_sorted_lines(self, ro='line_implicit'):
        """
        Returns ordered baselines from particular reading order.
        """
        if ro not in self.reading_orders:
            raise ValueError(f'Unknown reading order {ro}')

        def _traverse_ro(el):
            _ro = []
            if isinstance(el, list):
                _ro = [_traverse_ro(x) for x in el]
            else:
                # if line directly append to ro
                if el in self.lines:
                    return self.lines[el]
                # substitute lines if region in RO
                elif el in [reg['id'] for regs in self.regions.values() for reg in regs]:
                    _ro.extend(self.get_sorted_lines_by_region(el))
                else:
                    raise ValueError(f'Invalid reading order {ro}')
            return _ro

        _ro = self.reading_orders[ro]
        return _traverse_ro(_ro['order'])

    def get_sorted_regions(self, ro='region_implicit'):
        """
        Returns ordered regions from particular reading order.
        """
        if ro not in self.reading_orders:
            raise ValueError(f'Unknown reading order {ro}')

        regions = {reg.id: key for key, regs in self.regions.items() for reg in regs}

        def _traverse_ro(el):
            _ro = []
            if isinstance(el, list):
                _ro = [_traverse_ro(x) for x in el]
            else:
                # if region directly append to ro
                if el in regions.keys():
                    return [reg for reg in self.regions[regions[el]] if reg.id == el][0]
                else:
                    raise ValueError(f'Invalid reading order {ro}')
            return _ro

        _ro = self.reading_orders[ro]
        return _traverse_ro(_ro['order'])

    def get_sorted_lines_by_region(self, region, ro='line_implicit'):
        """
        Returns ordered lines in region.
        """
        if ro not in self.reading_orders:
            raise ValueError(f'Unknown reading order {ro}')
        if self.reading_orders[ro]['is_total'] is False:
            raise ValueError('Fetching lines by region of a non-total order is not supported')
        lines = [(id, line) for id, line in self._lines.items() if line.regions[0] == region]
        for line in lines:
            if line[0] not in self.reading_orders[ro]['order']:
                raise ValueError('Fetching lines by region is only possible for flat orders')
        return sorted(lines, key=lambda k: self.reading_orders[ro]['order'].index(k[0]))

    def get_lines_by_tag(self, key, value):
        return {k: v for k, v in self._lines.items() if v.tags.get(key) == value}

    def get_lines_by_split(self, split: Literal['train', 'validation', 'test']):
        return {k: v for k, v in self._lines.items() if v.tags.get('split') == split}

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
        try:
            points = [int(float(point.group())) for point in float_re.finditer(coords)]
        except (ValueError, TypeError):
            raise ValueError(f'Invalid points sequence string: {coords}')
        if len(points) % 2:
            raise ValueError(f'Odd number of points in points sequence: {points}')
        pts = zip(points[::2], points[1::2])
        return [k for k, g in groupby(pts)]

    @staticmethod
    def _parse_page_custom(s):
        o = defaultdict(list)
        s = s.strip()
        l_chunks = [l_chunk for l_chunk in s.split('}') if l_chunk.strip()]
        if l_chunks:
            for chunk in l_chunks:
                tag, vals = chunk.split('{')
                tag_vals = {}
                vals = [val.strip() for val in vals.split(';') if val.strip()]
                for val in vals:
                    key, *val = val.split(':')
                    tag_vals[key] = ":".join(val).strip()
                o[tag.strip()].append(tag_vals)
        return dict(o)

    def _parse_page_langs(self,
                          el,
                          default_lang: Optional[List[str]] = None):
        """
        Determines the language(s) of an element from custom string,
        attributes, and any inherited values.
        """
        el_langs = []
        if (custom_str := el.get('custom')) is not None:
            cs = self._parse_page_custom(custom_str)
            if (lang := cs.get('language')) is not None:
                for lang_val in lang:
                    if (lang_val_type := lang_val.get('type')) is not None:
                        try:
                            lang_val_type = Lang(lang_val_type).pt3
                        except InvalidLanguageValue:
                            pass
                        el_langs.append(lang_val_type)
        if (lang := el.get('primaryLanguage')) is not None:
            try:
                lang = Lang(lang).pt3
            except InvalidLanguageValue:
                pass
            el_langs.append(lang)
        if (lang := el.get('secondaryLanguage')) is not None:
            try:
                lang = Lang(lang).pt3
            except InvalidLanguageValue:
                pass
            el_langs.append(lang)
        if not len(el_langs):
            return default_lang
        return el_langs

    @staticmethod
    def _parse_page_coords(coords):
        points = [x for x in coords.split(' ')]
        points = [int(c) for point in points for c in point.split(',')]
        pts = zip(points[::2], points[1::2])
        return [k for k, g in groupby(pts)]

    def _parse_alto_tagrefs(self, tag_map, tagrefs, **kwargs):
        tags = {}
        if tagrefs is not None:
            for tagref in tagrefs.split():
                tref, tag_type, tag_label = tag_map.get(tagref, (None, None, None))
                if not tag_type and not tag_label:
                    continue
                elif not tag_type and tag_label:
                    tag_type = 'type'
                tag_label = [{'type': tag_label}]
                self._tag_set.add(tag_label[0]['type'])
                tag_val = tags.pop(tag_type, None)
                if isinstance(tag_val, list):
                    tag_val.extend(tag_label)
                elif tag_val is not None:
                    tag_val = [tag_val] + tag_label
                else:
                    tag_val = tag_label
                tags[tag_type] = tag_val
        # set default values
        for k, v in kwargs.items():
            if k not in tags:
                tags[k] = v
        return tags

    def _parse_alto_langs(self,
                          el,
                          tag_map: dict[str, str],
                          default_lang: Optional[list[str]] = None):
        el_langs = []
        tags = self._parse_alto_tagrefs(tag_map, el.get('TAGREFS'))
        if (tag_langs := tags.get('language')) is not None:
            if isinstance(tag_langs, list):
                el_langs.extend([tl['type'] for tl in tag_langs])
            else:
                el_langs.append(tag_langs['type'])
        if (el_lang := el.get('LANG')) is not None:
            el_langs.append(el_lang)

        if not len(el_langs):
            return default_lang

        return el_langs

    def __str__(self):
        return f'XMLPage {self.filename} (format: {self.filetype}, image: {self.imagename})'

    def __repr__(self):
        return f'XMLPage(filename={self.filename}, filetype={self.filetype})'

    def to_container(self) -> Segmentation:
        """
        Returns a Segmentation object.
        """
        return Segmentation(type=self.type,
                            imagename=self.imagename,
                            text_direction='horizontal_lr',
                            script_detection=True,
                            lines=self.get_sorted_lines(),
                            regions=self._regions,
                            line_orders=list(self.reading_orders.values()))
