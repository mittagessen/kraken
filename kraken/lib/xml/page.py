#
# Copyright 2026 Benjamin Kiessling
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
PageXML parsing.
"""
import logging
from collections import defaultdict
from typing import Any, Optional

from lxml import etree

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

from kraken.containers import BBoxLine, BaselineLine, Region
from kraken.lib.xml.common import page_regions, parse_page_coords, parse_page_custom

logger = logging.getLogger(__name__)


def parse_page_langs(el, default_lang=None):
    """
    Determines the language(s) of an element from custom string,
    attributes, and any inherited values.
    """
    el_langs = []
    if (custom_str := el.get('custom')) is not None:
        cs = parse_page_custom(custom_str)
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


def parse_page(doc, filename, linetype):
    """
    Parse a PageXML document.

    Args:
        doc: Parsed lxml document.
        filename: Path to the XML file (for error messages and resolving image paths).
        linetype: 'baselines' or 'bbox'.

    Returns:
        Dict with keys: imagename, image_size, regions, lines, orders, tag_set,
                        raw_orders
    """
    base_directory = filename.parent

    if (image := doc.find('.//{*}Page')) is None or image.get('imageFilename') is None:
        raise ValueError(f'No valid image filename found in PageXML file {filename}')
    page_default_direction = {'left-to-right': 'L',
                              'right-to-left': 'R',
                              'top-to-bottom': 'L',
                              'bottom-to-top': 'R'}.get(image.get('readingDirection'), None)

    page_default_lang = parse_page_langs(image)

    imagename = base_directory.joinpath(image.get('imageFilename'))
    image_size = int(image.get('imageWidth')), int(image.get('imageHeight'))

    if not image_size[0] or not image_size[1]:
        logger.warning(f'Invalid image dimensions {image_size} in {filename}. Attempting to read from image file.')
        try:
            from PIL import Image
            with Image.open(imagename) as im:
                image_size = im.size
        except Exception as e:
            raise ValueError(f'Invalid image dimensions {image_size} in {filename} and unable to read image file {imagename}: {e}')

    # parse region type and coords
    region_data = defaultdict(list)
    tr_region_order = []
    missing_region_ids: set[str] = set()

    tag_set = set(('default',))
    tmp_transkribus_line_order = defaultdict(list)

    lines = {}
    line_implicit_order = []
    region_implicit_order = []

    for region in image.iterfind('./{*}*'):
        if not any([True if region.tag.endswith(k) else False for k in page_regions.keys()]):
            continue
        region_id = region.get('id')
        coords = region.find('./{*}Coords')
        try:
            coords = parse_page_coords(coords.get('points'))
        except Exception:
            logger.info(f'Region {region_id} without coordinates')
            coords = None
        region_has_coords = coords is not None
        tags = {}
        rtype = region.get('type')
        # parse transkribus-style custom field if possible
        region_default_lang = parse_page_langs(region, page_default_lang)
        if (custom_str := region.get('custom')) is not None:
            cs = parse_page_custom(custom_str)
            if not rtype and 'structure' in cs and 'type' in cs['structure'][0]:
                rtype = cs['structure'][0]['type']
            # transkribus-style reading order
            if (reg_ro := cs.get('readingOrder')) is not None and (reg_ro_idx := reg_ro[0].get('index')) is not None:
                if region_has_coords:
                    tr_region_order.append((region_id, int(reg_ro_idx)))
                else:
                    logger.warning(f'Region {region_id} in custom reading order lacks coordinates; skipping.')
            tags.update(cs)

        if region_default_lang is None:
            region_default_lang = page_default_lang

        # fall back to default region type if nothing is given
        if not rtype:
            rtype = page_regions[region.tag.split('}')[-1]]

        tags['type'] = [{'type': rtype}]
        if region_has_coords:
            region_data[rtype].append(Region(id=region_id, boundary=coords, tags=tags, language=region_default_lang))
        else:
            missing_region_ids.add(region_id)

        region_default_direction = {'left-to-right': 'L',
                                    'right-to-left': 'R',
                                    'top-to-bottom': 'L',
                                    'bottom-to-top': 'R'}.get(region.get('readingDirection'))

        # register implicit reading order
        if region_has_coords:
            region_implicit_order.append(region_id)

        # parse line information
        for line in region.iterfind('./{*}TextLine'):
            line_id = line.get('id')
            base = line.find('./{*}Baseline')
            baseline = None
            try:
                baseline = parse_page_coords(base.get('points'))
            except Exception:
                logger.info(f'TextLine {line_id} without baseline')
                if linetype == 'baselines':
                    continue

            pol = line.find('./{*}Coords')
            boundary = None
            try:
                boundary = parse_page_coords(pol.get('points'))
            except Exception:
                logger.info(f'TextLine {line_id} without polygon')
                if linetype == 'bbox':
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
            line_tags = {}
            if (custom_str := line.get('custom')) is not None:
                cs = parse_page_custom(custom_str)
                if (structure := cs.get('structure')) is not None and (ltype := structure[0].get('type')):
                    line_tags['type'] = [{'type': ltype}]
                if (line_ro := cs.get('readingOrder')) is not None and (line_ro_idx := line_ro[0].get('index')) is not None:
                    # look up region index from parent
                    reg_cus = {}
                    if (parent_str := line.getparent().get('custom')) is not None:
                        reg_cus = parse_page_custom(parent_str)
                        if 'readingOrder' not in reg_cus or 'index' not in reg_cus['readingOrder']:
                            logger.info('Incomplete `custom` attribute reading order found.')
                        elif not region_has_coords:
                            logger.warning(f'Region {region_id} in custom reading order lacks coordinates; skipping.')
                        else:
                            tmp_transkribus_line_order[int(reg_cus['readingOrder'][0]['index'])].append((int(line_ro_idx), line_id))
                line_tags.update(cs)

            # get base text direction
            line_dir = {'left-to-right': 'L',
                        'right-to-left': 'R',
                        'top-to-bottom': 'L',
                        'bottom-to-top': 'R'}.get(line.get('readingDirection'), None)
            if region_default_direction and line_dir is None:
                line_dir = region_default_direction
            elif page_default_direction and line_dir is None:
                line_dir = page_default_direction

            line_langs = parse_page_langs(line, region_default_lang)
            line_split = None
            if (split := line_tags.get('split', None)) is not None and len(split) == 1:
                line_split = split[0]['type']
                line_tags.pop('split')

            if linetype == 'baselines':
                line_obj = BaselineLine(id=line_id,
                                        baseline=baseline,
                                        boundary=boundary,
                                        text=text,
                                        tags=line_tags,
                                        language=line_langs,
                                        split=line_split,
                                        base_dir=line_dir,
                                        regions=[region_id] if region_has_coords else [])
            elif linetype == 'bbox':
                flat_box = [point for pol in boundary for point in pol]
                xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
                ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
                line_obj = BBoxLine(id=line_id,
                                    bbox=(xmin, ymin, xmax, ymax),
                                    text=text,
                                    tags=line_tags,
                                    language=line_langs,
                                    split=line_split,
                                    base_dir=line_dir,
                                    regions=[region_id] if region_has_coords else [])

            lines[line_id] = line_obj
            # register implicit reading order
            line_implicit_order.append(line_id)

    # add transkribus-style region order
    tr_region_order_sorted = [x[0] for x in sorted(tr_region_order, key=lambda k: k[1])]
    transkribus_orders = {}
    transkribus_orders['region_transkribus'] = {'order': tr_region_order_sorted,
                                                'is_total': len(set(map(lambda x: x[0], tr_region_order))) == len(tr_region_order),
                                                'description': 'Explicit region order from `custom` attribute',
                                                'level': 'region'}

    if tmp_transkribus_line_order:
        # sort by regions
        tmp_reg_order = sorted(((k, v) for k, v in tmp_transkribus_line_order.items()), key=lambda k: k[0])
        # flatten
        tr_line_order = []
        for _, tlines in tmp_reg_order:
            tr_line_order.extend([x[1] for x in sorted(tlines, key=lambda k: k[0])])
        transkribus_orders['line_transkribus'] = {'order': tr_line_order,
                                                  'is_total': True,
                                                  'description': 'Explicit line order from `custom` attribute',
                                                  'level': 'line'}

    # parse explicit reading orders if they exist
    raw_orders = {}
    ro_el = doc.find('.//{*}ReadingOrder')
    if ro_el is not None:
        reading_orders = ro_el.getchildren()
        # UnorderedGroup at top-level => treated as multiple reading orders
        if len(reading_orders) == 1 and reading_orders[0].tag.endswith('UnorderedGroup'):
            reading_orders = reading_orders[0].getchildren()

        def _parse_group(el):
            _ro = []
            if el.tag.endswith('UnorderedGroup'):
                # Nested UnorderedGroup: flatten in document order with warning
                logger.warning('Nested UnorderedGroup found in reading order, flattening in document order.')
                for child in el.iterchildren():
                    child_result = _parse_group(child)
                    if isinstance(child_result, list):
                        _ro.extend(child_result)
                    else:
                        _ro.append(child_result)
            elif el.tag.endswith('OrderedGroup'):
                for child in el.iterchildren():
                    child_result = _parse_group(child)
                    if isinstance(child_result, list):
                        _ro.extend(child_result)
                    else:
                        _ro.append(child_result)
            else:
                return el.get('regionRef')
            return _ro

        for ro in reading_orders:
            is_total = True
            raw_order = _parse_group(ro)
            if isinstance(raw_order, str):
                raw_order = [raw_order]
            # Check if this is a child of an UnorderedGroup => partial order
            parent = ro.getparent()
            if parent is not None and parent.tag.endswith('UnorderedGroup'):
                is_total = False

            raw_orders[ro.get('id')] = {'order': raw_order,
                                        'is_total': is_total,
                                        'description': ro.get('caption') if ro.get('caption') else ''}

    return {
        'imagename': imagename,
        'image_size': image_size,
        'regions': dict(region_data),
        'lines': lines,
        'line_implicit_order': line_implicit_order,
        'region_implicit_order': region_implicit_order,
        'tag_set': tag_set,
        'raw_orders': raw_orders,
        'transkribus_orders': transkribus_orders,
        'missing_region_ids': missing_region_ids,
    }
