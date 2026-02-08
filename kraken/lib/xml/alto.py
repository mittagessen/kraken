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
ALTO XML parsing.
"""
import logging
from collections import defaultdict
from typing import Any, Optional

from lxml import etree

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

from kraken.containers import BBoxLine, BaselineLine, Region
from kraken.lib.xml.common import alto_regions, parse_alto_pointstype

logger = logging.getLogger(__name__)


def parse_alto_tagrefs(tag_map, tagrefs, tag_set, **kwargs):
    """
    Parse ALTO tag references.

    Args:
        tag_map: Mapping from tag IDs to (type, label) tuples.
        tagrefs: Space-separated tag reference string.
        tag_set: Mutable set to track seen tags.
    """
    tags = {}
    if tagrefs is not None:
        for tagref in tagrefs.split():
            tref, tag_type, tag_label = tag_map.get(tagref, (None, None, None))
            if not tag_type and not tag_label:
                continue
            elif not tag_type and tag_label:
                tag_type = 'type'
            tag_label = [{'type': tag_label}]
            tag_set.add(tag_label[0]['type'])
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


def parse_alto_langs(el, tag_map, tag_set, default_lang=None):
    """
    Determine languages of an ALTO element.
    """
    el_langs = []
    tags = parse_alto_tagrefs(tag_map, el.get('TAGREFS'), tag_set)
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


def parse_alto(doc, filename, linetype):
    """
    Parse an ALTO XML document.

    Args:
        doc: Parsed lxml document.
        filename: Path to the XML file (for error messages and resolving image paths).
        linetype: 'baselines' or 'bbox'.

    Returns:
        dict with keys: imagename, image_size, regions, lines, orders, tag_set,
                        raw_orders, string_to_line_map
    """
    base_directory = filename.parent

    if (mu := doc.find('.//{*}MeasurementUnit')) is not None and mu.text.strip() != 'pixel':
        raise ValueError(f'Measurement unit in ALTO file {filename} is "{mu.text.strip()}" not "pixel".')

    if (image := doc.find('.//{*}fileName')) is None or not image.text:
        raise ValueError(f'No valid image filename found in ALTO file {filename}')

    imagename = base_directory.joinpath(image.text)

    if (page := doc.find('.//{*}Page')) is None:
        raise ValueError(f'No "Page" element found in ALTO file {filename}')

    try:
        image_size = int(page.get('HEIGHT')), int(page.get('WIDTH'))
    except (ValueError, TypeError) as e:
        raise ValueError(f'Invalid image dimensions in {filename}: {e}')

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

    tag_set = set(('default',))

    lines = {}
    region_data = defaultdict(list)
    line_implicit_order = []
    region_implicit_order = []
    missing_region_ids: set[str] = set()
    # Map string IDs to their parent TextLine IDs
    string_to_line_map = {}

    # parse region type and coords
    for region in regions:
        region_id = region.get('ID')
        region_default_direction = {'ltr': 'L',
                                    'rtl': 'R',
                                    'ttb': 'L',
                                    'btt': 'R'}.get(region.get('BASEDIRECTION'), None)

        # try to find shape object
        boundary = None
        if (coords := region.find('./{*}Shape/{*}Polygon')) is not None:
            boundary = parse_alto_pointstype(coords.get('POINTS'))
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
        region_has_coords = boundary is not None
        # parse region tags
        reg_tags = parse_alto_tagrefs(cls_map, region.get('TAGREFS'), tag_set)
        tag_type = reg_tags.pop('region') if 'region' in reg_tags else reg_tags.pop('type', None)
        if (rtype := region.get('TYPE')) is None:
            rtype = tag_type
        else:
            rtype = [{'type': rtype}]
        if rtype is None:
            rtype = [{'type': alto_regions[region.tag.split('}')[-1]]}]
        reg_tags['type'] = rtype

        region_default_lang = parse_alto_langs(region,
                                               cls_map,
                                               tag_set,
                                               [page_default_lang] if page_default_lang is not None else None)

        if region_has_coords:
            region_data[rtype[0]['type']].append(Region(id=region_id,
                                                        boundary=boundary,
                                                        tags=reg_tags,
                                                        language=region_default_lang))
            # register implicit reading order
            region_implicit_order.append(region_id)
        else:
            missing_region_ids.add(region_id)

        # parse lines in region
        for line in region.iterfind('./{*}TextLine'):
            line_id = line.get('ID')
            if linetype == 'baselines':
                try:
                    baseline = parse_alto_pointstype(line.get('BASELINE'))
                except ValueError:
                    logger.info(f'TextLine {line_id} without baseline')
                    continue

                boundary = None
                try:
                    pol = line.find('./{*}Shape/{*}Polygon')
                    boundary = parse_alto_pointstype(pol.get('POINTS'))
                except (ValueError, AttributeError):
                    logger.info(f'TextLine {line_id} without polygon')

            elif linetype == 'bbox':
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

            # Build string_to_line_map
            for string_el in line.iterfind('./{*}String'):
                string_id = string_el.get('ID')
                if string_id:
                    string_to_line_map[string_id] = line_id

            # parse tags
            line_tags = parse_alto_tagrefs(cls_map, line.get('TAGREFS'), tag_set)
            line_langs = parse_alto_langs(line, cls_map, tag_set, region_default_lang)
            line_split = None
            if (split := line_tags.get('split', None)) is not None and len(split) == 1:
                line_split = split[0]['type']
                line_tags.pop('split')

            # get base text direction
            line_dir = {'ltr': 'L',
                        'rtl': 'R',
                        'ttb': 'L',
                        'btt': 'R'}.get(line.get('BASEDIRECTION'), None)
            if region_default_direction and line_dir is None:
                line_dir = region_default_direction

            if linetype == 'baselines':
                line_obj = BaselineLine(id=line_id,
                                        baseline=baseline,
                                        boundary=boundary,
                                        text=text,
                                        tags=line_tags if line_tags else None,
                                        language=line_langs,
                                        split=line_split,
                                        base_dir=line_dir,
                                        regions=[region_id] if region_has_coords else [])
            elif linetype == 'bbox':
                line_obj = BBoxLine(id=line_id,
                                    bbox=bbox,
                                    text=text,
                                    tags=line_tags if line_tags else None,
                                    language=line_langs,
                                    split=line_split,
                                    base_dir=line_dir,
                                    regions=[region_id] if region_has_coords else [])

            lines[line_id] = line_obj
            # register implicit reading order
            line_implicit_order.append(line_id)

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
                ref = el.get('REF')
                return ref
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

            ro_tags = parse_alto_tagrefs(cls_map, ro.get('TAGREFS'), tag_set)
            raw_orders[ro.get('ID')] = {'order': raw_order,
                                        'is_total': is_total,
                                        'description': ro_tags.get('type', '')}

    return {
        'imagename': imagename,
        'image_size': image_size,
        'regions': dict(region_data),
        'lines': lines,
        'line_implicit_order': line_implicit_order,
        'region_implicit_order': region_implicit_order,
        'tag_set': tag_set,
        'raw_orders': raw_orders,
        'string_to_line_map': string_to_line_map,
        'missing_region_ids': missing_region_ids,
    }
