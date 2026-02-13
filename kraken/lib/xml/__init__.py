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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from lxml import etree

from kraken.containers import Segmentation
from kraken.lib.xml.common import (alto_regions,
                                   flatten_order_to_lines,
                                   flatten_order_to_regions,
                                   page_regions,
                                   validate_and_clean_order)
from kraken.lib.xml.alto import parse_alto
from kraken.lib.xml.page import parse_page

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['XMLPage', 'page_regions', 'alto_regions']


class XMLPage(object):
    """
    Parses XML facsimiles in ALTO or PageXML format.

    The parser is able to deal with most (but not all) features supported by
    those standards. In particular, any data below the line level is discarded.

    Args:
        filename: Path to the XML file
        filetype: Selector for explicit subparser choice.
        linetype: Parse line data as baselines or bounding box type.

    Attributes:
        type: Either 'baselines' or 'bbox'.
        imagename: Path to the image to the XML file.
        image_size: Size of the image as a (width, height) tuple
        has_tags: Indicates if the source document contains tag information
        has_splits: Indicates if the source document contains explicit training splits
    """
    type: Literal['baselines', 'bbox'] = 'baselines'
    base_dir: Optional[Literal['L', 'R']] = None
    imagename: 'PathLike' = None
    image_size: tuple[int, int] = None
    _orders: dict[str, dict[str, Any]] = None
    has_tags: bool = False
    _tag_set: Optional[dict] = None
    has_splits: bool = False
    _split_set: Optional[list] = None

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
        self._orders = {'line_implicit': {'order': [], 'is_total': True,
                                          'description': 'Implicit line order derived from element sequence',
                                          'level': 'line'},
                        'region_implicit': {'order': [], 'is_total': True,
                                            'description': 'Implicit region order derived from element sequence',
                                            'level': 'region'}}

        if filetype == 'xml':
            self._parse_xml()
        elif filetype == 'alto':
            self._do_parse_alto()
        elif filetype == 'page':
            self._do_parse_page()

    def _parse_xml(self):
        with open(self.filename, 'rb') as fp:
            try:
                doc = etree.parse(fp)
            except etree.XMLSyntaxError as e:
                raise ValueError(f'Parsing {self.filename} failed: {e}')
        if doc.getroot().tag.endswith('alto'):
            return self._do_parse_alto(doc)
        elif doc.getroot().tag.endswith('PcGts'):
            return self._do_parse_page(doc)
        else:
            raise ValueError(f'Unknown XML format in {self.filename}')

    def _do_parse_alto(self, doc=None):
        if doc is None:
            with open(self.filename, 'rb') as fp:
                try:
                    doc = etree.parse(fp)
                except etree.XMLSyntaxError as e:
                    raise ValueError(f'Parsing {self.filename} failed: {e}')

        result = parse_alto(doc, self.filename, self.type)
        self._apply_result(result, 'alto')

    def _do_parse_page(self, doc=None):
        if doc is None:
            with open(self.filename, 'rb') as fp:
                try:
                    doc = etree.parse(fp)
                except etree.XMLSyntaxError as e:
                    raise ValueError(f'Parsing {self.filename} failed: {e}')

        result = parse_page(doc, self.filename, self.type)
        self._apply_result(result, 'page')

    def _apply_result(self, result, filetype):
        """Apply parsed result to this XMLPage instance and flatten reading orders."""
        self.imagename = result['imagename']
        self.image_size = result['image_size']
        self._regions = result['regions']
        self._lines = result['lines']
        self._tag_set = result['tag_set']

        # Set implicit orders
        self._orders['line_implicit']['order'] = result['line_implicit_order']
        self._orders['region_implicit']['order'] = result['region_implicit_order']

        # Build set of all region IDs
        region_ids = set()
        for regs in self._regions.values():
            for reg in regs:
                region_ids.add(reg.id)
        missing_region_ids = set(result.get('missing_region_ids', set()))

        # Get string_to_line_map for ALTO
        string_to_line_map = result.get('string_to_line_map')

        # Add transkribus orders (PageXML only)
        if 'transkribus_orders' in result:
            self._orders.update(result['transkribus_orders'])

        # Flatten raw explicit reading orders
        raw_orders = result.get('raw_orders', {})
        for ro_id, ro_data in raw_orders.items():
            raw_order = ro_data['order']
            is_total = ro_data['is_total']
            description = ro_data['description']

            # Flatten to line-level
            flat_lines = flatten_order_to_lines(raw_order,
                                                self._lines,
                                                region_ids,
                                                result['line_implicit_order'],
                                                string_to_line_map,
                                                missing_region_ids)
            flat_lines, _ = validate_and_clean_order(flat_lines, set(self._lines.keys()))
            self._orders[ro_id] = {'order': flat_lines,
                                   'is_total': is_total,
                                   'description': description,
                                   'level': 'line'}

            # Flatten to region-level
            flat_regions = flatten_order_to_regions(raw_order,
                                                    self._lines,
                                                    region_ids,
                                                    string_to_line_map,
                                                    missing_region_ids)
            flat_regions, _ = validate_and_clean_order(flat_regions, region_ids)
            self._orders[f'{ro_id}:regions'] = {'order': flat_regions,
                                                'is_total': is_total,
                                                'description': description,
                                                'level': 'region'}

        if len(self._tag_set) > 1:
            self.has_tags = True
        else:
            self.has_tags = False

        self.filetype = filetype

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

        ro_data = self.reading_orders[ro]
        return [self._lines[lid] for lid in ro_data['order'] if lid in self._lines]

    def get_sorted_regions(self, ro='region_implicit'):
        """
        Returns ordered regions from particular reading order.
        """
        if ro not in self.reading_orders:
            raise ValueError(f'Unknown reading order {ro}')

        region_map = {reg.id: reg for regs in self.regions.values() for reg in regs}

        ro_data = self.reading_orders[ro]
        return [region_map[rid] for rid in ro_data['order'] if rid in region_map]

    def get_sorted_lines_by_region(self, region, ro='line_implicit'):
        """
        Returns ordered lines in region.
        """
        if ro not in self.reading_orders:
            raise ValueError(f'Unknown reading order {ro}')
        if self.reading_orders[ro]['is_total'] is False:
            raise ValueError('Fetching lines by region of a non-total order is not supported')
        region_lines = [line for line in self._lines.values() if line.regions and line.regions[0] == region]
        ro_order = self.reading_orders[ro]['order']
        for line in region_lines:
            if line.id not in ro_order:
                raise ValueError('Fetching lines by region is only possible for flat orders')
        return sorted(region_lines, key=lambda l: ro_order.index(l.id))

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

    def __str__(self):
        return f'XMLPage {self.filename} (format: {self.filetype}, image: {self.imagename})'

    def __repr__(self):
        return f'XMLPage(filename={self.filename}, filetype={self.filetype})'

    def to_container(self) -> Segmentation:
        """
        Returns a Segmentation object.
        """
        sorted_lines = self.get_sorted_lines()
        line_id_to_idx = {line.id: idx for idx, line in enumerate(sorted_lines)}
        line_orders = []
        for ro_name, ro_data in self._orders.items():
            if ro_data['level'] != 'line':
                continue
            indices = [line_id_to_idx[lid] for lid in ro_data['order'] if lid in line_id_to_idx]
            if indices:
                line_orders.append(indices)
        return Segmentation(type=self.type,
                            imagename=self.imagename,
                            text_direction='horizontal-lr',
                            script_detection=True,
                            lines=sorted_lines,
                            regions=self._regions,
                            line_orders=line_orders)
