#
# Copyright 2015 Benjamin Kiessling
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
Utility functions for data loading and training of VGSL networks.
"""
from collections import defaultdict
from math import factorial
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from kraken.lib.xml import XMLPage
from kraken.lib.dataset.utils import _get_type
from kraken.lib.exceptions import KrakenInputException

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['PairWiseROSet', 'PageWiseROSet']

import logging

logger = logging.getLogger(__name__)


def _num_classes_from_mapping(class_mapping: dict[str, int]) -> int:
    """Returns the number of classes needed for a potentially sparse mapping."""
    if not class_mapping:
        return 1
    return max(0, *class_mapping.values()) + 1


def _extract_features(element, image_size, class_mapping, num_classes):
    """Extract spatial features from a BaselineLine or Region object.

    Args:
        element: A BaselineLine or Region object.
        image_size: Tuple (width, height) of the source image.
        class_mapping: Dict mapping type tags to class indices.
        num_classes: Total number of classes (including the default class).

    Returns:
        A tuple (tag, features_tensor).
    """
    w, h = image_size
    tag = _get_type(element.tags)
    cl = torch.zeros(num_classes, dtype=torch.float)
    cl[class_mapping.get(tag, 0)] = 1

    if hasattr(element, 'baseline') and element.baseline is not None:
        # BaselineLine: use baseline coords
        coords = np.array(element.baseline) / (w, h)
        center = np.mean(coords, axis=0)
        start = coords[0, :]
        end = coords[-1, :]
    else:
        # Region: use boundary bounding box
        boundary = np.array(element.boundary)
        center = np.mean(boundary, axis=0) / (w, h)
        start = np.array([boundary[:, 0].min(), boundary[:, 1].min()]) / (w, h)
        end = np.array([boundary[:, 0].max(), boundary[:, 1].max()]) / (w, h)

    return tag, torch.cat((cl,
                           torch.tensor(center, dtype=torch.float),
                           torch.tensor(start, dtype=torch.float),
                           torch.tensor(end, dtype=torch.float)))


class PairWiseROSet(Dataset):
    """
    Dataset for training a reading order determination model.

    Returns random pairs of lines from the same page.
    """
    def __init__(self,
                 files: Sequence[Union['PathLike', str]],
                 class_mapping: dict[str, int],
                 mode: Optional[Literal['alto', 'page', 'xml']] = 'xml',
                 level: Literal['regions', 'baselines'] = 'baselines',
                 ro_id: Optional[str] = None) -> None:
        """
        Samples pairs lines/regions from XML files for training a reading order
        model .

        Args:
            mode: Selects type of data source files.
            level: Computes reading order tuples on line or region level.
            ro_id: ID of the reading order to sample from. Defaults to
                   `line_implicit`/`region_implicit`.
            class_mapping: Explicit class mapping to use. No sanity checks are
                           performed. Takes precedence over valid_*, merge_*.
        """
        super().__init__()

        self._num_pairs = 0
        self._num_classes = None
        self.failed_samples = []
        self.class_mapping = class_mapping
        self.class_stats = defaultdict(int)

        self.data = []

        if mode in ['alto', 'page', 'xml']:
            docs = []
            for file in files:
                try:
                    doc = XMLPage(file, filetype=mode)
                    if level == 'baselines':
                        if not ro_id:
                            ro_id = 'line_implicit'
                        order = doc.get_sorted_lines(ro_id)
                    elif level == 'regions':
                        if not ro_id:
                            ro_id = 'region_implicit'
                        order = doc.get_sorted_regions(ro_id)
                    else:
                        raise ValueError(f'Invalid RO type {level}')
                    _order = []
                    for el in order:
                        tag = _get_type(el.tags)
                        try:
                            self.class_mapping[tag]
                            _order.append(el)
                            self.class_stats[tag] += 1
                        except KeyError:
                            continue
                    docs.append((doc.image_size, _order))
                except KrakenInputException as e:
                    logger.warning(e)
                    continue

            num_classes = _num_classes_from_mapping(self.class_mapping)
            for image_size, order in docs:
                # traverse RO and substitute features.
                sorted_lines = []
                for element in order:
                    tag, features = _extract_features(element, image_size, self.class_mapping, num_classes)
                    line_data = {'type': tag, 'features': features}
                    sorted_lines.append(line_data)
                if len(sorted_lines) > 1:
                    self.data.append(sorted_lines)
                    self._num_pairs += int(factorial(len(sorted_lines))/factorial(len(sorted_lines)-2))
                else:
                    logger.info(f'Page {doc} has less than 2 lines. Skipping')
        else:
            raise Exception('invalid dataset mode')

    @property
    def num_classes(self):
        return _num_classes_from_mapping(self.class_mapping)

    @property
    def canonical_class_mapping(self) -> dict[str, int]:
        """Returns a one-to-one class mapping (one string per label index).

        For merged classes (multiple strings -> same index), the first string
        by insertion order is kept as the canonical name.
        """
        seen_indices = set()
        canonical = {}
        for key, idx in self.class_mapping.items():
            if idx not in seen_indices:
                seen_indices.add(idx)
                canonical[key] = idx
        return canonical

    @property
    def merged_classes(self) -> dict[str, list[str]]:
        """Returns merged class info: {canonical_name: [aliases]}.

        Only includes entries where multiple strings map to the same index.
        """
        idx_to_names: dict[int, list[str]] = defaultdict(list)
        for key, idx in self.class_mapping.items():
            idx_to_names[idx].append(key)
        merged = {}
        for idx, names in idx_to_names.items():
            if len(names) > 1:
                merged[names[0]] = names[1:]
        return merged

    def __getitem__(self, idx):
        lines = []
        while len(lines) < 2:
            lines = self.data[torch.randint(len(self.data), (1,))[0]]
        idx0, idx1 = 0, 0
        while idx0 == idx1:
            idx0, idx1 = torch.randint(len(lines), (2,))
        x = torch.cat((lines[idx0]['features'], lines[idx1]['features']))
        y = torch.tensor(0 if idx0 >= idx1 else 1, dtype=torch.float)
        return {'sample': x, 'target': y}

    def get_feature_dim(self):
        return 2 * self.num_classes + 12

    def __len__(self):
        return self._num_pairs


class PageWiseROSet(Dataset):
    """
    Dataset for training a reading order determination model.

    Returns all lines from the same page.
    """
    def __init__(self,
                 files: Sequence[Union['PathLike', str]],
                 class_mapping: dict[str, int],
                 mode: Optional[Literal['alto', 'page', 'xml']] = 'xml',
                 level: Literal['regions', 'baselines'] = 'baselines',
                 ro_id: Optional[str] = None) -> None:
        """
        Samples pairs lines/regions from XML files for evaluating a reading order
        model.

        Args:
            mode: Selects type of data source files.
            level: Computes reading order tuples on line or region level.
            ro_id: ID of the reading order to sample from. Defaults to
                   `line_implicit`/`region_implicit`.
            class_mapping: Explicit class mapping to use. No sanity checks are performed.
        """
        super().__init__()

        self._num_classes = None
        self.failed_samples = []
        self.class_mapping = class_mapping
        self.class_stats = defaultdict(int)

        self.data = []

        if mode in ['alto', 'page', 'xml']:
            docs = []
            for file in files:
                try:
                    doc = XMLPage(file, filetype=mode)
                    if level == 'baselines':
                        if not ro_id:
                            ro_id = 'line_implicit'
                        order = doc.get_sorted_lines(ro_id)
                    elif level == 'regions':
                        if not ro_id:
                            ro_id = 'region_implicit'
                        order = doc.get_sorted_regions(ro_id)
                    else:
                        raise ValueError(f'Invalid RO type {level}')
                    _order = []
                    for el in order:
                        tag = _get_type(el.tags)
                        try:
                            self.class_mapping[tag]
                            _order.append(el)
                            self.class_stats[tag] += 1
                        except KeyError:
                            continue
                    docs.append((doc.image_size, _order))
                except KrakenInputException as e:
                    logger.warning(e)
                    continue

            num_classes = _num_classes_from_mapping(self.class_mapping)
            for image_size, order in docs:
                # traverse RO and substitute features.
                sorted_lines = []
                for element in order:
                    tag, features = _extract_features(element, image_size, self.class_mapping, num_classes)
                    line_data = {'type': tag, 'features': features}
                    sorted_lines.append(line_data)
                if len(sorted_lines) > 1:
                    self.data.append(sorted_lines)
                else:
                    logger.info(f'Page {doc} has less than 2 lines. Skipping')
        else:
            raise Exception('invalid dataset mode')

    @property
    def num_classes(self):
        return _num_classes_from_mapping(self.class_mapping)

    @property
    def canonical_class_mapping(self) -> dict[str, int]:
        """Returns a one-to-one class mapping (one string per label index).

        For merged classes (multiple strings -> same index), the first string
        by insertion order is kept as the canonical name.
        """
        seen_indices = set()
        canonical = {}
        for key, idx in self.class_mapping.items():
            if idx not in seen_indices:
                seen_indices.add(idx)
                canonical[key] = idx
        return canonical

    @property
    def merged_classes(self) -> dict[str, list[str]]:
        """Returns merged class info: {canonical_name: [aliases]}.

        Only includes entries where multiple strings map to the same index.
        """
        idx_to_names: dict[int, list[str]] = defaultdict(list)
        for key, idx in self.class_mapping.items():
            idx_to_names[idx].append(key)
        merged = {}
        for idx, names in idx_to_names.items():
            if len(names) > 1:
                merged[names[0]] = names[1:]
        return merged

    def __getitem__(self, idx):
        xs = []
        ys = []
        for i in range(len(self.data[idx])):
            for j in range(len(self.data[idx])):
                if i == j and len(self.data[idx]) != 1:
                    continue
                xs.append(torch.cat((self.data[idx][i]['features'],
                                     self.data[idx][j]['features'])))
                ys.append(torch.tensor(0 if i >= j else 1, dtype=torch.float))
        return {'sample': torch.stack(xs), 'target': torch.stack(ys), 'num_lines': len(self.data[idx])}

    def get_feature_dim(self):
        return 2 * self.num_classes + 12

    def __len__(self):
        return len(self.data)
