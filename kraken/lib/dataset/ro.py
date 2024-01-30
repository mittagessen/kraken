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
from math import factorial
from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from kraken.lib.exceptions import KrakenInputException
from kraken.lib.xml import XMLPage

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['PairWiseROSet', 'PageWiseROSet']

import logging

logger = logging.getLogger(__name__)


class PairWiseROSet(Dataset):
    """
    Dataset for training a reading order determination model.

    Returns random pairs of lines from the same page.
    """
    def __init__(self, files: Sequence[Union['PathLike', str]] = None,
                 mode: Optional[Literal['alto', 'page', 'xml']] = 'xml',
                 level: Literal['regions', 'baselines'] = 'baselines',
                 ro_id: Optional[str] = None,
                 class_mapping: Optional[Dict[str, int]] = None):
        """
        Samples pairs lines/regions from XML files for training a reading order
        model .

        Args:
            mode: Either alto, page, xml, None. In alto, page, and xml
                  mode the baseline paths and image data is retrieved from an
                  ALTO/PageXML file. In `None` mode data is iteratively added
                  through the `add` method.
            ro_id: ID of the reading order to sample from. Defaults to
                   `line_implicit`/`region_implicit`.
        """
        super().__init__()

        self._num_pairs = 0
        self.failed_samples = []
        if class_mapping:
            self.class_mapping = class_mapping
            self.num_classes = len(class_mapping) + 1
        else:
            self.num_classes = 1
            self.class_mapping = {}

        self.data = []

        if mode in ['alto', 'page', 'xml']:
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
                    for el in order:
                        for tag in el.tags.values():
                            if tag not in self.class_mapping:
                                self.class_mapping[tag] = self.num_classes
                                self.num_classes += 1
                    # traverse RO and substitute features.
                    w, h = doc.image_size
                    sorted_lines = []
                    for line in order:
                        line_coords = np.array(line.baseline) / (w, h)
                        line_center = np.mean(line_coords, axis=0)
                        cl = torch.zeros(self.num_classes, dtype=torch.float)
                        # if class is not in class mapping default to None class (idx 0)
                        cl[self.class_mapping.get(line.tags['type'], 0)] = 1
                        line_data = {'type': line.tags['type'],
                                     'features': torch.cat((cl,  # one hot encoded line type
                                                            torch.tensor(line_center, dtype=torch.float),  # line center
                                                            torch.tensor(line_coords[0, :], dtype=torch.float),  # start_point coord
                                                            torch.tensor(line_coords[-1, :], dtype=torch.float),  # end point coord)
                                                            )
                                                           )
                                     }
                        sorted_lines.append(line_data)
                    if len(sorted_lines) > 1:
                        self.data.append(sorted_lines)
                        self._num_pairs += int(factorial(len(sorted_lines))/factorial(len(sorted_lines)-2))
                    else:
                        logger.info(f'Page {doc} has less than 2 lines. Skipping')
                except KrakenInputException as e:
                    logger.warning(e)
                    continue
        else:
            raise Exception('invalid dataset mode')

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
    def __init__(self, files: Sequence[Union['PathLike', str]] = None,
                 mode: Optional[Literal['alto', 'page', 'xml']] = 'xml',
                 level: Literal['regions', 'baselines'] = 'baselines',
                 ro_id: Optional[str] = None,
                 class_mapping: Optional[Dict[str, int]] = None):
        """
        Samples pairs lines/regions from XML files for training a reading order
        model .

        Args:
            mode: Either alto, page, xml, None. In alto, page, and xml
                  mode the baseline paths and image data is retrieved from an
                  ALTO/PageXML file. In `None` mode data is iteratively added
                  through the `add` method.
            ro_id: ID of the reading order to sample from. Defaults to
                   `line_implicit`/`region_implicit`.
        """
        super().__init__()

        self.failed_samples = []
        if class_mapping:
            self.class_mapping = class_mapping
            self.num_classes = len(class_mapping) + 1
        else:
            self.num_classes = 1
            self.class_mapping = {}

        self.data = []

        if mode in ['alto', 'page', 'xml']:
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
                    for el in order:
                        for tag in el.tags.values():
                            if tag not in self.class_mapping:
                                self.class_mapping[tag] = self.num_classes
                                self.num_classes += 1
                    # traverse RO and substitute features.
                    w, h = doc.image_size
                    sorted_lines = []
                    for line in order:
                        line_coords = np.array(line.baseline) / (w, h)
                        line_center = np.mean(line_coords, axis=0)
                        cl = torch.zeros(self.num_classes, dtype=torch.float)
                        # if class is not in class mapping default to None class (idx 0)
                        cl[self.class_mapping.get(line.tags['type'], 0)] = 1
                        line_data = {'type': line.tags['type'],
                                     'features': torch.cat((cl,  # one hot encoded line type
                                                            torch.tensor(line_center, dtype=torch.float),  # line center
                                                            torch.tensor(line_coords[0, :], dtype=torch.float),  # start_point coord
                                                            torch.tensor(line_coords[-1, :], dtype=torch.float),  # end point coord)
                                                            )
                                                           )
                                     }
                        sorted_lines.append(line_data)
                    if len(sorted_lines) > 1:
                        self.data.append(sorted_lines)
                    else:
                        logger.info(f'Page {doc} has less than 2 lines. Skipping')
                except KrakenInputException as e:
                    logger.warning(e)
                    continue
        else:
            raise Exception('invalid dataset mode')

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
