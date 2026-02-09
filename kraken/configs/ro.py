#
# Copyright 2025 Benjamin Kiessling
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
kraken.configs.vgsl
~~~~~~~~~~~~~~~~~~~

Configurations for RO model training.
"""
from collections import defaultdict

from kraken.configs.base import (TrainingConfig,
                                 SegmentationTrainingDataConfig,
                                 _Counter)

__all__ = ['ROTrainingDataConfig',
           'ROTrainingConfig']


class ROTrainingDataConfig(SegmentationTrainingDataConfig):
    """
    Base configuration for training a reading order model.

    Arg:
        level (Literal['baselines', 'regions'], defaults to 'baselines'):
            Whether to train line or region RO.
        reading_order (str, defaults to line_implicit/region_implicit):
            Identifier of the reading order to train. If None is selected the
            implicit order is used.
    """
    def __init__(self, **kwargs):
        self.level = kwargs.pop('level', 'baselines')
        self.reading_order = kwargs.pop('reading_order', None)
        self.class_mapping = kwargs.pop('class_mapping', defaultdict(_Counter(1)))
        kwargs.setdefault('batch_size', 15000)
        super().__init__(**kwargs)


class ROTrainingConfig(TrainingConfig):
    """
    Base configuration for training a BLLA VGSL recognition model.

    Arg:
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('lrate', 0.001)
        kwargs.setdefault('min_epochs', 500)
        kwargs.setdefault('epochs', 3000)
        kwargs.setdefault('lag', 300)
        kwargs.setdefault('weight_decay', 0.01)
        kwargs.setdefault('schedule', 'cosine')
        kwargs.setdefault('cos_t_max', 100)
        kwargs.setdefault('cos_min_lr', 0.001)
        super().__init__(**kwargs)
