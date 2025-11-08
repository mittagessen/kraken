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
kraken.configs.pretrain
~~~~~~~~~~~~~~~~~~~~~~~

Configurations for semi-supervised model pretraining.
"""
from kraken.configs.vgsl import VGSLRecognitionTrainingConfig

__all__ = ['VGSLPreTrainingConfig', 'VGSLPreTrainingDataConfig']


class VGSLPreTrainingDataConfig(VGSLRecognitionTrainingConfig):
    """
    Base configuration for pretraining a reading order model.

    Arg:
        mask_width (int, defaults to 4):
        mask_prob (float, defaults to 0.5):
        num_negatives (int, defaults to 100):
        logit_temp (float, defaults to 0.1):
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('batch_size', 64)
        super().__init__(**kwargs)


class VGSLPreTrainingConfig(VGSLRecognitionTrainingConfig):
    """
    Base configuration for pretraining a reading order model.

    Arg:
        mask_width (int, defaults to 4):
        mask_prob (float, defaults to 0.5):
        num_negatives (int, defaults to 100):
        logit_temp (float, defaults to 0.1):
    """
    def __init__(self, **kwargs):
        self.mask_width = kwargs.pop('mask_width', 4)
        self.mask_prob = kwargs.pop('mask_prob', 0.5)
        self.num_negatives = kwargs.pop('num_negatives', 100)
        self.logit_temp = kwargs.pop('logit_temp', 0.1)
        kwargs.setdefault('min_epochs', 100)
        kwargs.setdefault('lrate', 1e-6)
        kwargs.setdefault('weight_decay', 0.01)
        kwargs.setdefault('schedule', 'cosine')
        kwargs.setdefault('cos_t_max', 100)
        kwargs.setdefault('cos_min_lr', 1e-7)
        kwargs.setdefault('warmup', 32000)
        super().__init__(**kwargs)
