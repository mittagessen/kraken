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
kraken.configs.ppocr
~~~~~~~~~~~~~~~~~~~~

Configuration objects for PP-OCRv6 recognition training. These subclass
:class:`~kraken.configs.base.TrainingConfig` directly so that `ketos train`
sees only PP-OCRv6 options, not VGSL-only fields.
"""
from kraken.configs.base import TrainingConfig
from kraken.configs.vgsl import VGSLRecognitionTrainingDataConfig

__all__ = ['PPOCRv6RecognitionTrainingConfig',
           'PPOCRv6RecognitionTrainingDataConfig']


_VARIANT_LRATE = {'tiny': 1e-3, 'small': 5e-4, 'medium': 5e-4}


class PPOCRv6RecognitionTrainingConfig(TrainingConfig):
    """
    Training configuration for a PP-OCRv6 recognition model.

    Args:
        variant (str, defaults to 'small'):
            Model size, one of ``tiny``, ``small``, or ``medium``.
        height (int, defaults to 96):
            Input line height.
        compile (bool, defaults to True):
            Compile the training forward pass with static shapes; every batch
            is padded to the data config's ``max_width``.
        resize (Literal['fail', 'new', 'union'], defaults to 'fail'):
            Codec/output layer resizing option when fine-tuning a loaded model.
    """
    def __init__(self, **kwargs):
        self.variant = kwargs.pop('variant', 'small')
        self.height = kwargs.pop('height', 96)
        # pop before Config.__init__ maps `compile` to its `compile_config` field
        self.compile = kwargs.pop('compile', True)
        self.resize = kwargs.pop('resize', 'fail')

        kwargs.setdefault('optimizer', 'AdamW+Muon')
        kwargs.setdefault('lrate', _VARIANT_LRATE.get(self.variant, 5e-4))
        kwargs.setdefault('weight_decay', 0.01)
        kwargs.setdefault('momentum', 0.95)  # Muon momentum
        kwargs.setdefault('schedule', 'cosine')
        kwargs.setdefault('warmup', 1000)
        kwargs.setdefault('cos_min_lr', 1e-6)
        kwargs.setdefault('quit', 'fixed')
        kwargs.setdefault('epochs', 100)
        # training batch size (Config defaults to 1 for inference)
        kwargs.setdefault('batch_size', 64)
        super().__init__(**kwargs)


class PPOCRv6RecognitionTrainingDataConfig(VGSLRecognitionTrainingDataConfig):
    """
    Training data configuration for PP-OCRv6 recognition models.

    Args:
        max_width (int, defaults to 4096):
            Maximum line width in pixels after height normalization. Longer
            lines are dropped.
    """
    def __init__(self, **kwargs):
        self.max_width = kwargs.pop('max_width', 4096)
        super().__init__(**kwargs)
