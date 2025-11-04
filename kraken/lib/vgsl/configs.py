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
from kraken.models import (TrainingConfig,
                           RecognitionTrainingDataConfig,
                           SegmentationTrainingDataConfig)


class VGSLRecognitionTrainingDataConfig(RecognitionTrainingDataConfig):
    """
    Training data configuration for training VGSL recognition models.

    Arg:
        normalization (str, defaults to None):
            Unicode normalization
        normalize_whitespace (bool, defaults to True):
            Flag to normalize all whitespace in training data to U+0020.
        bidi_reordering (bool, defaults to True):
            Reorder code points according to the Unicode bidirectional
            algorithm. Set to L|R to override default text direction.
        legacy_polygons (bool, defaults to False):
            Whether to use the slow legacy polygon extractor for training.
    """
    def __init__(**kwargs):
        self.normalization = kwargs.pop('normalization', None)
        self.normalize_whitespace = kwargs.pop('normalize_whitespace', True)
        self.bidi_reordering = kwargs.pop('bidi_reordering', True)
        self.legacy_polygons = kwargs.pop('legacy_polygons', False)
        super().__init__(**kwargs)


class BLLASegmentationTrainingDataConfig(SegmentationTrainingDataConfig):
    """
    Base configuration for training a BLLA VGSL recognition model.

    Arg:
        line_width (int, defaults to 8):
            Line width in the target segmentation map.
    """
    def __init__(self, **kwargs):
        self.line_width = kwargs.pop('line_width', 8)
        super().__init__(**kwargs)


class VGSLRecognitionTrainingConfig(TrainingConfig):
    """
    Base configuration for training a VGSL recognition model with CTC loss.

    Arg:
        paddding (int, defaults to 16):
            Padding around start/end of line image.
        freeze_backbone (int, defaults to 0):
            Freezes the backbone (everything before the recurrent layers) of
            the network for `n` iterations.
        resize (Literal['fail', 'new', 'union'], defaults to 'fail'):
    """
    def __init__(self, **kwargs):
        self.spec = kwargs.pop('spec', '[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]')
        self.padding = kwargs.pop('padding', 16)
        self.freeze_backbone = kwargs.pop('freeze_backbone', 0)
        self.resize = kwargs.pop('resize', 'fail')
        kwargs.setdefault('quit', 'early')
        kwargs.setdefault('lrate', 1e-3)
        super().__init__(**kwargs)


class BLLASegmentationTrainingConfig(TrainingConfig):
    """
    Base configuration for training a BLLA VGSL recognition model.

    Arg:
        padding (tuple[int, int], defaults to (0, 0)):
        resize (Literal['fail', 'new', 'union'], defaults to 'fail'):

    """
    def __init__(self, **kwargs):
        self.spec = kwargs.pop('spec', '[1,1800,0,3 Cr7,7,64,2,2 Gn32 Cr3,3,128,2,2 Gn32 Cr3,3,128 Gn32 Cr3,3,256 Gn32 Cr3,3,256 Gn32 Lbx32 Lby32 Cr1,1,32 Gn32 Lby32 Lbx32]')
        self.padding = kwargs.pop('padding', (0, 0))
        self.resize = kwargs.pop('resize', 'fail')

        kwargs.setdefault('quit', 'fixed')
        kwargs.setdefault('epochs', 50)
        kwargs.setdefault('lrate', 2e-4)
        kwargs.setdefault('weight_decay', 1e-5)
        kwargs.setdefault('cos_t_max', 50)
        kwargs.setdefault('cos_min_lr', 2e-5)
        super().__init__(**kwargs)
