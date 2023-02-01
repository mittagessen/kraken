#
# Copyright 2022 Benjamin Kiessling
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
Top-level module containing datasets for recognition and segmentation training.
"""
from .recognition import ArrowIPCRecognitionDataset, PolygonGTDataset, GroundTruthDataset # NOQA
from .segmentation import BaselineSet # NOQA
from .utils import ImageInputTransforms, collate_sequences, global_align, compute_confusions # NOQA
