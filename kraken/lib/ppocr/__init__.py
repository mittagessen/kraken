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
kraken.lib.ppocr
~~~~~~~~~~~~~~~~

PP-OCRv6 text recognition models.
"""
from .model import PPOCRv6Model
from .network import (MODEL_VARIANTS, PPOCRv6Recognizer, PPOCRv6Variant,
                     build_recognizer)

__all__ = ['MODEL_VARIANTS', 'PPOCRv6Model', 'PPOCRv6Recognizer',
           'PPOCRv6Variant', 'build_recognizer']
