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
Model constructor, task, and hyperparameter registries.
"""
import inspect
import importlib
from functools import partial
from collections import OrderedDict

from typing import Literal


OPTIMIZERS = ['Adam', 'AdamW', 'SGD', 'RMSprop']
SCHEDULERS = ['cosine', 'constant', 'exponential', 'step', '1cycle', 'reduceonplateau']
STOPPERS = ['early', 'fixed']
PRECISIONS = ['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true']
