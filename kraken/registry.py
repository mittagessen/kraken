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

MODEL_REGISTRY = {}
TASK_REGISTRY = {}
LOADER_REGISTRY = {}
WRITER_REGISTRY = OrderedDict()

_REGISTRIES = {'model': MODEL_REGISTRY,
               'loader': LOADER_REGISTRY,
               'writer': WRITER_REGISTRY,
               'task': TASK_REGISTRY}


def register(type: Literal['loader', 'model', 'task']):
    if type not in _REGISTRIES:
        raise ValueError(f'Unknown registry {type}')

    def wrapper(cls, registry=None):
        if cls.__name__ in registry:
            raise ValueError(f'{cls.__name__} already registered.')

        argspec = inspect.getfullargspec(cls.__init__)
        arg_names = list(filter(lambda x: x != 'self', argspec.args))

        default = len(argspec.defaults) if argspec.defaults is not None else 0
        required = len(arg_names) - default

        schema = {}
        schema['_module'] = importlib.import_module(cls.__module__)

        for i, name in enumerate(arg_names):
            if i >= required:
                value = argspec.defaults[i - required]
            else:
                value = None
            schema[name] = value

        registry[cls.__name__] = schema
        return cls

    return partial(wrapper, registry=_REGISTRIES[type])


def create_model(name, **kwargs):
    """
    Constructs an empty model from the model registry.
    """
    if not type(name) in (type, str):
        raise ValueError(f'`{name}` is neither type nor string.')

    if name not in MODEL_REGISTRY:
        raise ValueError(f'`{name}` is not in model registry.')

    cfg = MODEL_REGISTRY[name]
    cls = getattr(cfg['_module'], name)

    return cls(**kwargs)
