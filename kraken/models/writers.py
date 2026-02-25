"""
kraken.models.writers
~~~~~~~~~~~~~~~~~~~~~~~~~

Implementations for model writing in different formats.
"""
import json
import uuid
import logging
import importlib

from torch import nn
from os import PathLike
from pathlib import Path
from typing import Union, TYPE_CHECKING

from kraken.models.base import BaseModel

if TYPE_CHECKING:
    from kraken.lib.vgsl import TorchVGSLModel

logger = logging.getLogger(__name__)

__all__ = ['write_models', 'write_safetensors', 'write_coreml']


def write_models(objs: list[BaseModel], path: Union[str, PathLike]) -> PathLike:
    """
    Tries to write a list of models to a file. Writers are tried in order of
    registration.

    """
    if (path := Path(path)).exists():
        raise ValueError(f'{path} already exists.')

    for writer in importlib.metadata.entry_points(group='kraken.writers'):
        try:
            return writer.load()(objs, path)
        except ValueError:
            continue
    raise ValueError(f'No writer found for {path}')


def write_safetensors(objs: list[BaseModel], path: Union[str, PathLike]) -> PathLike:
    """
    Writes a set of models as a safetensors.
    """
    from safetensors.torch import save_model
    # assign unique prefixes to each model in model list
    prefixes = nn.ModuleDict({str(uuid.uuid4()): model for model in objs})
    metadata = {k: {'_kraken_min_version': v._kraken_min_version,
                    '_tasks': v.model_type,
                    '_model': v.__class__.__name__,
                    **v.user_metadata} for k, v in prefixes.items()}

    save_model(prefixes,
               filename=path,
               metadata={'kraken_meta': json.dumps(metadata)})
    return Path(path)


def write_coreml(obj: list['TorchVGSLModel'], path: Union[str, PathLike]) -> PathLike:
    """
    Writes a single model as a CoreML file.
    """
    from kraken.lib.vgsl import TorchVGSLModel

    if len(obj) != 1:
        raise ValueError('CoreML writer only support writing one model at a time.')
    model = obj[0]
    if not isinstance(model, TorchVGSLModel):
        raise ValueError(f'CoreML writer only serializes TorchVGSLModel objects, not {model}.')

    path = Path(path)
    # coreml refuses to serialize into a path that doesn't have a '.mlmodel'
    # suffix
    if path.suffix != '.mlmodel':
        path = path.with_suffix('.mlmodel')
    if path.exists():
        raise ValueError(f'{path} already exists.')

    model.save_model(path.as_posix())
    return path
