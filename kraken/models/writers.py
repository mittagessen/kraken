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
    from collections import defaultdict
    from safetensors.torch import save_model

    # Clone tensors in packed storage groups (e.g. from LSTM/GRU
    # flatten_parameters) where no single tensor covers the entire storage.
    # See https://github.com/huggingface/safetensors/issues/657.
    for model in objs:
        state_dict = model.state_dict()
        storage_to_names = defaultdict(set)
        for name, tensor in state_dict.items():
            storage_to_names[tensor.untyped_storage().data_ptr()].add(name)
        for names in storage_to_names.values():
            if len(names) <= 1:
                continue
            has_complete = any(
                state_dict[n].data_ptr() == state_dict[n].untyped_storage().data_ptr()
                and state_dict[n].nelement() * state_dict[n].element_size() == state_dict[n].untyped_storage().size()
                for n in names
            )
            if not has_complete:
                for name in names:
                    parts = name.split('.')
                    mod = model
                    for part in parts[:-1]:
                        mod = getattr(mod, part)
                    old = getattr(mod, parts[-1])
                    if isinstance(old, nn.Parameter):
                        setattr(mod, parts[-1], nn.Parameter(old.data.clone(), requires_grad=old.requires_grad))
                    else:
                        mod.register_buffer(parts[-1], old.clone())

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
