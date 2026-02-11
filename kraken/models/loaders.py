"""
kraken.models.loaders
~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation for model metadata and weight loading from various formats.
"""
import json
import logging
import importlib

from os import PathLike
from typing import Union, NewType, Literal, Optional
from pathlib import Path
from collections.abc import Sequence
from packaging.version import Version

from kraken.models.base import BaseModel
from kraken.models.utils import create_model
logger = logging.getLogger(__name__)


_T_tasks = NewType('_T_tasks', Literal['segmentation', 'recognition', 'reading_order'])

__all__ = ['load_models', 'load_coreml', 'load_safetensors']


def load_models(path: Union[str, 'PathLike'], tasks: Optional[Sequence[_T_tasks]] = None) -> list[BaseModel]:
    """
    Tries all loaders in sequence to deserialize models found in file at path.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'{path} is not a regular file.')
    for loader in importlib.metadata.entry_points(group='kraken.loaders'):
        try:
            return loader.load()(path, tasks=tasks)
        except ValueError:
            continue
    raise ValueError(f'No loader found for {path}')


def load_safetensors(path: Union[str, PathLike], tasks: Optional[Sequence[_T_tasks]] = None) -> list[BaseModel]:
    """
    Loads one or more models in safetensors format and returns them.

    Args:
        path: Path to the safetensors file.
        tasks: Filter for model types to load from file.

    Returns:
        A list of models.

    Raises:
        ValueError: When model metadata is incomplete or the safetensors file
        is invalid.
    """
    from torch import nn
    from safetensors import safe_open, SafetensorError
    from safetensors.torch import load_model
    models = nn.ModuleDict()
    try:
        with safe_open(path, framework="pt") as f:
            if (metadata := f.metadata()) is not None:
                model_map = json.loads(metadata.get('kraken_meta', 'null'))
                prefixes = list(model_map.keys())
                # construct models
                for prefix in prefixes:
                    if (min_ver := Version(model_map[prefix].get('_kraken_min_version'))) > (inst_ver := Version(importlib.metadata.version('kraken'))):
                        logger.warning(f'Model {prefix} in model file {path} requires minimum kraken version {min_ver} (installed {inst_ver})')
                        # XXX: Don't skip check on release.
#                        continue
                    if tasks and not set(tasks).intersection(set(model_map[prefix].get('_tasks', []))):
                        logger.info(f'Model {prefix} in model file {path} not in demanded tasks {tasks}')
                        continue
                    model_map[prefix].pop('_tasks')
                    models[prefix] = create_model(model_map[prefix].get('_model'), **model_map[prefix])
            else:
                raise ValueError(f'No model metadata found in {path}.')
    except SafetensorError as e:
        raise ValueError(f'Invalid model file {path}') from e
    # load weights into models with strict=False to allow dtype mismatches
    # between saved weights and model parameters (e.g. bfloat16 weights
    # into a float32 model). PyTorch's load_state_dict handles the
    # conversion internally via copy_().
    missing, unexpected = load_model(models, path, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Error(s) in loading state_dict for {models.__class__.__name__}:\n'
                           f'    Missing key(s): {missing}\n'
                           f'    Unexpected key(s): {unexpected}')
    return list(models.values())


def load_coreml(path: Union[str, PathLike], tasks: Optional[Sequence[_T_tasks]] = None) -> list[BaseModel]:
    """
    Loads a model in CoreML format.

    Args:
        path: Path to the coreml file.
        tasks: Filter for model types to load from file.

    Returns:
        A list of models.

    Raises:
        ValueError: When model metadata is incomplete or the coreml file
        is invalid.
    """
    root_logger = logging.getLogger()
    level = root_logger.getEffectiveLevel()
    root_logger.setLevel(logging.ERROR)
    from coremltools.models import MLModel
    root_logger.setLevel(level)
    from google.protobuf.message import DecodeError

    models = []

    if isinstance(path, PathLike):
        path = path.as_posix()
    try:
        mlmodel = MLModel(path)
    except TypeError as e:
        raise ValueError(str(e)) from e
    except DecodeError as e:
        raise ValueError(f'Failure parsing model protobuf: {e}') from e

    metadata = json.loads(mlmodel.user_defined_metadata.get('kraken_meta', '{}'))
    # convert kraken < 7 string syle model_type to list
    metadata['model_type'] = [metadata['model_type']] if metadata['model_type'] else []
    vgsl_spec = mlmodel.user_defined_metadata.get('vgsl') or metadata.get('vgsl')
    # avoid passing codec/vgsl twice (once in metadata, once as explicit argument)
    metadata.pop('codec', None)
    metadata.pop('vgsl', None)
    if not vgsl_spec:
        raise ValueError(f'No VGSL spec in model metadata for {path}')

    if tasks and not set(metadata.get('model_type', [])).intersection(set(tasks)):
        logger.info(f'Model file {path} not in demanded tasks {tasks}')
        return []

    model = create_model('TorchVGSLModel',
                         vgsl=vgsl_spec,
                         codec=json.loads(mlmodel.user_defined_metadata.get('codec', 'null')),
                         **metadata)

    # construct state dict
    weights = {}
    spec = mlmodel.get_spec().neuralNetwork.layers
    from ._coreml import _coreml_parsers
    for cml_parser in _coreml_parsers:
        weights.update(cml_parser(spec))

    model.load_state_dict(weights)
    models.append(model)

    # construct additional models if auxiliary layers are defined.

    if 'aux_layers' in mlmodel.user_defined_metadata:
        logger.info('Deserializing auxiliary layers.')

        nn.aux_layers = {k: cls(v).nn.get_submodule(k) for k, v in json.loads(mlmodel.user_defined_metadata['aux_layers']).items()}

    return models
