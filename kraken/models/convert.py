import json
import logging
import importlib

from pathlib import Path
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional, Union
from kraken.models.loaders import load_models

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)

__all__ = ['convert_models', 'load_from_checkpoint',
           'find_checkpoint_module', 'find_weights_archs']


def _register_safe_globals() -> None:
    import torch.serialization
    from collections import defaultdict

    from kraken.containers import BaselineLine, BBoxLine, Region, Segmentation
    from kraken.lib.codec import PytorchCodec

    safe_globals = [defaultdict, Segmentation, BaselineLine, BBoxLine, Region, PytorchCodec]
    for ep in importlib.metadata.entry_points(group='kraken.configs'):
        try:
            safe_globals.append(ep.load())
        except Exception as e:
            logger.debug(f'Config entry point {ep.name} ({ep.value}) failed to load: {e}')
    torch.serialization.add_safe_globals(safe_globals)


def find_checkpoint_module(path: Union[str, 'PathLike']) -> type:
    """
    Identifies the KrakenTrainerModule subclass that produced a checkpoint
    without instantiating it, by matching the pickled module configuration
    against each registered lightning module's `_config_class`.

    Raises:
        ValueError: when the checkpoint contains no configuration or no
        registered lightning module matches it.
    """
    import torch

    _register_safe_globals()
    ckpt = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    config = ckpt.get('_module_config')
    if config is None:
        config = ckpt.get('hyper_parameters', {}).get('config')
    if config is None:
        raise ValueError(f'{path} contains no kraken module configuration.')
    candidates = []
    for ep in importlib.metadata.entry_points(group='kraken.lightning_modules'):
        cls = ep.load()
        config_cls = getattr(cls, '_config_class', None)
        if config_cls is None:
            continue
        if type(config) is config_cls:
            return cls
        if isinstance(config, config_cls):
            candidates.append(cls)
    if candidates:
        # third-party config subclasses: most-derived _config_class wins
        return max(candidates, key=lambda c: len(c._config_class.__mro__))
    raise ValueError(f'No registered lightning module matches the configuration '
                     f'class {type(config).__name__} in {path}.')


def find_weights_archs(path: Union[str, 'PathLike'],
                       task: Optional[str] = None) -> Optional[set[str]]:
    """
    Returns the set of architecture names matching the models in a weights
    file, determined by matching the safetensors `kraken_meta` per-model
    `_model` class names against the `_model_class` of each registered
    architecture entry point.

    Architectures are drawn from the `kraken.archs.<task>` entry point groups:
    only `task`'s when given, otherwise every `kraken.archs.*` group, so the
    result may span tasks when families share a model class (VGSL recognition
    and BLLA segmentation both use `TorchVGSLModel`).

    Returns:
        The set of matching arch names, or None when the architecture cannot
        be determined (non-safetensors file, missing metadata, or unknown
        model classes).
    """
    if Path(path).suffix != '.safetensors':
        return None
    try:
        from safetensors import safe_open
        with safe_open(path, framework='pt') as f:
            metadata = f.metadata()
        model_map = json.loads((metadata or {}).get('kraken_meta', 'null'))
    except Exception as e:
        logger.debug(f'Failed to read weights metadata from {path}: {e}')
        return None
    if not model_map:
        return None
    model_names = {entry.get('_model') for entry in model_map.values() if isinstance(entry, dict)}
    if task is not None:
        groups = [f'kraken.archs.{task}']
    else:
        groups = [g for g in importlib.metadata.entry_points().groups
                  if g.startswith('kraken.archs.')]
    archs = set()
    for group in groups:
        for ep in importlib.metadata.entry_points(group=group):
            model_class = getattr(ep.load(), '_model_class', None)
            if model_class is not None and model_class.__name__ in model_names:
                archs.add(ep.name)
    return archs or None


def load_from_checkpoint(path):
    module = find_checkpoint_module(path)
    return module.load_from_checkpoint(path, weights_only=True, map_location='cpu')


def convert_models(paths: Iterable[Union[str, 'PathLike']],
                   output: Union[str, 'PathLike'],
                   weights_format: str = 'safetensors') -> 'PathLike':
    """
    Converts the models in a set of checkpoint or weights files into a single
    output weights file.

    It accepts checkpoints and weights files interchangeably for all supported
    formats and models.

    This function has a number of uses:

        * it can be used to convert checkpoints into weights.

          convert_models(['model.ckpt'], 'model.safetensors')

        * it can be used to convert multiple related models into a single
          weights file for joint inference:

          convert_models(['blla_line.ckpt', 'blla_region.ckpt'], 'model.safetensors')

        * it can convert models between coreml and safetensors formats:

          convert_models(['blla.mlmodel'], 'blla.safetensors')

    Args:
        paths: Paths to checkpoint or weights files.
        output: Output path to the combined/converted file. The actual output
                path may be modified.
        weights_format: Serialization format to write the weights to.

    Returns:
        The path the actual weights file was written to.
    """
    try:
        (entry_point,) = importlib.metadata.entry_points(group='kraken.writers', name=weights_format)
        writer = entry_point.load()
    except ValueError:
        raise ValueError(f'No writer for format {weights_format} found.')

    models = []
    for ckpt in paths:
        ckpt = Path(ckpt)
        if ckpt.suffix == '.ckpt':
            models.append(load_from_checkpoint(ckpt).net)
        else:
            models.extend(load_models(ckpt))

    return writer(models, output)
