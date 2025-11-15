import importlib

from pathlib import Path
from collections.abc import Iterable
from typing import TYPE_CHECKING, Union
from kraken.models.loaders import load_models

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['convert_models']


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
        raise ValueError('No writer for format {weights_format} found.')

    def _find_module(path):
        for entry_point in importlib.metadata.entry_points(group='kraken.lightning_modules'):
            module = entry_point.load()
            try:
                return module.load_from_checkpoint(path)
            except ValueError:
                continue
        raise ValueError(f'No lightning module found for checkpoint {path}')

    models = []
    for ckpt in paths:
        ckpt = Path(ckpt)
        if ckpt.suffix == '.ckpt':
            models.append(_find_module(ckpt).net)
        else:
            models.extend(load_models(ckpt))

    return writer(models, output)
