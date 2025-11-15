import importlib

from typing import TYPE_CHECKING


__all__ = ['create_model']

if TYPE_CHECKING:
    from .base import BaseModel


def create_model(name, *args, **kwargs) -> 'BaseModel':
    """
    Constructs an empty model from the model registry.
    """
    if not type(name) in (type, str):
        raise ValueError(f'`{name}` is neither type nor string.')

    try:
        (entry_point,) = importlib.metadata.entry_points(group='kraken.models', name=name)
    except ValueError:
        raise ValueError(f'`{name}` is not in model registry.')

    cls = entry_point.load()
    return cls(*args, **kwargs)
