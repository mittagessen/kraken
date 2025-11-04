from kraken.models.base import BaseModel
from kraken.registry import MODEL_REGISTRY

__all__ = ['create_model']


def create_model(name, *args, **kwargs) -> BaseModel:
    """
    Constructs an empty model from the model registry.
    """
    if not type(name) in (type, str):
        raise ValueError(f'`{name}` is neither type nor string.')

    if name not in MODEL_REGISTRY:
        raise ValueError(f'`{name}` is not in model registry.')

    cfg = MODEL_REGISTRY[name]
    cls = getattr(cfg['_module'], name)

    return cls(*args, **kwargs)
