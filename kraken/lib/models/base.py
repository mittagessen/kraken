"""
kraken.lib.models.base
~~~~~~~~~~~~~~~~~~~~~~

Base metaclass for models
"""
from abc import ABC, abstractmethod
from typing import Union, Literal, NewType

__all__ = ['BaseModel']

import logging

logger = logging.getLogger(__name__)

_T_tasks = NewType('_T_tasks', list[Literal['segmentation', 'recognition', 'reading_order']])


class BaseModel(ABC):
    """
    Base model metaclass that all models inherit from.
    """
    @property
    @abstractmethod
    def _kraken_min_version(self) -> str:
        """
        Contains the minimum kraken version this model needs to run.
        """
        pass

    @property
    @abstractmethod
    def user_metadata(self) -> dict:
        """
        A dictionary containing all hyperparameters, statistics, etc. of the
        model.
        """
        pass

    @property
    @abstractmethod
    def hyper_params(self):
        """
        A filtered version of the `user_metadata` field containing only values
        needed to instantiate the model.
        """
        pass

    @property
    def one_channel_mode(self) -> Union[None, Literal['1', 'L']]:
        """
        An auxiliary attribute of VGSL models to distinguish between b/w and
        grayscale input image modes.
        """
        pass

    @property
    def seg_type(self) -> Union[None, Literal['bbox', 'baseline']]:
        """
        An auxiliary attribute of text recognition models to determine if they
        have been trained on bounding box or baseline data.
        """
        return None

    @property
    def use_legacy_polygons(self) -> Union[None, bool]:
        """
        An auxiliary attribute of text recognition models to determine if they
        have been trained with data produced using the slow legacy polygon
        extraction implementation.
        """
        return None

    @property
    @abstractmethod
    def model_type(self) -> _T_tasks:
        """
        An attribute containing the tasks the model can perform.
        """
        pass

    @abstractmethod
    def load_state_dict(self):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def prepare_for_inference(self, config: 'InferenceConfig'):
        """
        Prepares the model for inference.
        """
        pass
