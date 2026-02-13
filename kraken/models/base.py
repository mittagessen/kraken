"""
kraken.models.base
~~~~~~~~~~~~~~~~~~~~~~

Base metaclass for models
"""
import logging

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Union, Literal, NewType, TYPE_CHECKING

__all__ = ['BaseModel',
           'RecognitionBaseModel',
           'SegmentationBaseModel']

if TYPE_CHECKING:
    from kraken.configs import Config
    from PIL.Image import Image
    from kraken.containers import Segmentation, ocr_record

logger = logging.getLogger(__name__)

_T_tasks = NewType('_T_tasks', list[Literal['segmentation', 'recognition', 'reading_order', 'pretrain']])


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
    def user_metadata(self) -> dict:
        """
        A dictionary containing all hyperparameters, statistics, etc. of the
        model.
        """
        if not hasattr(self, '_user_metadata'):
            self._user_metadata = {}
        return self._user_metadata

    @user_metadata.setter
    def user_metadata(self, val: dict) -> None:
        self._user_metadata = val

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
    def prepare_for_inference(self, config: 'Config'):
        """
        Prepares the model for inference.
        """
        pass


class SegmentationBaseModel(BaseModel):
    """
    Base model metaclass for layout analysis models.
    """
    @abstractmethod
    def predict(self, im: 'Image.Image') -> 'Segmentation':
        """
        Computes a segmentation on an input image.
        """
        pass


class RecognitionBaseModel(BaseModel):
    """
    Base model metaclass for text recognition models.
    """
    @abstractmethod
    def predict(self, im: 'Image.Image', segmentation: 'Segmentation') -> Generator['ocr_record', None, None]:
        pass
