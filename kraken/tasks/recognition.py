"""
kraken.tasks.recognition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrappers around models for specific tasks.
"""
import torch
from torch import nn

from PIL import Image
from collections.abc import Generator
from typing import TYPE_CHECKING, Union
from kraken.models import load_models
from kraken.configs import RecognitionInferenceConfig

if TYPE_CHECKING:
    from os import PathLike
    from kraken.containers import Segmentation, ocr_record

__all__ = ['RecognitionTaskModel']

import logging

logger = logging.getLogger(__name__)


class RecognitionTaskModel(nn.Module):
    """
    A wrapper for model performing a recognition task.

    Raises:
        ValueError: Is raised when the model type is not a sequence recognizer.
    """
    net: nn.Module

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        # only use recognition models.
        models = [net for net in models if 'recognition' in net.model_type]

        if not len(models):
            raise ValueError('No recognition model in model list {models}.')
        if len(models) > 1:
            logger.warning('More than one recognition model in model collection. Using first model.')

        self.net = models[0]
        self.one_channel_mode = self.net.one_channel_mode
        self.seg_type = self.net.seg_type

    @torch.inference_mode()
    def predict(self,
                im: 'Image.Image',
                segmentation: 'Segmentation',
                config: RecognitionInferenceConfig) -> Generator['ocr_record', None, None]:
        """
        Inference using a recognition model.

        Args:
            im: Input image
            segmentation: The segmentation corresponding to the input image.
            config: A configuration object containing inference parameters.

        Yields:
            One ocr_record for each line.
        """
        if config.precision in ['bf16-true', '16-true']:
            logger.warning(f'Selected float precision {config.precision} is '
                           'fixed length 16 bit and likely to cause unstable '
                           'recogntion. Proceed with caution.')
        self.net.prepare_for_inference(config)
        return self.net.predict(im=im, segmentation=segmentation)

    @classmethod
    def load_model(cls, path: Union[str, 'PathLike']):
        models = load_models(path)
        return cls(models)
