"""
kraken.tasks.recognition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrappers around models for specific tasks.
"""
import torch
from torch import nn

from PIL import Image
from dataclasses import dataclass
from collections.abc import Generator, Counter
from typing import TYPE_CHECKING, Union
from kraken.registry import register
from kraken.models import load_models, RecognitionInferenceConfig

if TYPE_CHECKING:
    from os import PathLike
    from kraken.containers import Segmentation, ocr_record

__all__ = ['RecognitionTaskModel']

import logging

logger = logging.getLogger(__name__)


@dataclass
class FileRecognitionTestMetrics:
    """
    A container class of text recognition test metrics for a single file.
    """
    character_counts: Counter
    num_errors: int
    cer: float
    wer: float
    case_insensitive_cer: float
    confusions:  int
    scripts: int
    insertions: int
    deletes: int
    substitutions: int


@dataclass
class RecognitionTestMetrics:
    """
    A container class of text recognition test metrics for a collection of
    pages.
    """
    character_counts: Counter
    num_errors: int
    cer: float
    wer: float
    case_insensitive_cer: float
    confusions:  int
    scripts: int
    insertions: int
    deletes: int
    substitutions: int
    per_file_metrics: dict[str, FileRecognitionTestMetrics]


@register(type='task')
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
        self.net.prepare_for_inference(config)
        return self.net.predict(im=im, segmentation=segmentation)

    @torch.inference_mode
    def test(self,
             test_data: list['Segmentation'],
             config: RecognitionInferenceConfig) -> RecognitionTestMetrics:
        """
        Tests a recognition model against a set of existing transcriptions.

        Args:
            test_data: A list of segmentation objects containing transcriptions.
            config: A configuration object containing inference parameters.

        Returns:
        """
        from torchmetrics.text import CharErrorRate, WordErrorRate
        from kraken.lib.dataset import compute_confusions, global_align

        self.net_prepare_for_inference(config)

        test_cer = CharErrorRate()
        test_cer_case_insensitive = CharErrorRate()
        test_wer = WordErrorRate()

        errors = 0
        per_file_metrics = {}
        characters = Counter()
        algn_gt: list[str] = []
        algn_pred: list[str] = []

        for segmentation in test_data:
            try:
                im = Image.open(segmentation)
            except FileNotFoundError as e:
                logger.warning(f'{e.strerror} {e.filename}. Skipping.')
                continue

            file_test_cer = CharErrorRate()
            file_test_cer_case_insensitive = CharErrorRate()
            file_test_wer = WordErrorRate()
            file_characters = Counter()
            file_errors = 0

            file_algn_gt: list[str] = []
            file_algn_pred: list[str] = []

            for record in self.predict(im=im, segmentation=segmentation, config=config):
                for x, y in zip(record.prediction, record.text):
                    file_characters.add(y)
                    c, algn1, algn2 = global_align(y, x)
                    algn_gt.extend(algn1)
                    algn_pred.extend(algn2)
                    file_errors += c
                    test_cer.update(x, y)
                    test_cer_case_insensitive.update(x.lower(), y.lower())
                    test_wer.update(x, y)

                    file_test_cer.update(x, y)
                    file_test_cer_case_insensitive.update(x.lower(), y.lower())
                    file_test_wer.update(x, y)

            confusions, scripts, ins, dels, subs = compute_confusions(file_algn_gt, file_algn_pred)

            per_file_metrics[segmentation.imagename] = FileRecognitionTestMetrics(character_counts=file_characters,
                                                                                  num_errors=file_errors,
                                                                                  cer=1.0-file_test_cer.compute(),
                                                                                  wer=1.0-file_test_wer.compute(),
                                                                                  case_insensitive_cer=1.0-file_test_cer_case_insensitive.compute(),
                                                                                  confusions=confusions,
                                                                                  scripts=scripts,
                                                                                  insertions=ins,
                                                                                  deletes=dels,
                                                                                  substitutions=subs)
            algn_gt.extend(file_algn_gt)
            algn_pred.extend(file_algn_pred)

            characters.add(file_characters)
            errors += file_errors

        confusions, scripts, ins, dels, subs = compute_confusions(algn_gt, algn_pred)

        return RecognitionTestMetrics(character_counts=characters,
                                      num_errors=errors,
                                      cer=1.0-test_cer.compute(),
                                      wer=1.0-test_wer.compute(),
                                      case_insensitive_cer=1.0-test_cer_case_insensitive.compute(),
                                      confusions=confusions,
                                      scripts=scripts,
                                      insertions=ins,
                                      deletes=dels,
                                      substitutions=subs,
                                      per_file_metrics=per_file_metrics)

    @classmethod
    def load_model(cls, path: Union[str, 'PathLike']):
        models = load_models(path)
        return cls(models)
