#
# Copyright 2026 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
kraken.lib.ppocr.model
~~~~~~~~~~~~~~~~~~~~~~~~

A PP-OCRv6 recognition model.
"""
import logging
from typing import Iterable, Optional

import torch
import torch.nn as nn

from lightning.fabric import Fabric

from kraken.models.base import RecognitionBaseModel
from kraken.lib.codec import PytorchCodec
from kraken.models.ctc import CTCRecognitionInferenceMixin

from .network import PPOCRv6Variant, build_recognizer

logger = logging.getLogger(__name__)

__all__ = ['PPOCRv6Model']


class PPOCRv6Model(nn.Module, CTCRecognitionInferenceMixin, RecognitionBaseModel):
    """
    A kraken recognition model wrapping a PP-OCRv6 CTC network.

        variant: PP-OCRv6 size (``tiny``/``small``/``medium``).
        num_classes: number of CTC classes including the blank (class 0).
        height: input line height.
        codec: codec as a ``c2l`` dict (grapheme -> list[int]).
        seg_type: ``baselines`` or ``bbox``.
        one_channel_mode: ``'1'``/``'L'``/``None`` — expected grayscale mode.
        legacy_polygons: whether the model was trained with the legacy polygon
            extractor.
    """

    _kraken_min_version = '7.0.0'

    def __init__(self,
                 variant: PPOCRv6Variant = 'small',
                 num_classes: Optional[int] = None,
                 height: int = 96,
                 codec: Optional[dict] = None,
                 seg_type: Optional[str] = 'baselines',
                 one_channel_mode: Optional[str] = None,
                 legacy_polygons: bool = False,
                 **kwargs):
        super().__init__()
        if num_classes is None:
            raise ValueError('num_classes is required to build a PPOCRv6Model.')

        self.variant = variant
        self.num_classes = num_classes
        # (batch, channels, height, width); line images are always 3-channel
        # (RGB), width 0 == variable
        self.input = (1, 3, height, 0)

        # the wrapped network: self.nn(line, lens) -> (logits, olens)
        self.nn = build_recognizer(variant, num_classes=num_classes)

        self.codec = PytorchCodec(codec) if codec is not None else None

        # everything needed to reconstruct the model on load
        self.user_metadata = {
            'variant': variant,
            'num_classes': num_classes,
            'height': height,
            'codec': self.codec.c2l if self.codec is not None else None,
            'seg_type': seg_type,
            'one_channel_mode': one_channel_mode,
            'legacy_polygons': legacy_polygons,
            'accuracy': kwargs.get('accuracy', []),
            'metrics': kwargs.get('metrics', []),
        }

    # -------------------------------------------------------- BaseModel API
    @property
    def model_type(self):
        return ['recognition']

    @property
    def seg_type(self):
        return self.user_metadata.get('seg_type')

    @seg_type.setter
    def seg_type(self, value: Optional[str]):
        self.user_metadata['seg_type'] = value

    @property
    def one_channel_mode(self):
        return self.user_metadata.get('one_channel_mode')

    @one_channel_mode.setter
    def one_channel_mode(self, value: Optional[str]):
        self.user_metadata['one_channel_mode'] = value

    @property
    def use_legacy_polygons(self):
        return self.user_metadata.get('legacy_polygons', False)

    @use_legacy_polygons.setter
    def use_legacy_polygons(self, value: bool):
        self.user_metadata['legacy_polygons'] = value

    def forward(self, x, seq_lens=None):
        return self.nn(x, seq_lens)

    # -------------------------------------------------------- training API
    def add_codec(self, codec: PytorchCodec) -> None:
        """
        Adds a PytorchCodec to the model, mirroring it into the serialization
        metadata.
        """
        self.codec = codec
        self.user_metadata['codec'] = codec.c2l

    @property
    def _output_proj(self) -> nn.Linear:
        """The codec-sized final projection of the CTC head."""
        head = self.nn.head
        return head.fc if hasattr(head, 'fc') else head.fc2

    def resize_output(self, output_size: int, del_indices: Optional[Iterable[int]] = None) -> None:
        """
        Resizes the final CTC projection, retaining the existing weights.

        Args:
            output_size: New size/output classes of the last layer.
            del_indices: list of output rows to delete from the layer.
        """
        if not del_indices:
            del_indices = []
        proj = self._output_proj
        logger.debug(f'Resizing CTC output projection to {output_size}')
        old_shape = proj.weight.size(0)
        idx = torch.tensor([x for x in range(old_shape) if x not in del_indices],
                           device=proj.weight.device)
        weight = proj.weight.index_select(0, idx)
        rweight = torch.empty((output_size - weight.size(0), weight.size(1)),
                              dtype=weight.dtype,
                              device=weight.device)
        if rweight.shape[0] > 0:
            nn.init.trunc_normal_(rweight, std=0.02)
        weight = torch.cat([weight, rweight])
        bias = proj.bias.index_select(0, idx)
        bias = torch.cat([bias, torch.zeros(output_size - bias.size(0),
                                            dtype=bias.dtype,
                                            device=bias.device)])
        proj.out_features = output_size
        proj.weight = nn.Parameter(weight)
        proj.bias = nn.Parameter(bias)
        self.num_classes = output_size
        self.nn.num_classes = output_size
        self.user_metadata['num_classes'] = output_size

    # -------------------------------------------------------- inference API
    def prepare_for_inference(self, config):
        """Configure the model for inference (mirrors TorchVGSLModel)."""
        if 'recognition' not in self.model_type:
            raise ValueError(f'{self} is not a recognition model.')
        if self.codec is None:
            raise ValueError('Model has no codec; cannot decode predictions.')
        self.eval()
        self._inf_config = config

        if getattr(self, '_line_extraction_pool', None) is None:
            if getattr(config, 'num_line_workers', 0) == 0:
                class _InProcessPool:
                    def imap_unordered(self, func, iterable):
                        for item in iterable:
                            yield func(item)

                    def terminate(self):
                        return None

                self._line_extraction_pool = _InProcessPool()
            else:
                from torch.multiprocessing import Pool
                self._line_extraction_pool = Pool(config.num_line_workers)
            import weakref
            weakref.finalize(self, self._line_extraction_pool.terminate)

        self._fabric = Fabric(accelerator=config.accelerator,
                              devices=config.device,
                              precision=config.precision)
        self.nn = self._fabric._precision.convert_module(self.nn)
        self.nn = self._fabric.to_device(self.nn)
        self._m_dtype = next(self.parameters()).dtype

    @torch.inference_mode()
    def predict(self, im, segmentation):
        """Yield one ocr_record per line in ``segmentation``."""
        return self._recognition_pred(im, segmentation)
