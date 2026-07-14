#
# Copyright 2026
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
kraken.lib.ppocr.network
~~~~~~~~~~~~~~~~~~~~~~~~

Assembles the PP-OCRv6 recognition network (PPLCNetV4 backbone → sequence neck →
CTC head). The auxiliary NRTR head is added separately by the trainer and
discarded at inference.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .backbone import PPLCNetV4, PPOCRv6Variant
from .necks import build_neck
from .heads import CTCHead

__all__ = ['MODEL_VARIANTS', 'PPOCRv6Variant', 'PPOCRv6Recognizer',
           'WIDTH_SUBSAMPLING', 'build_recognizer']


# total width subsampling of the recognizer (stem stride 4 × final pooling 2)
WIDTH_SUBSAMPLING = 8


# Architecture hyper-parameters for each PP-OCRv6 recognition variant.
MODEL_VARIANTS = {
    'tiny': {
        'backbone_size': 'tiny',
        'neck': {'type': 'reshape'},
        'head': {'mid_channels': 80, 'use_guide': True},
        'nrtr': {'dim': 384, 'num_decoder_layers': 4},
    },
    'small': {
        'backbone_size': 'small',
        'neck': {'type': 'lightsvtr', 'dims': 120, 'depth': 2,
                 'mlp_ratio': 2.0, 'local_kernel': 7, 'use_guide': False},
        'head': {'mid_channels': None, 'use_guide': False},
        'nrtr': {'dim': 384, 'num_decoder_layers': 4},
    },
    'medium': {
        'backbone_size': 'medium',
        'neck': {'type': 'lightsvtr', 'dims': 192, 'depth': 2,
                 'mlp_ratio': 4.0, 'local_kernel': 7, 'use_guide': False},
        'head': {'mid_channels': None, 'use_guide': False},
        'nrtr': {'dim': 512, 'num_decoder_layers': 4},
    },
}


@torch.compiler.disable
def _lengths_and_mask(seq_lens, w_in, w_out, device):
    """
    Map per-sample input widths to output sequence lengths and a per-timestep
    validity mask (N, W') for masking the neck attention.

    Runs eagerly (``torch.compiler.disable``) to keep the ``w_out / w_in`` ratio
    out of the compile trace, which otherwise spams value-range warnings.
    """
    scaled = (seq_lens.to(torch.float32) * (w_out / float(w_in))).floor().long()
    out_lens = scaled.clamp(min=1, max=w_out)
    positions = torch.arange(w_out, device=device)
    mask = positions[None, :] < out_lens[:, None]
    return out_lens, mask


class PPOCRv6Recognizer(nn.Module):
    """
    Full PP-OCRv6 CTC recognition network.

    ``forward(image, seq_lens)`` returns ``(logits, out_lens)`` where ``image``
    is ``(N, C, H, W)``, ``logits`` is ``(N, num_classes, 1, W')`` and
    ``out_lens`` is ``(N,)`` (or ``None``). Class 0 is the CTC blank.
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module,
                 variant: PPOCRv6Variant, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.variant = variant
        self.num_classes = num_classes

    def forward_train(self, x: torch.Tensor,
                      seq_lens: Optional[torch.Tensor] = None):
        """
        Like :meth:`forward` but also returns the raw backbone feature map so an
        auxiliary head (NRTR/GTC) can share it without recomputing the backbone.

        Returns ``(logits (N, num_classes, 1, W'), out_lens, feat (N, C, 1, W'))``.
        """
        w_in = x.shape[3]
        feat = self.backbone(x)                # (N, C, 1, W')
        w_out = feat.shape[3]

        out_lens = None
        mask = None
        if seq_lens is not None:
            out_lens, mask = _lengths_and_mask(seq_lens, w_in, w_out, feat.device)

        seq = self.neck(feat, mask)            # (N, W', Cn)
        logits = self.head(seq)                # (N, W', num_classes)
        logits = logits.permute(0, 2, 1).unsqueeze(2)  # (N, num_classes, 1, W')
        return logits, out_lens, feat

    def forward(self, x: torch.Tensor,
                seq_lens: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits, out_lens, _ = self.forward_train(x, seq_lens)
        return logits, out_lens


def build_recognizer(variant: PPOCRv6Variant, num_classes: int) -> PPOCRv6Recognizer:
    """
    Build a PP-OCRv6 recognizer.

    Args:
        variant: one of ``tiny``, ``small`` or ``medium``.
        num_classes: number of CTC output classes including the blank (class 0).
    """
    if variant not in MODEL_VARIANTS:
        raise ValueError(f'variant must be one of {list(MODEL_VARIANTS)}, got {variant!r}')
    spec = MODEL_VARIANTS[variant]
    backbone = PPLCNetV4(model_size=spec['backbone_size'])
    neck = build_neck(spec['neck'], backbone.out_channels)
    head = CTCHead(neck.out_channels, num_classes, **spec['head'])
    return PPOCRv6Recognizer(backbone, neck, head, variant, num_classes)
