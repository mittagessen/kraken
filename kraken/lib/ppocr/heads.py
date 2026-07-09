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
kraken.lib.ppocr.heads
~~~~~~~~~~~~~~~~~~~~~~~~

The CTC recognition head. Consumes a ``(N, W, C)`` sequence and emits
per-timestep class logits ``(N, W, num_classes)``; class 0 is the CTC blank.
"""
import torch.nn as nn

__all__ = ['CTCHead']


class _GuideLayer(nn.Module):
    """Depthwise+pointwise 1-D conv block applied along the sequence (tiny variant)."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.Hardswish(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Hardswish(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.block(x)
        return x.transpose(1, 2)


class CTCHead(nn.Module):
    """
    CTC head with an optional intermediate projection and guide layer.

    Args:
        in_channels: feature dimension of the incoming sequence.
        out_channels: number of output classes (including the CTC blank).
        mid_channels: if set, a two-layer projection ``in -> mid -> out`` is
            used (tiny variant); otherwise a single linear layer.
        use_guide: enable the depthwise 1-D guide conv before projection.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, use_guide=False):
        super().__init__()
        self.use_guide = use_guide
        self.mid_channels = mid_channels
        if use_guide:
            self.guide_layer = _GuideLayer(in_channels)
        if mid_channels is None:
            self.fc = nn.Linear(in_channels, out_channels)
        else:
            self.fc1 = nn.Linear(in_channels, mid_channels)
            self.fc2 = nn.Linear(mid_channels, out_channels)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_guide:
            x = self.guide_layer(x)
        if self.mid_channels is None:
            return self.fc(x)
        return self.fc2(self.fc1(x))
