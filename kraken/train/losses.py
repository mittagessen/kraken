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
Loss functions for segmentation training.
"""
import torch

from torch import nn


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss for binary/multi-label segmentation.

    Computes the Dice coefficient per channel and averages, providing
    robustness to class imbalance compared to BCE alone.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities [B, C, H, W] in [0, 1].
            target: Binary ground truth [B, C, H, W].

        Returns:
            Scalar loss value (1 - mean Dice across channels).
        """
        pred = pred.flatten(2)
        target = target.flatten(2)

        intersection = (pred * target).sum(dim=2)
        cardinality = pred.sum(dim=2) + target.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()
