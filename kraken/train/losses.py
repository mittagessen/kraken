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

Implements soft Dice loss for region segmentation and soft clDice
(Shit et al., CVPR 2021) for connectivity preservation in thin
curvilinear structures such as text baselines.
"""
import torch
import torch.nn.functional as F

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


class SoftSkeletonize(nn.Module):
    """
    Differentiable soft skeletonization via iterative morphological erosion.

    Approximates the morphological skeleton by iteratively computing:
        skeleton += x - opening(x)
        x = erosion(x)

    where erosion uses max-pooling on (1-x) and opening is erosion followed
    by dilation.
    """
    def __init__(self, num_iter: int = 3):
        super().__init__()
        self.num_iter = num_iter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C, H, W] with values in [0, 1].

        Returns:
            Soft skeleton of same shape.
        """
        skeleton = torch.zeros_like(x)
        for _ in range(self.num_iter):
            eroded = self._erode(x)
            opened = self._dilate(eroded)
            skeleton = skeleton + F.relu(x - opened)
            x = eroded
        return skeleton

    @staticmethod
    def _erode(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        pad = kernel_size // 2
        return 1.0 - F.max_pool2d(1.0 - x, kernel_size, stride=1, padding=pad)

    @staticmethod
    def _dilate(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        pad = kernel_size // 2
        return F.max_pool2d(x, kernel_size, stride=1, padding=pad)


class SoftClDiceLoss(nn.Module):
    """
    Soft centerline Dice loss for tubular structure segmentation.

    Computes the Dice coefficient on morphological soft skeletons of
    prediction and target, penalizing broken connectivity and spurious
    branches in thin structures.

    Reference:
        Shit et al., "clDice -- A Novel Topology-Preserving Loss Function
        for Tubular Structure Segmentation", CVPR 2021.
    """
    def __init__(self, num_iter: int = 3, smooth: float = 1.0):
        super().__init__()
        self.skeletonize = SoftSkeletonize(num_iter)
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities [B, C, H, W] in [0, 1].
            target: Binary ground truth [B, C, H, W].

        Returns:
            Scalar loss value (1 - clDice).
        """
        skel_pred = self.skeletonize(pred)
        skel_target = self.skeletonize(target)

        tprec = ((skel_pred * target).sum() + self.smooth) / (skel_pred.sum() + self.smooth)
        tsens = ((skel_target * pred).sum() + self.smooth) / (skel_target.sum() + self.smooth)

        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-7)
        return 1.0 - cl_dice
