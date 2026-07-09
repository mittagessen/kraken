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
kraken.lib.dataset.augment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Historical-document line augmentation for recognition training. Runs on the
dataset output: float ``(C, H, W)`` tensors in ``[0, 1]``, height-normalized,
edge-padded with zeros and inverted (ink ≈ 1, background ≈ 0).
"""
import math

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode, v2

__all__ = ['DefaultAugmenter']


def _u(a: float, b: float) -> float:
    return float(torch.empty(()).uniform_(a, b))


class DefaultAugmenter:
    """
    Augmentation recipe for historical manuscript lines.

    Ops: ``warp`` (mild affine/perspective/rotation/elastic), ``morph`` (ink
    erosion/dilation), ``show_through`` (mirrored verso bleed), ``ruling``
    (faint ruling lines), ``intrusion`` (neighbour-line ascenders/descenders),
    ``grayscale``, ``color`` (brightness/contrast/saturation/hue),
    ``gamma``, ``shading`` (low-frequency illumination field), ``microfilm``
    (harsh-contrast grayscale + grain), ``noise``, ``blur``, ``erase``.

    Args:
        p: global gate — probability that a sample is augmented at all.
        max_ops: cap on the number of ops applied to a single sample.
        probs: per-op probability overrides, e.g. ``{'microfilm': 0.0}``.
    """

    _RECIPE = (('warp', 0.25),
               ('morph', 0.20),
               ('show_through', 0.20),
               ('ruling', 0.15),
               ('intrusion', 0.20),
               ('grayscale', 0.25),
               ('color', 0.30),
               ('gamma', 0.25),
               ('shading', 0.20),
               ('microfilm', 0.10),
               ('noise', 0.20),
               ('blur', 0.20),
               ('erase', 0.10))

    def __init__(self,
                 p: float = 0.5,
                 max_ops: int = 3,
                 probs: dict[str, float] | None = None):
        self.p = p
        self.max_ops = max_ops
        self.probs = dict(self._RECIPE)
        if probs:
            unknown = set(probs) - set(self.probs)
            if unknown:
                raise ValueError(f'Unknown augmentation op(s): {sorted(unknown)}')
            self.probs.update(probs)

        self._warp_t = v2.RandomChoice([
            v2.RandomAffine(degrees=0,
                            translate=(0.02, 0.04),
                            scale=(0.92, 1.08),
                            shear=(-3.0, 3.0),
                            interpolation=InterpolationMode.BILINEAR,
                            fill=0.0),
            v2.RandomPerspective(distortion_scale=0.15, p=1.0, fill=0.0),
            v2.RandomRotation(degrees=2.0,
                              interpolation=InterpolationMode.BILINEAR,
                              fill=0.0),
            v2.ElasticTransform(alpha=20.0, sigma=5.0, fill=0.0),
        ])
        self._grayscale_t = v2.Grayscale(num_output_channels=3)
        self._color_t = v2.ColorJitter(brightness=(0.6, 1.15),
                                       contrast=(0.7, 1.3),
                                       saturation=(0.5, 1.5),
                                       hue=0.1)
        # blur capped at ~1px and erasing kept small-area to preserve fine glyphs
        self._blur_t = v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        self._noise_t = v2.GaussianNoise(mean=0.0, sigma=0.04, clip=False)
        self._erase_t = v2.RandomErasing(p=1.0, scale=(0.01, 0.08),
                                         ratio=(0.3, 3.3), value=0.0)

    def __call__(self, image: torch.Tensor, index: int = 0) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return image
        chosen = [name for name, prob in self.probs.items()
                  if prob > 0 and torch.rand(()) < prob]
        if len(chosen) > self.max_ops:
            keep = sorted(torch.randperm(len(chosen))[:self.max_ops].tolist())
            chosen = [chosen[i] for i in keep]
        im = image
        for name in chosen:
            im = getattr(self, f'_{name}')(im)
        return im.clamp(0.0, 1.0)

    def _warp(self, im: torch.Tensor) -> torch.Tensor:
        return self._warp_t(im)

    def _morph(self, im: torch.Tensor) -> torch.Tensor:
        # ink = high: max-pool dilates (blotting), min-pool erodes (fading)
        x = im.unsqueeze(0)
        if torch.rand(()) < 0.5:
            morphed = F.max_pool2d(x, 3, stride=1, padding=1)
        else:
            morphed = -F.max_pool2d(-x, 3, stride=1, padding=1)
        return im + _u(0.5, 1.0) * (morphed.squeeze(0) - im)

    def _show_through(self, im: torch.Tensor) -> torch.Tensor:
        # verso bleed: a mirrored, blurred, offset copy blended faintly under the ink
        c, h, w = im.shape
        ghost = torch.flip(im.mean(0, keepdim=True), dims=(2,))
        ghost = F.avg_pool2d(ghost.unsqueeze(0), 5, stride=1,
                             padding=2).squeeze(0)
        ghost = torch.roll(ghost,
                           shifts=(int(_u(-h / 6, h / 6)), int(_u(0, w))),
                           dims=(1, 2))
        return torch.maximum(im, _u(0.08, 0.25) * ghost)

    def _ruling(self, im: torch.Tensor) -> torch.Tensor:
        c, h, w = im.shape
        out = im
        yy = torch.arange(h, dtype=im.dtype).unsqueeze(1)
        xs = torch.arange(w, dtype=im.dtype)
        for _ in range(int(torch.randint(1, 3, (1,)))):
            centers = _u(0, h - 1) + xs * (_u(-2.0, 2.0) / max(w - 1, 1))
            dist = (yy - centers.unsqueeze(0)).abs()
            band = (_u(1.0, 2.0) - dist).clamp(0.0, 1.0)
            out = torch.maximum(out, _u(0.1, 0.3) * band.unsqueeze(0))
        return out

    def _intrusion(self, im: torch.Tensor) -> torch.Tensor:
        # ascenders/descenders of adjacent lines: a shifted copy pasted into a
        # thin band at the top and/or bottom edge
        c, h, w = im.shape
        out = im
        top, bottom = torch.rand(()) < 0.5, torch.rand(()) < 0.5
        if not top and not bottom:
            top = True
        if top:
            dy = int(_u(0.65, 0.9) * h)
            band = min(int(_u(0.08, 0.22) * h), dy)
            shifted = torch.roll(im, shifts=(dy, int(_u(0, w))), dims=(1, 2))
            frag = torch.zeros_like(im)
            frag[:, :band] = shifted[:, :band]
            out = torch.maximum(out, frag)
        if bottom:
            dy = int(_u(0.65, 0.9) * h)
            band = min(int(_u(0.08, 0.22) * h), dy)
            shifted = torch.roll(im, shifts=(-dy, int(_u(0, w))), dims=(1, 2))
            frag = torch.zeros_like(im)
            frag[:, h - band:] = shifted[:, h - band:]
            out = torch.maximum(out, frag)
        return out

    def _grayscale(self, im: torch.Tensor) -> torch.Tensor:
        if im.shape[0] == 1:
            return im
        return self._grayscale_t(im)

    def _color(self, im: torch.Tensor) -> torch.Tensor:
        if im.shape[0] == 1:
            return im * _u(0.6, 1.15)
        return self._color_t(im)

    def _gamma(self, im: torch.Tensor) -> torch.Tensor:
        # gamma is defined on the un-inverted image
        return 1 - v2.functional.adjust_gamma((1 - im).clamp(0.0, 1.0),
                                              _u(0.6, 1.6))

    def _shading(self, im: torch.Tensor) -> torch.Tensor:
        # low-frequency multiplicative illumination field on the paper
        c, h, w = im.shape
        field = 1 - torch.rand(1, 1, 2, max(2, w // 192 + 2)) * _u(0.1, 0.35)
        field = F.interpolate(field, size=(h, w), mode='bilinear',
                              align_corners=True).squeeze(0)
        return 1 - (1 - im).clamp(0.0, 1.0) * field

    def _microfilm(self, im: torch.Tensor) -> torch.Tensor:
        # harsh-contrast grayscale capture with film grain (sigmoid contrast curve)
        c = im.shape[0]
        o = 1 - im.mean(0, keepdim=True)
        tau, k = _u(0.45, 0.7), _u(6.0, 14.0)
        lo = 1 / (1 + math.exp(tau * k))
        hi = 1 / (1 + math.exp((tau - 1) * k))
        o = (torch.sigmoid((o - tau) * k) - lo) / (hi - lo)
        t = 1 - o + torch.randn_like(o) * _u(0.02, 0.05)
        return t.expand(c, -1, -1).clone() if c > 1 else t

    def _noise(self, im: torch.Tensor) -> torch.Tensor:
        return self._noise_t(im)

    def _blur(self, im: torch.Tensor) -> torch.Tensor:
        return self._blur_t(im)

    def _erase(self, im: torch.Tensor) -> torch.Tensor:
        return self._erase_t(im)
