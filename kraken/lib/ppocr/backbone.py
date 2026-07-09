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
kraken.lib.ppocr.backbone
~~~~~~~~~~~~~~~~~~~~~~~~

The PPLCNetV4 recognition backbone used by PP-OCRv6.
"""
from typing import Literal, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PPLCNetV4', 'NET_CONFIG_REC', 'PPOCRv6Variant']


# The available PP-OCRv6 recognition size variants.
PPOCRv6Variant = Literal['tiny', 'small', 'medium']


# Per-variant stage configuration for the recognition backbone.
# Each block entry is [kernel_size, in_channels, out_channels, stride, use_se].
# A stride given as a 2-tuple ``(2, 1)`` downsamples the height only, preserving
# the width (and therefore the CTC sequence length).
NET_CONFIG_REC = {
    'tiny': {
        'stem': (24, 48),
        'stem_type': 'simple',
        'blocks2': [[3, 48, 48, 1, True]],
        'blocks3': [[3, 48, 48, 1, False]],
        'blocks4': [[3, 48, 96, (2, 1), False],
                    [3, 96, 96, 1, True],
                    [3, 96, 96, 1, False]],
        'blocks5': [[3, 96, 160, (2, 1), False],
                    [3, 160, 160, 1, True],
                    [3, 160, 160, 1, False],
                    [3, 160, 160, 1, False]],
        'blocks6': [],
    },
    'small': {
        'stem': (48, 96),
        'stem_type': 'branch',
        'blocks2': [[3, 96, 96, 1, True]],
        'blocks3': [[3, 96, 96, 1, False], [3, 96, 96, 1, False]],
        'blocks4': [[3, 96, 192, (2, 1), False],
                    [3, 192, 192, 1, True],
                    [3, 192, 192, 1, False],
                    [3, 192, 192, 1, True],
                    [3, 192, 192, 1, False],
                    [3, 192, 192, 1, True],
                    [3, 192, 192, 1, False]],
        'blocks5': [[3, 192, 384, (2, 1), False],
                    [3, 384, 384, 1, True],
                    [3, 384, 384, 1, False]],
        'blocks6': [],
    },
    'medium': {
        'stem': (64, 128),
        'stem_type': 'branch',
        'blocks2': [[3, 128, 128, 1, True]],
        'blocks3': [[3, 128, 256, 1, False],
                    [3, 256, 256, 1, False],
                    [3, 256, 256, 1, True]],
        'blocks4': [[3, 256, 512, (2, 1), False],
                    [3, 512, 512, 1, True],
                    [3, 512, 512, 1, False],
                    [3, 512, 512, 1, True],
                    [3, 512, 512, 1, False],
                    [3, 512, 512, 1, True],
                    [3, 512, 512, 1, False]],
        'blocks5': [[3, 512, 768, (2, 1), False],
                    [3, 768, 768, 1, True],
                    [3, 768, 768, 1, False]],
        'blocks6': [],
    },
}


def _pair(v: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _same_pad(x: torch.Tensor, kernel_size: int, stride: int = 1, value: float = 0.0) -> torch.Tensor:
    """
    ``SAME`` padding for a square kernel/stride.

    For the even kernels used in the branch stem (k=2, s=1) this pads the right
    and bottom edges by one pixel (asymmetric padding).
    """
    ih, iw = x.shape[-2:]
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    pad_h = max((-(-ih // sh) - 1) * sh + kh - ih, 0)
    pad_w = max((-(-iw // sw) - 1) * sw + kw - iw, 0)
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                     pad_h // 2, pad_h - pad_h // 2], value=value)


class Conv2DBN(nn.Module):
    """Conv2d (no bias) followed by BatchNorm2d."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, groups=1, bn_weight_init=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(self.bn.weight, 1.0 if bn_weight_init == 1.0 else 0.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBNAct(nn.Module):
    """
    Conv2d + BatchNorm2d + optional ReLU, used by the branch stem.

    Supports ``padding="SAME"`` for the even-kernel convolutions in
    :class:`StemBlock`.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, use_act=True):
        super().__init__()
        self.same = padding == 'SAME'
        self.kernel_size = kernel_size
        self.stride = stride
        pad = 0 if self.same else (padding if padding is not None else (kernel_size - 1) // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if use_act else nn.Identity()

    def forward(self, x):
        if self.same:
            x = _same_pad(x, self.kernel_size, self.stride)
        return self.act(self.bn(self.conv(x)))


class SELayer(nn.Module):
    """Squeeze-and-excitation block (reduction 4, ReLU + Hardsigmoid gating)."""

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = x.mean(dim=(2, 3), keepdim=True)
        x = self.relu(self.conv1(x))
        x = self.hardsigmoid(self.conv2(x))
        return identity * x


class RepDWConv(nn.Module):
    """
    Reparameterisable depthwise convolution: a k×k depthwise Conv-BN, a 1×1
    depthwise conv and an identity branch, summed and passed through a trailing
    BatchNorm.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.conv = Conv2DBN(channels, channels, kernel_size, 1, padding, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x):
        return self.bn(self.conv(x) + self.conv1(x) + x)


class LCNetV4Block(nn.Module):
    """
    Core PPLCNetV4 block: a token mixer (depthwise conv, optionally SE) followed
    by an inverted-bottleneck channel mixer (expand → act → compress), with a
    residual connection around the channel mixer when the spatial size and
    channel count are preserved.
    """

    def __init__(self, in_channels, out_channels, stride, dw_size, use_se=False,
                 expand_ratio=2, act_type='gelu'):
        super().__init__()
        stride_t = _pair(stride)
        self.has_residual = in_channels == out_channels and stride_t == (1, 1)
        self.use_rep_dw = self.has_residual

        token_mixer = nn.Sequential()
        if self.use_rep_dw:
            token_mixer.add_module('rep_dw', RepDWConv(in_channels, dw_size))
        else:
            padding = (dw_size - 1) // 2
            token_mixer.add_module('dw_conv',
                                   Conv2DBN(in_channels, in_channels, dw_size,
                                            stride, padding, groups=in_channels))
        if use_se:
            token_mixer.add_module('se', SELayer(in_channels))
        self.token_mixer = token_mixer

        hidden_channels = int(in_channels * expand_ratio)
        compress_bn_init = 0.0 if self.has_residual else 1.0
        channel_mixer = nn.Sequential()
        channel_mixer.add_module('expand', Conv2DBN(in_channels, hidden_channels, 1, 1, 0))
        if act_type == 'gelu':
            channel_mixer.add_module('act', nn.GELU())
        elif act_type == 'hswish':
            channel_mixer.add_module('act', nn.Hardswish())
        elif act_type == 'relu':
            channel_mixer.add_module('act', nn.ReLU())
        channel_mixer.add_module('compress',
                                 Conv2DBN(hidden_channels, out_channels, 1, 1, 0,
                                          bn_weight_init=compress_bn_init))
        self.channel_mixer = channel_mixer

    def forward(self, x):
        x = self.token_mixer(x)
        if self.has_residual:
            return x + self.channel_mixer(x)
        return self.channel_mixer(x)


class _SimpleStem(nn.Module):
    """Two stride-2 Conv-BN layers with a GELU in between (tiny variant)."""

    def __init__(self, in_channels, stem_mid, stem_out):
        super().__init__()
        self.conv1 = Conv2DBN(in_channels, stem_mid, 3, 2, 1)
        self.act = nn.GELU()
        self.conv2 = Conv2DBN(stem_mid, stem_out, 3, 2, 1)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class StemBlock(nn.Module):
    """Multi-branch stem (total stride 4) used by the small/medium variants."""

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.stem1 = ConvBNAct(in_channels, mid_channels, 3, 2)
        self.stem2a = ConvBNAct(mid_channels, mid_channels // 2, 2, 1, padding='SAME')
        self.stem2b = ConvBNAct(mid_channels // 2, mid_channels, 2, 1, padding='SAME')
        self.stem3 = ConvBNAct(mid_channels * 2, mid_channels, 3, 2)
        self.stem4 = ConvBNAct(mid_channels, out_channels, 1, 1)

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2b(self.stem2a(x))
        # max-pool with SAME padding (k=2, s=1): pad with -inf so the padded
        # cells never win the max.
        x1 = F.max_pool2d(_same_pad(x, 2, 1, value=float('-inf')), kernel_size=2, stride=1)
        x = self.stem4(self.stem3(torch.cat([x1, x2], dim=1)))
        return x


class PPLCNetV4(nn.Module):
    """
    PPLCNetV4 recognition backbone.

    Args:
        model_size: one of ``tiny``, ``small`` or ``medium``.

    Input line images are always 3-channel (RGB). The forward pass collapses
    the feature height to 1 and halves the width, returning a
    ``(N, out_channels, 1, W')`` tensor suitable for an Im2Seq + CTC head.
    """

    def __init__(self, model_size: PPOCRv6Variant = 'small'):
        super().__init__()
        if model_size not in NET_CONFIG_REC:
            raise ValueError(f'model_size must be one of {list(NET_CONFIG_REC)}, got {model_size!r}')
        cfg = NET_CONFIG_REC[model_size]
        stem_mid, stem_out = cfg['stem']
        # line images are always 3-channel (RGB)
        if cfg['stem_type'] == 'branch':
            self.conv1 = StemBlock(3, stem_mid, stem_out)
        else:
            self.conv1 = _SimpleStem(3, stem_mid, stem_out)

        def make_stage(name):
            return nn.Sequential(*[LCNetV4Block(in_c, out_c, s, k, se, expand_ratio=2)
                                   for k, in_c, out_c, s, se in cfg.get(name, [])])

        self.blocks2 = make_stage('blocks2')
        self.blocks3 = make_stage('blocks3')
        self.blocks4 = make_stage('blocks4')
        self.blocks5 = make_stage('blocks5')
        self.blocks6 = make_stage('blocks6')

        self.out_channels = None
        for name in reversed(['blocks2', 'blocks3', 'blocks4', 'blocks5', 'blocks6']):
            if cfg.get(name):
                self.out_channels = cfg[name][-1][2]
                break

        # Kaiming (fan-out) init for conv weights; BatchNorm gains are left as
        # set by the block constructors.
        self.apply(self._init_conv)

    @staticmethod
    def _init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)
        h = x.shape[2]
        x = F.avg_pool2d(x, kernel_size=(h, 2))
        return x
