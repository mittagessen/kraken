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
kraken.lib.ppocr.necks
~~~~~~~~~~~~~~~~~~~~~~~~

Sequence-encoder necks bridging the 4-D backbone feature map and the CTC head.
Each takes a ``(N, C, 1, W)`` feature map and returns a ``(N, W, out_channels)``
sequence.
"""
import torch
import torch.nn as nn

__all__ = ['Im2Seq', 'ReshapeNeck', 'LightSVTRNeck', 'build_neck']


class Im2Seq(nn.Module):
    """Squeeze the (unit) height dimension and turn a feature map into a sequence."""

    def __init__(self, in_channels):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        N, C, H, W = x.shape
        if H != 1:
            raise ValueError(f'Im2Seq expects feature height 1, got {H}')
        return x.squeeze(2).permute(0, 2, 1)  # (N, C, W) -> (N, W, C)


class ReshapeNeck(nn.Module):
    """The trivial ``reshape`` neck (tiny variant): just Im2Seq."""

    def __init__(self, in_channels):
        super().__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = in_channels

    def forward(self, x, mask=None):
        # no global mixing, so the padding mask is unused
        return self.encoder_reshape(x)


class ConvBNLayer(nn.Module):
    """Conv2d (no bias) + BatchNorm2d + activation, as used inside the SVTR neck."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, groups=1, act=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.SiLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    """Standard multi-head self attention (the ``Global`` SVTR mixer)."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            # mask: (B, N) bool, True == valid key; forbid attending to padded keys
            attn = attn.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class SVTRBlock(nn.Module):
    """Post-norm SVTR transformer block (global attention + MLP, SiLU/Swish acts)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, act_layer=nn.SiLU, eps=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.mixer(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class LightSVTRNeck(nn.Module):
    """
    The ``lightsvtr`` neck (small/medium variants).

    A 1×1 channel reduction primed with a depthwise local (1×k) conv, followed
    by a stack of global SVTR blocks and a lightweight 1×1 skip connection from
    the input. Returns a ``(N, W, dims)`` sequence.
    """

    def __init__(self, in_channels, dims=64, depth=2, num_heads=8, qkv_bias=True,
                 mlp_ratio=4.0, drop_rate=0.1, attn_drop_rate=0.1, qk_scale=None,
                 local_kernel=7, use_guide=False):
        super().__init__()
        self.use_guide = use_guide
        self.conv_reduce = ConvBNLayer(in_channels, dims, kernel_size=1, act=nn.SiLU)
        self.local_conv = nn.Sequential(
            nn.Conv2d(dims, dims, (1, local_kernel), padding=(0, local_kernel // 2),
                      groups=dims, bias=False),
            nn.BatchNorm2d(dims),
            nn.SiLU(),
        )
        self.svtr_block = nn.ModuleList([
            SVTRBlock(dim=dims, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                      attn_drop=attn_drop_rate, act_layer=nn.SiLU, eps=1e-5)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dims, eps=1e-6)
        self.skip_conv = ConvBNLayer(in_channels, dims, kernel_size=1, act=nn.SiLU)
        self.out_channels = dims
        self.im2seq = Im2Seq(dims)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # transformer-style init: truncated-normal Linear, Kaiming conv, std LN
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        if self.use_guide:
            x = x.detach()
        skip = self.skip_conv(x)
        z = self.conv_reduce(x)
        if mask is not None:
            # zero padded columns so the depthwise local conv doesn't mix them in
            m = mask[:, None, None, :].to(z.dtype)  # (B, 1, 1, W)
            z = z * m
        z = z + self.local_conv(z)
        if mask is not None:
            z = z * m
        B, C, H, W = z.shape
        z = z.flatten(2).transpose(1, 2)  # (B, H*W, C)
        for blk in self.svtr_block:
            z = blk(z, mask)
        z = self.norm(z)
        z = z.reshape(B, H, W, C).permute(0, 3, 1, 2)
        z = z + skip
        return self.im2seq(z)


def build_neck(neck_cfg, in_channels):
    """Construct a neck from a config dict (see ``MODEL_VARIANTS`` in network.py)."""
    cfg = dict(neck_cfg)
    neck_type = cfg.pop('type')
    if neck_type == 'reshape':
        return ReshapeNeck(in_channels)
    elif neck_type == 'lightsvtr':
        return LightSVTRNeck(in_channels, **cfg)
    raise ValueError(f'unknown neck type {neck_type!r}')
