# Copyright (c) 2024, Benjamin Kiessling
# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .modules import Transpose


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout

    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            nn.GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)


class Conv2dSubsampling(nn.Module):
    """
    Depthwise convolutional subsampling with variable subsampling factor.

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        in_feats: Number of features in the height dimension of the input image
        conv_channels: Channels of convolutio filter(s).
        input_dropout_p: Dropout probability after final linear layer.
        subsampling_factor: The subsampling factor which should be a power of 2

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 in_feats: int = 80,
                 conv_channels: int = 256,
                 input_dropout_p: float = 0.1,
                 subsampling_factor: int = 8) -> None:
        super(Conv2dSubsampling, self).__init__()

        self._conv_channels = conv_channels
        self._in_channels = in_channels
        self._in_feats = in_feats
        self._input_dropout_p = input_dropout_p

        if not math.log(subsampling_factor, 2).is_integer():
            raise ValueError('Sampling factor should be a power of 2.')
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor

        self._stride = 2
        self._kernel_size = 3

        self._padding = (self._kernel_size - 1) // 2

        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=conv_channels,
                                kernel_size=self._kernel_size,
                                stride=self._stride,
                                padding=self._padding))
        in_channels = conv_channels
        layers.append(nn.ReLU(True))
        for i in range(self._sampling_num - 1):
            layers.append(nn.Conv2d(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=self._kernel_size,
                          stride=self._stride,
                          padding=self._padding,
                          groups=in_channels))

            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=conv_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1))
            layers.append(nn.ReLU(True))
            in_channels = conv_channels

        self.conv = nn.Sequential(*layers)
        in_length = torch.tensor(in_feats, dtype=torch.float)
        out_length = calc_length(lengths=in_length,
                                 all_paddings=2*self._padding,
                                 kernel_size=self._kernel_size,
                                 stride=self._stride,
                                 repeat_num=self._sampling_num)

        self.out = nn.Sequential(nn.Linear(conv_channels * int(out_length), out_channels),
                                 nn.Dropout(p=input_dropout_p))

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        output_lengths = calc_length(input_lengths,
                                     all_paddings=2*self._padding,
                                     kernel_size=self._kernel_size,
                                     stride=self._stride,
                                     repeat_num=self._sampling_num)
        x = inputs.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, output_lengths


def calc_length(lengths, all_paddings, kernel_size, stride, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)
