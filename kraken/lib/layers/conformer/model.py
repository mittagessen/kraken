#
# Copyright 2024 Benjamin Kiessling
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
kraken.lib.layers.conformer.model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FasterConformer model
"""
import torch

from torch import nn
from typing import Tuple

from .encoder import ConformerEncoder


class ConformerRecognitionModule(nn.Module):
    def __init__(self,
                 num_classes: int,
                 height: int,
                 encoder_dim: int,
                 num_encoder_layers: int,
                 num_attention_heads: int,
                 feed_forward_expansion_factor: int,
                 conv_expansion_factor: int,
                 input_dropout_p: float,
                 feed_forward_dropout_p: float,
                 attention_dropout_p: float,
                 conv_dropout_p: float,
                 conv_kernel_size: int,
                 half_step_residual: bool,
                 subsampling_conv_channels: int,
                 subsampling_factor: int,
                 **kwargs):
        """
        A nn.Module version of a conformer_ocr.model.RecognitionModel for
        inference.
        """
        super().__init__()
        encoder = ConformerEncoder(in_channels=1,
                                   input_dim=height,
                                   encoder_dim=encoder_dim,
                                   num_layers=num_encoder_layers,
                                   num_attention_heads=num_attention_heads,
                                   feed_forward_expansion_factor=feed_forward_expansion_factor,
                                   conv_expansion_factor=conv_expansion_factor,
                                   input_dropout_p=input_dropout_p,
                                   feed_forward_dropout_p=feed_forward_dropout_p,
                                   attention_dropout_p=attention_dropout_p,
                                   conv_dropout_p=conv_dropout_p,
                                   conv_kernel_size=conv_kernel_size,
                                   half_step_residual=half_step_residual,
                                   subsampling_conv_channels=subsampling_conv_channels,
                                   subsampling_factor=subsampling_factor)
        decoder = nn.Linear(encoder_dim, num_classes, bias=True)
        self.nn = nn.ModuleDict({'encoder': encoder,
                                 'decoder': decoder})

    def forward(self, line: torch.Tensor, lens: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass on a torch tensor of one or more lines with
        shape (N, C, H, W) and returns a numpy array (N, W, C).

        Args:
            line: NCHW line(s) tensor
            lens: N-shape Tensor containing sequence lengths

        Returns:
            Tuple with (N, W, C) shaped numpy array and final output sequence
            lengths.
        """
        if self.device:
            line = line.to(self.device)
        line = line.squeeze(1).transpose(1, 2)
        encoder_outputs, encoder_lens = self.nn.encoder(line, lens)
        logits = self.nn.decoder(encoder_outputs)
        return logits, encoder_lens
