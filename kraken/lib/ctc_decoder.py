#
# Copyright 2017 Benjamin Kiessling
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
Decoders for softmax outputs of CTC trained networks.

Decoders extract label sequences out of the raw output matrix of the line
recognition network. There are multiple different approaches implemented here,
from a simple greedy decoder, to the legacy ocropy thresholding decoder, and a
more complex beam search decoder.

Extracted label sequences are converted into the code point domain using kraken.lib.codec.PytorchCodec.
"""

import collections
from itertools import groupby

import torch
import numpy as np
from scipy.ndimage import measurements
from scipy.special import logsumexp
from typing import Union

__all__ = ['greedy_decoder']


def greedy_decoder(outputs: Union[torch.Tensor, np.ndarray],
                   seq_lens: torch.Tensor = None) -> list[list[tuple[int, int, int, float]]]:
    """
    Translates back the network output to a label sequence using greedy/best
    path decoding as described in [0].

    [0] Graves, Alex, et al. "Connectionist temporal classification: labelling
    unsegmented sequence data with recurrent neural networks." Proceedings of
    the 23rd international conference on Machine learning. ACM, 2006.

    Args:
        output: (C, W) or (N, C, W) shaped softmax output tensor
        seq_lens: Sequence lengths in batch. Can be left unset if `outputs` has
                  batch size 1.

    Returns:
        A list of lists with tuples (class, start, end, max). max is the
        maximum value of the time steps corresponding to a single decoded
        label.
    """
    if isinstance(outputs, np.ndarray):
        outputs = torch.from_numpy(outputs)
    if outputs.dim() == 2:
        outputs = outputs.unsqueeze(0)
    if outputs.shape[0] == 1 and seq_lens is None:
        seq_lens = torch.tensor([outputs.shape[-1]])
    elif seq_lens is None:
        raise ValueError(f'seq_lens need to be set for batch decoding.')

    dec = []
    for seq, seq_len in zip(outputs, seq_lens):
        confs, labels = seq[..., :seq_len].max(dim=0)
        classes = []
        for label, group in groupby(zip(range(seq_len), labels.tolist(), confs.tolist()), key=lambda x: x[1]):
            lgroup = list(group)
            if label != 0:
                classes.append((label, lgroup[0][0], lgroup[-1][0], max(x[2] for x in lgroup)))
        dec.append(classes)
    return dec
