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

import torch
import collections
import numpy as np

from typing import List, Tuple
from scipy.special import logsumexp
from scipy.ndimage import measurements

from itertools import groupby

__all__ = ['beam_decoder', 'greedy_decoder', 'blank_threshold_decoder']


def beam_decoder(outputs: np.ndarray, beam_size: int = 15) -> List[Tuple[int, int, int, float]]:
    """
    Translates back the network output to a label sequence using
    same-prefix-merge beam search decoding as described in [0].

    [0] Hannun, Awni Y., et al. "First-pass large vocabulary continuous speech
    recognition using bi-directional recurrent DNNs." arXiv preprint
    arXiv:1408.2873 (2014).

    Args:
        output: (C, W) shaped softmax output tensor
        beam_size: Size of the beam

    Returns:
        A list with tuples (class, start, end, prob). max is the maximum value
        of the softmax layer in the region.
    """
    try:
        from torchaudio.models.decoder import ctc_decoder as beam_search_decoder
    except ImportError:
        logger.error('Beam search decoding requires torchaudio')
        raise

    outputs = torch.tensor(outputs)
    c, w = outputs.shape

    # construct decoder with dummy tokens
    tokens = ['-'] + [str(idx) for idx in range(c-1)] + ['|']
    decoder = beam_search_decoder(nbest=1, beam_size=beam_size, lexicon=None, tokens=tokens)
    hypotheses = decoder(outputs.T.unsqueeze(0))[0][0]
    dec = []
    for l, timestep in zip(hypotheses.tokens, hypotheses.timesteps):
        # filter out silence token that the implementation *really* wants to
        # put out for some reason.
        if l != len(tokens)-1:
            dec.append((int(l),
                        int(min(timestep, w-1)),
                        int(min(timestep, w-1)),
                        float(outputs[l, min(timestep, w-1)])))
    return dec

def greedy_decoder(outputs: np.ndarray) -> List[Tuple[int, int, int, float]]:
    """
    Translates back the network output to a label sequence using greedy/best
    path decoding as described in [0].

    [0] Graves, Alex, et al. "Connectionist temporal classification: labelling
    unsegmented sequence data with recurrent neural networks." Proceedings of
    the 23rd international conference on Machine learning. ACM, 2006.

    Args:
        output: (C, W) shaped softmax output tensor

    Returns:
        A list with tuples (class, start, end, max). max is the maximum value
        of the softmax layer in the region.
    """
    labels = np.argmax(outputs, 0)
    seq_len = outputs.shape[1]
    mask = np.eye(outputs.shape[0], dtype='bool')[labels].T
    classes = []
    for label, group in groupby(zip(np.arange(seq_len), labels, outputs[mask]), key=lambda x: x[1]):
        lgroup = list(group)
        if label != 0:
            classes.append((label, lgroup[0][0], lgroup[-1][0], max(x[2] for x in lgroup)))
    return classes


def blank_threshold_decoder(outputs: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int, int, float]]:
    """
    Translates back the network output to a label sequence as the original
    ocropy/clstm.

    Thresholds on class 0, then assigns the maximum (non-zero) class to each
    region.

    Args:
        output: (C, W) shaped softmax output tensor
        threshold: Threshold for 0 class when determining possible label
                   locations.

    Returns:
        A list with tuples (class, start, end, max). max is the maximum value
        of the softmax layer in the region.
    """
    outputs = outputs.T
    labels, n = measurements.label(outputs[:, 0] < threshold)
    mask = np.tile(labels.reshape(-1, 1), (1, outputs.shape[1]))
    maxima = measurements.maximum_position(outputs, mask, np.arange(1, np.amax(mask)+1))
    p = 0
    start = None
    x = []
    for idx, val in enumerate(labels):
        if val != 0 and start is None:
            start = idx
            p += 1
        if val == 0 and start is not None:
            if maxima[p-1][1] == 0:
                start = None
            else:
                x.append((maxima[p-1][1], start, idx, outputs[maxima[p-1]]))
                start = None
    # append last non-zero region to list of no zero region occurs after it
    if start:
        x.append((maxima[p-1][1], start, len(outputs), outputs[maxima[p-1]]))
    return [y for y in x if x[0] != 0]
