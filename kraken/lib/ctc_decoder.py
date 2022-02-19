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
import numpy as np

from typing import List, Tuple
from scipy.special import logsumexp
from scipy.ndimage import measurements

from itertools import groupby

__all__ = ['beam_decoder', 'greedy_decoder', 'blank_threshold_decoder']


def beam_decoder(outputs: np.ndarray, beam_size: int = 3) -> List[Tuple[int, int, int, float]]:
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
    c, w = outputs.shape
    probs = np.log(outputs)
    beam = [(tuple(), (0.0, float('-inf')))]  # type: List[Tuple[Tuple, Tuple[float, float]]]

    # loop over each time step
    for t in range(w):
        next_beam = collections.defaultdict(lambda: 2*(float('-inf'),))  # type: dict
        # p_b -> prob for prefix ending in blank
        # p_nb -> prob for prefix not ending in blank
        for prefix, (p_b, p_nb) in beam:
            # only update ending-in-blank-prefix probability for blank
            n_p_b, n_p_nb = next_beam[prefix]
            n_p_b = logsumexp((n_p_b, p_b + probs[0, t], p_nb + probs[0, t]))
            next_beam[prefix] = (n_p_b, n_p_nb)
            # loop over non-blank classes
            for s in range(1, c):
                # only update the not-ending-in-blank-prefix probability for prefix+s
                l_end = prefix[-1][0] if prefix else None
                n_prefix = prefix + ((s, t, t),)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s == l_end:
                    # substitute the previous non-blank-ending-prefix
                    # probability for repeated labels
                    n_p_nb = logsumexp((n_p_nb, p_b + probs[s, t]))
                else:
                    n_p_nb = logsumexp((n_p_nb, p_b + probs[s, t], p_nb + probs[s, t]))

                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == l_end:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp((n_p_nb, p_nb + probs[s, t]))
                    # rewrite both new and old prefix positions
                    next_beam[prefix[:-1] + ((prefix[-1][0], prefix[-1][1], t),)] = (n_p_b, n_p_nb)
                    next_beam[n_prefix[:-1] + ((n_prefix[-1][0], n_prefix[-1][1], t),)] = next_beam.pop(n_prefix)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                      key=lambda x: logsumexp(x[1]),
                      reverse=True)
        beam = beam[:beam_size]
    return [(c, start, end, max(outputs[c, start:end+1])) for (c, start, end) in beam[0][0]]


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
