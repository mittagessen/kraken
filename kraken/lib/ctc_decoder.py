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

# -*- coding: utf-8 -*-
"""
Decoders for softmax outputs of CTC trained networks.
"""

import numpy as np

from scipy.ndimage import measurements

__all__ = ['beam_decoder', 'greedy_decoder', 'blank_threshold_decoder']

def beam_decoder(outputs):
    """
    Translates back the network output to a label sequence using beam search
    decoding as described in [0].

    Thresholds on class 0, then assigns the maximum (non-zero) class to each
    region.

    [0] Graves, Alex, et al. "Connectionist temporal classification: labelling
    unsegmented sequence data with recurrent neural networks." Proceedings of
    the 23rd international conference on Machine learning. ACM, 2006.

    Args:
        output (numpy.array): (C, W) shaped softmax output tensor

    Returns:
        A list with tuples (class, start, end, max). max is the maximum value
        of the softmax layer in the region.
    """
    pass

def greedy_decoder(outputs):
    """
    Translates back the network output to a label sequence using greedy/best
    path decoding as described in [0]

    Thresholds on class 0, then assigns the maximum (non-zero) class to each
    region.

    [0] Graves, Alex, et al. "Connectionist temporal classification: labelling
    unsegmented sequence data with recurrent neural networks." Proceedings of
    the 23rd international conference on Machine learning. ACM, 2006.

    Args:
        output (numpy.array): (C, W) shaped softmax output tensor

    Returns:
        A list with tuples (class, start, end, max). max is the maximum value
        of the softmax layer in the region.
    """
    m = np.argmax(outputs, 0)
    start = None
    x = []
    for i in range(len(m)):
        if start is None and dec[i] != 0 and (not (i > 0 and dec[i-1] == dec[i])):
            start = i
            code = dec[i]
        if start is not None and (dec[i-1] != dec[i]):
            x.append((code, start, i , val[start:i+1].max().data[-1]))
            start = None
    return x

def blank_threshold_decoder(outputs, threshold=0.5):
    """
    Translates back the network output to a label sequence as the original
    ocropy/clstm.

    Thresholds on class 0, then assigns the maximum (non-zero) class to each
    region.

    Args:
        output (numpy.array): (C, W) shaped softmax output tensor
        threshold (float): Threshold for 0 class when determining possible
                           label locations.

    Returns:
        A list with tuples (class, start, end, max). max is the maximum value
        of the softmax layer in the region.
    """
    outputs = outputs.T
    labels, n = measurements.label(outputs[:,0] < threshold)
    mask = np.tile(labels.reshape(-1,1), (1,outputs.shape[1]))
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
    return x

