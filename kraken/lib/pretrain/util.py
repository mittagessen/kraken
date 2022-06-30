# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Optional, Tuple, Sequence, Union

import torch
import random
import numpy as np

def positive_integers_with_sum(n, total):
    ls = [0]
    rv = []
    while len(ls) < n:
        c = random.randint(0, total)
        ls.append(c)
    ls = sorted(ls)
    ls.append(total)
    for i in range(1, len(ls)):
        rv.append(ls[i] - ls[i-1])
    return rv

def compute_masks(mask_prob: int,
                  mask_width: int,
                  num_neg_samples: int,
                  seq_lens: Union[torch.Tensor, Sequence[int]],
                  max_len: int):
    """
    Samples num_mask non-overlapping random masks of length mask_width in
    sequence of length seq_len.

    Args:
        mask_prob: Probability of each individual token being chosen as start
                   of a masked sequence. Overall number of masks is mask_prob *
                   sum(seq_lens) / mask_width.
        mask_width: width of each mask
        num_neg_samples: Number of samples in unmasked sequence parts.
        seq_len: sequence length

    Returns
    """
    mask_samples = np.zeros(max_len * len(seq_lens))
    neg_samples = []
    num_masks = int(mask_prob * sum(seq_lens.numpy()) // mask_width + num_neg_samples)
    for idx, seq_len in enumerate(seq_lens):
        base_idx = idx * max_len
        indices = [x+mask_width for x in positive_integers_with_sum(num_masks, (seq_len)-num_masks*mask_width)]
        start = 0
        sample_mask = []
        for i in indices:
            i_start = random.randint(start, i+start-mask_width)
            sample_mask.append(slice(base_idx+i_start, base_idx+i_start+mask_width))
            start+=i
        neg_idx = random.sample(range(len(sample_mask)), num_neg_samples)
        sample_neg = [sample_mask.pop(idx) for idx in sorted(neg_idx, reverse=True)]
        mask_samples[np.r_[tuple(sample_mask)]] = 1
        mask_samples[np.r_[tuple(sample_neg)]] = 2

    return mask_samples, num_masks - num_neg_samples
