# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Sequence, Union

import numpy as np
import torch


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
                  seq_lens: Union[torch.Tensor, Sequence[int]]):
    """
    Samples num_mask non-overlapping random masks of length mask_width in
    sequence of length seq_len.

    Args:
        mask_prob: Probability of each individual token being chosen as start
                   of a masked sequence. Overall number of masks num_masks is
                   mask_prob * sum(seq_lens) / mask_width.
        mask_width: width of each mask
        num_neg_samples: Number of samples from unmasked sequence parts (gets
                         multiplied by num_mask)
        seq_lens: sequence lengths

    Returns:
        An index array containing 1 for masked bits, 2 for negative samples,
        the number of masks, and the actual number of negative samples.
    """
    mask_samples = np.zeros(sum(seq_lens))
    num_masks = int(mask_prob * sum(seq_lens.numpy()) // mask_width)
    num_neg_samples = num_masks * num_neg_samples
    num_masks += num_neg_samples

    indices = [x+mask_width for x in positive_integers_with_sum(num_masks, sum(seq_lens)-num_masks*mask_width)]
    start = 0
    mask_slices = []
    for i in indices:
        i_start = random.randint(start, i+start-mask_width)
        mask_slices.append(slice(i_start, i_start+mask_width))
        start += i

    neg_idx = random.sample(range(len(mask_slices)), num_neg_samples)
    neg_slices = [mask_slices.pop(idx) for idx in sorted(neg_idx, reverse=True)]

    mask_samples[np.r_[tuple(mask_slices)]] = 1
    mask_samples[np.r_[tuple(neg_slices)]] = 2

    return mask_samples, num_masks - num_neg_samples, num_neg_samples
