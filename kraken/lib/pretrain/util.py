# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Union, Tuple

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


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def sample_negatives(y, num_samples, num_neg_samples: int):
    B, W, C = y.shape
    y = y.view(-1, C)  # BTC => (BxT)C

    with torch.no_grad():
        tszs = (buffered_arange(num_samples).unsqueeze(-1).expand(-1, num_neg_samples).flatten())

        neg_idxs = torch.randint(low=0, high=W - 1, size=(B, num_neg_samples * num_samples))
        neg_idxs[neg_idxs >= tszs] += 1

    for i in range(1, B):
        neg_idxs[i] += i * W

    negs = y[neg_idxs.view(-1)]
    negs = negs.view(B, num_samples, num_neg_samples, C).permute(2, 0, 1, 3)  # to NxBxTxC
    return negs


def compute_mask_indices(shape: Tuple[int, int], mask_prob: float, mask_length: int = 4, mask_min_space: int = 2) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller.
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(mask_prob * all_sz / float(mask_length) + np.random.rand())

    mask_idcs = []
    for i in range(bsz):
        # import ipdb; ipdb.set_trace()
        sz = all_sz
        num_mask = all_num_mask

        lengths = np.full(num_mask, mask_length)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        mask_idc = []

        def arrange(s, e, length, keep_length):
            span_start = np.random.randint(s, e - length)
            mask_idc.extend(span_start + i for i in range(length))

            new_parts = []
            if span_start - s - mask_min_space >= keep_length:
                new_parts.append((s, span_start - mask_min_space + 1))
            if e - span_start - keep_length - mask_min_space > keep_length:
                new_parts.append((span_start + length + mask_min_space, e))
            return new_parts

        parts = [(0, sz)]
        min_length = min(lengths)
        for length in sorted(lengths, reverse=True):
            lens = np.fromiter(
                (e - s if e - s >= length + mask_min_space else 0 for s, e in parts),
                np.int,
            )
            l_sum = np.sum(lens)
            if l_sum == 0:
                break
            probs = lens / np.sum(lens)
            c = np.random.choice(len(parts), p=probs)
            s, e = parts.pop(c)
            parts.extend(arrange(s, e, length, min_length))
        mask_idc = np.asarray(mask_idc)

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    # make sure all masks are the same length in the batch by removing masks
    # if they are greater than the min length mask
    min_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        assert len(mask_idc) == min_len
        mask[i, mask_idc] = True

    return mask
