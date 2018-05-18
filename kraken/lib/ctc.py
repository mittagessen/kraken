# -*- coding: utf-8 -*-
#
# Copyright 2018 Benjamin Kiessling
# Copyright 2015 Preferred Infrastructure, Inc.
# Copyright 2015 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
kraken.lib.layers
~~~~~~~~~~~~~~~~~

Various pytorch layers compatible with the dimension ordering and
inputs/outputs of VGSL-defined networks.
"""

import numpy as np

import torch
import torch.nn.functional as F

from torch.nn import Module
from torch.autograd import Function

__all__ = ['CTCCriterion']


# ~33% faster than scipy implementation with checks
def _logsumexp(a, axis=None):
    vmax = np.amax(a, axis=axis, keepdims=True)
    vmax += np.log(np.sum(np.exp(a - vmax), axis=axis,
                          keepdims=True, dtype=a.dtype))
    return np.squeeze(vmax, axis=axis)


def _label_to_path(labels, blank_symbol):
    path = np.full((len(labels), labels.shape[1]*2+1),
                   blank_symbol, dtype=np.int32)
    path[:, 1::2] = labels
    return path


def _flip_path(path, path_length):
    """Flips label sequence.

    This function rotates a label sequence and flips it.
    ``path[b, t]`` stores a label at time ``t`` in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[b, t] = path[b, t + path_length[b]]``

    .. ::

       a b c d .     . a b c d    d c b a .
       e f . . .  -> . . . e f -> f e . . .
       g h i j k     g h i j k    k j i h g

    """
    n_batch, n_label = path.shape
    rotate = (np.arange(n_label) + path_length[:, None]) % n_label
    return path[np.arange(n_batch, dtype='i')[:, None],
                rotate][:, ::-1]


def _flip_label_probability(y, input_length):
    """Flips a label probability matrix.

    This function rotates a label probability matrix and flips it.
    ``y[i, b, l]`` stores log probability of label ``l`` at ``i``-th
    input in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, b, l] = y[i + input_length[b], b, l]``

    """
    seq, n_batch, n_vocab = y.shape
    rotate = (np.arange(seq, dtype='i')[:, None] + input_length) % seq
    return y[
        rotate[:, :, None],
        np.arange(n_batch, dtype='i')[None, :, None],
        np.arange(n_vocab, dtype='i')[None, None, :]][::-1]


def _flip_path_probability(prob, input_length, path_length):
    """Flips a path probability matrix.

    This function returns a path probability matrix and flips it.
    ``prob[i, b, t]`` stores log probability at ``i``-th input and
    at time ``t`` in a output sequence in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, j, k] = prob[i + input_length[j], j, k + path_length[j]]``

    """
    seq, n_batch, n_label = prob.shape
    rotate_input = (np.arange(seq, dtype='i')[:, None] + input_length) % seq
    rotate_label = (
        np.arange(n_label, dtype='i') + path_length[:, None]) % n_label
    return prob[
        rotate_input[:, :, None],
        np.arange(n_batch, dtype='i')[None, :, None],
        rotate_label][::-1, :, ::-1]


class _CTC(Function):
    """
    Implementation of Connectionist Temporal Classification loss.
    """

    def __init__(self, size_average=True, reduce=True):
        self.blank_symbol = 0
        self.zero_padding = -10000000000.0

        self.size_average = size_average
        self.reduce = reduce

    def log_matrix(self, x):
        res = np.ma.log(x).filled(fill_value=self.zero_padding)
        return res.astype(np.float32)

    def label_probability(self, label_size, path, path_length,
                          multiply_seq):
        """
        Converts path probability to label probability.
        """
        seq_length = len(multiply_seq)
        n_batch = len(path)
        dtype = multiply_seq.dtype

        ret = np.zeros((seq_length, n_batch, label_size), dtype)
        for b in range(len(path)):
            target_path = path[b, :path_length[b]]
            chars = {c for c in target_path}
            for c in chars:
                ret[:, b, c] = np.sum(
                    multiply_seq[:, b, 0:path_length[b]]
                    [:, target_path == c], axis=1)
        return ret

    def _computes_transition(self, prev_prob, path, path_length, cum_prob, y):
        n_batch, max_path_length = path.shape
        mat = np.full(
            (3, n_batch, max_path_length), self.zero_padding, 'f')
        mat[0, :, :] = prev_prob
        mat[1, :, 1:] = prev_prob[:, :-1]
        mat[2, :, 2:] = prev_prob[:, :-2]
        # disable transition between the same symbols
        # (including blank-to-blank)
        same_transition = (path[:, :-2] == path[:, 2:])
        mat[2, :, 2:][same_transition] = self.zero_padding
        prob = _logsumexp(mat, axis=0)
        outside = np.arange(max_path_length) >= path_length[:, None]
        prob[outside] = self.zero_padding
        cum_prob += prob
        batch_index = np.arange(n_batch, dtype='i')
        prob += y[batch_index[:, None], path]
        return prob

    def calc_trans(self, yseq, input_length,
                   label, path, path_length):
        max_input_length, n_batch, n_unit = yseq.shape
        max_label_length = label.shape[1]
        max_path_length = path.shape[1]
        assert label.shape == (n_batch, max_label_length), (label.shape, n_batch)
        assert path.shape == (n_batch, max_label_length * 2 + 1)

        forward_prob = np.full(
            (n_batch, max_path_length), self.zero_padding, dtype='f')
        forward_prob[:, 0] = 0
        backward_prob = forward_prob

        batch_index = np.arange(n_batch, dtype='i')
        seq_index = np.arange(len(yseq), dtype='i')
        prob = yseq[seq_index[:, None, None], batch_index[:, None], path]
        # forward computation.
        for i, y in enumerate(yseq):
            forward_prob = self._computes_transition(
                forward_prob, path, path_length, prob[i], y)

        r_path = _flip_path(path, path_length)

        yseq_inv = _flip_label_probability(yseq, input_length)
        prob = _flip_path_probability(prob, input_length, path_length)

        for i, y_inv in enumerate(yseq_inv):
            backward_prob = self._computes_transition(
                backward_prob, r_path, path_length, prob[i], y_inv)

        return _flip_path_probability(prob, input_length, path_length)

    def forward(self, xs, t):
        t = t.numpy()
        # permute to (seq, batch, feat)
        xs = xs.permute(2, 0, 1).detach()
        self.yseq = F.softmax(xs, dim=2).numpy()
        xs = xs.numpy()
        self.input_length = np.full(len(xs[0]), len(xs), dtype=np.int32)
        self.batch_size = len(xs[0])
        label_length = np.full(len(t), t.shape[1], dtype=np.int32)
        self.path_length = 2 * label_length + 1

        log_yseq = self.log_matrix(self.yseq)
        self.path = _label_to_path(t, self.blank_symbol)
        self.prob_trans = self.calc_trans(log_yseq, self.input_length, t,
                                          self.path, self.path_length)
        loss = -_logsumexp(self.prob_trans[0], axis=1)
        return torch.tensor(loss)

    def backward(self, grad_output):
        total_probability = _logsumexp(self.prob_trans[0], axis=1)
        label_prob = self.label_probability(self.yseq.shape[2],
                                            self.path,
                                            self.path_length,
                                            np.exp(self.prob_trans - total_probability[:, None]))
        self.yseq -= label_prob
        # mask
        self.yseq *= (np.arange(len(self.yseq))[:, None] < self.input_length)[..., None]
        return torch.tensor(self.yseq).permute(1, 2, 0), None


class CTCCriterion(Module):
    r"""
    Connectionist Temporal Classification loss function.

    This class performs the softmax operation (increases numerical stability)
    for you, so inputs should be unnormalized linear projections from an RNN.
    For the same reason, forward-backward computations are performed in log
    domain.

    Shape:
        - Input: :math:`(N, C, S)` where `C` number of classes, `S` sequence
          length, and `N` number of batches.
        - Target: :math:`(N, l)`, `N` number of label sequences `l`.
        - Output: scalar. If reduce is False, then :math:`(N)`
    """
    def __init__(self):
        super(CTCCriterion, self).__init__()

    def forward(self, input, targets):
        return _CTC()(input, targets)
