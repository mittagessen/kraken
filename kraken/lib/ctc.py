"""
CTC for pytorch. Adapted from ocropy/clstm
"""

import torch
from torch.autograd import Function
from torch.nn import Module
from torch.nn.modules.loss import _assert_no_grad


import numpy as np

def log_mul(x,y):
    "Perform multiplication in the log domain (i.e., addition)."
    return x+y

def log_add(x,y):
    "Perform addition in the log domain."
    return np.where(abs(x - y) > 10,
                    np.maximum(x, y),
                    np.log(np.exp(np.clip(x-y, -20, 20))+1) + y)

def forward_algorithm(match,skip=-5.0):
    """Apply the forward algorithm to an array of log state
    correspondence probabilities."""
    v = skip * np.arange(len(match[0]))
    result = []
    # This is a fairly straightforward dynamic programming problem and
    # implemented in close analogy to the edit distance:
    # we either stay in the same state at no extra cost or make a diagonal
    # step (transition into new state) at no extra cost; the only costs come
    # from how well the symbols match the network output.
    for i in range(0, len(match)):
        w = np.roll(v, 1).copy()
        # extra cost for skipping initial symbols
        w[0] = skip * i
        # total cost is match cost of staying in same state
        # plus match cost of making a transition into the next state
        v = log_add(log_mul(v, match[i]), log_mul(w, match[i]))
        result.append(v)
    return np.array(result, 'f')

def ctc_align_targets(outputs, targets, lo=1e-5):
    """Perform alignment between the `outputs` of a neural network
    classifier and some targets. The targets themselves are a time sequence
    of vectors, usually a unary representation of each target class (but
    possibly sequences of arbitrary posterior probability distributions
    represented as vectors)."""

    outputs = np.maximum(lo, outputs)
    outputs = outputs * 1.0/np.sum(outputs, axis=1)[:,np.newaxis]

    # first, we compute the match between the outputs and the targets
    # and put the result in the log domain
    match = np.dot(outputs, targets.T)
    lmatch = np.log(match)

    # Now, we compute a forward-backward algorithm over the matches between
    # the input and the output states.
    lr = forward_algorithm(lmatch)
    # backward is just forward applied to the reversed sequence
    rl = forward_algorithm(lmatch[::-1,::-1])[::-1,::-1]
    both = lr + rl
    # We need posterior probabilities for the states, so we need to normalize
    # the output. Instead of keeping track of the normalization
    # factors, we just normalize the posterior distribution directly.
    epath = np.exp(both - np.amax(both))
    l = np.sum(epath, axis=0)[np.newaxis,:]
    epath /= np.where(l==0.0,1e-9,l)
    # The previous computation gives us an alignment between input time
    # and output sequence position as posteriors over states.
    # However, we actually want the posterior probability distribution over
    # output classes at each time step. This dot product gives
    # us that result. We renormalize again afterwards.
    aligned = np.maximum(lo, np.dot(epath, targets))
    l = np.sum(aligned, axis=1)[:,np.newaxis]
    aligned /= np.where(l==0.0,1e-9,l)
    return -log_add(lr[-1,-1], lr[-1,-2]), outputs-aligned

class _CTC(Function):
    def forward(self, inputs, targets, size_average=True, reduce=True):
        targets = make_targets(targets, inputs.size()[2])
        self.grads = torch.zeros(inputs.size()).type_as(inputs)
        loss = torch.FloatTensor(inputs.size()[1])

        for idx, (input, target) in enumerate(zip(inputs.split(1, dim=1), targets.split(1, dim=1))):
            l, g = ctc_align_targets(input.squeeze().numpy(), target.squeeze().numpy())
            loss[idx] = float(l)
            self.grads[:,idx,:] = torch.FloatTensor(g)

        if reduce:
            loss = torch.FloatTensor([torch.sum(loss)])
            self.grads = torch.sum(self.grads, 1)
            if size_average:
                loss /= inputs.size()[1]
                self.grads /= inputs.size()[1]
        return loss

    def backward(self, grad_output):
        return self.grads, None, None, None


class CTCCriterion(Module):
    r"""
    Python CTC loss pulled from ocropy.

    The targets are label sequences. The implementation will convert them to
    one-hot representation and pad them with blank labels.  Gradients are NOT
    computed with regard to the output gradients, so this implementation has to
    be used as the final layer in a network.

    Inputs can be either from a softmax layer or direct linear projections.

    Args:
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However if the field size_average
            is set to False, the losses are instead summed for each minibatch.
            Ignored when reduce is False. Default:True
        reduce (bool, optional): By default, the losses are averaged or summed
            for each minibatch. When reduce is False, the loss function returns
            a loss per batch element instead and ignores size_average. Default:
            True

    Shape:
        - Input: :math:`(S, N, C)` where `C` number of classes, `S` sequence
          length, and `N` number of batches.
        - Target: :math:`(N, l)`, `N` number of label sequences `l`.
        - Output: scalar. If reduce is False, then :math:`(N)`
    """
    def __init__(self, size_average=True, reduce=True):
        super(CTCCriterion, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, targets):
        """
        CTC
        """
        _assert_no_grad(targets)
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {} dimensions)'.format(input.dim()))
        if targets.dim() != 2:
            raise ValueError('expected 2D targets (got {} dimensions)'.format(targets.dim()))
        if len(targets) > len(input):
            raise ValueError('target label sequence ({}) has to be shorter than input sequence ({})'.format(len(targets), len(input)))
        return _CTC()(input, targets)

def make_targets(labels, l_num):
    """
    Produces a CTC target tensor from a list of labels and the dimension of an
    output layer.

    Args:
        labels (torch.LongTensor): Input label sequences (seqlen, batch)
        l_num (int): dimension of the output layer
    """
    # pad label sequence with blanks
    bl = np.zeros(labels.size())
    l = np.hstack([bl, labels.numpy()]).reshape((labels.size()[0]*2,) + labels.size()[1:])
    l = np.concatenate((l, np.zeros((1, labels.size()[1]))))
    labels = torch.LongTensor(l.astype('int'))

    # one hot encode padded label sequence
    onehot = torch.FloatTensor(labels.size() + (l_num,))
    labels.unsqueeze_(2)
    onehot.zero_()
    onehot.scatter_(2, labels, 1)
    return onehot
