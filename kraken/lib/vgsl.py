"""
VGSL plumbing
"""

import re
import warnings
import torch
import numpy as np
import torch.nn.functional as F


from torch.nn import Module

from torch.autograd import Variable
from kraken.lib.ctc import CTCCriterion
from torch.nn.modules.loss import _assert_no_grad

# all tensors are ordered NCHW, the "feature" dimension is C, so the output of
# an LSTM will be put into C same as the filters of a CNN.

__all__ = ['TorchVGSLModel']

class TransposedSummarizingRNN(Module):
    """
    An RNN wrapper allowing time axis transpositions and other
    """
    def __init__(self, input_size, hidden_size, type='l', direction='b', transpose=True, summarize=True):
        """
        A wrapper around torch.nn.LSTM/GRU optionally transposing inputs and
        returning only the last column of output.

        Args:
            input_size:
            hidden_size:
            type (str):
            direction (str):
            transpose (bool):
            summarize (bool):

        Shape:
            - Inputs: :math:`(N, C, H, W)` where `N` batches, `C` channels, `H`
              height, and `W` width.
            - Outputs output :math:`(N, hidden_size * num_directions, H, S)`
              with S (or H) being 1 if summarize (and transpose) are true
        """
        super(TransposedSummarizingRNN, self).__init__()
        self.transpose = transpose
        self.summarize = summarize
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidi = direction == 'b'
        self.output_size = hidden_size if not self.bidi else 2*hidden_size

        l = torch.nn.GRU if type == 'g' else torch.nn.LSTM

        self.layer = l(input_size, hidden_size, bidirectional=self.bidi, batch_first=True)

    def forward(self, inputs):
        # NCHW -> HNWC
        inputs = inputs.permute(2, 0, 3, 1)
        if self.transpose:
            # HNWC -> WNHC
            inputs = inputs.transpose(0, 2)
        # HNWC -> (H*N)WC
        siz = inputs.size()
        inputs = inputs.contiguous().view(-1, siz[2], siz[3])
        # (H*N)WO
        o, _ = self.layer(inputs, self.init_hidden(inputs.size(0)))
        # resize to HNWO
        o = o.resize(siz[0], siz[1], siz[2], self.output_size)
        if self.summarize:
            # HN1O
            o = o[:,:,-1,:].unsqueeze(2)
        if self.transpose:
            o = o.transpose(0, 2)
        # HNWO -> NOHW
        return o.permute(1, 3, 0, 2)

    def init_hidden(self, bsz=1):
        return (Variable(torch.zeros(2 if self.bidi else 1, bsz, self.hidden_size)),
                Variable(torch.zeros(2 if self.bidi else 1, bsz, self.hidden_size)))

    def get_shape(self, input):
        """
        Calculates the output shape from input 4D tuple (batch, channel, input_size, seq_len).
        """
        if self.summarize:
            if self.transpose:
                l = (1, input[3])
            else:
                l = (input[2], 1)
        else:
            l = (input[2], input[3])
        return (input[0], self.output_size) + l


class LinSoftmax(Module):
    """
    A wrapper for linear projection + softmax dealing with dimensionality mangling.
    """
    def __init__(self, input_size, output_size):
        """

        Args:
            input_size: Number of inputs in the feature dimension
            output_size: Number of outputs in the feature dimension

        Shape:
            - Inputs: :math:`(N, C, H, W)` where `N` batches, `C` channels, `H`
              height, and `W` width.
            - Outputs output :math:`(N, output_size, H, S)`
        """
        super(LinSoftmax, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.lin = torch.nn.Linear(input_size, output_size, bias=False)

    def forward(self, inputs):
        # move features (C) to last dimension for linear activation
        o = F.softmax(self.lin(inputs.transpose(1, 3)), dim=3)
        # and swap again
        return o.transpose(3,1)

    def get_shape(self, input):
        """
        Calculates the output shape from input 4D tuple NCHW.
        """
        return (input[0], self.output_size, input[2], input[3])


class ActConv2D(Module):
    """
    A wrapper for convolution + activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, nl='l'):
        super(ActConv2D, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.nl = None
        if nl == 's':
            self.nl = F.sigmoid
        elif nl == 't':
            self.nl = F.tanh
        elif nl == 'm':
            self.nl = F.softmax
        elif nl == 'r':
            self.nl = F.relu

        self.co = torch.nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, inputs):
        return self.nl(self.co(inputs))

    def get_shape(self, input):
        return (input[0],
                self.out_channels,
                int(np.floor((input[2]-(self.kernel_size[0]-1)-1)+1) if input[2] != 0 else 0),
                int(np.floor((input[3]-(self.kernel_size[1]-1)-1)+1) if input[3] != 0 else 0))


class TorchVGSLModel(object):
    """
    Class building a torch module from a VSGL spec.

    The initialized class will contain a variable number of layers and a loss
    function. Inputs and outputs are always 4D tensors in order (batch,
    channels, height, width) with channels always being the feature dimension.

    Importantly this means that a recurrent network will be fed the channel
    vector at each step along its time axis, i.e. either put the non-time-axis
    dimension into the channels dimension or use a summarizing RNN squashing
    the time axis to 1 and putting the output into the channels dimension
    respectively.

    Attributes:
        input (tuple): Expected input tensor as a 4-tuple.
        nn (torch.nn.Sequential): Stack of layers parsed from the spec.
        criterion (torch.nn.Module): Fully parametrized loss function.

    """

    def __init__(self, spec):
        """
        Constructs a torch module from a (subset of) VSGL spec.

        Args:
            spec (str): Model definition similar to tesseract as follows:
                        ============ FUNCTIONAL OPS ============
                        C(s|t|r|l|m)[{name}]<y>,<x>,<d> Convolves using a y,x window, with no
                          shrinkage, SAME infill, d outputs, with s|t|r|l|m non-linear layer.
                          (s|t|r|l|m) specifies the type of non-linearity:
                          s = sigmoid
                          t = tanh
                          r = relu
                          l = linear (i.e., None)
                          m = softmax
                        F(s|t|r|l|m)[{name}]<d> Fully-connected with s|t|r|l|m non-linearity and
                          d outputs. Reduces height, width to 1. Input height and width must be
                          constant.
                        L(f|r|b)(x|y)[s][{name}]<n> LSTM cell with n outputs.
                          f runs the LSTM forward only.
                          r runs the LSTM reversed only.
                          b runs the LSTM bidirectionally.
                          x runs the LSTM in the x-dimension (on data with or without the
                             y-dimension).
                          y runs the LSTM in the y-dimension (data must have a y dimension).
                          s (optional) summarizes the output in the requested dimension,
                             outputting only the final step, collapsing the dimension to a
                             single element.
                          Examples:
                          Lfx128 runs a forward-only LSTM in the x-dimension with 128
                                 outputs, treating any y dimension independently.
                          Lfys64 runs a forward-only LSTM in the y-dimension with 64 outputs
                                 and collapses the y-dimension to 1 element.
                        G(f|r|b)(x|y)[s][{name}]<n> GRU cell with n outputs.
                          Arguments are equivalent to LSTM specs.
                        Do[{name}] Insert a 1D dropout layer with 0.5 drop probability.
                        ============ PLUMBING OPS ============
                        [...] Execute ... networks in series (layers).
                        Mp[{name}]<y>,<x>[y_stride][x_stride] Maxpool the input, reducing the (y,x) rectangle to a
                          single vector value.


        """
        self.spec = spec
        self.ops = [self.build_rnn, self.build_dropout, self.build_maxpool, self.build_conv, self.build_output]
        self.criterion = None

        #
        self.idx = 0
        spec = spec.strip()
        if spec[0] != '[' or spec[-1] != ']':
            raise ValueError('Non-sequential models not supported')
        spec = spec[1:-1]
        blocks = spec.split(' ')
        pattern = re.compile(r'(\d+),(\d+),(\d+),(\d+)')
        m = pattern.match(blocks.pop(0))
        if not m:
            raise ValueError('Invalid input spec.')
        batch, height, width, channels = [int(x) for x in m.groups()]
        input = [batch, channels, height, width]
        self.input = tuple(input)
        self.nn = torch.nn.Sequential()
        for block in blocks:
            oshape = None
            layer = None
            for op in self.ops:
                oshape, name, layer = op(input, block)
                if oshape:
                    break
            if oshape:
                input = oshape
                self.nn.add_module(name, layer)
            else:
                raise ValueError('{} invalid layer definition'.format(block))

    def cuda(self):
        self.nn.cuda()
        if self.criterion:
            self.criterion.cuda()

    def save_model(self, path):
        """
        Serializes the model into path.

        Args:
            path (unicode): Target destination
        """
        pass

    def init_weights(self):
        """
        Initializes weights for all layers of the graph.

        LSTM/GRU layers are orthogonally initialized, convolutional layers
        uniformly from (-0.1,0.1).
        """
        def _wi(m):
            if isinstance(m, torch.nn.Linear):
                m.weight.data.fill_(1.0)
            elif isinstance(m, torch.nn.LSTM):
                for p in m.parameters():
                    # weights
                    if p.data.dim() == 2:
                        torch.nn.init.orthogonal(p.data)
                    # initialize biases to 1 (jozefowicz 2015)
                    else:
                        p.data[len(p)//4:len(p)//2].fill_(1.0)
            elif isinstance(m, torch.nn.GRU):
                for p in m.parameters():
                    torch.nn.init.orthogonal(p.data)
            elif isinstance(m, torch.nn.Conv2d):
                for p in m.parameters():
                    torch.nn.init.uniform(p.data, -0.1, 0.1)
        self.nn.apply(_wi)

    def get_layer_name(self, layer, name=None):
        """
        Generates a unique identifier for the layer optionally using a supplied
        name.

        Args:
            layer (str): Identifier of the layer type
            name (str): user-supplied {name} with {} that need removing.

        Returns:
            (str) network unique layer name
        """
        if name:
            return name[1:-1]
        else:
            return '{}_{}'.format(re.sub(r'\W+', '_', layer), self.idx)

    def build_rnn(self, input, block):
        """
        Builds an LSTM/GRU layer returning number of outputs and layer.
        """
        pattern = re.compile(r'(?P<type>L|G)(?P<dir>f|r|b)(?P<dim>x|y)(?P<sum>s)?(?P<name>{\w+})?(?P<out>\d+)')
        m = pattern.match(block)
        if not m:
            return None, None, None
        type = m.group(1)
        direction = m.group(2)
        dim = m.group(3)  == 'y'
        summarize = m.group(4) == 's'
        hidden = int(m.group(6))
        l = TransposedSummarizingRNN(input[1], hidden, type, direction, dim, summarize)

        return l.get_shape(input), self.get_layer_name(type, m.group('name')), l

    def build_dropout(self, input, block):
        pattern = re.compile(r'(?P<type>Do)(?P<name>{\w+})?')
        m = pattern.match(block)
        if not m:
            return None, None, None
        else:
            return input, self.get_layer_name(m.group('type'), m.group('name')), torch.nn.Dropout()

    def build_conv(self, input, block):
        """
        Builds a 2D convolution layer.
        """
        pattern = re.compile(r'(?P<type>C)(?P<nl>s|t|r|l|m)(?P<name>{\w+})?(\d+),(\d+),(?P<out>\d+)')
        m = pattern.match(block)
        if not m:
            return None, None, None
        kernel_size = (int(m.group(4)), int(m.group(5)))
        filters = int(m.group('out'))
        nl = m.group('nl')
        fn = ActConv2D(input[1], filters, kernel_size, nl)
        return fn.get_shape(input), self.get_layer_name(m.group('type'), m.group('name')), fn

    def build_maxpool(self, input, block):
        """
        Builds a maxpool layer.
        """
        pattern = re.compile(r'(?P<type>Mp)(?P<name>{\w+})?(\d+),(\d+)(?:,(\d+),(\d+))?')
        m = pattern.match(block)
        if not m:
            return None, None, None
        kernel_size = (int(m.group(3)), int(m.group(4)))
        stride = (kernel_size[0] if not m.group(5) else int(m.group(5)),
                  kernel_size[1] if not m.group(6) else int(m.group(6)))
        output = (input[0],
                  input[1],
                  int(np.floor((input[2]-(kernel_size[0]-1)-1)/stride[0]+1) if input[2] != 0 else 0),
                  int(np.floor((input[3]-(kernel_size[1]-1)-1)/stride[1]+1) if input[3] != 0 else 0))
        return output, self.get_layer_name(m.group('type'), m.group('name')), torch.nn.MaxPool2d(kernel_size, stride)

    def build_output(self, input, block):
        """
        Builds an output layer.
        """
        pattern = re.compile(r'(O)(?P<name>{\w+})?(?P<dim>2|1|0)(?P<type>l|s|c)(?P<out>\d+)')
        m = pattern.match(block)
        if not m:
            return None, None, None
        if int(m.group('dim')) != 1:
            raise ValueError('non-2d output not supported, yet')
        nl = m.group('type')
        if nl not in ['s', 'c']:
            raise ValueError('only softmax and ctc supported in output')
        if nl == 'c':
            self.criterion = CTCCriterion()
        lin = LinSoftmax(input[1], int(m.group('out')))

        return lin.get_shape(input), self.get_layer_name(m.group(0), m.group('name')), lin
