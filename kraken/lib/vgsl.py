"""
VGSL plumbing
"""

import torch
import torch.nn.functional as F

from torch.autograd import Function
from torch.nn import Module
from torch.nn.modules.loss import _assert_no_grad

# all tensors are ordered NCHW

class TransposedSummarizingRNN(nn.Module):
    """
    An RNN wrapper allowing time axis transpositions and other 
    """
    def __init__(self, input_size, hidden_size, type='LSTM', direction='bidirectional', transpose=True, summarize=True):
        """
        A wrapper around torch.nn.LSTM/GRU optionally transposing inputs and
        returning only the last column of output.

        Channels in input are distributed into the height dimension, the
        outputs are put into the same. Output is `(N, 1, hidden_size, w)` or `(N, 1, hidden_size, 1)` if summarized.

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
              NCHW NHWC <- tf
        """
        super(TransposedSummarizingRNN, self).__init__()
        self.transpose = transpose
        self.summarize = summarize
        self.input_size = input_size
        self.hidden_size = hidden_size
        bidi = direction == 'bidirectional'
        self.output_size = hidden_size if not bidi else 2*hidden_size

        l = torch.nn.GRU if type == 'GRU' else torch.nn.LSTM

        self.layer = l(input_size, hidden_size, bidirectional=bidi)

    ### for current implementation x layer the input height is in the channels!
    def forward(self, inputs, hidden):
        # first resize from (batch, channels, height, width) into (batch, channels * height, width)
        # inputs = inputs.view(inputs.size(0), -1, inputs.size(3))
        # and into (width, batch, height)
        # inputs = inputs.permute(2, 0, 1)
        # batches = inputs.size(1)
        if self.transpose:
            # create (height, batch * width, channels) inputs
            
            # (height, batch * width, output_size) outputs

            # into (height, batch, width, output_size)

            # summarize

            # permute to (batch, output_size, height, width)
        else:
        # create (width, batch * height, channels) inputs
        # (width, batch * height, output_size) outputs
        # into (width, batch, height, output_size)


        
        # (width, batch, self.output_size)
        o, hidden = self.layer(inputs, hidden)
        if self.summarize:
            # take final output and unsqueeze to make 3D again
            o = o[-1].unsqueeze(0)
        if self.transpose:
            # switch height and width again
            o.transpose(0, 2)

    def get_shape(self, input):
        """
        Calculates the output shape from input 4D tuple (batch, input_size, seq_len, channel).
        """
        if self.transpose and self.summarize:
            return (input[0], self.hidden_size, input[2], 1)
        elif self.summarize:
            return (input[0], self.hidden_size, 1, 1)
        else:
            return (input[0], self.hidden_size, input[2], input[3])


class LinSoftmax(nn.Module):
    """
    A wrapper for linear projection + softmax dealing with dimensionality mangling.
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.lin = torch.nn.Linear(input_size, output_size, bias=False)

    def forward(self, inputs):
        # move features (H) to last dimension for linear activation
        o = F.softmax(self.lin(inputs.transpose(3, 2)), dim=3)
        # and swap again
        return o.transpose(3,2)

    def get_shape(self, input):
        """
        Calculates the output shape from input 4D tuple NCHW.
        """
        return (input[0], input[1], self.output_size, input[3])


class ActConv2D(nn.Module):
    """
    A wrapper for convolution + activation with dimensionality mangling.
    """
    def __init__(self, in_channnels, out_channels, kernel_size, nl='l'):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.nl = None
        if nl == 's':
            self.nl = F.sigmoid()
        elif nl == 't':
            self.nl = F.tanh()
        elif nl == 'm':
            self.nl = F.softmax()
        elif nl == 'r':
            self.nl = F.relu()

        self.co = torch.nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, inputs):
        return self.nl(self.co(inputs))

    def get_shape(input):
        return (input[0],
                self.out_channels,
                np.floor((input[1]-(kernel_size[0]-1)-1)+1),
                np.floor((input[2]-(kernel_size[1]-1)-1)+1))


class TorchVGSLModel(object):
    """
    Class building a torch module from a VSGL spec.
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

        Returns:
            nn.Module
        """
        self.spec = spec

        @classmethod
        def _parse_spec(spec):
            spec = spec.strip()
            if spec[0] != '[' or spec[-1] != ']':
                raise ValueError('Non-sequential models not supported')
            spec = spec[1:-1]
            blocks = spec.split(' ')
            batch, height, width, channels = blocks.pop(0).split(',')
            
        def build_rnn(self, input, block):
            """
            Builds an LSTM/GRU layer returning number of outputs and layer.
            """
            pattern = re.compile(r'(?P<type>L|G)(?P<dir>f|r|b)(?P<dim>x|y)(?P<sum>s)?(?P<name>{\w+})?(?P<out>\d+)')
            m = pattern.match(block)
            if not m:
                return None, None
            direction = m.group(2)
            dim = m.group(3)
            summarize = m.group(4) == 's'
            

        def build_dropout(self, input, block):
            pattern = re.compile(r'(?P<type>Do)(?P<name>{\w+})?')
            m = pattern.match(block)
            if not m:
                return None, None
            else:
                return input, torch.nn.Dropout()

        def build_conv(self, input, block):
            """
            Builds a 2D convolution layer.
            """
            pattern = re.compile(r'(C)(?P<nl>s|t|r|l|m)(?P<name>{\w+})?(\d+,(\d+),(?P<out>\d+)')
            m = pattern.match(block)
            kernel_size = (int(m.group(4)), int(m.group(5)))
            filters = int(m.group(6))
            nl = m.group(2)
            fn = [torch.nn.Conv2D(input[4], filters, kernel_size)]
            output = (input[0],
                      np.floor((input[1]-(kernel_size[0]-1)-1)+1),
                      np.floor((input[2]-(kernel_size[1]-1)-1)+1),
                      filters)
            if nl == 's':
                fn.append(torch.nn.Sigmoid())
            elif nl == 't':
                fn.append(torch.nn.Tanh())
            elif nl == 'm':
                fn.append(torch.nn.Softmax())
            elif nl == 'r':
                fn.append(torch.nn.ReLu())
            return output, fn

        def build_maxpool(self, input, block):
            """
            Builds a maxpool layer.
            """
            pattern = re.compile(r'(Mp)(?P<name>{\w+})?(\d+),(\d+)(?:,(\d+),(\d+))?')
            m = pattern.match(block)
            if not m:
                return None, None
            kernel_size = (int(m.group(3)), int(m.group(4)))
            stride = (kernel_size[0] if not m.group(5) else int(m.group(5)),
                      kernel_size[1] if not m.group(6) else int(m.group(6)))
            output = (input[0],
                      np.floor((input[1]-(kernel_size[0]-1)-1)/stride[0]+1),
                      np.floor((input[2]-(kernel_size[1]-1)-1)/stride[1]+1),
                      filters)
            return output, [torch.nn.MaxPool2D(kernel_size, stride)]

        def build_output(self, input, block):
            """
            Builds an output layer.
            """
            pattern = re.compile(r'(O)(?P<dim>2|1|0)(?P<type>l|s|c)(?P<out>\d+)')
            m = pattern.match(block)
            if not m:
                return None, None
            if input[3] != 1:
                raise ValueError('input depth of output layer is not 1 (got {} instead)'.format(input[3]))
            if m.group(2) != 1:
                raise ValueError('non-2d output not supported, yet')
            if m.group(3) not in ['s', 'c']:
                raise ValueError('only softmax and ctc supported in output')
            nl = m.group(3)
            if nl == 'c':
                warnings.warn('CTC is loss not layer. Just adding softmax to network.')
            lin = LinSoftmax(input[1], int(m.group(4)))

            return lin.get_shape(input), [lin]
