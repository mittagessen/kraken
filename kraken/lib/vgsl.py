"""
VGSL plumbing
"""

import re
import json
import torch
import warnings
import numpy as np
import torch.nn.functional as F

from torch.nn import Module

from torch.autograd import Variable
from kraken.lib import clstm_pb2
from kraken.lib import pyrnn_pb2

from kraken.lib.ctc import CTCCriterion
from kraken.lib.codec import PytorchCodec

from torch.nn import functional as F
from torch.nn.modules.loss import _assert_no_grad

from coremltools.models import MLModel
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder


# all tensors are ordered NCHW, the "feature" dimension is C, so the output of
# an LSTM will be put into C same as the filters of a CNN.

__all__ = ['TorchVGSLModel']

def PeepholeLSTMCell(input, hidden, w_ih, w_hh, w_ip, w_fp, w_op):
    """
    An LSTM cell with peephole connections without biases.

    Mostly ripped from the pytorch autograd lstm implementation.
    """
    hx, cx = hidden
    gates = F.linear(input, w_ih) + F.linear(hx, w_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    peep_i = w_ip.unsqueeze(0).expand_as(cx) * cx
    ingate = ingate + peep_i
    peep_f = w_fp.unsqueeze(0).expand_as(cx) * cx
    forgetgate = forgetgate + peep_f

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    peep_o = w_op.unsqueeze(0).expand_as(cy) * cy
    outgate = outgate + peep_o
    hy = outgate * F.tanh(cy)

    return hy, cy


def StackedRNN(inners, num_layers, num_directions):
    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight):
        assert (len(weight) == total_layers)
        next_hidden = []
        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j
                hy, output = inner(input, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)
            input = torch.cat(all_output, input.dim() - 1)
        next_h, next_c = zip(*next_hidden)
        next_hidden = (
            torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
            torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
        )
        return next_hidden, input

    return forward


def Recurrent(inner, reverse=False):
    def forward(input, hidden, weight):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        return hidden, output

    return forward


class PeepholeBidiLSTM(Module):

    def __init__(self, input_size, hidden_size):
        super(PeepholeBidiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self._all_weights = []
        gate_size = 4 * hidden_size
        for direction in range(2):
            w_ih = torch.nn.Parameter(torch.Tensor(gate_size, input_size))
            w_hh = torch.nn.Parameter(torch.Tensor(gate_size, hidden_size))

            w_ci = torch.nn.Parameter(torch.Tensor(hidden_size))
            w_cf = torch.nn.Parameter(torch.Tensor(hidden_size))
            w_co = torch.nn.Parameter(torch.Tensor(hidden_size))

            layer_params = (w_ih, w_hh, w_ci, w_cf, w_co)

            suffix = '_reverse' if direction == 1 else ''
            param_names = ['weight_ih_l0{}', 'weight_hh_l0{}', 'weight_ip_l0{}', 'weight_fp_l0{}', 'weight_op_l0{}']
            param_names = [x.format(suffix) for x in param_names]

            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)

    def init_hidden(self, bsz=1):
        return (Variable(torch.zeros(2, bsz, self.hidden_size)),
                Variable(torch.zeros(2, bsz, self.hidden_size)))

    def forward(self, input, hidden):
        layer = (Recurrent(PeepholeLSTMCell), Recurrent(PeepholeLSTMCell, reverse=True))
        func = StackedRNN(layer, 1, 2)
        input = input.transpose(0, 1)
        hidden, output = func(input, hidden, self.all_weights)
        output = output.transpose(0, 1)
        return output, hidden

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class MaxPool(Module):
    """
    A simple wrapper for MaxPool layers
    """
    def __init__(self, kernel_size, stride):
        """
        A wrapper around MaxPool layers with serialization and layer arithmetic.
        """
        super(MaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer = torch.nn.MaxPool2d(kernel_size, stride)

    def forward(self, inputs):
        return self.layer(inputs)

    def get_shape(self, input):
        return (input[0],
                input[1],
                int(np.floor((input[2]-(self.kernel_size[0]-1)-1)/self.stride[0]+1) if input[2] != 0 else 0),
                int(np.floor((input[3]-(self.kernel_size[1]-1)-1)/self.stride[1]+1) if input[3] != 0 else 0))

    def deserialize(self, name, spec):
        """
        Noop for MaxPool deserialization
        """
        pass

    def serialize(self, name, input, builder):
        builder.add_pooling(name,
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.stride[0],
                            self.stride[1],
                            layer_type='MAX',
                            padding_type='SAME',
                            input_name=input,
                            output_name=name)
        return name

class TransposedSummarizingRNN(Module):
    """
    An RNN wrapper allowing time axis transpositions and other
    """
    def __init__(self, input_size, hidden_size, direction='b', transpose=True, summarize=True, legacy=None):
        """
        A wrapper around torch.nn.LSTM optionally transposing inputs and
        returning only the last column of output.

        Args:
            input_size:
            hidden_size:
            direction (str):
            transpose (bool): Transpose width/height dimension
            summarize (bool): Only return the last time step.
            legacy (str): Set to `clstm` for clstm rnns and `ocropy` for ocropus models.

        Shape:
            - Inputs: :math:`(N, C, H, W)` where `N` batches, `C` channels, `H`
              height, and `W` width.
            - Outputs output :math:`(N, hidden_size * num_directions, H, S)`
              S (or H) being 1 if summarize (and transpose) are true
        """
        super(TransposedSummarizingRNN, self).__init__()
        self.transpose = transpose
        self.summarize = summarize
        self.legacy = legacy
        self.input_size = input_size
        if self.legacy:
            self.input_size += 1
        self.hidden_size = hidden_size
        self.bidi = direction == 'b'
        self.output_size = hidden_size if not self.bidi else 2*hidden_size

        if legacy == 'ocropy':
            self.layer = PeepholeBidiLSTM(self.input_size, hidden_size)
        else:
            self.layer = torch.nn.LSTM(self.input_size,
                                       hidden_size,
                                       bidirectional=self.bidi,
                                       batch_first=True,
                                       bias=False if legacy else True)

    def forward(self, inputs):
        # NCHW -> HNWC
        inputs = inputs.permute(2, 0, 3, 1)
        if self.transpose:
            # HNWC -> WNHC
            inputs = inputs.transpose(0, 2)
        if self.legacy:
            ones = Variable(torch.ones(inputs.shape[:3] + (1,)))
            inputs = torch.cat([ones, inputs], dim=3)
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

    def deserialize(self, name, spec):
        """
        Sets the weights of an initialized layer from a coreml spec.
        """
        nn = [x for x in spec.neuralNetwork.layers if x.name == name][0]
        arch = nn.WhichOneof('layer')
        l = getattr(nn, arch)
        if arch == 'permute':
            nn = [x for x in spec.neuralNetwork.layers if x.input == nn.output][0]
            arch = nn.WhichOneof('layer')
            l = getattr(nn, arch)

        fwd_params = l.weightParams if arch == 'uniDirectionalLSTM' else l.weightParams[0]
        # ih_matrix
        weight_ih = torch.FloatTensor([fwd_params.inputGateWeightMatrix.floatValue, # wi
                                       fwd_params.forgetGateWeightMatrix.floatValue, # wf
                                       fwd_params.blockInputWeightMatrix.floatValue, # wz/wg
                                       fwd_params.outputGateWeightMatrix.floatValue]) # wo

        self.layer.weight_ih_l0 = torch.nn.Parameter(weight_ih.resize_as_(self.layer.weight_ih_l0.data))

        # hh_matrix
        weight_hh = torch.FloatTensor([fwd_params.inputGateRecursionMatrix.floatValue, # wi
                                       fwd_params.forgetGateRecursionMatrix.floatValue, # wf
                                       fwd_params.blockInputRecursionMatrix.floatValue, #wz/wg
                                       fwd_params.outputGateRecursionMatrix.floatValue]) # wo
        self.layer.weight_hh_l0 = torch.nn.Parameter(weight_hh.resize_as_(self.layer.weight_hh_l0.data))

        if not self.legacy:
            # ih biases
            biases = torch.FloatTensor([fwd_params.inputGateBiasVector.floatValue, #bi
                                        fwd_params.forgetGateBiasVector.floatValue, # bf
                                        fwd_params.blockInputBiasVector.floatValue, # bz/bg
                                        fwd_params.outputGateBiasVector.floatValue]) #bo
            self.layer.bias_hh_l0 = torch.nn.Parameter(biases.resize_as_(self.layer.bias_hh_l0.data))
            # no hh_biases
            self.layer.bias_ih_l0 = torch.nn.Parameter(torch.zeros(self.layer.bias_ih_l0.size()))

        # get backward weights
        if arch == 'biDirectionalLSTM':
            bwd_params = l.weightParams[1]
            weight_ih_rev = torch.FloatTensor([bwd_params.inputGateWeightMatrix.floatValue, # wi
                                               bwd_params.forgetGateWeightMatrix.floatValue, # wf
                                               bwd_params.blockInputWeightMatrix.floatValue, # wz/wg
                                               bwd_params.outputGateWeightMatrix.floatValue]) # wo
            self.layer.weight_ih_l0_reverse = torch.nn.Parameter(weight_ih.resize_as_(self.layer.weight_ih_l0.data))

            weight_hh_rev = torch.FloatTensor([bwd_params.inputGateRecursionMatrix.floatValue, # wi
                                               bwd_params.forgetGateRecursionMatrix.floatValue, # wf
                                               bwd_params.blockInputRecursionMatrix.floatValue, #wz/wg
                                               bwd_params.outputGateRecursionMatrix.floatValue]) # wo
            self.layer.weight_hh_l0_reverse = torch.nn.Parameter(weight_hh.resize_as_(self.layer.weight_hh_l0.data))

            if not self.legacy:
                biases_rev = torch.FloatTensor([bwd_params.inputGateBiasVector.floatValue, #bi
                                                bwd_params.forgetGateBiasVector.floatValue, # bf
                                                bwd_params.blockInputBiasVector.floatValue, # bz/bg
                                                bwd_params.outputGateBiasVector.floatValue]) #bo
                self.layer.bias_hh_l0_reverse = torch.nn.Parameter(biases.resize_as_(self.layer.bias_hh_l0.data))
                self.layer.bias_ih_l0 = torch.nn.Parameter(torch.zeros(self.layer.bias_ih_l0.size()))

    def serialize(self, name, input, builder):
        """
        Serializes the module using a NeuralNetworkBuilder.
        """
        # coreml weight order is IFOG while pytorch uses IFGO
        # it also uses a single bias while pytorch splits them for some reason
        def _reorder_indim(tensor, splits=4, idx=[0, 1, 3, 2]):
            """
            Splits the first dimension into `splits` chunks, reorders them
            according to idx, and convert them to a numpy array.
            """
            s = tensor.chunk(splits)
            return [s[i].data.numpy() for i in idx]

        if self.transpose:
            ninput = '{}_transposed'.format(name)
            builder.add_permute(name=name,
                                dim=[0, 1, 3, 2],
                                input_name=input,
                                output_name=ninput)
            input = ninput
            name = ninput
        if self.bidi:
            builder.add_bidirlstm(name=name,
                                  W_h=_reorder_indim(self.layer.weight_hh_l0),
                                  W_x=_reorder_indim(self.layer.weight_ih_l0),
                                  b=_reorder_indim((self.layer.bias_ih_l0 + self.layer.bias_hh_l0)) if not self.legacy else None,
                                  W_h_back=_reorder_indim(self.layer.weight_hh_l0_reverse),
                                  W_x_back=_reorder_indim(self.layer.weight_ih_l0_reverse),
                                  b_back=_reorder_indim((self.layer.bias_ih_l0_reverse + self.layer.bias_hh_l0_reverse)) if not self.legacy else None,
                                  hidden_size=self.hidden_size,
                                  input_size=self.input_size,
                                  input_names=[input],
                                  output_names=[name],
                                  peep=[self.layer.weight_ip_l0.data.numpy(),
                                        self.layer.weight_fp_l0.data.numpy(),
                                        self.layer.weight_op_l0.data.numpy()] if self.legacy == 'ocropy' else None,
                                  peep_back=[self.layer.weight_ip_l0_reverse.data.numpy(),
                                             self.layer.weight_fp_l0_reverse.data.numpy(),
                                             self.layer.weight_op_l0_reverse.data.numpy()] if self.legacy == 'ocropy' else None,
                                  output_all=not self.summarize)
        else:
            builder.add_unilstm(name=name,
                                W_h=_reorder_indim(self.layer.weight_hh_l0),
                                W_x=_reorder_indim(self.layer.weight_ih_l0),
                                b=_reorder_indim((self.layer.bias_ih_l0 + self.layer.bias_hh_l0)) if not self.legacy else None,
                                hidden_size=self.hidden_size,
                                input_size=self.input_size,
                                input_names=[input],
                                output_names=[name],
                                peep=[self.layer.weight_ip_l0.data.numpy(),
                                      self.layer.weight_fp_l0.data.numpy(),
                                      self.layer.weight_op_l0.data.numpy()] if self.legacy == 'ocropy' else None,
                                output_all=not self.summarize)
        return name

class LinSoftmax(Module):
    """
    A wrapper for linear projection + softmax dealing with dimensionality mangling.
    """
    def __init__(self, input_size, output_size, augmentation=False):
        """

        Args:
            input_size: Number of inputs in the feature dimension
            output_size: Number of outputs in the feature dimension
            augmentation (bool): Enables 1-augmentation of input vectors

        Shape:
            - Inputs: :math:`(N, C, H, W)` where `N` batches, `C` channels, `H`
              height, and `W` width.
            - Outputs output :math:`(N, output_size, H, S)`
        """
        super(LinSoftmax, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.augmentation = augmentation
        if self.augmentation:
            self.input_size += 1

        self.lin = torch.nn.Linear(self.input_size, output_size)

    def forward(self, inputs):
        # move features (C) to last dimension for linear activation
        inputs = inputs.transpose(1, 3)
        # augment with ones along the input (C) axis
        if self.augmentation:
            inputs = torch.cat([Variable(torch.ones(inputs.shape[:3] + (1,))), inputs], dim=3)
        o = F.softmax(self.lin(inputs), dim=3)
        # and swap again
        return o.transpose(3,1)

    def get_shape(self, input):
        """
        Calculates the output shape from input 4D tuple NCHW.
        """
        return (input[0], self.output_size, input[2], input[3])

    def deserialize(self, name, spec):
        """
        Sets the weights of an initialized module from a CoreML protobuf spec.
        """
        # extract conv parameters
        lin = [x for x in spec.neuralNetwork.layers if x.name == '{}_lin'.format(name)][0].innerProduct
        weights = torch.FloatTensor(lin.weights.floatValue).resize_as_(self.lin.weight.data)
        bias = torch.FloatTensor(lin.bias.floatValue)
        self.lin.weight = torch.nn.Parameter(weights)
        self.lin.bias = torch.nn.Parameter(bias)

    def serialize(self, name, input, builder):
        """
        Serializes the module using a NeuralNetworkBuilder.
        """
        lin_name = '{}_lin'.format(name)
        softmax_name = '{}_softmax'.format(name)
        builder.add_inner_product(lin_name, self.lin.weight.data.numpy(),
                                  self.lin.bias.data.numpy(),
                                  self.input_size, self.output_size,
                                  has_bias=True, input_name=input, output_name=lin_name)
        builder.add_softmax(softmax_name, lin_name, name)
        return name


class ActConv2D(Module):
    """
    A wrapper for convolution + activation with automatic padding ensuring no
    dropped columns.
    """
    def __init__(self, in_channels, out_channels, kernel_size, nl='l'):
        super(ActConv2D, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.padding = tuple((k - 1) // 2 for k in kernel_size)
        self.nl = None
        self.nl_name = None
        if nl == 's':
            self.nl = F.sigmoid
            self.nl_name = 'SIGMOID'
        elif nl == 't':
            self.nl = F.tanh
            self.nl_name = 'TANH'
        elif nl == 'm':
            self.nl = F.softmax
            self.nl_name='SOFTMAX'
        elif nl == 'r':
            self.nl = F.relu
            self.nl_name='RELU'

        self.co = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                  padding=self.padding)

    def forward(self, inputs):
        return self.nl(self.co(inputs))

    def get_shape(self, input):
        return (input[0],
                self.out_channels,
                int(min(np.floor((input[2]+2*self.padding[0]-(self.kernel_size[0]-1)-1)+1), 1) if input[2] != 0 else 0),
                int(min(np.floor((input[3]+2*self.padding[1]-(self.kernel_size[1]-1)-1)+1), 1) if input[3] != 0 else 0))

    def deserialize(self, name, spec):
        """
        Sets the weight of an initialized model from a CoreML protobuf spec.
        """
        conv = [x for x in spec.neuralNetwork.layers if x.name == '{}_conv'.format(name)][0].convolution
        self.co.weight = torch.nn.Parameter(torch.FloatTensor(conv.weights.floatValue).view(self.out_channels,
                                                                                            self.in_channels,
                                                                                            *self.kernel_size))
        self.co.bias = torch.nn.Parameter(torch.FloatTensor(conv.bias.floatValue))

    def serialize(self, name, input, builder):
        """
        Serializes the module using a NeuralNetworkBuilder.
        """
        conv_name = '{}_conv'.format(name)
        act_name = '{}_act'.format(name)
        builder.add_convolution(name=conv_name,
                                kernel_channels=self.in_channels,
                                output_channels=self.out_channels,
                                height=self.kernel_size[0],
                                width=self.kernel_size[1],
                                stride_height=1,
                                stride_width=1,
                                border_mode='same',
                                groups=1,
                                W=self.co.weight.permute(2, 3, 1, 0).data.numpy(),
                                b=self.co.bias.data.numpy(),
                                has_bias=True,
                                input_name=input,
                                output_name=conv_name)
        if self.nl_name != 'SOFTMAX':
            builder.add_activation(act_name, self.nl_name, conv_name, name)
        else:
            builder.add_softmax(act_name, conv_name, name)
        return name

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
        self.named_spec = []
        self.ops = [self.build_rnn, self.build_dropout, self.build_maxpool, self.build_conv, self.build_output]
        self.codec = None
        self.criterion = None

        self.idx = -1
        spec = spec.strip()
        if spec[0] != '[' or spec[-1] != ']':
            raise ValueError('Non-sequential models not supported')
        spec = spec[1:-1]
        blocks = spec.split(' ')
        self.named_spec.append(blocks[0])
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
                self.named_spec.append(self.set_layer_name(block, name))
                self.nn.add_module(name, layer)
            else:
                raise ValueError('{} invalid layer definition'.format(block))
        self.output = oshape

    def cuda(self):
        self.nn.cuda()
        if self.criterion:
            self.criterion.cuda()

    @classmethod
    def load_pronn_model(cls, path):
        """
        Loads an pronn model to VGSL.
        """
        with open(path, 'rb') as fp:
            net = pyrnn_pb2.pyrnn()
            try:
                net.ParseFromString(fp.read())
            except:
                raise KrakenInvalidModelException('File does not contain valid proto msg')
            if not net.IsInitialized():
                raise KrakenInvalidModelException('Model incomplete')

        # extract codec
        codec = PytorchCodec(net.codec)

        input = net.ninput
        hidden = net.fwdnet.wgi.dim[0]

        # extract weights
        weightnames = ('wgi', 'wgf', 'wci', 'wgo', 'wip', 'wfp', 'wop')

        fwd_w = []
        rev_w = []
        for w in weightnames:
            fwd_ar = getattr(net.fwdnet, w)
            rev_ar = getattr(net.revnet, w)
            fwd_w.append(torch.FloatTensor(fwd_ar.value).view(list(fwd_ar.dim)))
            rev_w.append(torch.FloatTensor(rev_ar.value).view(list(rev_ar.dim)))

        t = torch.cat(fwd_w[:4])
        weight_ih_l0 = t[:, :input+1]
        weight_hh_l0 = t[:, input+1:]

        t = torch.cat(rev_w[:4])
        weight_ih_l0_rev = t[:, :input+1]
        weight_hh_l0_rev = t[:, input+1:]

        weight_lin = torch.FloatTensor(net.softmax.w2.value).view(list(net.softmax.w2.dim))

        # build vgsl spec and set weights
        nn = cls('[1,1,0,{} Lbxo{} O1ca{}]'.format(input, hidden, len(net.codec)))

        nn.nn.L_0.layer.weight_ih_l0 = torch.nn.Parameter(weight_ih_l0)
        nn.nn.L_0.layer.weight_hh_l0 = torch.nn.Parameter(weight_hh_l0)
        nn.nn.L_0.layer.weight_ih_l0_reverse = torch.nn.Parameter(weight_ih_l0_rev)
        nn.nn.L_0.layer.weight_hh_l0_reverse = torch.nn.Parameter(weight_hh_l0_rev)
        nn.nn.L_0.layer.weight_ip_l0 = torch.nn.Parameter(fwd_w[4])
        nn.nn.L_0.layer.weight_fp_l0 = torch.nn.Parameter(fwd_w[5])
        nn.nn.L_0.layer.weight_op_l0 = torch.nn.Parameter(fwd_w[6])
        nn.nn.L_0.layer.weight_ip_l0_reverse = torch.nn.Parameter(rev_w[4])
        nn.nn.L_0.layer.weight_fp_l0_reverse = torch.nn.Parameter(rev_w[5])
        nn.nn.L_0.layer.weight_op_l0_reverse = torch.nn.Parameter(rev_w[6])

        nn.nn.O_1.lin.weight = torch.nn.Parameter(weight_lin)

        nn.add_codec(codec)

        return nn

    @classmethod
    def load_clstm_model(cls, path):
        """
        Loads an CLSTM model to VGSL.
        """
        net = clstm_pb2.NetworkProto()
        with open(path, 'rb') as fp:
            try:
                net.ParseFromString(fp.read())
            except:
                raise KrakenInvalidModelException('File does not contain valid proto msg')
            if not net.IsInitialized():
                raise KrakenInvalidModelException('Model incomplete')

        input = net.ninput
        output = net.noutput
        attrib = {a.key: a.value for a in list(net.attribute)}
        # mainline clstm model
        if len(attrib) > 1:
            mode = 'clstm'
        else:
            mode = 'clstm_compat'

        # extract codec
        codec = PytorchCodec([u''] + [unichr(x) for x in net.codec[1:]])

        # separate layers
        nets = {}
        nets['softm'] = [n for n in list(net.sub) if n.kind == 'SoftmaxLayer'][0]
        parallel = [n for n in list(net.sub) if n.kind == 'Parallel'][0]
        nets['lstm1'] = [n for n in list(parallel.sub) if n.kind.startswith('NPLSTM')][0]
        rev = [n for n in list(parallel.sub) if n.kind == 'Reversed'][0]
        nets['lstm2'] = rev.sub[0]

        hidden = int(nets['lstm1'].attribute[0].value)

        weights = {}
        for n in nets:
            weights[n] = {}
            for w in list(nets[n].weights):
                weights[n][w.name] = torch.FloatTensor(w.value).view(list(w.dim))

        if mode == 'clstm_compat':
            weightnames = ('.WGI', '.WGF', '.WCI', '.WGO')
            weightname_softm = '.W'
        else:
            weightnames = ('WGI', 'WGF', 'WCI', 'WGO')
            weightname_softm = 'W1'

        # input hidden and hidden-hidden weights are in one matrix. also
        # CLSTM/ocropy likes 1-augmenting every other tensor so the ih weights
        # are input+1 in one dimension.
        t = torch.cat(w for w in [weights['lstm1'][wn] for wn in weightnames])
        weight_ih_l0 = t[:, :input+1]
        weight_hh_l0 = t[:, input+1:]

        t = torch.cat(w for w in [weights['lstm2'][wn] for wn in weightnames])
        weight_ih_l0_rev = t[:, :input+1]
        weight_hh_l0_rev = t[:, input+1:]

        weight_lin = weights['softm'][weightname_softm]
        if mode == 'clstm_compat':
            weight_lin = torch.cat([torch.zeros(len(weight_lin), 1), weight_lin], 1)

        # build vgsl spec and set weights
        nn = cls('[1,1,0,{} Lbxc{} O1ca{}]'.format(input, hidden, len(net.codec)))
        nn.nn.L_0.layer.weight_ih_l0 = torch.nn.Parameter(weight_ih_l0)
        nn.nn.L_0.layer.weight_hh_l0 = torch.nn.Parameter(weight_hh_l0)
        nn.nn.L_0.layer.weight_ih_l0_reverse = torch.nn.Parameter(weight_ih_l0_rev)
        nn.nn.L_0.layer.weight_hh_l0_reverse = torch.nn.Parameter(weight_hh_l0_rev)
        nn.nn.O_1.lin.weight = torch.nn.Parameter(weight_lin)

        nn.add_codec(codec)

        return nn

    @classmethod
    def load_model(cls, path):
        """
        Deserializes a VGSL model from a CoreML file.

        Args:
            path (str): CoreML file
        """
        mlmodel = MLModel(path)
        if 'vgsl' not in mlmodel.user_defined_metadata:
            raise ValueError('No VGSL spec in model metadata')
        vgsl_spec = mlmodel.user_defined_metadata['vgsl']
        nn = cls(vgsl_spec)
        for name, layer in nn.nn.named_children():
            layer.deserialize(name, mlmodel.get_spec())

        if 'codec' in mlmodel.user_defined_metadata:
            nn.add_codec(PytorchCodec(json.loads(mlmodel.user_defined_metadata['codec'])))
        return nn

    def save_model(self, path):
        """
        Serializes the model into path.

        Args:
            path (str): Target destination
        """
        inputs = [('input', datatypes.Array(*self.input))]
        outputs = [('output', datatypes.Array(*self.output))]
        net_builder = NeuralNetworkBuilder(inputs, outputs)
        input = 'input'
        for name, layer in self.nn.named_children():
            input = layer.serialize(name, input, net_builder)
        mlmodel = MLModel(net_builder.spec)
        mlmodel.short_description = 'kraken recognition model'
        mlmodel.user_defined_metadata['vgsl'] = '[' + ' '.join(self.named_spec) + ']'
        if self.codec:
            mlmodel.user_defined_metadata['codec'] = json.dumps(self.codec.c2l)
        mlmodel.save(path)

    def add_codec(self, codec):
        """
        Adds a PytorchCodec to the model.
        """
        self.codec = codec

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

    @staticmethod
    def set_layer_name(layer, name):
        """
        Sets the name field of an VGSL layer definition.

        Args:
            layer (str): VGSL definition
            name (str): Layer name
        """
        if '{' in layer and '}' in layer:
            return
        l = re.split(r'(^[^\d]+)', layer)
        l.insert(-1, '{{{}}}'.format(name))
        return ''.join(l)

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
            self.idx += 1
            return '{}_{}'.format(re.sub(r'\W+', '_', layer), self.idx)

    def build_rnn(self, input, block):
        """
        Builds an LSTM/GRU layer returning number of outputs and layer.
        """
        pattern = re.compile(r'(?P<type>L|G)(?P<dir>f|r|b)(?P<dim>x|y)(?P<sum>s)?(?P<legacy>c|o)?(?P<name>{\w+})?(?P<out>\d+)')
        m = pattern.match(block)
        if not m:
            return None, None, None
        type = m.group('type')
        direction = m.group('dir')
        dim = m.group('dim')  == 'y'
        summarize = m.group('sum') == 's'
        legacy = None
        if m.group('legacy') == 'c':
            legacy = 'clstm'
        elif m.group('legacy') == 'o':
            legacy = 'ocropy'
        hidden = int(m.group(7))
        l = TransposedSummarizingRNN(input[1], hidden, direction, dim, summarize, legacy)

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
        fn = MaxPool(kernel_size, stride)
        return fn.get_shape(input), self.get_layer_name(m.group('type'), m.group('name')), fn

    def build_output(self, input, block):
        """
        Builds an output layer.
        """
        pattern = re.compile(r'(O)(?P<name>{\w+})?(?P<dim>2|1|0)(?P<type>l|s|c)(?P<aug>a)?(?P<out>\d+)')
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
        lin = LinSoftmax(input[1], int(m.group('out')), True if m.group('aug') else False)

        return lin.get_shape(input), self.get_layer_name(m.group(1), m.group('name')), lin
