"""
Layers for VGSL models
"""
import torch
import numpy as np

from torch.nn import Module
from torch.autograd import Variable
from torch.nn import functional as F
from coremltools.proto import NeuralNetwork_pb2

# all tensors are ordered NCHW, the "feature" dimension is C, so the output of
# an LSTM will be put into C same as the filters of a CNN.

__all__ = ['MaxPool', 'Dropout', 'TransposedSummarizingRNN', 'LinSoftmax', 'ActConv2D']


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
                length = i * num_directions + j
                hy, output = inner(input, hidden[length], weight[length])
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
        self.output_shape = (input[0],
                             input[1],
                             int(np.floor((input[2]-(self.kernel_size[0]-1)-1)/self.stride[0]+1) if input[2] != 0 else 0),
                             int(np.floor((input[3]-(self.kernel_size[1]-1)-1)/self.stride[1]+1) if input[3] != 0 else 0))
        return self.output_shape

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


class Dropout(Module):
    """
    A simple wrapper for dropout layers
    """
    def __init__(self, p, dim):
        """
        A wrapper around dropout layers with serialization and layer arithmetic.
        """
        super(Dropout, self).__init__()
        self.p = p
        self.dim = dim
        if dim == 1:
            self.layer = torch.nn.Dropout(p)
        elif dim == 2:
            self.layer = torch.nn.Dropout2D(p)

    def forward(self, inputs):
        return self.layer(inputs)

    def get_shape(self, input):
        self.output_shape = input
        return input

    def deserialize(self, name, spec):
        """
        Noop for deserialization
        """
        pass

    def serialize(self, name, input, builder):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = 'dropout'
        params.description = 'An n-dimensional dropout layer'
        params.parameters['dim'].intValue = self.dim
        params.parameters['p'].doubleValue = self.p
        builder.add_custom(name,
                           input_names=[input],
                           output_names=[name],
                           custom_proto_spec=params)
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
        o = o.view(siz[0], siz[1], siz[2], self.output_size)
        if self.summarize:
            # HN1O
            o = o[:, :, -1, :].unsqueeze(2)
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
                layer = (1, input[3])
            else:
                layer = (input[2], 1)
        else:
            layer = (input[2], input[3])
        self.output_shape = (input[0], self.output_size) + layer
        return self.output_shape

    def deserialize(self, name, spec):
        """
        Sets the weights of an initialized layer from a coreml spec.
        """
        nn = [x for x in spec.neuralNetwork.layers if x.name == name][0]
        arch = nn.WhichOneof('layer')
        layer = getattr(nn, arch)
        if arch == 'permute':
            nn = [x for x in spec.neuralNetwork.layers if x.input == nn.output][0]
            arch = nn.WhichOneof('layer')
            layer = getattr(nn, arch)


        def _deserialize_weights(params, layer, direction):
            # ih_matrix
            weight_ih = torch.Tensor([params.inputGateWeightMatrix.floatValue,  # wi
                                      params.forgetGateWeightMatrix.floatValue,  # wf
                                      params.blockInputWeightMatrix.floatValue,  # wz/wg
                                      params.outputGateWeightMatrix.floatValue])  # wo
            # hh_matrix
            weight_hh = torch.Tensor([params.inputGateRecursionMatrix.floatValue,  # wi
                                      params.forgetGateRecursionMatrix.floatValue,  # wf
                                      params.blockInputRecursionMatrix.floatValue,  # wz/wg
                                      params.outputGateRecursionMatrix.floatValue])  # wo

            if direction == 'fwd':
                layer.weight_ih_l0 = torch.nn.Parameter(weight_ih.resize_as_(layer.weight_ih_l0.data))
                layer.weight_hh_l0 = torch.nn.Parameter(weight_hh.resize_as_(layer.weight_hh_l0.data))
            elif direction == 'bwd':
                layer.weight_ih_l0_reverse = torch.nn.Parameter(weight_ih.resize_as_(layer.weight_ih_l0.data))
                layer.weight_hh_l0_reverse = torch.nn.Parameter(weight_hh.resize_as_(layer.weight_hh_l0.data))

        def _deserialize_biases(params, layer, direction):
            # ih biases
            biases = torch.Tensor([params.inputGateBiasVector.floatValue,  # bi
                                   params.forgetGateBiasVector.floatValue,  # bf
                                   params.blockInputBiasVector.floatValue,  # bz/bg
                                   params.outputGateBiasVector.floatValue])  # bo
            if direction == 'fwd':
                layer.bias_hh_l0 = torch.nn.Parameter(biases.resize_as_(layer.bias_hh_l0.data))
                # no ih_biases
                layer.bias_ih_l0 = torch.nn.Parameter(torch.zeros(layer.bias_ih_l0.size()))
            elif direction == 'bwd':
                layer.bias_hh_l0_reverse = torch.nn.Parameter(biases.resize_as_(layer.bias_hh_l0.data))
                # no ih_biases
                layer.bias_ih_l0_reverse = torch.nn.Parameter(torch.zeros(layer.bias_ih_l0.size()))

        fwd_params = layer.weightParams if arch == 'uniDirectionalLSTM' else layer.weightParams[0]
        _deserialize_weights(fwd_params, self.layer, 'fwd')
        if not self.legacy:
            _deserialize_biases(fwd_params, self.layer, 'fwd')

        # get backward weights
        if arch == 'biDirectionalLSTM':
            bwd_params = layer.weightParams[1]
            _deserialize_weights(bwd_params, self.layer, 'bwd')
            if not self.legacy:
                _deserialize_biases(bwd_params, self.layer, 'bwd')

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
        # NCHW -> NWHC
        inputs = inputs.transpose(1, 3)
        # augment with ones along the input (C) axis
        if self.augmentation:
            inputs = torch.cat([Variable(torch.ones(inputs.shape[:3] + (1,))), inputs], dim=3)
        # only enable softmax during inference (CTC loss calculates softmax internally)
        o = self.lin(inputs)
        if not self.training:
            o = F.softmax(o, dim=3)
        # and swap again
        return o.transpose(1, 3)

    def get_shape(self, input):
        """
        Calculates the output shape from input 4D tuple NCHW.
        """
        self.output_shape = (input[0], self.output_size, input[2], input[3])
        return self.output_shape

    def deserialize(self, name, spec):
        """
        Sets the weights of an initialized module from a CoreML protobuf spec.
        """
        # extract conv parameters
        lin = [x for x in spec.neuralNetwork.layers if x.name == '{}_lin'.format(name)][0].innerProduct
        weights = torch.Tensor(lin.weights.floatValue).resize_as_(self.lin.weight.data)
        bias = torch.Tensor(lin.bias.floatValue)
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

    def resize(self, output_size, del_indices=None):
        """
        Resizes the linear layer with minimal disturbance to the existing
        weights.

        First removes the weight and bias at the output positions in
        del_indices, then resizes both tensors to a new output size.

        Args:
            output_size (int): Desired output size after resizing
            del_indices (list): List of connection to outputs to remove.
        """
        if not del_indices:
            del_indices = []
        old_shape = self.lin.weight.size(0)
        self.output_size = output_size
        idx = torch.tensor([x for x in range(old_shape) if x not in del_indices])
        weight = self.lin.weight.index_select(0, idx)
        rweight = torch.zeros((output_size - weight.size(0), weight.size(1)))
        torch.nn.init.xavier_uniform_(rweight)
        weight = torch.cat([weight, rweight])
        bias = self.lin.bias.index_select(0, idx)
        bias = torch.cat([bias, torch.zeros(output_size - bias.size(0))])
        self.lin = torch.nn.Linear(self.input_size, output_size)
        self.lin.weight = torch.nn.Parameter(weight)
        self.lin.bias = torch.nn.Parameter(bias)


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
            self.nl_name = 'SOFTMAX'
        elif nl == 'r':
            self.nl = F.relu
            self.nl_name = 'RELU'

        self.co = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                  padding=self.padding)

    def forward(self, inputs):
        return self.nl(self.co(inputs))

    def get_shape(self, input):
        self.output_shape = (input[0],
                             self.out_channels,
                             int(min(np.floor((input[2]+2*self.padding[0]-(self.kernel_size[0]-1)-1)+1), 1) if input[2] != 0 else 0),
                             int(min(np.floor((input[3]+2*self.padding[1]-(self.kernel_size[1]-1)-1)+1), 1) if input[3] != 0 else 0))
        return self.output_shape

    def deserialize(self, name, spec):
        """
        Sets the weight of an initialized model from a CoreML protobuf spec.
        """
        conv = [x for x in spec.neuralNetwork.layers if x.name == '{}_conv'.format(name)][0].convolution
        self.co.weight = torch.nn.Parameter(torch.Tensor(conv.weights.floatValue).view(self.out_channels,
                                                                                       self.in_channels,
                                                                                       *self.kernel_size))
        self.co.bias = torch.nn.Parameter(torch.Tensor(conv.bias.floatValue))

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
