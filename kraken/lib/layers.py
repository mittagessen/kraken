"""
Layers for VGSL models
"""
import torch
import numpy as np

from typing import List, Tuple, Optional, Iterable
from torch.nn import Module, Sequential
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from coremltools.proto import NeuralNetwork_pb2

# all tensors are ordered NCHW, the "feature" dimension is C, so the output of
# an LSTM will be put into C same as the filters of a CNN.

__all__ = ['Addition', 'MaxPool', 'Reshape', 'Dropout', 'TransposedSummarizingRNN', 'LinSoftmax', 'ActConv2D']


class MultiParamSequential(Sequential):
    """
    Sequential variant accepting multiple arguments.
    """

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MultiParamParallel(Module):
    """
    Parallel module.
    """
    def forward(self, *inputs):
        outputs = []
        seq_lens = None
        for module in self._modules.values():
            if type(inputs) == tuple:
                output, seq_lens = module(*inputs)
                outputs.append(output)
            else:
                outputs.append(module(inputs))
        return torch.cat(outputs, dim=1), seq_lens


def PeepholeLSTMCell(input: torch.Tensor,
                     hidden: Tuple[torch.Tensor, torch.Tensor],
                     w_ih: torch.Tensor,
                     w_hh: torch.Tensor,
                     w_ip: torch.Tensor,
                     w_fp: torch.Tensor,
                     w_op: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


def StackedRNN(inners, num_layers: int, num_directions: int):
    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight):
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
        next_hidden = [
            torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
            torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
        ]
        return next_hidden, input

    return forward


def Recurrent(inner, reverse: bool = False):
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

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self._all_weights = []  # type: List[List[str]]
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        layer = (Recurrent(PeepholeLSTMCell), Recurrent(PeepholeLSTMCell, reverse=True))
        func = StackedRNN(layer, 1, 2)
        input = input.transpose(0, 1)
        hidden = (torch.zeros(2, input.shape[1], self.hidden_size).to(input.device),
                  torch.zeros(2, input.shape[1], self.hidden_size).to(input.device))
        hidden, output = func(input, hidden, self.all_weights)
        output = output.transpose(0, 1)
        return output, hidden

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class Addition(Module):
    """
    An addition module
    """
    def __init__(self, dim: int, chunk_size: int) -> None:
        """
        An addition module

        Shape:
             - Inputs: :math:`(N, C, H, W)` where `N` batches, `C` channels, `H`
              height, and `W` width.
            - Outputs output :math:`(N, C, H, W)`
        """
        self.dim = dim
        self.chunk_size = chunk_size
        super().__init__()

    def forward(self, inputs: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = inputs.unfold(self.dim, self.chunk_size, self.chunk_size)
        out = out.sum(self.dim, keepdim=True)
        out = out.transpose(-1, self.dim).squeeze(-1)
        return out, seq_len

    def get_shape(self, input: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Calculates the output shape from input 4D tuple NCHW.
        """
        input = list(input)
        input[self.dim] = self.chunk_size
        self.output_shape = tuple(input)
        return self.output_shape

    def deserialize(self, name, spec):
        """
        Noop for deserialization
        """
        pass

    def serialize(self, name, input, builder):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = 'addition'
        params.description = 'An addition layer'
        params.parameters['dim'].intValue = self.dim
        params.parameters['chunk_size'].intValue = self.chunk_size

        builder.add_custom(name,
                           input_names=[input],
                           output_names=[name],
                           custom_proto_spec=params)
        return name


class Identity(Module):
    """
    A placeholder identity operator.
    """
    def __init__(self) -> None:
        """
        A placeholder identity operator (mostly used for residual connections and similar).

        Shape:
             - Inputs: :math:`(N, C, H, W)` where `N` batches, `C` channels, `H`
              height, and `W` width.
            - Outputs output :math:`(N, C, H, W)`
        """
        super().__init__()

    def forward(self, inputs: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return inputs, seq_len

    def get_shape(self, input: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        self.output_shape = input
        return input

    def deserialize(self, name, spec):
        """
        Noop for deserialization
        """
        pass

    def serialize(self, name, input, builder):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = 'identity'
        params.description = 'An identity layer'
        builder.add_custom(name,
                           input_names=[input],
                           output_names=[name],
                           custom_proto_spec=params)
        return name


class Reshape(Module):
    """
    Reshapes input and moves it into other dimensions.
    """

    def __init__(self, src_dim: int, part_a: int, part_b: int, high: int, low: int) -> None:
        """
        A wrapper around reshape with serialization and layer arithmetic.

        Args:
            src_dim (int): Source dimension
            part_a (int): Size of split dim to move to `high`
            part_b (int): Size of split dim to move to `low`
            high (int): Target dimension 1
            low (int): Target dimension 2

        Shape:
            - Inputs: :math:`(N, C, H, W)` where `N` batches, `C` channels, `H`
              height, and `W` width.
            - Outputs output :math:`(N, C, H, W)`
        """
        super().__init__()
        self.src_dim = src_dim
        self.part_a = part_a
        self.part_b = part_b
        self.high = high
        self.low = low

    def forward(self, input: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        initial_len = input.shape[3]
        # split dimension src_dim into part_a x part_b
        input = input.reshape(input.shape[:self.src_dim] + (self.part_a, self.part_b) + input.shape[self.src_dim + 1:])
        dest = self.low
        src_dim = self.src_dim
        if self.high != src_dim:
            dest = self.high
        else:
            src_dim += 1
        # rotate dimension permutation list
        perm = list(range(len(input.shape)))
        step = 1 if dest > src_dim else -1
        for x in range(src_dim, dest, step):
            perm[x], perm[x + step] = perm[x + step], perm[x]
        input = input.permute(perm)
        o = input.reshape(input.shape[:dest] + (input.shape[dest] * input.shape[dest + 1],) + input.shape[dest + 2:])
        if seq_len is not None:
            seq_len = (seq_len * (float(initial_len)/o.shape[3])).int()
        return o, seq_len

    def get_shape(self, input: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        input_shape = torch.zeros([x if x else 1 for x in input])
        with torch.no_grad():
            o, _ = self.forward(input_shape)
        self.output_shape = tuple(o.shape)
        return self.output_shape  # type: ignore

    def deserialize(self, name, spec):
        """
        Noop for deserialization
        """
        pass

    def serialize(self, name: str, input: str, builder) -> str:
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = 'reshape'
        params.description = 'A generalized reshape layer'
        params.parameters['src_dim'].intValue = self.src_dim
        params.parameters['part_a'].intValue = self.part_a
        params.parameters['part_b'].intValue = self.part_b
        params.parameters['high'].intValue = self.high
        params.parameters['low'].intValue = self.low

        builder.add_custom(name,
                           input_names=[input],
                           output_names=[name],
                           custom_proto_spec=params)
        return name


class MaxPool(Module):
    """
    A simple wrapper for MaxPool layers
    """

    def __init__(self, kernel_size: Tuple[int, int], stride: Tuple[int, int]) -> None:
        """
        A wrapper around MaxPool layers with serialization and layer arithmetic.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer = torch.nn.MaxPool2d(kernel_size, stride)

    def forward(self, inputs: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        o = self.layer(inputs)
        if seq_len is not None:
            seq_len = torch.floor((seq_len-(self.kernel_size[1]-1)-1).float()/self.stride[1]+1).int()
        return o, seq_len

    def get_shape(self, input: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        self.output_shape = (input[0],
                             input[1],
                             int(np.floor((input[2]-(self.kernel_size[0]-1)-1)/self.stride[0]+1) if input[2] != 0 else 0),
                             int(np.floor((input[3]-(self.kernel_size[1]-1)-1)/self.stride[1]+1) if input[3] != 0 else 0))
        return self.output_shape

    def deserialize(self, name, spec) -> None:
        """
        Noop for MaxPool deserialization
        """
        pass

    def serialize(self, name: str, input: str, builder) -> str:
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

    def __init__(self, p: float, dim: int) -> None:
        """
        A wrapper around dropout layers with serialization and layer arithmetic.
        """
        super().__init__()
        self.p = p
        self.dim = dim
        if dim == 1:
            self.layer = torch.nn.Dropout(p)
        elif dim == 2:
            self.layer = torch.nn.Dropout2d(p)

    def forward(self, inputs: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.layer(inputs), seq_len

    def get_shape(self, input: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
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

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 direction: str = 'b',
                 transpose: bool = True,
                 summarize: bool = True,
                 legacy: Optional[str] = None) -> None:
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
        super().__init__()
        self.transpose = transpose
        self.summarize = summarize
        self.legacy = legacy
        self.input_size = input_size
        if self.legacy is not None:
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

    def forward(self, inputs: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # NCHW -> HNWC
        inputs = inputs.permute(2, 0, 3, 1)
        if self.transpose:
            # HNWC -> WNHC
            inputs = inputs.transpose(0, 2)
        if self.legacy is not None:
            ones = torch.ones(inputs.shape[:3] + (1,))
            inputs = torch.cat([ones, inputs], dim=3)
        # HNWC -> (H*N)WC
        siz = inputs.size()
        inputs = inputs.contiguous().view(-1, siz[2], siz[3])
        if not self.transpose and seq_len is not None:
            if inputs.shape[0] != len(seq_len):
                raise Exception(f'Height has to be 1 (not f{inputs.shape[0]} for batching/multi-sequences.')
            seq_len = seq_len.cpu()
            inputs = pack_padded_sequence(inputs, seq_len, batch_first=True, enforce_sorted=False)
        # (H*N)WO
        o, _ = self.layer(inputs)
        if not self.transpose and seq_len is not None:
            o, seq_len = pad_packed_sequence(o, batch_first=True)
        # resize to HNWO
        o = o.view(siz[0], siz[1], siz[2], self.output_size)
        if self.summarize:
            # HN1O
            o = o[:, :, -1, :].unsqueeze(2)
        if self.transpose:
            o = o.transpose(0, 2)
        # HNWO -> NOHW
        if seq_len is not None and seq_len.max() > o.shape[2]:
            raise Exception('Do not use summarizing layer in x-axis with batching/sequences')
        return o.permute(1, 3, 0, 2), seq_len

    def get_shape(self, input: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
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
        return self.output_shape  # type: ignore

    def deserialize(self, name: str, spec) -> None:
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

    def serialize(self, name: str, input: str, builder) -> str:
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
                                  b=_reorder_indim((self.layer.bias_ih_l0 + self.layer.bias_hh_l0)
                                                   ) if not self.legacy else None,
                                  W_h_back=_reorder_indim(self.layer.weight_hh_l0_reverse),
                                  W_x_back=_reorder_indim(self.layer.weight_ih_l0_reverse),
                                  b_back=_reorder_indim((self.layer.bias_ih_l0_reverse +
                                                        self.layer.bias_hh_l0_reverse)) if not self.legacy else None,
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

    def __init__(self, input_size: int, output_size: int, augmentation: bool = False) -> None:
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
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.augmentation = augmentation
        if self.augmentation:
            self.input_size += 1

        self.lin = torch.nn.Linear(self.input_size, output_size)

    def forward(self, inputs: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # move features (C) to last dimension for linear activation
        # NCHW -> NWHC
        inputs = inputs.transpose(1, 3)
        # augment with ones along the input (C) axis
        if self.augmentation:
            inputs = torch.cat([torch.ones(inputs.shape[:3] + (1,)), inputs], dim=3)
        o = self.lin(inputs)
        # switch between log softmax (needed by ctc) and regular (for inference).
        if not self.training:
            o = F.softmax(o, dim=3)
        else:
            o = F.log_softmax(o, dim=3)
        # and swap again
        return o.transpose(1, 3), seq_len

    def get_shape(self, input: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Calculates the output shape from input 4D tuple NCHW.
        """
        self.output_shape = (input[0], self.output_size, input[2], input[3])
        return self.output_shape

    def deserialize(self, name: str, spec) -> None:
        """
        Sets the weights of an initialized module from a CoreML protobuf spec.
        """
        # extract conv parameters
        lin = [x for x in spec.neuralNetwork.layers if x.name == '{}_lin'.format(name)][0].innerProduct
        weights = torch.Tensor(lin.weights.floatValue).resize_as_(self.lin.weight.data)
        bias = torch.Tensor(lin.bias.floatValue)
        self.lin.weight = torch.nn.Parameter(weights)
        self.lin.bias = torch.nn.Parameter(bias)

    def serialize(self, name: str, input: str, builder):
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

    def resize(self, output_size: int, del_indices: Optional[Iterable[int]] = None) -> None:
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

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride: Tuple[int, int], nl: str = 'l') -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.padding = tuple((k - 1) // 2 for k in kernel_size)
        if nl == 's':
            self.nl = torch.sigmoid
            self.nl_name = 'SIGMOID'
        elif nl == 't':
            self.nl = torch.tanh
            self.nl_name = 'TANH'
        elif nl == 'm':
            self.nl = torch.nn.Softmax(dim=1)
            self.nl_name = 'SOFTMAX'
        elif nl == 'r':
            self.nl = torch.relu
            self.nl_name = 'RELU'
        elif nl == 'lr':
            self.nl = torch.nn.LeakyReLU()
            self.nl_name = 'LEAKYRELU'
        else:
            self.nl_name = 'LINEAR'
            self.nl = lambda x: x
        self.co = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=self.padding)

    def forward(self, inputs: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        o = self.co(inputs)
        # return logits for sigmoid activation during training
        if not (self.nl_name == 'SIGMOID' and self.training):
            o = self.nl(o)

        if seq_len is not None:
            seq_len = torch.clamp(torch.floor(
                (seq_len+2*self.padding[1]-(self.kernel_size[1]-1)-1).float()/self.stride[1]+1), min=1).int()
        return o, seq_len

    def get_shape(self, input: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        self.output_shape = (input[0],
                             self.out_channels,
                             int(max(np.floor((input[2]+2*self.padding[0]-(self.kernel_size[0]-1)-1) /
                                 self.stride[0]+1), 1) if input[2] != 0 else 0),
                             int(max(np.floor((input[3]+2*self.padding[1]-(self.kernel_size[1]-1)-1)/self.stride[1]+1), 1) if input[3] != 0 else 0))
        return self.output_shape

    def deserialize(self, name: str, spec) -> None:
        """
        Sets the weight of an initialized model from a CoreML protobuf spec.
        """
        conv = [x for x in spec.neuralNetwork.layers if x.name == '{}_conv'.format(name)][0].convolution
        self.co.weight = torch.nn.Parameter(torch.Tensor(conv.weights.floatValue).view(self.out_channels,
                                                                                       self.in_channels,
                                                                                       *self.kernel_size))
        self.co.bias = torch.nn.Parameter(torch.Tensor(conv.bias.floatValue))

    def serialize(self, name: str, input: str, builder) -> str:
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
                                stride_height=self.stride[0],
                                stride_width=self.stride[1],
                                border_mode='same',
                                groups=1,
                                W=self.co.weight.permute(2, 3, 1, 0).data.numpy(),
                                b=self.co.bias.data.numpy(),
                                has_bias=True,
                                input_name=input,
                                output_name=conv_name)
        if self.nl_name != 'SOFTMAX':
            builder.add_activation(act_name, self.nl_name, conv_name, name, params=None if self.nl_name != 'LEAKYRELU' else [self.nl.negative_slope])
        else:
            builder.add_softmax(act_name, conv_name, name)
        return name

    def resize(self, output_size: int, del_indices: Optional[Iterable[int]] = None) -> None:
        """
        Resizes the convolutional filters of the layer

        First removes the filters at output positions in del_indices, then
        resizes both tensors to a new output size.

        Args:
            output_size (int): Desired output dimensionality after resizing
            del_indices (list): List of connection to outputs to remove.
        """
        if not del_indices:
            del_indices = []
        old_shape = self.co.weight.size(0)
        self.out_channels = output_size
        idx = torch.tensor([x for x in range(old_shape) if x not in del_indices])
        weight = self.co.weight.index_select(0, idx)
        rweight = torch.zeros((output_size - weight.size(0), weight.size(1), weight.size(2), weight.size(3)))
        if rweight.shape[0] > 0:
            torch.nn.init.xavier_uniform_(rweight)
        weight = torch.cat([weight, rweight], dim=0)
        bias = self.co.bias.index_select(0, idx)
        bias = torch.cat([bias, torch.zeros(output_size - bias.size(0))])
        self.co = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                                  stride=self.stride, padding=self.padding)
        self.co.weight = torch.nn.Parameter(weight)
        self.co.bias = torch.nn.Parameter(bias)


class GroupNorm(Module):
    """
    A group normalization layer
    """

    def __init__(self, in_channels: int, num_groups: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = num_groups

        self.layer = torch.nn.GroupNorm(num_groups, in_channels)

    def forward(self, inputs: torch.Tensor, seq_len: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        o = self.layer(inputs)
        return o, seq_len

    def get_shape(self, input: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        self.output_shape = input
        return self.output_shape  # type: ignore

    def deserialize(self, name: str, spec) -> None:
        """
        Sets the weight of an initialized model from a CoreML protobuf spec.
        """
        gn = [x for x in spec.neuralNetwork.layers if x.name == '{}'.format(name)][0].custom
        self.layer.weight = torch.nn.Parameter(torch.Tensor(gn.weights[0].floatValue).resize_as_(self.layer.weight))
        self.layer.bias = torch.nn.Parameter(torch.Tensor(gn.weights[1].floatValue).resize_as_(self.layer.bias))

    def serialize(self, name: str, input: str, builder) -> str:
        """
        Serializes the module using a NeuralNetworkBuilder.
        """
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = 'groupnorm'
        params.description = 'A Group Normalization layer'
        params.parameters['in_channels'].intValue = self.in_channels
        params.parameters['num_groups'].intValue = self.num_groups

        weight = params.weights.add()
        weight.floatValue.extend(self.layer.weight.data.numpy())
        bias = params.weights.add()
        bias.floatValue.extend(self.layer.bias.data.numpy())

        builder.add_custom(name,
                           input_names=[input],
                           output_names=[name],
                           custom_proto_spec=params)
        return name
