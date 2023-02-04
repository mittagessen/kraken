"""
VGSL plumbing
"""
import re
import json
import torch
import logging
import warnings

from torch import nn
from os import PathLike
from typing import Sequence, List, Tuple, Union, Optional, Iterable, Callable, Dict, Any

from kraken.lib import layers
from kraken.lib.codec import PytorchCodec
from kraken.lib.exceptions import KrakenInvalidModelException

# filter out coreml warnings coming from their conversion routines (which we don't use).
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', message='has not been tested with coremltools')
    warnings.filterwarnings(action='ignore', message='is not supported')

    from coremltools.models import MLModel
    from coremltools.models import datatypes
    from coremltools.models.neural_network import NeuralNetworkBuilder

from google.protobuf.message import DecodeError

# all tensors are ordered NCHW, the "feature" dimension is C, so the output of
# an LSTM will be put into C same as the filters of a CNN.

__all__ = ['TorchVGSLModel']

logger = logging.getLogger(__name__)


class VGSLBlock(object):
    def __init__(self, block: str, layer: str, name: str, idx: int):
        if name:
            name = name[1:-1]
        else:
            name = '{}_{}'.format(re.sub(r'\W+', '_', layer), idx)
        block = re.sub(r'\{.+\}', '', block)
        lsplits = re.split(r'(^[^\d]+)', block)
        lsplits.insert(-1, '{{{}}}'.format(name))
        self._block = ''.join(lsplits)
        self._name = name
        self._layer = layer

    def __str__(self):
        return self._block

    @property
    def name(self):
        return self._name

    @property
    def layer(self):
        return self._layer


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
        input: Expected input tensor as a 4-tuple.
        nn: Stack of layers parsed from the spec.
        criterion: Fully parametrized loss function.
        user_metadata: dict with user defined metadata. Is flushed into
                             model file during saving/overwritten by loading
                             operations.
        one_channel_mode: Field indicating the image type used during
                                training of one-channel images. Is '1' for
                                models trained on binarized images, 'L' for
                                grayscale, and None otherwise.
    """
    def __init__(self, spec: str) -> None:
        """
        Constructs a torch module from a (subset of) VSGL spec.

        Args:
            spec: Model definition similar to tesseract as follows:
                  ============ FUNCTIONAL OPS ============
                  C(s|t|r|l|rl|m)[{name}]<y>,<x>,<d>[,<y_stride>,<x_stride>]
                    Convolves using a y,x window, with no shrinkage, SAME
                    infill, d outputs, with s|t|r|l|m non-linear layer.
                    (s|t|r|l|m) specifies the type of non-linearity:
                    s = sigmoid
                    t = tanh
                    r = relu
                    lr = leaky relu
                    l = linear (i.e., None)
                    m = softmax
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
                  Do[{name}][<p>,<d>] Insert a dropout layer operating in
                                      <d> dimensions with probability
                                      <p>. Defaults to 1D with 0.5
                                      probability.
                  Gn[{name}]<n> A group normalization layer with n groups
                  ============ PLUMBING OPS ============
                  [...] Execute ... networks in series (layers).
                  (...) Execute ... networks in parallel.
                  I[{name}] Identity function to build residual connections in parallel layers.
                  Mp[{name}]<y>,<x>[<y_stride>,<x_stride>] Maxpool the input, reducing the (y,x) rectangle to a
                    single vector value.
                  S[{name}]<d>(<a>x<b>)<e>,<f> Splits one dimension, moves one part to another
                    dimension.
        """
        self.spec = spec
        self.named_spec = []  # type:  List[str]
        self.ops = [self.build_addition, self.build_identity, self.build_rnn,
                    self.build_dropout, self.build_maxpool, self.build_conv,
                    self.build_output, self.build_reshape, self.build_wav2vec2,
                    self.build_groupnorm, self.build_series,
                    self.build_parallel]
        self.codec = None  # type: Optional[PytorchCodec]
        self.criterion = None  # type: Any
        self.nn = layers.MultiParamSequential()
        self.user_metadata = {'accuracy': [],
                              'metrics': [],
                              'seg_type': None,
                              'one_channel_mode': None,
                              'model_type': None,
                              'hyper_params': {}}  # type: dict[str, Any]
        self._aux_layers = nn.ModuleDict()

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
        self.input = (batch, channels, height, width)
        named_spec, self.nn, self.output = self._parse(self.input, blocks)
        self.named_spec.extend(str(x) for x in named_spec)
        self.init_weights()

    def _parse(self, input: Tuple[int, int, int, int], blocks: Sequence[str], parallel=False) -> None:
        """
        Parses VGSL spec and appends layers to nn
        """
        logger.debug('layer\t\ttype\tparams')
        named_spec = []
        if not parallel:
            nn = layers.MultiParamSequential()
        else:
            nn = layers.MultiParamParallel()
            prev_oshape = None
            channels = 0
        idx = 0
        while idx < len(blocks):
            oshape = None
            layer = None
            for op in self.ops:
                oshape, name, layer = op(input, blocks, idx)
                if oshape:
                    break
            if oshape:
                if not parallel:
                    input = oshape
                else:
                    if prev_oshape and prev_oshape[2:] != oshape[2:]:
                        raise ValueError('Output shape in parallel block not equal!')
                    else:
                        prev_oshape = oshape
                        channels += oshape[1]
                named_spec.extend(name)  # type: ignore
                idx += len(name)
                nn.add_module(' '.join(n.name for n in name), layer)
            else:
                raise ValueError('{} invalid layer definition'.format(blocks[idx]))
        if parallel:
            return named_spec, nn, (oshape[0], channels, *oshape[2:])
        else:
            return named_spec, nn, oshape

    def append(self, idx: int, spec: str) -> None:
        """
        Splits a model at layer `idx` and append layers `spec`.

        New layers are initialized using the init_weights method.

        Args:
            idx (int): Index of layer to append spec to starting with 1.  To
                       select the whole layer stack set idx to None.
            spec (str): VGSL spec without input block to append to model.
        """
        self.nn = self.nn[:idx]
        self.idx = idx-1
        spec = spec[1:-1]
        blocks = spec.split(' ')
        self.named_spec = self.named_spec[:idx+1]
        named_spec, nn, self.output = self._parse(self.nn[-1].output_shape, blocks)
        self.named_spec.extend(str(x) for x in named_spec)
        for module in nn.named_children():
            self.nn.add_module(*module)
        self.spec = '[' + ' '.join(self.named_spec) + ']'
        self.init_weights(slice(idx, -1))

    def to(self, device: Union[str, torch.device]) -> None:
        self.nn = self.nn.to(device)
        if self.criterion:
            self.criterion = self.criterion.to(device)

    def eval(self) -> None:
        """
        Sets the model to evaluation/inference mode, disabling dropout and
        gradient calculation.
        """
        self.nn.eval()
        torch.set_grad_enabled(False)

    def train(self) -> None:
        """
        Sets the model to training mode (enables dropout layers and disables
        softmax on CTC layers).
        """
        self.nn.train()
        # set last layer back to eval mode if not CTC output layer
        # (log_softmax/softmax switch).
        if not self.criterion:
            self.nn[-1].eval()
        torch.set_grad_enabled(True)

    def set_num_threads(self, num: int) -> None:
        """
        Sets number of OpenMP threads to use.
        """
        torch.set_num_threads(num)

    @classmethod
    def load_model(cls, path: Union[str, PathLike]):
        """
        Deserializes a VGSL model from a CoreML file.

        Args:
            path: CoreML file

        Returns:
            A TorchVGSLModel instance.

        Raises:
            KrakenInvalidModelException if the model data is invalid (not a
            string, protobuf file, or without appropriate metadata).
            FileNotFoundError if the path doesn't point to a file.
        """
        if isinstance(path, PathLike):
            path = path.as_posix()
        try:
            mlmodel = MLModel(path)
        except TypeError as e:
            raise KrakenInvalidModelException(str(e)) from e
        except DecodeError as e:
            raise KrakenInvalidModelException('Failure parsing model protobuf: {}'.format(str(e))) from e
        if 'vgsl' not in mlmodel.user_defined_metadata:
            raise KrakenInvalidModelException('No VGSL spec in model metadata')
        vgsl_spec = mlmodel.user_defined_metadata['vgsl']
        nn = cls(vgsl_spec)

        def _deserialize_layers(name, layer):
            logger.debug(f'Deserializing layer {name} with type {type(layer)}')
            if type(layer) in (layers.MultiParamParallel, layers.MultiParamSequential):
                for name, l in layer.named_children():
                    _deserialize_layers(name, l)
            else:
                layer.deserialize(name, mlmodel.get_spec())

        try:
            _deserialize_layers('', nn.nn)
        except Exception as exc:
            raise KrakenInvalidModelException('Failed parsing out layers from model weights') from exc

        if 'aux_layers' in mlmodel.user_defined_metadata:
            logger.info('Deserializing auxiliary layers.')
            nn.aux_layers = {k: cls(v).nn.get_submodule(k) for k, v in json.loads(mlmodel.user_defined_metadata['aux_layers']).items()}

        if 'codec' in mlmodel.user_defined_metadata:
            nn.add_codec(PytorchCodec(json.loads(mlmodel.user_defined_metadata['codec'])))

        nn.user_metadata = {'accuracy': [],
                            'metrics': [],
                            'seg_type': 'bbox',
                            'one_channel_mode': '1',
                            'model_type': None,
                            'hyper_params': {}}  # type: dict[str, str]

        if 'kraken_meta' in mlmodel.user_defined_metadata:
            nn.user_metadata.update(json.loads(mlmodel.user_defined_metadata['kraken_meta']))
        return nn

    @property
    def one_channel_mode(self):
        return self.user_metadata['one_channel_mode']

    @one_channel_mode.setter
    def one_channel_mode(self, val: str):
        if val not in ['1', 'L', None]:
            raise ValueError('one_channel_mode {} is not one of [1, L, None]'.format(val))
        self.user_metadata['one_channel_mode'] = val

    @property
    def model_type(self):
        return self.user_metadata['model_type']

    @model_type.setter
    def model_type(self, val: str):
        if val not in ['recognition', 'segmentation']:
            raise ValueError('model_type {} is not one of [recognition, segmentation]'.format(val))
        self.user_metadata['model_type'] = val

    @property
    def seg_type(self):
        return self.user_metadata['seg_type']

    @seg_type.setter
    def seg_type(self, val: str):
        if val not in ['bbox', 'baselines', None]:
            raise ValueError('segmentation type {} is not one of [bbox, baselines, None]'.format(val))
        self.user_metadata['seg_type'] = val

    @property
    def hyper_params(self, **kwargs):
        return self.user_metadata['hyper_params']

    @hyper_params.setter
    def hyper_params(self, val: Dict[str, Any]):
        self.user_metadata['hyper_params'].update(val)

    @property
    def aux_layers(self, **kwargs):
        return self._aux_layers

    @aux_layers.setter
    def aux_layers(self, val: Dict[str, torch.nn.Module]):
        self._aux_layers.update(val)

    def save_model(self, path: str):
        """
        Serializes the model into path.

        Args:
            path: Target destination
        """
        inputs = [('input', datatypes.Array(*self.input))]
        outputs = [('output', datatypes.Array(*self.output))]
        net_builder = NeuralNetworkBuilder(inputs, outputs)
        input = 'input'
        prev_device = next(self.nn.parameters()).device
        try:
            self.nn.to('cpu')

            def _serialize_layer(net, input, net_builder):
                for name, l in net.named_children():
                    logger.debug(f'Serializing layer {name} with type {type(l)}')
                    if type(l) in (layers.MultiParamParallel, layers.MultiParamSequential):
                        _serialize_layer(l, input, net_builder)
                    else:
                        l.serialize(name, input, net_builder)
            _serialize_layer(self.nn, input, net_builder)
            if self.aux_layers:
                prev_aux_device = next(self.aux_layers.parameters()).device
                try:
                    logger.debug(f'Serializing {len(self.aux_layers)} auxiliary layers')
                    self.aux_layers.to('cpu')
                    _serialize_layer(self.aux_layers, input, net_builder)
                finally:
                    self.aux_layers.to(prev_aux_device)

            mlmodel = MLModel(net_builder.spec)
            mlmodel.short_description = 'kraken model'
            mlmodel.user_defined_metadata['vgsl'] = '[' + ' '.join(self.named_spec) + ']'
            if self.codec:
                mlmodel.user_defined_metadata['codec'] = json.dumps(self.codec.c2l)
            if self.user_metadata:
                mlmodel.user_defined_metadata['kraken_meta'] = json.dumps(self.user_metadata)
            if self.aux_layers:
                mlmodel.user_defined_metadata['aux_layers'] = json.dumps({k: v.get_spec(k) for k, v in self.aux_layers.items()})
            mlmodel.save(path)
        finally:
            self.nn.to(prev_device)

    def add_codec(self, codec: PytorchCodec) -> None:
        """
        Adds a PytorchCodec to the model.
        """
        self.codec = codec

    def init_weights(self, idx: slice = slice(0, None)) -> None:
        """
        Initializes weights for all or a subset of layers in the graph.

        LSTM/GRU layers are orthogonally initialized, convolutional layers
        uniformly from (-0.1,0.1).

        Args:
            idx (slice): A slice object representing the indices of layers to
                         initialize.
        """
        def _wi(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, torch.nn.LSTM):
                for p in m.parameters():
                    # weights
                    if p.data.dim() == 2:
                        torch.nn.init.orthogonal_(p.data)
                    # initialize biases to 1 (jozefowicz 2015)
                    else:
                        torch.nn.init.constant_(p.data[len(p)//4:len(p)//2], 1.0)
            elif isinstance(m, torch.nn.GRU):
                for p in m.parameters():
                    torch.nn.init.orthogonal_(p.data)
            elif isinstance(m, torch.nn.Conv2d):
                for p in m.parameters():
                    torch.nn.init.uniform_(p.data, -0.1, 0.1)
        self.nn[idx].apply(_wi)

    def resize_output(self, output_size: int, del_indices: Optional[Iterable] = None) -> None:
        """
        Resizes an output layer.

        Args:
            output_size (int): New size/output channels of last layer
            del_indices (list): list of outputs to delete from layer
        """
        if type(self.nn[-1]) not in [layers.ActConv2D, layers.LinSoftmax]:
            raise ValueError('last layer is neither linear nor convolutional layer')
        logger.debug('Resizing output layer to {}'.format(output_size))
        self.nn[-1].resize(output_size, del_indices)
        pattern = re.compile(r'(O)(?P<name>{\w+})?(?P<dim>2|1|0)(?P<type>l|s|c)(?P<aug>a)?(?P<out>\d+)')
        m = pattern.match(self.named_spec[-1])
        if not m:
            raise ValueError('Output specification is not parsable')
        aug = m.group('aug') if m.group('aug') else ''
        self.named_spec[-1] = 'O{}{}{}{}{}'.format(m.group('name'), m.group('dim'), m.group('type'), aug, output_size)
        self.spec = '[' + ' '.join(self.named_spec) + ']'

    def build_rnn(self,
                  input: Tuple[int, int, int, int],
                  blocks: List[str],
                  idx: int) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        """
        Builds an LSTM/GRU layer returning number of outputs and layer.
        """
        pattern = re.compile(r'(?P<type>L|G)(?P<dir>f|r|b)(?P<dim>x|y)(?P<sum>s)?(?P<legacy>c|o)?(?P<name>{\w+})?(?P<out>\d+)')
        m = pattern.match(blocks[idx])
        if not m:
            return None, None, None
        type = m.group('type')
        direction = m.group('dir')
        dim = m.group('dim') == 'y'
        summarize = m.group('sum') == 's'
        legacy = None
        if m.group('legacy') == 'c':
            legacy = 'clstm'
        elif m.group('legacy') == 'o':
            legacy = 'ocropy'
        hidden = int(m.group(7))
        fn = layers.TransposedSummarizingRNN(input[1], hidden, direction, dim, summarize, legacy)
        self.idx += 1
        logger.debug(f'{self.idx}\t\trnn\tdirection {direction} transposed {dim} '
                     f'summarize {summarize} out {hidden} legacy {legacy}')
        return fn.get_shape(input), [VGSLBlock(blocks[idx], type, m.group('name'), self.idx)], fn

    def build_dropout(self,
                      input: Tuple[int, int, int, int],
                      blocks: List[str],
                      idx: int) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        pattern = re.compile(r'(?P<type>Do)(?P<name>{\w+})?(?P<p>(\d+(\.\d*)?|\.\d+))?(,(?P<dim>\d+))?')
        m = pattern.match(blocks[idx])
        if not m:
            return None, None, None
        prob = float(m.group('p')) if m.group('p') else 0.5
        dim = int(m.group('dim')) if m.group('dim') else 1
        fn = layers.Dropout(prob, dim)
        self.idx += 1
        logger.debug('{}\t\tdropout\tprobability {} dims {}'.format(self.idx, prob, dim))
        return fn.get_shape(input), [VGSLBlock(blocks[idx], m.group('type'), m.group('name'), self.idx)], fn

    def build_addition(self,
                       input: Tuple[int, int, int, int],
                       blocks: List[str],
                       idx: int) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        pattern = re.compile(r'(?P<type>A)(?P<name>{\w+})?(?P<dim>\d+),(?P<chunk_size>\d+)')
        m = pattern.match(blocks[idx])
        if not m:
            return None, None, None
        dim_map = {0: 0, 1: 2, 2: 3, 3: 1}
        dim = int(m.group('dim'))
        chunk_size = int(m.group('chunk_size'))
        if dim > 3:
            raise ValueError(f'Invalid dimension {dim} in addition block')
        dim = dim_map[dim]
        fn = layers.Addition(dim=dim, chunk_size=chunk_size)
        self.idx += 1
        logger.debug(f'{self.idx}\t\taddition dim: {dim} chunk: {chunk_size}')
        return fn.get_shape(input), [VGSLBlock(blocks[idx], m.group('type'), m.group('name'), self.idx)], fn

    def build_identity(self,
                       input: Tuple[int, int, int, int],
                       blocks: List[str],
                       idx: int) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        pattern = re.compile(r'(?P<type>I)(?P<name>{\w+})?')
        m = pattern.match(blocks[idx])
        if not m:
            return None, None, None
        fn = layers.Identity()
        self.idx += 1
        logger.debug(f'{self.idx}\t\tidentity')
        return fn.get_shape(input), [VGSLBlock(blocks[idx], m.group('type'), m.group('name'), self.idx)], fn

    def build_groupnorm(self,
                        input: Tuple[int, int, int, int],
                        blocks: List[str],
                        idx: int) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        pattern = re.compile(r'(?P<type>Gn)(?P<name>{\w+})?(?P<groups>\d+)')
        m = pattern.match(blocks[idx])
        if not m:
            return None, None, None
        groups = int(m.group('groups'))
        fn = layers.GroupNorm(input[1], groups)
        self.idx += 1
        logger.debug('{}\t\tgroupnorm\tgroups {}'.format(self.idx, groups))
        return fn.get_shape(input), [VGSLBlock(blocks[idx], m.group('type'), m.group('name'), self.idx)], fn

    def build_wav2vec2(self,
                       input: Tuple[int, int, int, int],
                       blocks: List[str],
                       idx: int) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        """
        Builds a Wav2Vec2 masking layer.
        """
        pattern = re.compile(r'(?P<type>W)(?P<name>{\w+})(?P<final_dim>\d+),(?P<mask_width>\d+),(?P<mask_prob>(\d+(\.\d*)?|\.\d+)),(?P<num_negatives>\d+)')
        m = pattern.match(blocks[idx])
        if not m:
            return None, None, None
        final_dim = int(m.group('final_dim'))
        mask_width = int(m.group('mask_width'))
        mask_prob = float(m.group('mask_prob'))
        num_negatives = int(m.group('num_negatives'))
        from kraken.lib import pretrain
        fn = pretrain.layers.Wav2Vec2Mask(input[1], final_dim, mask_width, mask_prob, num_negatives)
        self.idx += 1
        logger.debug(f'{self.idx}\t\twav2vec2\tmask width {mask_width}, prob '
                     f'{mask_prob}, negative samples {num_negatives}')
        return fn.get_shape(input), [VGSLBlock(blocks[idx], m.group('type'), m.group('name'), self.idx)], fn

    def build_conv(self,
                   input: Tuple[int, int, int, int],
                   blocks: List[str],
                   idx: int) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        """
        Builds a 2D convolution layer.
        """
        pattern = re.compile(r'(?P<type>C)(?P<nl>s|t|r|l|lr|m)(?P<name>{\w+})?(\d+),'
                             r'(\d+),(?P<out>\d+)(,(?P<stride_y>\d+),(?P<stride_x>\d+))?')
        m = pattern.match(blocks[idx])
        if not m:
            return None, None, None
        kernel_size = (int(m.group(4)), int(m.group(5)))
        filters = int(m.group('out'))
        stride = (int(m.group('stride_y')), int(m.group('stride_x'))) if m.group('stride_x') else (1, 1)
        nl = m.group('nl')
        fn = layers.ActConv2D(input[1], filters, kernel_size, stride, nl)
        self.idx += 1
        logger.debug(f'{self.idx}\t\tconv\tkernel {kernel_size[0]} x {kernel_size[1]} '
                     f'filters {filters} stride {stride} activation {nl}')
        return fn.get_shape(input), [VGSLBlock(blocks[idx], m.group('type'), m.group('name'), self.idx)], fn

    def build_maxpool(self,
                      input: Tuple[int, int, int, int],
                      blocks: List[str],
                      idx: int) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        """
        Builds a maxpool layer.
        """
        pattern = re.compile(r'(?P<type>Mp)(?P<name>{\w+})?(\d+),(\d+)(?:,(\d+),(\d+))?')
        m = pattern.match(blocks[idx])
        if not m:
            return None, None, None
        kernel_size = (int(m.group(3)), int(m.group(4)))
        stride = (kernel_size[0] if not m.group(5) else int(m.group(5)),
                  kernel_size[1] if not m.group(6) else int(m.group(6)))
        fn = layers.MaxPool(kernel_size, stride)
        self.idx += 1
        logger.debug(f'{self.idx}\t\tmaxpool\tkernel {kernel_size[0]} x {kernel_size[1]} stride {stride[0]} x {stride[1]}')
        return fn.get_shape(input), [VGSLBlock(blocks[idx], m.group('type'), m.group('name'), self.idx)], fn

    def build_reshape(self,
                      input: Tuple[int, int, int, int],
                      blocks: List[str],
                      idx: int) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        """
        Builds a reshape layer
        """
        pattern = re.compile(r'(?P<type>S)(?P<name>{\w+})?(?P<dim>\d+)\((?P<part_a>\d+)x'
                             r'(?P<part_b>\d+)\)(?P<high>\d+),(?P<low>\d+)')
        m = pattern.match(blocks[idx])
        if not m:
            return None, None, None
        src_dim = int(m.group('dim'))
        part_a = int(m.group('part_a'))
        part_b = int(m.group('part_b'))
        high = int(m.group('high'))
        low = int(m.group('low'))
        dim_map = {0: 0, 1: 2, 2: 3, 3: 1}

        if part_a == 0:
            part_a = -1
        elif part_b == 0:
            part_b = -1

        if src_dim != high and src_dim != low:
            raise ValueError('Either high ({}) or low ({}) must be source dimension ({})'.format(high, low, src_dim))
        if part_a == 0 or part_b == 0:
            raise ValueError('Expected non-zero size for part_a ({}) or part_b ({})'.format(part_a, part_b))
        if part_a == -1 and part_b == -1:
            raise ValueError('Only one size may be -1')
        self.idx += 1
        logger.debug('{}\t\treshape from {} {} x {} to {}/{}'.format(self.idx, src_dim, part_a, part_b, high, low))
        src_dim = dim_map[src_dim]
        high = dim_map[high]
        low = dim_map[low]
        fn = layers.Reshape(src_dim, part_a, part_b, high, low)
        return fn.get_shape(input), [VGSLBlock(blocks[idx], m.group('type'), m.group('name'), self.idx)], fn

    def build_output(self,
                     input: Tuple[int, int, int, int],
                     blocks: List[str],
                     idx: int) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        """
        Builds an output layer.
        """
        pattern = re.compile(r'(O)(?P<name>{\w+})?(?P<dim>2|1|0)(?P<type>l|s|c)(?P<aug>a)?(?P<out>\d+)')
        m = pattern.match(blocks[idx])
        if not m:
            return None, None, None
        dim = int(m.group('dim'))
        nl = m.group('type')
        outdim = int(m.group('out'))
        if dim == 0:
            raise ValueError('categorical output not supported, yet.')
        if nl == 'c' and dim == 2:
            raise ValueError('CTC not supported for heatmap output')
        if nl in ['l', 's'] and int(m.group('out')) >= 1:
            self.criterion = nn.BCEWithLogitsLoss()
        elif nl == 'c':
            self.criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)
        else:
            raise ValueError('unsupported output specification')
        # heatmap output
        if dim == 2:
            act = 's' if nl == 'l' else 'm'
            fn = layers.ActConv2D(input[1], outdim, (1, 1), (1, 1), act)
            self.idx += 1
            logger.debug('{}\t\tconv\tkernel 1 x 1 filters {} stride 1 activation {}'.format(self.idx, outdim, nl))
            return fn.get_shape(input), [VGSLBlock(blocks[idx], m.group('type'), m.group('name'), self.idx)], fn
        else:
            aug = True if m.group('aug') else False
            lin = layers.LinSoftmax(input[1], int(m.group('out')), aug)
            self.idx += 1
            logger.debug('{}\t\tlinear\taugmented {} out {}'.format(self.idx, aug, m.group('out')))
            return lin.get_shape(input), [VGSLBlock(blocks[idx], m.group(1), m.group('name'), self.idx)], lin

    def build_series(self,
                     input: Tuple[int, int, int, int],
                     blocks: List[str],
                     idx: int) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        """
        Builds a serial block of layers.
        """
        if not blocks[idx] or blocks[idx][0] != '[':
            return None, None, None
        # single layer in serial block
        if blocks[idx][0] == '[' and blocks[idx][-1] == ']':
            named_spec, nn, oshape = self._parse(input, [blocks[idx][1:-1]])
            named_spec[0]._block = '[' + named_spec[0]._block + ']'
            return oshape, named_spec, nn
        # multiple layers in serial block
        block_depth = 0
        for bl_idx, block in enumerate(blocks[idx:]):
            if block[0] == '[':
                block_depth += 1
            if block[-1] == ']':
                block_depth -= 1
                if block_depth == 0:
                    break
        if block_depth:
            raise ValueError('Unbalanced parentheses in VGSL spec')
        named_spec, nn, oshape = self._parse(input, [blocks[idx][1:]] + blocks[idx+1:idx+bl_idx] + [blocks[idx+bl_idx][:-1]])
        named_spec[0]._block = '[' + named_spec[0]._block
        named_spec[-1]._block = named_spec[-1]._block + ']'
        return oshape, named_spec, nn

    def build_parallel(self,
                       input: Tuple[int, int, int, int],
                       blocks: List[str],
                       idx: int) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        """
        Builds a block of parallel layers.
        """
        if not blocks[idx] or blocks[idx][0] != '(':
            return None, None, None
        # single layer in parallel block
        if blocks[idx][0] == '(' and blocks[idx][-1] == ')':
            named_spec, nn, oshape = self._parse(input, [blocks[idx][1:-1]], parallel=True)
            named_spec[0]._block = '(' + named_spec[0]._block + ')'
            return oshape, named_spec, nn
        block_depth = 0
        for bl_idx, block in enumerate(blocks[idx:]):
            if block[0] == '(':
                block_depth += 1
            if block[-1] == ')':
                block_depth -= 1
                if block_depth == 0:
                    break
        if block_depth:
            raise ValueError('Unbalanced parentheses in VGSL spec')
        named_spec, nn, oshape = self._parse(input, [blocks[idx][1:]] + blocks[idx+1:idx+bl_idx] + [blocks[idx+bl_idx][:-1]], parallel=True)
        named_spec[0]._block = '(' + named_spec[0]._block
        named_spec[-1]._block = named_spec[-1]._block + ')'
        return oshape, named_spec, nn
