"""
VGSL plumbing
"""
from future.utils import PY2

import io
import re
import sys
import json
import gzip
import torch
import logging

from torch import nn
from typing import Sequence, List, Tuple, Union, Optional, Iterable, Callable, Dict, Any

import kraken.lib.lstm

from kraken.lib import layers
from kraken.lib import clstm_pb2
from kraken.lib import pyrnn_pb2
from kraken.lib.codec import PytorchCodec
from kraken.lib.exceptions import KrakenInvalidModelException

from coremltools.models import MLModel
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder

from google.protobuf.message import DecodeError

# all tensors are ordered NCHW, the "feature" dimension is C, so the output of
# an LSTM will be put into C same as the filters of a CNN.

__all__ = ['TorchVGSLModel']

logger = logging.getLogger(__name__)


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
        user_metdata (dict): dict with user defined metadata. Is flushed into
                             model file during saving/overwritten by loading
                             operations.
        one_channel_mode (str): Field indicating the image type used during
                                training of one-channel images. Is '1' for
                                models trained on binarized images, 'L' for
                                grayscale, and None otherwise.
    """
    def __init__(self, spec: str) -> None:
        """
        Constructs a torch module from a (subset of) VSGL spec.

        Args:
            spec (str): Model definition similar to tesseract as follows:
                        ============ FUNCTIONAL OPS ============
                        C(s|t|r|l|m)[{name}]<y>,<x>,<d>[,<y_stride>,<x_stride>]
                          Convolves using a y,x window, with no shrinkage, SAME
                          infill, d outputs, with s|t|r|l|m non-linear layer.
                          (s|t|r|l|m) specifies the type of non-linearity:
                          s = sigmoid
                          t = tanh
                          r = relu
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
                        G(f|r|b)(x|y)[s][{name}]<n> GRU cell with n outputs.
                          Arguments are equivalent to LSTM specs.
                        Do[{name}] Insert a 1D dropout layer with 0.5 drop probability.
                        ============ PLUMBING OPS ============
                        [...] Execute ... networks in series (layers).
                        Mp[{name}]<y>,<x>[<y_stride>,<x_stride>] Maxpool the input, reducing the (y,x) rectangle to a
                          single vector value.
                        S[{name}]<d>(<a>x<b>)<e>,<f> Splits one dimension, moves one part to another
                          dimension.
        """
        self.spec = spec
        self.named_spec = []  # type:  List[str]
        self.ops = [self.build_rnn, self.build_dropout, self.build_maxpool, self.build_conv, self.build_output, self.build_reshape, self.build_groupnorm]
        self.codec = None  # type: Optional[PytorchCodec]
        self.criterion = None  # type: Any
        self.nn = layers.MultiParamSequential()
        self.user_metadata = {'accuracy': [], 'seg_type': None, 'one_channel_mode': None, 'model_type': None, 'hyper_params': {}}  # type: dict[str, str]

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
        self._parse(self.input, blocks)

    def _parse(self, input: Tuple[int, int, int, int], blocks: Sequence[str]) -> None:
        """
        Parses VGSL spec and appends layers to self.nn
        """
        logger.debug('layer\t\ttype\tparams')
        for block in blocks:
            oshape = None
            layer = None
            for op in self.ops:
                oshape, name, layer = op(input, block)
                if oshape:
                    break
            if oshape:
                input = oshape
                self.named_spec.append(self.set_layer_name(block, name))  # type: ignore
                self.nn.add_module(name, layer)
            else:
                raise ValueError('{} invalid layer definition'.format(block))
        self.output = oshape

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
        self._parse(self.nn[-1].output_shape, blocks)
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
    def load_pyrnn_model(cls, path: str):
        """
        Loads an pyrnn model to VGSL.
        """
        if not PY2:
            raise KrakenInvalidModelException('Loading pickle models is not supported on python 3')

        import cPickle

        def find_global(mname, cname):
            aliases = {
                'lstm.lstm': kraken.lib.lstm,
                'ocrolib.lstm': kraken.lib.lstm,
                'ocrolib.lineest': kraken.lib.lineest,
            }
            if mname in aliases:
                return getattr(aliases[mname], cname)
            return getattr(sys.modules[mname], cname)

        of = io.open
        if path.endswith('.gz'):
            of = gzip.open
        with io.BufferedReader(of(path, 'rb')) as fp:
            unpickler = cPickle.Unpickler(fp)
            unpickler.find_global = find_global
            try:
                net = unpickler.load()
            except Exception as e:
                raise KrakenInvalidModelException(str(e))
            if not isinstance(net, kraken.lib.lstm.SeqRecognizer):
                raise KrakenInvalidModelException('Pickle is %s instead of '
                                                  'SeqRecognizer' %
                                                  type(net).__name__)
        # extract codec
        codec = PytorchCodec({k: [v] for k, v in net.codec.char2code.items()})

        input = net.Ni
        parallel, softmax = net.lstm.nets
        fwdnet, revnet = parallel.nets
        revnet = revnet.net

        hidden = fwdnet.WGI.shape[0]

        # extract weights
        weightnames = ('WGI', 'WGF', 'WCI', 'WGO', 'WIP', 'WFP', 'WOP')

        fwd_w = []
        rev_w = []
        for w in weightnames:
            fwd_w.append(torch.Tensor(getattr(fwdnet, w)))
            rev_w.append(torch.Tensor(getattr(revnet, w)))

        t = torch.cat(fwd_w[:4])
        weight_ih_l0 = t[:, :input+1]
        weight_hh_l0 = t[:, input+1:]

        t = torch.cat(rev_w[:4])
        weight_ih_l0_rev = t[:, :input+1]
        weight_hh_l0_rev = t[:, input+1:]

        weight_lin = torch.Tensor(softmax.W2)

        # build vgsl spec and set weights
        nn = cls('[1,1,0,{} Lbxo{} O1ca{}]'.format(input, hidden, len(net.codec.code2char)))

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
    def load_pronn_model(cls, path: str):
        """
        Loads an pronn model to VGSL.
        """
        with open(path, 'rb') as fp:
            net = pyrnn_pb2.pyrnn()
            try:
                net.ParseFromString(fp.read())
            except Exception:
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
            fwd_w.append(torch.Tensor(fwd_ar.value).view(list(fwd_ar.dim)))
            rev_w.append(torch.Tensor(rev_ar.value).view(list(rev_ar.dim)))

        t = torch.cat(fwd_w[:4])
        weight_ih_l0 = t[:, :input+1]
        weight_hh_l0 = t[:, input+1:]

        t = torch.cat(rev_w[:4])
        weight_ih_l0_rev = t[:, :input+1]
        weight_hh_l0_rev = t[:, input+1:]

        weight_lin = torch.Tensor(net.softmax.w2.value).view(list(net.softmax.w2.dim))

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
    def load_clstm_model(cls, path: str):
        """
        Loads an CLSTM model to VGSL.
        """
        net = clstm_pb2.NetworkProto()
        with open(path, 'rb') as fp:
            try:
                net.ParseFromString(fp.read())
            except Exception:
                raise KrakenInvalidModelException('File does not contain valid proto msg')
            if not net.IsInitialized():
                raise KrakenInvalidModelException('Model incomplete')

        input = net.ninput
        attrib = {a.key: a.value for a in list(net.attribute)}
        # mainline clstm model
        if len(attrib) > 1:
            mode = 'clstm'
        else:
            mode = 'clstm_compat'

        # extract codec
        codec = PytorchCodec([''] + [chr(x) for x in net.codec[1:]])

        # separate layers
        nets = {}
        nets['softm'] = [n for n in list(net.sub) if n.kind == 'SoftmaxLayer'][0]
        parallel = [n for n in list(net.sub) if n.kind == 'Parallel'][0]
        nets['lstm1'] = [n for n in list(parallel.sub) if n.kind.startswith('NPLSTM')][0]
        rev = [n for n in list(parallel.sub) if n.kind == 'Reversed'][0]
        nets['lstm2'] = rev.sub[0]

        hidden = int(nets['lstm1'].attribute[0].value)

        weights = {}  # type: Dict[str, torch.Tensor]
        for n in nets:
            weights[n] = {}
            for w in list(nets[n].weights):
                weights[n][w.name] = torch.Tensor(w.value).view(list(w.dim))

        if mode == 'clstm_compat':
            weightnames = ('.WGI', '.WGF', '.WCI', '.WGO')
            weightname_softm = '.W'
        else:
            weightnames = ('WGI', 'WGF', 'WCI', 'WGO')
            weightname_softm = 'W1'

        # input hidden and hidden-hidden weights are in one matrix. also
        # CLSTM/ocropy likes 1-augmenting every other tensor so the ih weights
        # are input+1 in one dimension.
        t = torch.cat(list(w for w in [weights['lstm1'][wn] for wn in weightnames]))
        weight_ih_l0 = t[:, :input+1]
        weight_hh_l0 = t[:, input+1:]

        t = torch.cat(list(w for w in [weights['lstm2'][wn] for wn in weightnames]))
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
    def load_model(cls, path: str):
        """
        Deserializes a VGSL model from a CoreML file.

        Args:
            path (str): CoreML file

        Returns:
            A TorchVGSLModel instance.

        Raises:
            KrakenInvalidModelException if the model data is invalid (not a
            string, protobuf file, or without appropriate metadata).
            FileNotFoundError if the path doesn't point to a file.
        """
        try:
            mlmodel = MLModel(path)
        except TypeError as e:
            raise KrakenInvalidModelException(str(e))
        except DecodeError as e:
            raise KrakenInvalidModelException('Failure parsing model protobuf: {}'.format(str(e)))
        if 'vgsl' not in mlmodel.user_defined_metadata:
            raise KrakenInvalidModelException('No VGSL spec in model metadata')
        vgsl_spec = mlmodel.user_defined_metadata['vgsl']
        nn = cls(vgsl_spec)
        for name, layer in nn.nn.named_children():
            layer.deserialize(name, mlmodel.get_spec())

        if 'codec' in mlmodel.user_defined_metadata:
            nn.add_codec(PytorchCodec(json.loads(mlmodel.user_defined_metadata['codec'])))

        nn.user_metadata = {'accuracy': [], 'seg_type': 'bbox', 'one_channel_mode': '1', 'model_type': None, 'hyper_params': {}}  # type: dict[str, str]
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

    def save_model(self, path: str):
        """
        Serializes the model into path.

        Args:
            path (str): Target destination
        """
        inputs = [('input', datatypes.Array(*self.input))]
        outputs = [('output', datatypes.Array(*self.output))]
        net_builder = NeuralNetworkBuilder(inputs, outputs)
        input = 'input'
        prev_device = next(next(self.nn.children()).parameters()).device
        try:
            for name, layer in self.nn.to('cpu').named_children():
                input = layer.serialize(name, input, net_builder)
            mlmodel = MLModel(net_builder.spec)
            mlmodel.short_description = 'kraken model'
            mlmodel.user_defined_metadata['vgsl'] = '[' + ' '.join(self.named_spec) + ']'
            if self.codec:
                mlmodel.user_defined_metadata['codec'] = json.dumps(self.codec.c2l)
            if self.user_metadata:
                mlmodel.user_defined_metadata['kraken_meta'] = json.dumps(self.user_metadata)
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

    @staticmethod
    def set_layer_name(layer: str, name: str) -> str:
        """
        Sets the name field of an VGSL layer definition.

        Args:
            layer (str): VGSL definition
            name (str): Layer name
        """
        if '{' in layer and '}' in layer:
            return layer
        lsplits = re.split(r'(^[^\d]+)', layer)
        lsplits.insert(-1, '{{{}}}'.format(name))
        return ''.join(lsplits)

    def get_layer_name(self, layer: str, name: Optional[str] = None) -> str:
        """
        Generates a unique identifier for the layer optionally using a supplied
        name.

        Args:
            layer (str): Identifier of the layer type
            name (str): user-supplied {name} with {} that need removing.

        Returns:
            (str) network unique layer name
        """
        self.idx += 1
        if name:
            return name[1:-1]
        else:
            return '{}_{}'.format(re.sub(r'\W+', '_', layer), self.idx)

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

    def build_rnn(self, input: Tuple[int, int, int, int], block: str) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        """
        Builds an LSTM/GRU layer returning number of outputs and layer.
        """
        pattern = re.compile(r'(?P<type>L|G)(?P<dir>f|r|b)(?P<dim>x|y)(?P<sum>s)?(?P<legacy>c|o)?(?P<name>{\w+})?(?P<out>\d+)')
        m = pattern.match(block)
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
        logger.debug('{}\t\trnn\tdirection {} transposed {} summarize {} out {} legacy {}'.format(self.idx+1, direction, dim, summarize, hidden, legacy))
        return fn.get_shape(input), self.get_layer_name(type, m.group('name')), fn

    def build_dropout(self, input: Tuple[int, int, int, int], block: str) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        pattern = re.compile(r'(?P<type>Do)(?P<name>{\w+})?(?P<p>(\d+(\.\d*)?|\.\d+))?(,(?P<dim>\d+))?')
        m = pattern.match(block)
        if not m:
            return None, None, None
        prob = float(m.group('p')) if m.group('p') else 0.5
        dim = int(m.group('dim')) if m.group('dim') else 1
        fn = layers.Dropout(prob, dim)
        logger.debug('{}\t\tdropout\tprobability {} dims {}'.format(self.idx+1, prob, dim))
        return fn.get_shape(input), self.get_layer_name(m.group('type'), m.group('name')), fn

    def build_groupnorm(self, input: Tuple[int, int, int, int], block: str) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        pattern = re.compile(r'(?P<type>Gn)(?P<name>{\w+})?(?P<groups>\d+)')
        m = pattern.match(block)
        if not m:
            return None, None, None
        groups = int(m.group('groups'))
        fn = layers.GroupNorm(input[1], groups)
        logger.debug('{}\t\tgroupnorm\tgroups {}'.format(self.idx+1, groups))
        return fn.get_shape(input), self.get_layer_name(m.group('type'), m.group('name')), fn

    def build_conv(self, input: Tuple[int, int, int, int], block: str) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        """
        Builds a 2D convolution layer.
        """
        pattern = re.compile(r'(?P<type>C)(?P<nl>s|t|r|l|m)(?P<name>{\w+})?(\d+),(\d+),(?P<out>\d+)(,(?P<stride_y>\d+),(?P<stride_x>\d+))?')
        m = pattern.match(block)
        if not m:
            return None, None, None
        kernel_size = (int(m.group(4)), int(m.group(5)))
        filters = int(m.group('out'))
        stride = (int(m.group('stride_y')), int(m.group('stride_x'))) if m.group('stride_x') else (1, 1)
        nl = m.group('nl')
        fn = layers.ActConv2D(input[1], filters, kernel_size, stride, nl)
        logger.debug('{}\t\tconv\tkernel {} x {} filters {} stride {} activation {}'.format(self.idx+1, kernel_size[0], kernel_size[1], filters, stride, nl))
        return fn.get_shape(input), self.get_layer_name(m.group('type'), m.group('name')), fn

    def build_maxpool(self, input: Tuple[int, int, int, int], block: str) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
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
        fn = layers.MaxPool(kernel_size, stride)
        logger.debug('{}\t\tmaxpool\tkernel {} x {} stride {} x {}'.format(self.idx+1, kernel_size[0], kernel_size[1], stride[0], stride[1]))
        return fn.get_shape(input), self.get_layer_name(m.group('type'), m.group('name')), fn

    def build_reshape(self, input: Tuple[int, int, int, int], block: str) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        """
        Builds a reshape layer
        """
        pattern = re.compile(r'(?P<type>S)(?P<name>{\w+})?(?P<dim>\d+)\((?P<part_a>\d+)x(?P<part_b>\d+)\)(?P<high>\d+),(?P<low>\d+)')
        m = pattern.match(block)
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
        if part_b == 0:
            part_b = -1

        if src_dim != high and src_dim != low:
            raise ValueError('Either high ({}) or low ({}) must be source dimension ({})'.format(high, low, src_dim))
        if part_a == 0 or part_b == 0:
            raise ValueError('Expected non-zero size for part_a ({}) or part_b ({})'.format(part_a, part_b))
        if part_a == -1 and part_b == -1:
            raise ValueError('Only one size may be -1')
        logger.debug('{}\t\treshape from {} {} x {} to {}/{}'.format(self.idx+1, src_dim, part_a, part_b, high, low))
        src_dim = dim_map[src_dim]
        high = dim_map[high]
        low = dim_map[low]
        fn = layers.Reshape(src_dim, part_a, part_b, high, low)
        return fn.get_shape(input), self.get_layer_name(m.group('type'), m.group('name')), fn

    def build_output(self, input: Tuple[int, int, int, int], block: str) -> Union[Tuple[None, None, None], Tuple[Tuple[int, int, int, int], str, Callable]]:
        """
        Builds an output layer.
        """
        pattern = re.compile(r'(O)(?P<name>{\w+})?(?P<dim>2|1|0)(?P<type>l|s|c)(?P<aug>a)?(?P<out>\d+)')
        m = pattern.match(block)
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
            self.criterion = nn.BCELoss()
        elif nl == 'c':
            self.criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)
        else:
            raise ValueError('unsupported output specification')
        # heatmap output
        if dim == 2:
            act = 's' if nl == 'l' else 'm'
            fn = layers.ActConv2D(input[1], outdim, (1, 1), (1, 1), act)
            logger.debug('{}\t\tconv\tkernel 1 x 1 filters {} stride 1 activation {}'.format(self.idx+1, outdim, nl))
            return fn.get_shape(input), self.get_layer_name(m.group('type'), m.group('name')), fn
        else:
            aug = True if m.group('aug') else False
            lin = layers.LinSoftmax(input[1], int(m.group('out')), aug)
            logger.debug('{}\t\tlinear\taugmented {} out {}'.format(self.idx+1, aug, m.group('out')))
            return lin.get_shape(input), self.get_layer_name(m.group(1), m.group('name')), lin
