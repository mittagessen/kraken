"""
kraken.models.loaders
~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation for model metadata and weight loading from various formats.
"""
import json
import torch
import logging
import importlib

from os import PathLike
from typing import Union, NewType, Literal, Optional
from pathlib import Path
from collections import defaultdict
from collections.abc import Sequence
from packaging.version import Version

from kraken.registry import register, create_model, LOADER_REGISTRY
from kraken.models.base import BaseModel

logger = logging.getLogger(__name__)


_T_tasks = NewType('_T_tasks', Literal['segmentation', 'recognition', 'reading_order'])

__all__ = ['load_models', 'load_coreml', 'load_safetensors']


# deserializers for coreml layers with weights
def _coreml_lin(spec):
    weights = {}
    for layer in spec:
        if layer.WhichOneof('layer') == 'innerProduct':
            name = layer.name.removesuffix('_lin')
            lin = layer.innerProduct
            weights[f'nn.{name}.lin.weight'] = torch.Tensor(lin.weights.floatValue).view(lin.outputChannels, lin.inputChannels)
            weights[f'nn.{name}.lin.bias'] = torch.Tensor(lin.bias.floatValue)
    return weights


def _coreml_rnn(spec):
    weights = {}
    for layer in spec:
        if (arch := layer.WhichOneof('layer')) in ['uniDirectionalLSTM', 'biDirectionalLSTM']:
            rnn = getattr(layer, arch)
            output_size = rnn.outputVectorSize
            input_size = rnn.inputVectorSize
            name = layer.name.removesuffix('_transposed')

            def _deserialize_weights(params, direction):
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
                weights[f'nn.{name}.layer.weight_ih_l0{"_reverse" if direction == "bwd" else ""}'] = weight_ih.view(-1, input_size)
                weights[f'nn.{name}.layer.weight_hh_l0{"_reverse" if direction == "bwd" else ""}'] = weight_hh.view(-1, output_size)
                biases = torch.Tensor([params.inputGateBiasVector.floatValue,  # bi
                                       params.forgetGateBiasVector.floatValue,  # bf
                                       params.blockInputBiasVector.floatValue,  # bz/bg
                                       params.outputGateBiasVector.floatValue]).view(-1)  # bo
                weights[f'nn.{name}.layer.bias_hh_l0{"_reverse" if direction == "bwd" else ""}'] = biases
                # no ih_biases
                weights[f'nn.{name}.layer.bias_ih_l0{"_reverse" if direction == "bwd" else ""}'] = torch.zeros_like(biases)

            fwd_params = rnn.weightParams if arch == 'uniDirectionalLSTM' else rnn.weightParams[0]
            _deserialize_weights(fwd_params, 'fwd')

            # get backward weights
            if arch == 'biDirectionalLSTM':
                _deserialize_weights(rnn.weightParams[1], 'bwd')
    return weights


def _coreml_conv(spec):
    weights = {}
    for layer in spec:
        if layer.WhichOneof('layer') == 'convolution':
            name = layer.name.removesuffix('_conv')
            conv = layer.convolution
            in_channels = conv.kernelChannels
            out_channels = conv.outputChannels
            kernel_size = conv.kernelSize
            if conv.isDeconvolution:
                weights[f'nn.{name}.co.weight'] = torch.Tensor(conv.weights.floatValue).view(in_channels, out_channels, *kernel_size)
            else:
                weights[f'nn.{name}.co.weight'] = torch.Tensor(conv.weights.floatValue).view(out_channels, in_channels, *kernel_size)
            weights[f'nn.{name}.co.bias'] = torch.Tensor(conv.bias.floatValue)
    return weights


def _coreml_groupnorm(spec):
    weights = {}
    for layer in spec:
        if layer.WhichOneof('layer') == 'custom' and layer.custom.className == 'groupnorm':
            gn = layer.custom
            in_channels = gn.parameters['in_channels'].intValue
            weights[f'nn.{layer.name}.layer.weight'] = torch.Tensor(gn.weights[0].floatValue).view(in_channels)
            weights[f'nn.{layer.name}.layer.bias'] = torch.Tensor(gn.weights[1].floatValue).view(in_channels)
    return weights


def _coreml_romlp(spec):
    weights = {}
    return weights


def _coreml_wav2vec2mask(spec):
    weights = {}
    # extract embedding parameters
    if len(emb := [x for x in spec if x.name.endswith('_wave2vec2_emb')]):
        emb = emb[0].embedding
        weights['nn._wave2vec2mask.mask_emb.weight'] = torch.Tensor(emb.weights.floatValue)
    # extract linear projection parameters
    if len(lin := [x for x in spec if x.name.endswith('_wave2vec2_lin')]):
        lin = lin[0].innerProduct
        weights['nn._wave2vec2mask.project_q.weight'] = torch.Tensor(lin.weights.floatValue).view(lin.outputChannels, lin.inputChannels)
        weights['nn._wave2vec2mask.project_q.bias'] = torch.Tensor(lin.bias.floatValue)
    return weights


_coreml_parsers = [_coreml_conv, _coreml_rnn, _coreml_lin, _coreml_groupnorm,
                   _coreml_wav2vec2mask, _coreml_romlp]


def load_models(path: Union[str, 'PathLike'], tasks: Optional[Sequence[_T_tasks]] = None) -> list[BaseModel]:
    """
    Tries all loaders in sequence to deserialize models found in file at path.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'{path} is not a regular file.')
    for name, cfg in LOADER_REGISTRY.items():
        try:
            return getattr(cfg['_module'], name)(path)
        except ValueError:
            continue
    raise ValueError(f'No loader found for {path}')


@register(type='loader')
def load_safetensors(path: Union[str, PathLike], tasks: Optional[Sequence[_T_tasks]] = None) -> list[BaseModel]:
    """
    Loads one or more models in safetensors format and returns them.
    """
    from safetensors import safe_open, SafetensorError
    weights = defaultdict(dict)
    models = {}
    try:
        with safe_open(path, framework="pt") as f:
            if (metadata := f.metadata()) is not None:
                model_map = json.loads(metadata.get('kraken_meta', 'null'))
                prefixes = list(model_map.keys())
                # construct models
                for prefix in prefixes:
                    if (min_ver := Version(model_map[prefix].get('_kraken_min_version'))) > (inst_ver := Version(importlib.metadata.version('kraken'))):
                        logger.warning(f'Model {prefix} in model file {path} requires minimum kraken version {min_ver} (installed {inst_ver})')
                        continue
                    if tasks and not set(tasks).intersection(set(model_map[prefix].get('model_type', []))):
                        logger.info(f'Model {prefix} in model file {path} not in demanded tasks {tasks}')
                        continue
                    model_map[prefix].pop('_tasks')
                    models[prefix] = create_model(model_map[prefix].get('_model'), **model_map[prefix])
            else:
                raise ValueError(f'No model metadata found in {path}.')
            for k in f.offset_keys():
                try:
                    prefix = prefixes[list(map(k.startswith, prefixes)).index(True)]
                    weights[prefix][k.removeprefix(f'{prefix}.')] = f.get_tensor(k)
                except ValueError:
                    continue
    except SafetensorError as e:
        raise ValueError(f'Invalid model file {path}') from e
    # load weights into models
    for prefix, weight in weights.items():
        models[prefix].load_state_dict(weight)
    return list(models.values())


@register(type='loader')
def load_coreml(path: Union[str, PathLike], tasks: Optional[Sequence[_T_tasks]] = None) -> list[BaseModel]:
    """
    Loads a model in coreml format.
    """
    root_logger = logging.getLogger()
    level = root_logger.getEffectiveLevel()
    root_logger.setLevel(logging.ERROR)
    from coremltools.models import MLModel
    root_logger.setLevel(level)
    from google.protobuf.message import DecodeError

    if isinstance(path, PathLike):
        path = path.as_posix()
    try:
        mlmodel = MLModel(path)
    except TypeError as e:
        raise ValueError(str(e)) from e
    except DecodeError as e:
        raise ValueError(f'Failure parsing model protobuf: {e}') from e

    metadata = json.loads(mlmodel.user_defined_metadata.get('kraken_meta', '{}'))

    if tasks and not metadata['model_type'] not in tasks:
        logger.info(f'Model file {path} not in demanded tasks {tasks}')
        return []

    model = create_model('TorchVGSLModel',
                         vgsl=mlmodel.user_defined_metadata['vgsl'],
                         codec=json.loads(mlmodel.user_defined_metadata.get('codec', 'null')),
                         **metadata)

    # construct state dict
    weights = {}
    spec = mlmodel.get_spec().neuralNetwork.layers
    for cml_parser in _coreml_parsers:
        weights.update(cml_parser(spec))

    model.load_state_dict(weights)

    # construct additional models if auxiliary layers are defined.

    #if 'aux_layers' in mlmodel.user_defined_metadata:
    #    logger.info('Deserializing auxiliary layers.')

    #    nn.aux_layers = {k: cls(v).nn.get_submodule(k) for k, v in json.loads(mlmodel.user_defined_metadata['aux_layers']).items()}

    return [model]
