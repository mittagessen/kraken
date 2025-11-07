"""
kraken.models.loaders
~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation for model metadata and weight loading from various formats.
"""
import torch


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
