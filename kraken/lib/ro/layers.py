"""
Layers for VGSL models
"""
from typing import TYPE_CHECKING, Tuple

import torch
from torch import nn

if TYPE_CHECKING:
    from kraken.lib.vgsl import VGSLBlock

# all tensors are ordered NCHW, the "feature" dimension is C, so the output of
# an LSTM will be put into C same as the filters of a CNN.

__all__ = ['MLP']


class MLP(nn.Module):
    """
    A simple 2 layer MLP for reading order determination.
    """
    def __init__(self, feature_size: int, hidden_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.class_mapping = None

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

    def get_shape(self, input: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Calculates the output shape from input 4D tuple NCHW.
        """
        return input

    def get_spec(self, name) -> 'VGSLBlock':
        """
        Generates a VGSL spec block from the layer instance.
        """
        return f'[1,0,0,1 RO{{{name}}}{self.feature_size},{self.hidden_size}]'

    def deserialize(self, name: str, spec) -> None:
        """
        Sets the weights of an initialized module from a CoreML protobuf spec.
        """
        # extract 1st linear projection parameters
        lin = [x for x in spec.neuralNetwork.layers if x.name == '{}_mlp_lin_0'.format(name)][0].innerProduct
        weights = torch.Tensor(lin.weights.floatValue).resize_as_(self.fc1.weight.data)
        bias = torch.Tensor(lin.bias.floatValue)
        self.fc1.weight = torch.nn.Parameter(weights)
        self.fc1.bias = torch.nn.Parameter(bias)
        # extract 2nd linear projection parameters
        lin = [x for x in spec.neuralNetwork.layers if x.name == '{}_mlp_lin_1'.format(name)][0].innerProduct
        weights = torch.Tensor(lin.weights.floatValue).resize_as_(self.fc2.weight.data)
        bias = torch.Tensor(lin.bias.floatValue)
        self.fc2.weight = torch.nn.Parameter(weights)
        self.fc2.bias = torch.nn.Parameter(bias)

    def serialize(self, name: str, input: str, builder):
        """
        Serializes the module using a NeuralNetworkBuilder.
        """
        builder.add_inner_product(f'{name}_mlp_lin_0', self.fc1.weight.data.numpy(),
                                  self.fc1.bias.data.numpy(),
                                  self.feature_size, self.hidden_size,
                                  has_bias=True, input_name=input, output_name=f'{name}_mlp_lin_0')
        builder.add_activation(f'{name}_mlp_lin_0_relu', 'RELU', f'{name}_mlp_lin_0', f'{name}_mlp_lin_0_relu')
        builder.add_inner_product(f'{name}_mlp_lin_1', self.fc2.weight.data.numpy(),
                                  self.fc2.bias.data.numpy(),
                                  self.hidden_size, 1,
                                  has_bias=True, input_name=f'{name}_mlp_lin_0_relu', output_name=f'{name}_mlp_lin_1')
        return name
