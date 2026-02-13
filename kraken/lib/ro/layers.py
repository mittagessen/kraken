"""
Layers for VGSL models
"""
from typing import TYPE_CHECKING

import torch
from torch import nn

from kraken.models import BaseModel

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from kraken.lib.vgsl import VGSLBlock

__all__ = ['ROMLP']


class ROMLP(nn.Module, BaseModel):
    """
    A simple 2 layer MLP for reading order determination.
    """
    _kraken_min_version = '5.0.0'
    model_type = ['reading_order']

    def __init__(self, **kwargs):
        super().__init__()
        self.user_metadata = {}
        self.class_mapping = kwargs.get('class_mapping', None)
        if self.class_mapping is None:
            raise ValueError('`class_mapping` missing in reading order model arguments.')
        self.level = kwargs.get('level', None)
        if self.level is None:
            raise ValueError('`level` missing in reading order model arguments.')

        num_classes = max(0, *self.class_mapping.values()) + 1
        self.feature_size = 2 * num_classes + 12
        self.hidden_size = self.feature_size * 2

        self.user_metadata = kwargs
        self.fc1 = nn.Linear(self.feature_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

    def prepare_for_inference(self, config):
        pass

    def get_shape(self, input: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """
        Calculates the output shape from input 4D tuple NCHW.
        """
        self.output_shape = input
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
