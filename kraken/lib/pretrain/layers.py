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

from kraken.lib.pretrain.util import compute_masks

# all tensors are ordered NCHW, the "feature" dimension is C, so the output of
# an LSTM will be put into C same as the filters of a CNN.

__all__ = ['Wav2Vec2Mask']


class Wav2Vec2Mask(Module):
    """
    A layer for Wav2Vec2-style masking. Needs to be placed just before
    recurrent/contextual layers.
    """

    def __init__(self,
                 context_encoder_input_dim: int,
                 final_dim: int,
                 mask_width: int,
                 mask_prob: float,
                 num_negatives: int) -> None:
        """

        Args:
            context_encoder_input_dim: size of the ``H` input dimension
            final_dim: size of the decoder `C` output dimension just before the
                       final linear projection.
            mask_width: width of the non-overlapping masked areas.
            mask_prob: probability of masking at each time step
            num_negatives: number of negative samples

        Shape:
            - Inputs: :math:`(N, C, H, W)` where `N` batches, `C` channels, `H`
              height, and `W` width.
            - Outputs output :math:`(N, C, H, W)`
        """
        super().__init__()

        self.context_encoder_input_dim = context_encoder_input_dim
        self.final_dim = final_dim
        self.mask_width = mask_width
        self.mask_prob = mask_prob
        self.num_negatives = num_negatives

        # mask embedding replacing the masked out areas
        self.mask_emb = nn.Embedding(mask_width, context_encoder_input_dim)
        self.project_q = nn.Linear(context_encoder_input_dim, final_dim)

    def forward(self, inputs: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        N, C, H, W = inputs.shape
        inputs = inputs.reshape(C, H, -1)
        mask, num_masks = compute_masks(self.mask_prob, self.mask_width, self.num_neg_samples, seq_len, W)
        inputs[..., mask == 1] = self.mask_emb.weight.T.repeat(num_masks)
        masked_inputs = self.project_q(inputs[..., mask == 1])
        negative_samples = self.project_q(inputs[..., mask == 2])
        return {'output': inputs,
                'masked_outputs': masked_inputs,
                'negative_samples': negative_samples,
                'seq_len': seq_len,
                'mask': mask}

    def get_shape(self, input: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Calculates the output shape from input 4D tuple NCHW.
        """
        return self.input

    def deserialize(self, name: str, spec) -> None:
        """
        Sets the weights of an initialized module from a CoreML protobuf spec.
        """
        # extract conv parameters
        emb = [x for x in spec.neuralNetwork.layers if x.name == '{}_wave2vec2'.format(name)][0].embedding
        weights = torch.Tensor(emb.weights.floatValue).resize_as_(self.mask_emb.weight.data)
        self.mask_emb.weight = torch.nn.Parameter(weights)

    def serialize(self, name: str, input: str, builder):
        """
        Serializes the module using a NeuralNetworkBuilder.
        """
        wave2vec2_name = '{}_wave2vec2'.format(name)
        builder.add_embedding(wave2vec2_name, self.mask_emb.weight.data.numpy(),
                              None,
                              self.height, self.mask_width,
                              has_bias=False, input_name=input, output_name=wave2vec2_name)
        return name
