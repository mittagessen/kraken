"""
Layers for VGSL models
"""
import torch

from typing import Tuple, Optional
from torch.nn import Module, Embedding, Linear
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from kraken.lib.vgsl import VGSLBlock
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
            context_encoder_input_dim: size of the `C` input dimension
            final_dim: size of the decoder `C` output dimension just before the
                       final linear projection.
            mask_width: width of the non-overlapping masked areas.
            mask_prob: probability of masking at each time step
            num_negatives: number of negative samples with width mask_width *
                           num_masks

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
        self.mask_emb = Embedding(mask_width, context_encoder_input_dim)
        self.project_q = Linear(context_encoder_input_dim, final_dim)

    def forward(self, inputs: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        N, C, H, W = inputs.shape
        if H != 1:
            raise Exception(f'Height has to be 1, not {H} for Wav2Vec2 masking layer.')

        # NCHW -> NWC
        inputs = inputs.transpose(1, 3).reshape(-1, W, C)
        seq_len = seq_len.cpu()
        packed_inputs = pack_padded_sequence(inputs, seq_len, batch_first=True, enforce_sorted=False)
        mask, num_masks, num_negatives = compute_masks(self.mask_prob, self.mask_width, self.num_negatives, seq_len)

        unmasked_samples = self.project_q(packed_inputs.data[mask == 1]).reshape(1, num_masks, self.mask_width, -1)
        negative_samples = self.project_q(packed_inputs.data[mask == 2]).reshape(self.num_negatives, num_masks, self.mask_width, -1)

        # mask features
        packed_inputs.data[mask == 1] = self.mask_emb.weight.repeat(num_masks, 1)
        inputs, seq_len = pad_packed_sequence(packed_inputs, batch_first=True)
        # NWC -> NCHW
        inputs = inputs.permute(0, 2, 1).unsqueeze(2)
        return {'output': inputs,
                'unmasked_samples': unmasked_samples,
                'negative_samples': negative_samples,
                'seq_len': seq_len,
                'mask': mask}

    def get_shape(self, input: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Calculates the output shape from input 4D tuple NCHW.
        """
        return input

    def get_spec(self, name) -> "VGSLBlock":
        """
        Generates a VGSL spec block from the layer instance.
        """
        return f'[1,{self.final_dim},0,{self.context_encoder_input_dim} W{{{name}}}{self.final_dim},{self.mask_width},{self.mask_prob},{self.num_negatives}]'

    def deserialize(self, name: str, spec) -> None:
        """
        Sets the weights of an initialized module from a CoreML protobuf spec.
        """
        # extract embedding parameters
        emb = [x for x in spec.neuralNetwork.layers if x.name == '{}_wave2vec2_emb'.format(name)][0].embedding
        weights = torch.Tensor(emb.weights.floatValue).resize_as_(self.mask_emb.weight.data)
        self.mask_emb.weight = torch.nn.Parameter(weights)
        # extract linear projection parameters
        lin = [x for x in spec.neuralNetwork.layers if x.name == '{}_wave2vec2_lin'.format(name)][0].innerProduct
        weights = torch.Tensor(lin.weights.floatValue).resize_as_(self.project_q.weight.data)
        bias = torch.Tensor(lin.bias.floatValue)
        self.project_q.weight = torch.nn.Parameter(weights)
        self.project_q.bias = torch.nn.Parameter(bias)

    def serialize(self, name: str, input: str, builder):
        """
        Serializes the module using a NeuralNetworkBuilder.
        """
        wave2vec2_emb_name = f'{name}_wave2vec2_emb'
        builder.add_embedding(wave2vec2_emb_name, self.mask_emb.weight.data.numpy(),
                              None,
                              self.context_encoder_input_dim, self.mask_width,
                              has_bias=False, input_name=input, output_name=wave2vec2_emb_name)
        wave2vec2_lin_name = f'{name}_wave2vec2_lin'
        builder.add_inner_product(wave2vec2_lin_name, self.project_q.weight.data.numpy(),
                                  self.project_q.bias.data.numpy(),
                                  self.context_encoder_input_dim, self.final_dim,
                                  has_bias=True, input_name=input, output_name=wave2vec2_lin_name)
        return name
