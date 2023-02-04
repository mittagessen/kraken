"""
kraken.lib.models
~~~~~~~~~~~~~~~~~

Wrapper around TorchVGSLModel including a variety of forward pass helpers for
sequence classification.
"""
from os import PathLike
from os.path import expandvars, expanduser, abspath

import torch
import numpy as np
import kraken.lib.lineest
import kraken.lib.ctc_decoder

from typing import List, Tuple, Optional, Union

from kraken.lib.vgsl import TorchVGSLModel
from kraken.lib.exceptions import KrakenInvalidModelException, KrakenInputException

__all__ = ['TorchSeqRecognizer', 'load_any']

import logging

logger = logging.getLogger(__name__)


class TorchSeqRecognizer(object):
    """
    A wrapper class around a TorchVGSLModel for text recognition.
    """
    def __init__(self,
                 nn: TorchVGSLModel,
                 decoder=kraken.lib.ctc_decoder.greedy_decoder,
                 train: bool = False,
                 device: str = 'cpu'):
        """
        Constructs a sequence recognizer from a VGSL model and a decoder.

        Args:
            nn: Neural network used for recognition.
            decoder: Decoder function used for mapping softmax activations to
                     labels and positions.
            train: Enables or disables gradient calculation and dropout.
            device: Device to run model on.

        Attributes:
            nn: Neural network used for recognition.
            codec: PytorchCodec extracted from the recognition model.

            decoder: Decoder function used for mapping softmax activations to
                     labels and positions.
            train: Enables or disables gradient calculation and dropout.
            device: Device to run model on.
            one_channel_mode: flag indicating if the model expects binary or
                              grayscale input images.
            seg_type: flag indicating if the model expects baseline- or bounding
                      box-derived text line images.

        Raises:
            ValueError: Is raised when the model type is not a sequence recognizer.
        """
        self.nn = nn
        self.kind = ''
        if train is True:
            self.nn.train()
        elif train is False:
            self.nn.eval()
        self.codec = self.nn.codec
        self.decoder = decoder
        self.train = train
        self.device = device
        if nn.model_type not in [None, 'recognition']:
            raise ValueError(f'Models of type {nn.model_type} are not supported by TorchSeqRecognizer')
        self.one_channel_mode = nn.one_channel_mode
        self.seg_type = nn.seg_type
        if self.device:
            self.nn.to(device)

    def to(self, device):
        """
        Moves model to device and automatically loads input tensors onto it.
        """
        self.device = device
        self.nn.to(device)

    def forward(self, line: torch.Tensor, lens: torch.Tensor = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Performs a forward pass on a torch tensor of one or more lines with
        shape (N, C, H, W) and returns a numpy array (N, W, C).

        Args:
            line: NCHW line tensor
            lens: Optional tensor containing sequence lengths if N > 1

        Returns:
            Tuple with (N, W, C) shaped numpy array and final output sequence
            lengths.

        Raises:
            KrakenInputException: Is raised if the channel dimension isn't of
                                  size 1 in the network output.
        """
        if self.device:
            line = line.to(self.device)
        o, olens = self.nn.nn(line, lens)
        if o.size(2) != 1:
            raise KrakenInputException('Expected dimension 3 to be 1, actual {}'.format(o.size()))
        self.outputs = o.detach().squeeze(2).cpu().numpy()
        if olens is not None:
            olens = olens.cpu().numpy()
        return self.outputs, olens

    def predict(self, line: torch.Tensor, lens: Optional[torch.Tensor] = None) -> List[List[Tuple[str, int, int, float]]]:
        """
        Performs a forward pass on a torch tensor of a line with shape (N, C, H, W)
        and returns the decoding as a list of tuples (string, start, end,
        confidence).

        Args:
            line: NCHW line tensor
            lens: Optional tensor containing sequence lengths if N > 1

        Returns:
            List of decoded sequences.
        """
        o, olens = self.forward(line, lens)
        dec_seqs = []
        if olens is not None:
            for seq, seq_len in zip(o, olens):
                locs = self.decoder(seq[:, :seq_len])
                dec_seqs.append(self.codec.decode(locs))
        else:
            locs = self.decoder(o[0])
            dec_seqs.append(self.codec.decode(locs))
        return dec_seqs

    def predict_string(self, line: torch.Tensor, lens: Optional[torch.Tensor] = None) -> List[str]:
        """
        Performs a forward pass on a torch tensor of a line with shape (N, C, H, W)
        and returns a string of the results.

        Args:
            line: NCHW line tensor
            lens: Optional tensor containing the sequence lengths of the input batch.
        """
        o, olens = self.forward(line, lens)
        dec_strs = []
        if olens is not None:
            for seq, seq_len in zip(o, olens):
                locs = self.decoder(seq[:, :seq_len])
                dec_strs.append(''.join(x[0] for x in self.codec.decode(locs)))
        else:
            locs = self.decoder(o[0])
            dec_strs.append(''.join(x[0] for x in self.codec.decode(locs)))
        return dec_strs

    def predict_labels(self, line: torch.tensor, lens: torch.Tensor = None) -> List[List[Tuple[int, int, int, float]]]:
        """
        Performs a forward pass on a torch tensor of a line with shape (N, C, H, W)
        and returns a list of tuples (class, start, end, max). Max is the
        maximum value of the softmax layer in the region.
        """
        o, olens = self.forward(line, lens)
        oseqs = []
        if olens is not None:
            for seq, seq_len in zip(o, olens):
                oseqs.append(self.decoder(seq[:, :seq_len]))
        else:
            oseqs.append(self.decoder(o[0]))
        return oseqs


def load_any(fname: Union[PathLike, str],
             train: bool = False,
             device: str = 'cpu') -> TorchSeqRecognizer:
    """
    Loads anything that was, is, and will be a valid ocropus model and
    instantiates a shiny new kraken.lib.lstm.SeqRecognizer from the RNN
    configuration in the file.

    Currently it recognizes the following kinds of models:

        * protobuf models containing VGSL segmentation and recognition
          networks.

    Additionally an attribute 'kind' will be added to the SeqRecognizer
    containing a string representation of the source kind. Current known values
    are:

        * vgsl for VGSL models

    Args:
        fname: Path to the model
        train: Enables gradient calculation and dropout layers in model.
        device: Target device

    Returns:
        A kraken.lib.models.TorchSeqRecognizer object.

    Raises:
        KrakenInvalidModelException: if the model is not loadable by any parser.
    """
    nn = None
    fname = abspath(expandvars(expanduser(fname)))
    logger.info('Loading model from {}'.format(fname))
    try:
        nn = TorchVGSLModel.load_model(str(fname))
    except Exception as e:
        raise KrakenInvalidModelException('File {} not loadable by any parser.'.format(fname)) from e
    seq = TorchSeqRecognizer(nn, train=train, device=device)
    seq.kind = 'vgsl'
    return seq


def validate_hyper_parameters(hyper_params):
    """
    Validate some model's hyper parameters and modify them in place if need be.
    """
    if (hyper_params['quit'] == 'dumb' and hyper_params['completed_epochs'] >= hyper_params['epochs']):
        logger.warning('Maximum epochs reached (might be loaded from given model), starting again from 0.')
        hyper_params['completed_epochs'] = 0
