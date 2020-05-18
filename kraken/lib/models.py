"""
kraken.lib.models
~~~~~~~~~~~~~~~~~

Wrapper around TorchVGSLModel including a variety of forward pass helpers for
sequence classification.
"""
from os.path import expandvars, expanduser, abspath

import torch
import numpy as np
import kraken.lib.lineest
import kraken.lib.ctc_decoder

from typing import List, Tuple

from kraken.lib.vgsl import TorchVGSLModel
from kraken.lib.exceptions import KrakenInvalidModelException, KrakenInputException

__all__ = ['TorchSeqRecognizer', 'load_any']

import logging

logger = logging.getLogger(__name__)


class TorchSeqRecognizer(object):
    """
    A class wrapping a TorchVGSLModel with a more comfortable recognition interface.
    """
    def __init__(self, nn, decoder=kraken.lib.ctc_decoder.greedy_decoder, train: bool = False, device: str = 'cpu') -> None:
        """
        Constructs a sequence recognizer from a VGSL model and a decoder.

        Args:
            nn (kraken.lib.vgsl.TorchVGSLModel): neural network used for recognition
            decoder (func): Decoder function used for mapping softmax
                            activations to labels and positions
            train (bool): Enables or disables gradient calculation
            device (torch.Device): Device to run model on
        """
        self.nn = nn
        self.kind = ''
        if train:
            self.nn.train()
        else:
            self.nn.eval()
        self.codec = self.nn.codec
        self.decoder = decoder
        self.train = train
        self.device = device
        if nn.model_type not in [None, 'recognition']:
            raise ValueError('Models of type {} are not supported by TorchSeqRecognizer'.format(nn.model_type))
        self.one_channel_mode = nn.one_channel_mode
        self.seg_type = nn.seg_type
        self.nn.to(device)

    def to(self, device):
        """
        Moves model to device and automatically loads input tensors onto it.
        """
        self.device = device
        self.nn.to(device)

    def forward(self, line: torch.Tensor, lens: torch.Tensor = None) -> np.array:
        """
        Performs a forward pass on a torch tensor of one or more lines with
        shape (N, C, H, W) and returns a numpy array (N, W, C).

        Args:
            line (torch.Tensor): NCHW line tensor
            lens (torch.Tensor): Optional tensor containing sequence lengths if N > 1

        Returns:
            Tuple with (N, W, C) shaped numpy array and final output sequence
            lengths.
        """
        line = line.to(self.device)
        o, olens = self.nn.nn(line, lens)
        if o.size(2) != 1:
            raise KrakenInputException('Expected dimension 3 to be 1, actual {}'.format(o.size()))
        self.outputs = o.detach().squeeze(2).cpu().numpy()
        if olens is not None:
            olens = olens.cpu().numpy()
        return self.outputs, olens

    def predict(self, line: torch.Tensor, lens: torch.Tensor = None) -> List[List[Tuple[str, int, int, float]]]:
        """
        Performs a forward pass on a torch tensor of a line with shape (N, C, H, W)
        and returns the decoding as a list of tuples (string, start, end,
        confidence).

        Args:
            line (torch.Tensor): NCHW line tensor
            lens (torch.Tensor): Optional tensor containing sequence lengths if N > 1

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

    def predict_string(self, line: torch.Tensor, lens: torch.Tensor = None) -> List[str]:
        """
        Performs a forward pass on a torch tensor of a line with shape (N, C, H, W)
        and returns a string of the results.
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


def load_any(fname: str, train: bool = False, device: str = 'cpu') -> TorchSeqRecognizer:
    """
    Loads anything that was, is, and will be a valid ocropus model and
    instantiates a shiny new kraken.lib.lstm.SeqRecognizer from the RNN
    configuration in the file.

    Currently it recognizes the following kinds of models:

        * pyrnn models containing BIDILSTMs
        * protobuf models containing converted python BIDILSTMs
        * protobuf models containing CLSTM networks

    Additionally an attribute 'kind' will be added to the SeqRecognizer
    containing a string representation of the source kind. Current known values
    are:

        * pyrnn for pickled BIDILSTMs
        * clstm for protobuf models generated by clstm

    Args:
        fname (str): Path to the model
        train (bool): Enables gradient calculation and dropout layers in model.
        device (str): Target device

    Returns:
        A kraken.lib.models.TorchSeqRecognizer object.
    """
    nn = None
    kind = ''
    fname = abspath(expandvars(expanduser(fname)))
    logger.info('Loading model from {}'.format(fname))
    try:
        nn = TorchVGSLModel.load_model(str(fname))
        kind = 'vgsl'
    except Exception:
        try:
            nn = TorchVGSLModel.load_clstm_model(fname)
            kind = 'clstm'
        except Exception:
            nn = TorchVGSLModel.load_pronn_model(fname)
            kind = 'pronn'
        try:
            nn = TorchVGSLModel.load_pyrnn_model(fname)
            kind = 'pyrnn'
        except Exception:
            pass
    if not nn:
        raise KrakenInvalidModelException('File {} not loadable by any parser.'.format(fname))
    seq = TorchSeqRecognizer(nn, train=train, device=device)
    seq.kind = kind
    return seq
