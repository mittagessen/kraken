#
# Copyright 2017 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Pytorch compatible codec with many-to-many mapping between labels and
graphemes.
"""
import logging
import numpy as np

from collections import Counter
from typing import List, Tuple, Set, Union, Dict, Sequence
from torch import IntTensor
from kraken.lib.exceptions import KrakenEncodeException, KrakenCodecException

__all__ = ['PytorchCodec']

logger = logging.getLogger(__name__)


class PytorchCodec(object):
    """
    Builds a codec converting between graphemes/code points and integer
    label sequences.

    charset may either be a string, a list or a dict. In the first case
    each code point will be assigned a label, in the second case each
    string in the list will be assigned a label, and in the final case each
    key string will be mapped to the value sequence of integers. In the
    first two cases labels will be assigned automatically. When a mapping
    is manually provided the label codes need to be a prefix-free code.

    As 0 is the blank label in a CTC output layer, output labels and input
    dictionaries are/should be 1-indexed.

    Args:
        charset: Input character set.
        strict: Flag indicating if encoding/decoding errors should be ignored
                or cause an exception.

    Raises:
        KrakenCodecException: If the character set contains duplicate
                              entries or the mapping is non-singular or
                              non-prefix-free.
    """
    def __init__(self, charset: Union[Dict[str, Sequence[int]], Sequence[str], str], strict=False):
        if isinstance(charset, dict):
            self.c2l = charset
        else:
            cc = Counter(charset)
            if len(cc) < len(charset):
                raise KrakenCodecException(f'Duplicate entry in codec definition string: {cc}')
            self.c2l = {k: [v] for v, k in enumerate(sorted(charset), start=1)}
        self.c_sorted = sorted(self.c2l.keys(), key=len, reverse=True)
        self.l2c = {tuple(v): k for k, v in self.c2l.items()}  # type: Dict[Tuple[int], str]
        self.strict = strict
        if not self.is_valid:
            raise KrakenCodecException('Codec is not valid (non-singular/non-prefix free).')

    def __len__(self) -> int:
        """
        Total number of input labels the codec can decode.
        """
        return len(self.l2c.keys())

    @property
    def is_valid(self) -> bool:
        """
        Returns True if the codec is prefix-free (in label space) and
        non-singular (in both directions).
        """
        # quick test for non-singularity
        if len(self.l2c.keys()) != len(self.c2l.keys()):
            return False

        for i, code_1 in enumerate(sorted(self.l2c.keys())):
            for j, code_2 in enumerate(sorted(self.l2c.keys())):
                if i != j and code_1[:len(code_2)] == code_2:
                    return False

        return True

    @property
    def max_label(self) -> int:
        """
        Returns the maximum label value.
        """
        return max(label for labels in self.c2l.values() for label in labels)

    def encode(self, s: str) -> IntTensor:
        """
        Encodes a string into a sequence of labels.

        If the code is non-singular we greedily encode the longest sequence first.

        Args:
            s: Input unicode string

        Returns:
            Ecoded label sequence

        Raises:
            KrakenEncodeException: if the a subsequence is not encodable and the
                                   codec is set to strict mode.
        """
        labels = []  # type: List[int]
        idx = 0
        while idx < len(s):
            encodable_suffix = False
            for code in self.c_sorted:
                if s[idx:].startswith(code):
                    labels.extend(self.c2l[code])
                    idx += len(code)
                    encodable_suffix = True
                    break
            if not encodable_suffix:
                if self.strict:
                    raise KrakenEncodeException(f'Non-encodable sequence {s[idx:idx+5]}... encountered.')
                logger.warning(f'Non-encodable sequence {s[idx:idx+5]}... encountered. Advancing one code point.')
                idx += 1

        return IntTensor(labels)

    def decode(self, labels: Sequence[Tuple[int, int, int, float]]) -> List[Tuple[str, int, int, float]]:
        """
        Decodes a labelling.

        Given a labelling with cuts and  confidences returns a string with the
        cuts and confidences aggregated across label-code point
        correspondences. When decoding multilabels to code points the resulting
        cuts are min/max, confidences are averaged.

        Args:
            labels: Input containing tuples (label, start, end,
                           confidence).

        Returns:
            A list of tuples (code point, start, end, confidence)
        """
        start = [x for _, x, _, _ in labels]
        end = [x for _, _, x, _ in labels]
        con = [x for _, _, _, x in labels]
        labels = tuple(x for x, _, _, _ in labels)
        decoded = []
        idx = 0
        while idx < len(labels):
            decodable_suffix = False
            for code in self.l2c.keys():
                if code == labels[idx:idx+len(code)]:
                    decoded.extend([(c, s, e, u) for c, s, e, u in zip(self.l2c[code],
                                                                       len(self.l2c[code]) * [start[idx]],
                                                                       len(self.l2c[code]) * [end[idx + len(code) - 1]],
                                                                       len(self.l2c[code]) * [np.mean(con[idx:idx + len(code)])])])
                    idx += len(code)
                    decodable_suffix = True
                    break
            if not decodable_suffix:
                if self.strict:
                    raise KrakenEncodeException(f'Non-decodable sequence {labels[idx:idx+5]}... encountered.')
                logger.debug(f'Non-decodable sequence {labels[idx:idx+5]}... encountered. Advancing one label.')
                idx += 1
        return decoded

    def merge(self, codec: 'PytorchCodec') -> Tuple['PytorchCodec', Set]:
        """
        Transforms this codec (c1) into another (c2) reusing as many labels as
        possible.

        The resulting codec is able to encode the same code point sequences
        while not necessarily having the same labels for them as c2.
        Retains matching character -> label mappings from both codecs, removes
        mappings not c2, and adds mappings not in c1. Compound labels in c2 for
        code point sequences not in c1 containing labels also in use in c1 are
        added as separate labels.

        Args:
            codec: PytorchCodec to merge with

        Returns:
            A merged codec and a list of labels that were removed from the
            original codec.
        """
        # find character sequences not encodable (exact match) by new codec.
        # get labels for these sequences as deletion candidates
        rm_candidates = {cseq: enc for cseq, enc in self.c2l.items() if cseq not in codec.c2l}
        c2l_cand = self.c2l.copy()
        for x in rm_candidates.keys():
            c2l_cand.pop(x)
        # remove labels from candidate list that are in use for other decodings
        rm_labels = [label for v in rm_candidates.values() for label in v]
        for v in c2l_cand.values():
            for label in rm_labels:
                if label in v:
                    rm_labels.remove(label)
        # iteratively remove labels, decrementing subsequent labels to close
        # (new) holes in the codec.
        offset_rm_labels = [v-idx for idx, v in enumerate(sorted(set(rm_labels)))]
        for rlabel in offset_rm_labels:
            c2l_cand = {k: [label-1 if label > rlabel else label for label in v] for k, v in c2l_cand.items()}
        # add mappings not in original codec
        add_list = {cseq: enc for cseq, enc in codec.c2l.items() if cseq not in self.c2l}
        # renumber
        start_idx = max((0,) + tuple(label for v in c2l_cand.values() for label in v)) + 1
        add_labels = {k: v for v, k in enumerate(sorted(set(label for v in add_list.values() for label in v)), start_idx)}
        for k, v in add_list.items():
            c2l_cand[k] = [add_labels[label] for label in v]
        return PytorchCodec(c2l_cand, self.strict), set(rm_labels)

    def add_labels(self, charset: Union[Dict[str, Sequence[int]], Sequence[str], str]) -> 'PytorchCodec':
        """
        Adds additional characters/labels to the codec.

        charset may either be a string, a list or a dict. In the first case
        each code point will be assigned a label, in the second case each
        string in the list will be assigned a label, and in the final case each
        key string will be mapped to the value sequence of integers. In the
        first two cases labels will be assigned automatically.

        As 0 is the blank label in a CTC output layer, output labels and input
        dictionaries are/should be 1-indexed.

        Args:
            charset: Input character set.
        """
        if isinstance(charset, dict):
            c2l = self.c2l.copy()
            c2l.update(charset)
        else:
            c2l = self.c2l.copy()
            c2l.update({k: [v] for v, k in enumerate(sorted(charset), start=self.max_label+1)})
        return PytorchCodec(c2l, self.strict)

    def __repr__(self):
        return f'PytorchCodec({self.c2l})'
