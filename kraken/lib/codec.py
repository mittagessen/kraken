# -*- coding: utf-8 -*-
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
pytorch compatible codec with many-to-many mapping between labels and
graphemes.
"""
import regex
import numpy as np

from typing import List, Tuple, Set, Union, Dict, Sequence
from torch import IntTensor
from kraken.lib.exceptions import KrakenEncodeException, KrakenCodecException

__all__ = ['PytorchCodec']


class PytorchCodec(object):
    """
    Translates between labels and graphemes.
    """
    def __init__(self, charset: Union[Dict[str, Sequence[int]], Sequence[str], str]) -> None:
        """
        Builds a codec converting between graphemes/code points and integer
        label sequences.

        charset may either be a string, a list or a dict. In the first case
        each code point will be assigned a label, in the second case each
        string in the list will be assigned a label, and in the final case each
        key string will be mapped to the value sequence of integers. In the
        first two cases labels will be assigned automatically.

        As 0 is the blank label in a CTC output layer, output labels and input
        dictionaries are/should be 1-indexed.

        Args:
            charset (unicode, list, dict): Input character set.
        """
        if isinstance(charset, dict):
            self.c2l = charset
        else:
            self.c2l = {k: [v] for v, k in enumerate(sorted(charset), start=1)}
        # map integer labels to code points because regex only works with strings
        self.l2c = {}  # type: Dict[str, str]
        for k, v in self.c2l.items():
            self.l2c[''.join(chr(c) for c in v)] = k

        # sort prefixes for c2l regex
        self.c2l_regex = regex.compile(r'|'.join(regex.escape(x) for x in sorted(self.c2l.keys(), key=len, reverse=True)))
        # sort prefixes for l2c regex
        self.l2c_regex = regex.compile(r'|'.join(regex.escape(x) for x in sorted(self.l2c.keys(), key=len, reverse=True)))

    def __len__(self) -> int:
        """
        Total number of input labels the codec can decode.
        """
        return len(self.l2c.keys())

    def max_label(self) -> int:
        """
        Returns the maximum label value.
        """
        return max(l for labels in self.c2l.values() for l in labels)

    def encode(self, s: str) -> IntTensor:
        """
        Encodes a string into a sequence of labels.

        Args:
            s (str): Input unicode string

        Returns:
            (torch.IntTensor) encoded label sequence

        Raises:
            KrakenEncodeException if encoding fails.
        """
        splits = self._greedy_split(s, self.c2l_regex)
        labels = []  # type: List[int]
        for c in splits:
            labels.extend(self.c2l[c])
        return IntTensor(labels)

    def decode(self, labels: Sequence[Tuple[int, int, int, float]]) -> List[Tuple[str, int, int, float]]:
        """
        Decodes a labelling.

        Given a labelling with cuts and  confidences returns a string with the
        cuts and confidences aggregated across label-code point
        correspondences. When decoding multilabels to code points the resulting
        cuts are min/max, confidences are averaged.

        Args:
            labels (list): Input containing tuples (label, start, end,
                           confidence).

        Returns:
            list: A list of tuples (code point, start, end, confidence)
        """
        # map into unicode space
        uni_labels = ''.join(chr(v) for v, _, _, _ in labels)
        start = [x for _, x, _, _ in labels]
        end = [x for _, _, x, _ in labels]
        con = [x for _, _, _, x in labels]
        splits = self._greedy_split(uni_labels, self.l2c_regex)
        decoded = []
        idx = 0
        for i in splits:
            decoded.extend([(c, s, e, u) for c, s, e, u in zip(self.l2c[i],
                                                               len(self.l2c[i]) * [start[idx]],
                                                               len(self.l2c[i]) * [end[idx + len(i) - 1]],
                                                               len(self.l2c[i]) * [np.mean(con[idx:idx + len(i)])])])
            idx += len(i)
        return decoded

    def _greedy_split(self, input: str, re: regex.Regex) -> List[str]:
        """
        Splits an input string greedily from a list of prefixes. Stops when no
        more matches are found.

        Args:
            input (str): input string
            re (regex.Regex): Prefix match object

        Returns:
            (list) of prefixes

        Raises:
            (KrakenEncodeException) if no prefix match is found for some part
            of the string.
        """
        r = []  # type: List[str]
        idx = 0
        while True:
            mo = re.match(input, idx)
            if mo is None or idx == len(input):
                if len(input) > idx:
                    raise KrakenEncodeException('No prefix matches for input after {}'.format(idx))
                return r
            r.append(mo.group())
            idx = mo.end()

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
            codec (kraken.lib.codec.PytorchCodec):

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
            for l in rm_labels:
                if l in v:
                    rm_labels.remove(l)
        # iteratively remove labels, decrementing subsequent labels to close
        # (new) holes in the codec.
        offset_rm_labels = [v-idx for idx, v in enumerate(sorted(set(rm_labels)))]
        for rlabel in offset_rm_labels:
            c2l_cand = {k: [l-1 if l > rlabel else l for l in v] for k, v in c2l_cand.items()}
        # add mappings not in original codec
        add_list = {cseq: enc for cseq, enc in codec.c2l.items() if cseq not in self.c2l}
        # renumber
        start_idx = max((0,) + tuple(label for v in c2l_cand.values() for label in v)) + 1
        add_labels = {k: v for v, k in enumerate(sorted(set(label for v in add_list.values() for label in v)), start_idx)}
        for k, v in add_list.items():
            c2l_cand[k] = [add_labels[label] for label in v]
        return PytorchCodec(c2l_cand), set(rm_labels)

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
            charset (unicode, list, dict): Input character set.
        """
        if isinstance(charset, dict):
            c2l = self.c2l.copy()
            c2l.update(charset)
        else:
            c2l = self.c2l.copy()
            c2l.update({k: [v] for v, k in enumerate(sorted(charset), start=self.max_label()+1)})
        return PytorchCodec(c2l)
