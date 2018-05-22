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

from torch import IntTensor

from kraken.lib.exceptions import KrakenEncodeException

__all__ = ['PytorchCodec']


class PytorchCodec(object):
    """
    Translates between labels and graphemes.
    """
    def __init__(self, charset):
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
        self.l2c = {}
        for k, v in self.c2l.items():
            self.l2c[''.join(chr(c) for c in v)] = k

        # sort prefixes for c2l regex
        self.c2l_regex = regex.compile(r'|'.join(regex.escape(x) for x in sorted(self.c2l.keys(), key=len, reverse=True)))
        # sort prefixes for l2c regex
        self.l2c_regex = regex.compile(r'|'.join(regex.escape(x) for x in sorted(self.l2c.keys(), key=len, reverse=True)))

    def __len__(self):
        """
        Total number of input labels the codec can decode.
        """
        return len(self.l2c.keys())

    def max_label(self):
        """
        Returns the maximum label value.
        """
        return max(l for labels in self.c2l.values() for l in labels)

    def encode(self, s):
        """
        Encodes a string into a sequence of labels.

        Args:
            s (unicode): Input unicode string

        Returns:
            (torch.IntTensor) encoded label sequence
        """
        splits = self._greedy_split(s, self.c2l_regex)
        labels = []
        for c in splits:
            labels.extend(self.c2l[c])
        return IntTensor(labels)

    def decode(self, labels):
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

    def _greedy_split(self, input, re):
        """
        Splits an input string greedily from a list of prefixes. Stops when no
        more matches are found.

        Args:
            input (unicode): input string
            re (regex.Regex): Prefix match object

        Returns:
            (list) of prefixes

        Raises:
            (KrakenEncodeException) if no prefix match is found for some part
            of the string.
        """
        r = []
        idx = 0
        while True:
            mo = re.match(input, idx)
            if mo is None or idx == len(input):
                if len(input) > idx:
                    raise KrakenEncodeException('No prefix matches for input after {}'.format(idx))
                return r
            r.append(mo.group())
            idx = mo.end()
