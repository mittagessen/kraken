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

# -*- coding: utf-8 -*-
"""
pytorch compatible codec with many-to-many mapping between labels and
graphemes.
"""

import regex

from torch import IntTensor

from kraken.lib.exceptions import KrakenEncodeException

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

        Args:
            charset (unicode, list, dict): Input character set.
        """
        if isinstance(charset, dict):
            self.c2l = charset
        else:
            self.c2l = {k: [v] for v, k in enumerate(sorted(charset))}
        # map integer labels to code points because regex only works with strings
        self.l2c = {}
        print(self.c2l)
        for k, v in self.c2l.items():
            self.l2c[''.join(unichr(c) for c in v)] = k

        # sort prefixes for c2l regex
        self.c2l_regex = regex.compile(r'|'.join(sorted(self.c2l.keys(), key=len, reverse=True)))
        # sort prefixes for l2c regex
        self.l2c_regex = regex.compile(r'|'.join(sorted(self.l2c.keys(), key=len, reverse=True)))

    def __len__(self):
        """
        Total number of input labels the codec can decode.
        """
        return len(self.l2c.keys())

    def encode(self, s):
        """
        Encodes a string into a sequence of labels.

        Args:
            s (unicode): Input unicode string

        Returns:
            (torch.IntTensor) encoded label sequence
        """
        splits = self._greedy_split(s, self.c2l_regex)
        l = []
        for c in splits:
            l.extend(self.c2l[c])
        return IntTensor(l)

    def decode(self, l):
        """
        Decodes a labelling into a string.

        Args:
            l (torch.IntTensor): Input vector containing the label sequence.

        Returns:
            (list) decoded sequence of unicode code points.
        """
        # map into unicode space
        l = ''.join(unichr(v) for v in l)
        splits = self._greedy_split(l, self.l2c_regex)
        c = []
        for i in splits:
            c.append(self.l2c[i])
        return c

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
            if mo is None:
                if len(input) > idx:
                    raise KrakenEncodeException('No prefix matches for {}'.format(input[idx:]))
                return r
            r.append(mo.group())
            idx = mo.end()
