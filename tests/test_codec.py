# -*- coding: utf-8 -*-
import unittest

from pytest import raises
from torch import IntTensor

from kraken.lib import codec
from kraken.lib.exceptions import KrakenEncodeException


class TestCodec(unittest.TestCase):

    """
    Testing codec mapping routines
    """

    def setUp(self):
        charsets = {'o2o': 'ab',
                    'm2o': ['aaa', 'aa', 'a', 'b'],
                    'o2m': {'a': [10, 11, 12], 'b': [12, 45, 80]},
                    'm2m': {'aaa': [10, 11, 12], 'aa': [9, 9], 'a': [11], 'bb': [15], 'b': [12]}}
        self.codecs = {kind: codec.PytorchCodec(charset) for kind, charset in charsets.items()}
        self.strict_codecs = {kind: codec.PytorchCodec(charset, strict=True) for kind, charset in charsets.items()}

        self.invalid_c_sequence = 'aaababbcaaa'
        self.valid_c_sequence = 'aaababbaaabbbb'

        self.ada_sequence = 'cdaabae'

        self.invalid_l_sequence = [(45, 78, 778, 0.3793492615638364),
                                   (10, 203, 859, 0.9485075253700872),
                                   (11, 70, 601, 0.7885297329523855),
                                   (12, 251, 831, 0.7216817042926938),
                                   (900, 72, 950, 0.27609823017048707)]

    def test_encode(self):
        """
        Test correct encoding of valid code point sequences.
        """
        cases = [('o2o', [1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2]),
                 ('m2o', [3, 4, 1, 4, 4, 3, 4, 4, 4, 4]),
                 ('o2m', [10, 11, 12, 10, 11, 12, 10, 11, 12, 12, 45, 80, 10,
                          11, 12, 12, 45, 80, 12, 45, 80, 10, 11, 12, 10, 11,
                          12, 10, 11, 12, 12, 45, 80, 12, 45, 80, 12, 45, 80,
                          12, 45, 80]),
                 ('m2m', [10, 11, 12, 12, 11, 15, 10, 11, 12, 15, 15])]
        for kind, expected in cases:
            with self.subTest(kind=kind):
                self.assertTrue(self.codecs[kind].encode(self.valid_c_sequence).eq(
                                IntTensor(expected)).all())

    def test_decode(self):
        """
        Test correct decoding of valid label sequences.
        """
        cases = [('o2o', [1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2], 'aaababbaaabbbb'),
                 ('m2o', [3, 4, 1, 4, 4, 3, 4, 4, 4, 4], 'aaababbaaabbbb'),
                 ('o2m', [10, 11, 12, 10, 11, 12, 10, 11, 12, 12, 45, 80, 10,
                          11, 12, 12, 45, 80, 12, 45, 80, 10, 11, 12, 10, 11,
                          12, 10, 11, 12, 12, 45, 80, 12, 45, 80, 12, 45, 80,
                          12, 45, 80], 'aaababbaaabbbb'),
                 ('m2m', [10, 11, 12, 12, 10, 15, 10, 11, 12, 15, 15], 'aaabbbaaabbbb')]
        for kind, labels, expected in cases:
            with self.subTest(kind=kind):
                decoded = self.codecs[kind].decode([(label, 0, 0, 0.0) for label in labels])
                self.assertEqual(''.join(x[0] for x in decoded), expected)

    def test_decode_invalid_nonstrict(self):
        """
        Test correct handling of undecodable sequences in non-strict mode.
        """
        partial = ('a', 203, 831, 0.8195729875383888)
        cases = [('o2o', []),
                 ('m2o', []),
                 ('o2m', [partial]),
                 ('m2m', [partial, partial, partial])]
        for kind, expected in cases:
            with self.subTest(kind=kind):
                self.assertEqual(self.codecs[kind].decode(self.invalid_l_sequence), expected)

    def test_encode_invalid_nonstrict(self):
        """
        Test correct handling of noisy character sequences in non-strict mode.
        """
        cases = [('o2o', [1, 1, 1, 2, 1, 2, 2, 1, 1, 1]),
                 ('m2o', [3, 4, 1, 4, 4, 3]),
                 ('o2m', [10, 11, 12, 10, 11, 12, 10, 11, 12, 12, 45,
                          80, 10, 11, 12, 12, 45, 80, 12, 45, 80, 10,
                          11, 12, 10, 11, 12, 10, 11, 12]),
                 ('m2m', [10, 11, 12, 12, 11, 15, 10, 11, 12])]
        for kind, expected in cases:
            with self.subTest(kind=kind):
                self.assertTrue(self.codecs[kind].encode(self.invalid_c_sequence).eq(
                                IntTensor(expected)).all())

    def test_decode_invalid_strict(self):
        """
        Test that undecodable sequences raise in strict mode.
        """
        for kind in ('o2o', 'm2o', 'o2m', 'm2m'):
            with self.subTest(kind=kind):
                with raises(KrakenEncodeException):
                    self.strict_codecs[kind].decode(self.invalid_l_sequence)

    def test_encode_invalid_strict(self):
        """
        Test that unencodable sequences raise in strict mode.
        """
        for kind in ('o2o', 'm2o', 'o2m', 'm2m'):
            with self.subTest(kind=kind):
                with raises(KrakenEncodeException):
                    self.strict_codecs[kind].encode(self.invalid_c_sequence)

    def test_codec_add_simple(self):
        """
        Test adding of new code points to codec.
        """
        prev_len = len(self.codecs['o2o'])
        new_codec = self.codecs['o2o'].add_labels('cde')
        self.assertEqual(len(new_codec), prev_len + 3)
        self.assertTrue(new_codec.encode(self.ada_sequence).eq(
                        IntTensor([3, 4, 1, 1, 2, 1, 5])).all())

    def test_codec_add_multiple(self):
        """
        Test adding of new code point sequences to codec as list and dict.
        """
        for form, added in (('list', ['cd', 'e']), ('dict', {'cd': [3], 'e': [4]})):
            with self.subTest(form=form):
                prev_len = len(self.codecs['o2o'])
                new_codec = self.codecs['o2o'].add_labels(added)
                self.assertEqual(len(new_codec), prev_len + 2)
                self.assertTrue(new_codec.encode(self.ada_sequence).eq(
                                IntTensor([3, 1, 1, 2, 1, 4])).all())

    def test_codec_merge_both(self):
        """
        Test merging of a codec adding and removing code points
        """
        merge_codec = codec.PytorchCodec('acde')
        new_codec, del_labels = self.codecs['o2o'].merge(merge_codec)
        self.assertEqual(del_labels, {2})
        self.assertEqual(new_codec.c2l, {'a': [1], 'c': [2], 'd': [3], 'e': [4]})

    def test_codec_merge_add(self):
        """
        Test merging of a codec adding and removing code points
        """
        merge_codec = codec.PytorchCodec('abcde')
        new_codec, del_labels = self.codecs['o2o'].merge(merge_codec)
        self.assertEqual(del_labels, set())
        self.assertEqual(new_codec.c2l, {'a': [1], 'b': [2], 'c': [3], 'd': [4], 'e': [5]})

    def test_codec_merge_remove(self):
        """
        Test merging of a codec removing code points
        """
        merge_codec = codec.PytorchCodec('a')
        new_codec, del_labels = self.codecs['o2o'].merge(merge_codec)
        self.assertEqual(del_labels, {2})
        self.assertEqual(new_codec.c2l, {'a': [1]})
