# -*- coding: utf-8 -*-
import unittest
import os

from future.utils import PY2
from nose.tools import raises

from torch import IntTensor

from kraken.lib import codec
from kraken.lib.exceptions import KrakenEncodeException

class TestCodec(unittest.TestCase):

    """
    Testing codec mapping routines
    """

    def setUp(self):
        # codec mapping one code point to one label
        self.o2o_codec = codec.PytorchCodec('ab')
        # codec mapping many code points to one label
        self.m2o_codec = codec.PytorchCodec(['aaa' , 'aa', 'a', 'b'])
        # codec mapping one code point to many labels
        self.o2m_codec = codec.PytorchCodec({'a': [10, 11, 12], 'b': [12, 45, 80]})
        # codec mapping many code points to many labels
        self.m2m_codec = codec.PytorchCodec({'aaa': [10, 11, 12], 'aa': [10, 10], 'a': [10], 'bb': [15], 'b': [12]})

        self.invalid_c_sequence = 'aaababbcaaa'
        self.valid_c_sequence = 'aaababbaaabbbb'

        self.invalid_l_sequence = IntTensor([45, 10, 11, 12, 900])

    def test_o2o_encode(self):
        """
        Test correct encoding of code point sequence
        """
        self.assertTrue(self.o2o_codec.encode(self.valid_c_sequence).eq(
                        IntTensor([0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1])).all())

    def test_m2o_encode(self):
        """
        Test correct encoding of code point sequence
        """
        self.assertTrue(self.m2o_codec.encode(self.valid_c_sequence).eq(
                        IntTensor([2, 3, 0, 3, 3, 2, 3, 3, 3, 3])).all())

    def test_o2m_encode(self):
        """
        Test correct encoding of code point sequence
        """
        self.assertTrue(self.o2m_codec.encode(self.valid_c_sequence).eq(
                        IntTensor([10, 11, 12, 10, 11, 12, 10, 11, 12,
                                   12, 45, 80, 10, 11, 12, 12, 45, 80, 12, 45,
                                   80, 10, 11, 12, 10, 11, 12, 10, 11, 12, 12,
                                   45, 80, 12, 45, 80, 12, 45, 80, 12, 45,
                                   80])).all())

    def test_m2m_encode(self):
        """
        Test correct encoding of code point sequence
        """
        self.assertTrue(self.m2m_codec.encode(self.valid_c_sequence).eq(
                        IntTensor([10, 11, 12, 12, 10, 15, 10, 11, 12,
                                   15, 15])).all())

    def test_o2o_decode(self):
        """
        Test correct decoding of label sequence
        """
        self.assertEqual(''.join(self.o2o_codec.decode(IntTensor([0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]))),
                         'aaababbaaabbbb')

    def test_m2o_decode(self):
        """
        Test correct decoding of label sequence
        """
        self.assertEqual(''.join(self.m2o_codec.decode(IntTensor([2, 3, 0, 3, 3, 2, 3, 3, 3, 3]))),
                         'aaababbaaabbbb')

    def test_o2m_decode(self):
        """
        Test correct decoding of label sequence
        """
        self.assertEqual(''.join(self.o2m_codec.decode(IntTensor([10, 11, 12, 10,
                                                                  11, 12, 10, 11,
                                                                  12, 12, 45, 80,
                                                                  10, 11, 12, 12,
                                                                  45, 80, 12, 45,
                                                                  80, 10, 11, 12,
                                                                  10, 11, 12, 10,
                                                                  11, 12, 12, 45,
                                                                  80, 12, 45, 80,
                                                                  12, 45, 80, 12,
                                                                  45, 80]))),
                         'aaababbaaabbbb')

    def test_m2m_decode(self):
        """
        Test correct decoding of label sequence
        """
        self.assertEqual(''.join(self.m2m_codec.decode(IntTensor([10, 11, 12, 12, 10, 15, 10, 11, 12, 15, 15]))),
                         'aaababbaaabbbb')

    @raises(KrakenEncodeException)
    def test_o2o_decode_invalid(self):
        """
        Test correct handling of undecodable sequences
        """
        self.o2o_codec.decode(self.invalid_l_sequence)

    @raises(KrakenEncodeException)
    def test_m2o_decode_invalid(self):
        """
        Test correct handling of undecodable sequences
        """
        self.m2o_codec.decode(self.invalid_l_sequence)

    @raises(KrakenEncodeException)
    def test_o2m_decode_invalid(self):
        """
        Test correct handling of undecodable sequences
        """
        self.o2m_codec.decode(self.invalid_l_sequence)

    @raises(KrakenEncodeException)
    def test_m2m_decode_invalid(self):
        """
        Test correct handling of undecodable sequences
        """
        self.m2m_codec.decode(self.invalid_l_sequence)

    @raises(KrakenEncodeException)
    def test_o2o_encode_invalid(self):
        """
        Test correct handling of unencodable sequences
        """
        self.o2o_codec.encode(self.invalid_c_sequence)

    @raises(KrakenEncodeException)
    def test_m2o_encode_invalid(self):
        """
        Test correct handling of unencodable sequences
        """
        self.m2o_codec.encode(self.invalid_c_sequence)

    @raises(KrakenEncodeException)
    def test_o2m_encode_invalid(self):
        """
        Test correct handling of unencodable sequences
        """
        self.o2m_codec.encode(self.invalid_c_sequence)

    @raises(KrakenEncodeException)
    def test_m2m_encode_invalid(self):
        """
        Test correct handling of unencodable sequences
        """
        self.m2m_codec.encode(self.invalid_c_sequence)
