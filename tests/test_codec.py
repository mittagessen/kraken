# -*- coding: utf-8 -*-
import unittest
import os

from future.utils import PY2
from nose.tools import raises

from torch import IntTensor

from kraken.lib import codec
from kraken.lib.exceptions import KrakenEncodeException, KrakenCodecException

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

        self.ada_sequence = 'cdaabae'

        self.invalid_l_sequence = [(45, 78, 778, 0.3793492615638364),
                                   (10, 203, 859, 0.9485075253700872),
                                   (11, 70, 601, 0.7885297329523855),
                                   (12, 251, 831, 0.7216817042926938),
                                   (900, 72, 950, 0.27609823017048707)]

    def test_o2o_encode(self):
        """
        Test correct encoding of one-to-one code point sequence
        """
        self.assertTrue(self.o2o_codec.encode(self.valid_c_sequence).eq(
                        IntTensor([1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2])).all())

    def test_m2o_encode(self):
        """
        Test correct encoding of many-to-one code point sequence
        """
        self.assertTrue(self.m2o_codec.encode(self.valid_c_sequence).eq(
                        IntTensor([3, 4, 1, 4, 4, 3, 4, 4, 4, 4])).all())

    def test_o2m_encode(self):
        """
        Test correct encoding of one-to-many code point sequence
        """
        self.assertTrue(self.o2m_codec.encode(self.valid_c_sequence).eq(
                        IntTensor([10, 11, 12, 10, 11, 12, 10, 11, 12,
                                   12, 45, 80, 10, 11, 12, 12, 45, 80, 12, 45,
                                   80, 10, 11, 12, 10, 11, 12, 10, 11, 12, 12,
                                   45, 80, 12, 45, 80, 12, 45, 80, 12, 45,
                                   80])).all())

    def test_m2m_encode(self):
        """
        Test correct encoding of many-to-many code point sequence
        """
        self.assertTrue(self.m2m_codec.encode(self.valid_c_sequence).eq(
                        IntTensor([10, 11, 12, 12, 10, 15, 10, 11, 12,
                                   15, 15])).all())

    def test_o2o_decode(self):
        """
        Test correct decoding of one-to-one label sequence
        """
        self.assertEqual(''.join(x[0] for x in self.o2o_codec.decode([(1, 288, 652, 0.8537325587315542),
                                                                      (1, 120, 861, 0.4968470297302481),
                                                                      (1, 372, 629, 0.008650773294205938),
                                                                      (2, 406, 831, 0.15637985875540783),
                                                                      (1, 3, 824, 0.26475146828232776),
                                                                      (2, 228, 959, 0.3062689368044844),
                                                                      (2, 472, 679, 0.8677848554329698),
                                                                      (1, 482, 771, 0.6055591197109657),
                                                                      (1, 452, 606, 0.40744265053745055),
                                                                      (1, 166, 879, 0.7509269177978337),
                                                                      (2, 92, 729, 0.34554103785480306),
                                                                      (2, 227, 959, 0.3006394689033981),
                                                                      (2, 341, 699, 0.07798704843315862),
                                                                      (2, 142, 513, 0.9933850573241767)])),
                         'aaababbaaabbbb')

    def test_m2o_decode(self):
        """
        Test correct decoding of many-to-one label sequence
        """
        self.assertEqual(''.join(x[0] for x in self.m2o_codec.decode([(3, 28, 967, 0.07761440833942468),
                                                                      (4, 282, 565, 0.4946281412618093),
                                                                      (1, 411, 853, 0.7767301050586806),
                                                                      (4, 409, 501, 0.47915609540996495),
                                                                      (4, 299, 637, 0.7755889399450564),
                                                                      (3, 340, 834, 0.726656062406549),
                                                                      (4, 296, 846, 0.2274859668684881),
                                                                      (4, 238, 695, 0.32982930128257815),
                                                                      (4, 187, 970, 0.43354272748701805),
                                                                      (4, 376, 863, 0.24483897879550764)])),
                         'aaababbaaabbbb')

    def test_o2m_decode(self):
        """
        Test correct decoding of one-to-many label sequence
        """
        self.assertEqual(''.join(x[0] for x in self.o2m_codec.decode([(10, 35, 959, 0.43819571289990644),
                                                                      (11, 361, 904, 0.1801115018592916),
                                                                      (12, 15, 616, 0.5987506334315549),
                                                                      (10, 226, 577, 0.6178248939780698),
                                                                      (11, 227, 814, 0.31531097360327787),
                                                                      (12, 390, 826, 0.7706594984014595),
                                                                      (10, 251, 579, 0.9442530315305507),
                                                                      (11, 269, 870, 0.4475979925584944),
                                                                      (12, 456, 609, 0.9396137478409995),
                                                                      (12, 60, 757, 0.06416607235266458),
                                                                      (45, 318, 918, 0.8129458423341515),
                                                                      (80, 15, 914, 0.49773432435726517),
                                                                      (10, 211, 648, 0.7919220961861382),
                                                                      (11, 326, 804, 0.7852387442556333),
                                                                      (12, 93, 978, 0.9376801123379804),
                                                                      (12, 23, 698, 0.915543635886972),
                                                                      (45, 71, 599, 0.8137750423628737),
                                                                      (80, 167, 980, 0.6501035181890226),
                                                                      (12, 259, 823, 0.3122860659712233),
                                                                      (45, 312, 948, 0.20582589628806058),
                                                                      (80, 430, 694, 0.3528792552966924),
                                                                      (10, 470, 866, 0.0685524032330419),
                                                                      (11, 459, 826, 0.39354887700146846),
                                                                      (12, 392, 926, 0.4102018609185847),
                                                                      (10, 271, 592, 0.1877915301623876),
                                                                      (11, 206, 995, 0.21614062190981576),
                                                                      (12, 466, 648, 0.3106914763314057),
                                                                      (10, 368, 848, 0.28715379701274113),
                                                                      (11, 252, 962, 0.5535299604896257),
                                                                      (12, 387, 709, 0.844810014550603),
                                                                      (12, 156, 916, 0.9803695305965802),
                                                                      (45, 150, 555, 0.5969071330809561),
                                                                      (80, 381, 922, 0.5608300913697513),
                                                                      (12, 35, 762, 0.5227506455088722),
                                                                      (45, 364, 931, 0.7205481732247938),
                                                                      (80, 341, 580, 0.536934566913969),
                                                                      (12, 79, 919, 0.5136066153481802),
                                                                      (45, 377, 773, 0.6507467790760987),
                                                                      (80, 497, 931, 0.7635100185309783),
                                                                      (12, 76, 580, 0.9542477438586341),
                                                                      (45, 37, 904, 0.4299813924853797),
                                                                      (80, 425, 638, 0.6825047210425983)])),
                         'aaababbaaabbbb')

    def test_m2m_decode(self):
        """
        Test correct decoding of many-to-many label sequence
        """
        self.assertEqual(''.join(x[0] for x in self.m2m_codec.decode([(10, 313, 788, 0.9379917930525369),
                                                                      (11, 117, 793, 0.9974374577004185),
                                                                      (12, 50, 707, 0.020074164253385374),
                                                                      (12, 382, 669, 0.525910770170754),
                                                                      (10, 458, 833, 0.4292373233167248),
                                                                      (15, 45, 831, 0.5759709886686226),
                                                                      (10, 465, 729, 0.8492104897235935),
                                                                      (11, 78, 800, 0.24733538459309445),
                                                                      (12, 375, 872, 0.26908722769105353),
                                                                      (15, 296, 889, 0.44251812620463726),
                                                                      (15, 237, 930, 0.5456105208117391)])),
                         'aaababbaaabbbb')

    @raises(KrakenEncodeException)
    def test_o2o_decode_invalid(self):
        """
        Test correct handling of undecodable sequences (one-to-one decoder)
        """
        self.o2o_codec.decode(self.invalid_l_sequence)

    @raises(KrakenEncodeException)
    def test_m2o_decode_invalid(self):
        """
        Test correct handling of undecodable sequences (many-to-one decoder)
        """
        self.m2o_codec.decode(self.invalid_l_sequence)

    @raises(KrakenEncodeException)
    def test_o2m_decode_invalid(self):
        """
        Test correct handling of undecodable sequences (one-to-many decoder)
        """
        self.o2m_codec.decode(self.invalid_l_sequence)

    @raises(KrakenEncodeException)
    def test_m2m_decode_invalid(self):
        """
        Test correct handling of undecodable sequences (many-to-many decoder)
        """
        self.m2m_codec.decode(self.invalid_l_sequence)

    @raises(KrakenEncodeException)
    def test_o2o_encode_invalid(self):
        """
        Test correct handling of unencodable sequences (one-to-one encoder)
        """
        self.o2o_codec.encode(self.invalid_c_sequence)

    @raises(KrakenEncodeException)
    def test_m2o_encode_invalid(self):
        """
        Test correct handling of unencodable sequences (many-to-one encoder)
        """
        self.m2o_codec.encode(self.invalid_c_sequence)

    @raises(KrakenEncodeException)
    def test_o2m_encode_invalid(self):
        """
        Test correct handling of unencodable sequences (one-to-many encoder)
        """
        self.o2m_codec.encode(self.invalid_c_sequence)

    @raises(KrakenEncodeException)
    def test_m2m_encode_invalid(self):
        """
        Test correct handling of unencodable sequences (many-to-many encoder)
        """
        self.m2m_codec.encode(self.invalid_c_sequence)

    def test_codec_add_simple(self):
        """
        Test adding of new code points to codec.
        """
        prev_len = len(self.o2o_codec)
        codec = self.o2o_codec.add_labels('cde')
        self.assertEqual(len(codec), prev_len + 3)
        self.assertTrue(codec.encode(self.ada_sequence).eq(
                        IntTensor([3, 4, 1, 1, 2, 1, 5])).all())

    def test_codec_add_list(self):
        """
        Test adding of new code points to codec.
        """
        prev_len = len(self.o2o_codec)
        codec = self.o2o_codec.add_labels(['cd', 'e'])
        self.assertEqual(len(codec), prev_len + 2)
        self.assertTrue(codec.encode(self.ada_sequence).eq(
                        IntTensor([3, 1, 1, 2, 1, 4])).all())

    def test_codec_add_dict(self):
        """
        Test adding of new code points to codec.
        """
        prev_len = len(self.o2o_codec)
        codec = self.o2o_codec.add_labels({'cd': [3], 'e': [4]})
        self.assertEqual(len(codec), prev_len + 2)
        self.assertTrue(codec.encode(self.ada_sequence).eq(
                        IntTensor([3, 1, 1, 2, 1, 4])).all())

    def test_codec_merge_both(self):
        """
        Test merging of a codec adding and removing code points
        """
        merge_codec = codec.PytorchCodec('acde')
        new_codec, del_labels = self.o2o_codec.merge(merge_codec)
        self.assertEqual(del_labels, {2})
        self.assertEqual(new_codec.c2l, {'a': [1], 'c': [2], 'd': [3], 'e': [4]})

    def test_codec_merge_add(self):
        """
        Test merging of a codec adding and removing code points
        """
        merge_codec = codec.PytorchCodec('abcde')
        new_codec, del_labels = self.o2o_codec.merge(merge_codec)
        self.assertEqual(del_labels, set())
        self.assertEqual(new_codec.c2l, {'a': [1], 'b': [2], 'c': [3], 'd': [4], 'e': [5]})

    def test_codec_merge_remove(self):
        """
        Test merging of a codec removing code points
        """
        merge_codec = codec.PytorchCodec('a')
        new_codec, del_labels = self.o2o_codec.merge(merge_codec)
        self.assertEqual(del_labels, {2})
        self.assertEqual(new_codec.c2l, {'a': [1]})
