# -*- coding: utf-8 -*-
import unittest
from pathlib import Path

from PIL import Image
from pytest import raises

from kraken.binarization import nlbin
from kraken.lib.exceptions import KrakenInputException

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


class TestBinarization(unittest.TestCase):
    """
    Tests of the nlbin function for binarization of images
    """
    def test_not_binarize_empty(self):
        """
        Test that mode '1' images aren't binarized again.
        """
        with raises(KrakenInputException):
            with Image.new('1', (1000, 1000)) as im:
                nlbin(im)

    def test_not_binarize_bw(self):
        """
        Test that mode '1' images aren't binarized again.
        """
        with Image.open(resources / 'bw.png') as im:
            self.assertEqual(im, nlbin(im))

    def test_binarize(self):
        """
        Tests binarization of RGB (JPG/WEBP) and grayscale input images.
        """
        cases = [('jpg', resources / 'input.jpg', None),
                 ('webp', resources / 'input.webp', None),
                 ('grayscale', resources / 'input.webp', 'L')]
        for name, path, mode in cases:
            with self.subTest(name):
                with Image.open(path) as im:
                    res = nlbin(im.convert(mode) if mode else im)
                    self.assertLessEqual(set(res.getdata()), {0, 255},
                                         msg='Output not binarized')
