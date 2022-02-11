# -*- coding: utf-8 -*-
import unittest

from pytest import raises

from PIL import Image
from pathlib import Path
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
            with Image.new('1', (1000,1000)) as im:
                nlbin(im)

    def test_not_binarize_bw(self):
        """
        Test that mode '1' images aren't binarized again.
        """
        with Image.open(resources / 'bw.png') as im:
            self.assertEqual(im, nlbin(im))

    def test_binarize_no_bw(self):
        """
        Tests binarization of image formats without a 1bpp mode (JPG).
        """
        with Image.open(resources / 'input.jpg') as im:
            res = nlbin(im)
            # calculate histogram and check if only pixels of value 0/255 exist
            self.assertEqual(254, res.histogram().count(0), msg='Output not '
                             'binarized')

    def test_binarize_tif(self):
        """
        Tests binarization of RGB TIFF images.
        """
        with Image.open(resources /'input.tif') as im:
            res = nlbin(im)
            # calculate histogram and check if only pixels of value 0/255 exist
            self.assertEqual(254, res.histogram().count(0), msg='Output not '
                             'binarized')

    def test_binarize_grayscale(self):
        """
        Test binarization of mode 'L' images.
        """
        with Image.open(resources / 'input.tif') as im:
            res = nlbin(im.convert('L'))
            # calculate histogram and check if only pixels of value 0/255 exist
            self.assertEqual(254, res.histogram().count(0), msg='Output not '
                             'binarized')
