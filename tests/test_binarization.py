# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import unittest
import os

from PIL import Image
from kraken.binarization import nlbin

thisfile = os.path.abspath(os.path.dirname(__file__))
resources = os.path.abspath(os.path.join(thisfile, 'resources'))

class TestBinarization(unittest.TestCase):

    """
    Tests of the nlbin function for binarization of images
    """
    def test_not_binarize_bw(self):
        """
        Test that mode '1' images aren't binarized again.
        """
        im = Image.new('1', (1000,1000))
        self.assertEqual(im, nlbin(im))

    def test_binarize(self):
        """
        Tests binarization of JPG image.
        """
        im = Image.open(os.path.join(resources, 'input.jpg'))
        res = nlbin(im)
        # calculate histogram and check if only pixels of value 0/255 exist
        self.assertEqual(254, res.histogram().count(0), msg='Output not '
                         'binarized')


