# -*- coding: utf-8 -*-
import unittest

from PIL import Image
from pathlib import Path
from pytest import raises

from kraken.lib import lineest

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestLineest(unittest.TestCase):

    """
    Testing centerline estimator
    """

    def setUp(self):
        self.lnorm = lineest.CenterNormalizer()

    def test_dewarp_bw(self):
        """
        Test dewarping of a single line in B/W
        """
        with Image.open(resources / '000236.png') as im:
            o = lineest.dewarp(self.lnorm, im.convert('1'))
            self.assertEqual(self.lnorm.target_height, o.size[1])

    def test_dewarp_gray(self):
        """
        Test dewarping of a single line in grayscale
        """
        with Image.open(resources /'000236.png') as im:
            o = lineest.dewarp(self.lnorm, im.convert('L'))
            self.assertEqual(self.lnorm.target_height, o.size[1])

    def test_dewarp_fail_color(self):
        """
        Test dewarping of a color line fails
        """
        with raises(ValueError):
            with Image.open(resources /'000236.png') as im:
                lineest.dewarp(self.lnorm, im.convert('RGB'))

    def test_dewarp_bw_undewarpable(self):
        """
        Test dewarping of an undewarpable line.
        """
        with Image.open(resources /'ONB_ibn_19110701_010.tif_line_1548924556947_449.png') as im:
            o = lineest.dewarp(self.lnorm, im)
            self.assertEqual(self.lnorm.target_height, o.size[1])

