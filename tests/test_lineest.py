# -*- coding: utf-8 -*-
import unittest
from pathlib import Path

from PIL import Image
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

    def test_dewarp(self):
        """
        Test dewarping of a single line in B/W and grayscale
        """
        for mode in ('1', 'L'):
            with self.subTest(mode=mode):
                with Image.open(resources / '000236.png') as im:
                    o = lineest.dewarp(self.lnorm, im.convert(mode))
                    self.assertEqual(self.lnorm.target_height, o.size[1])

    def test_dewarp_fail_color(self):
        """
        Test dewarping of a color line fails
        """
        with raises(ValueError):
            with Image.open(resources / '000236.png') as im:
                lineest.dewarp(self.lnorm, im.convert('RGB'))

    def test_dewarp_bw_undewarpable(self):
        """
        Test dewarping of an undewarpable line. Regression guard for a past
        crash; only the output height is asserted.
        """
        with Image.open(resources / 'ONB_ibn_19110701_010.tif_line_1548924556947_449.png') as im:
            o = lineest.dewarp(self.lnorm, im)
            self.assertEqual(self.lnorm.target_height, o.size[1])
