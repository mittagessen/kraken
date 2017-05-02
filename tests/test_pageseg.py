# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import unittest
import os

from PIL import Image
from nose.tools import raises

from kraken.pageseg import segment
from kraken.lib.exceptions import KrakenInputException

thisfile = os.path.abspath(os.path.dirname(__file__))
resources = os.path.abspath(os.path.join(thisfile, 'resources'))


class TestPageSeg(unittest.TestCase):

    """
    Tests of the page segmentation functionality
    """
    @raises(KrakenInputException)
    def test_segment_color(self):
        """
        Test correct handling of color input.
        """
        with Image.open(os.path.join(resources, 'input.jpg')) as im:
            segment(im)

    def test_segment_bw(self):
        """
        Tests segmentation of bi-level input.
        """
        with Image.open(os.path.join(resources, 'bw.png')) as im:
            lines = segment(im)
            # test if line count is roughly correct
            self.assertAlmostEqual(len(lines['boxes']), 30, msg='Segmentation differs '
                                   'wildly from true line count', delta=5)
            # check if lines do not extend beyond image
            for box in lines['boxes']:
                self.assertLess(0, box[0], msg='Line x0 < 0')
                self.assertLess(0, box[1], msg='Line y0 < 0')
                self.assertGreater(im.size[0], box[2], msg='Line x1 > {}'.format(im.size[0]))
                self.assertGreater(im.size[1], box[3], msg='Line y1 > {}'.format(im.size[1]))
