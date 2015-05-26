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
        im = Image.open(os.path.join(resources, 'input.jpg'))
        segment(im)

    def test_segment_bw(self):
        """
        Tests segmentation of bi-level input.
        """
        im = Image.open(os.path.join(resources, 'bw.png'))
        lines = segment(im)
        self.assertAlmostEqual(len(segment(im)), 30, msg='Segmentation differs'
                               'wildly from true line count', delta=5)
