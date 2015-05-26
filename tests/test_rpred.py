# -*- coding: utf-8 -*-
import unittest
import os

from PIL import Image

from kraken.lib import lstm
from kraken.rpred import rpred
from kraken.lib.exceptions import KrakenInputException

from nose.tools import raises

thisfile = os.path.abspath(os.path.dirname(__file__))
resources = os.path.abspath(os.path.join(thisfile, 'resources'))

class TestRecognition(unittest.TestCase):

    """
    Tests of the recognition facility and associated routines.
    """
    @raises(KrakenInputException)
    def test_rpred_outbounds(self):
        """
        Tests correct handling of invalid line coordinates.
        """
        im = Image.open(os.path.join(resources, 'bw.png'))
        pred = rpred(None, im, [(-1, -1, 10000, 10000)])
        pred.next()
