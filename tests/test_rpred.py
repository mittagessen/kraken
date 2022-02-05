# -*- coding: utf-8 -*-
import os
import pytest
import unittest

from PIL import Image
from pytest import raises
from pathlib import Path

from kraken.lib.models import load_any
from kraken.rpred import rpred
from kraken.lib.exceptions import KrakenInputException

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestRecognition(unittest.TestCase):

    """
    Tests of the recognition facility and associated routines.
    """
    def setUp(self):
        self.im = Image.open(resources / 'bw.png')
        self.overfit_line = Image.open(resources / '000236.png')
        self.model = load_any(resources / 'overfit.mlmodel')

    def tearDown(self):
        self.im.close()

    def test_rpred_bbox_outbounds(self):
        """
        Tests correct handling of invalid bbox line coordinates.
        """
        with raises(KrakenInputException):
            pred = rpred(self.model, self.im, {'boxes': [[-1, -1, 10000, 10000]], 'text_direction': 'horizontal'}, True)
            next(pred)

    @pytest.mark.xfail
    def test_rpred_bl_outbounds(self):
        """
        Tests correct handling of invalid baseline coordinates.
        """
        with raises(KrakenInputException):
            pred = rpred(self.model, self.im, {'lines': [{'tags': {'type': 'default'},
                                                          'baseline': [[0,0], [10000, 0]],
                                                          'boundary': [[-1, -1], [-1, 10000], [10000, 10000], [10000, -1]]}],
                                               'text_direction': 'horizontal',
                                               'type': 'baselines'}, True)
            next(pred)

    def test_simple_bbox_rpred(self):
        """
        Tests simple recognition without tags.
        """
        pred = rpred(self.model, self.overfit_line, {'boxes': [[0, 0, 2544, 156]], 'text_direction': 'horizontal'}, True)
        record = next(pred)
        self.assertEqual(record.prediction, 'ܡ ܘܡ ܗ ܡܕܐ ܐ ܐܐ ܡ ܗܗܐܐܐܕ')

    def test_simple_bl_rpred(self):
        """
        Tests simple recognition without tags.
        """
        pred = rpred(self.model, self.overfit_line, {'boxes': [[0, 0, 2544, 156]], 'text_direction': 'horizontal'}, True)
        record = next(pred)
        self.assertEqual(record.prediction, 'ܡ ܘܡ ܗ ܡܕܐ ܐ ܐܐ ܡ ܗܗܐܐܐܕ')

    def test_mm_rpred_bbox_missing_tags(self):
        """
        Test that mm_rpred fails when tags are missing
        """

    def test_mm_rpred_bl_missing_tags(self):
        """
        Test that mm_rpred fails when tags are missing
        """

    def test_rpred_ignore_tags(self):
        """
        Tests simple recognition with ignore tags.
        """
        pass

    def test_rpred_default_tags(self):
        """
        Tests recognition with default tag.
        """
        pass
