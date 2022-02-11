# -*- coding: utf-8 -*-
import os
import pytest
import unittest

from PIL import Image
from pytest import raises
from pathlib import Path
from collections import defaultdict

from kraken.lib.models import load_any
from kraken.rpred import rpred, mm_rpred
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
        with raises(KrakenInputException):
            pred = mm_rpred({'default': self.model},
                            self.overfit_line,
                            {'boxes': [[('default', [0, 0, 2544, 156])],
                                       [('foobar', [0, 0, 2544, 156])]],
                             'text_direction': 'horizontal',
                             'script_detection': True},
                            True)

    def test_mm_rpred_bl_missing_tags(self):
        """
        Test that mm_rpred fails when tags are missing
        """
        with raises(KrakenInputException):
            pred = mm_rpred({'default': self.model},
                            self.overfit_line,
                            {'lines': [{'tags': {'type': 'default'},
                                        'baseline': [[0,0], [10000, 0]],
                                        'boundary': [[-1, -1], [-1, 10000], [10000, 10000], [10000, -1]]},
                                        {'tags': {'type': 'foobar'},
                                        'baseline': [[0,0], [10000, 0]],
                                        'boundary': [[-1, -1], [-1, 10000], [10000, 10000], [10000, -1]]}],
                             'text_direction': 'horizontal',
                             'type': 'baselines'},
                            True)

    def test_mm_rpred_bbox_ignore_tags(self):
        """
        Tests mm_rpred recognition with ignore tags.
        """
        pred = mm_rpred({'default': self.model},
                        self.overfit_line,
                        {'boxes': [[('foobar', [0, 0, 2544, 156])],
                                   [('default', [0, 0, 2544, 156])]],
                         'text_direction': 'horizontal',
                         'script_detection': True},
                        True,
                        tags_ignore=['foobar'])
        record = next(pred)
        self.assertEqual(record.prediction, '')
        record = next(pred)
        self.assertEqual(record.prediction, 'ܡ ܘܡ ܗ ܡܕܐ ܐ ܐܐ ܡ ܗܗܐܐܐܕ')

    def test_mm_rpred_bbox_default_tags(self):
        """
        Tests recognition with default tag.
        """
        pred = mm_rpred(defaultdict(lambda: self.model),
                        self.overfit_line,
                        {'boxes': [[('foobar', [0, 0, 2544, 156])],
                                   [('default', [0, 0, 2544, 156])]],
                         'text_direction': 'horizontal',
                         'script_detection': True},
                        True)
        record = next(pred)
        self.assertEqual(record.prediction, 'ܡ ܘܡ ܗ ܡܕܐ ܐ ܐܐ ܡ ܗܗܐܐܐܕ')
        record = next(pred)
        self.assertEqual(record.prediction, 'ܡ ܘܡ ܗ ܡܕܐ ܐ ܐܐ ܡ ܗܗܐܐܐܕ')

    def test_mm_rpred_bl_ignore_tags(self):
        """
        Tests baseline recognition with ignore tags.
        """
        pred = mm_rpred({'default': self.model},
                        self.overfit_line,
                        {'lines': [{'tags': {'type': 'foobar'},
                                    'baseline': [[0, 10], [2543, 10]],
                                    'boundary': [[0, 0], [2543, 0], [2543, 155], [0, 155]]},
                                   {'tags': {'type': 'default'},
                                    'baseline': [[0, 10], [2543, 10]],
                                    'boundary': [[0, 0], [2543, 0], [2543, 155], [0, 155]]}],
                         'script_detection': True,
                         'type': 'baselines'},
                        True,
                        tags_ignore=['foobar'])
        record = next(pred)
        self.assertEqual(record.prediction, '')
        record = next(pred)
        self.assertEqual(record.prediction, '.ܗ ܣܗܐ  ܕ ܣ   ܗ ܕܗܗ ܟܕܗܣ    ܠ  ܐ .ܣܕܐܣ. ܗ ')

    def test_mm_rpred_bl_default_tags(self):
        """
        Tests baseline recognition with default tag.
        """
        pred = mm_rpred(defaultdict(lambda: self.model),
                        self.overfit_line,
                        {'lines': [{'tags': {'type': 'foobar'},
                                    'baseline': [[0, 10], [2543, 10]],
                                    'boundary': [[0, 0], [2543, 0], [2543, 155], [0, 155]]},
                                   {'tags': {'type': 'default'},
                                    'baseline': [[0, 10], [2543, 10]],
                                    'boundary': [[0, 0], [2543, 0], [2543, 155], [0, 155]]}],
                         'script_detection': True,
                         'type': 'baselines'},
                        True)
        record = next(pred)
        self.assertEqual(record.prediction, '.ܗ ܣܗܐ  ܕ ܣ   ܗ ܕܗܗ ܟܕܗܣ    ܠ  ܐ .ܣܕܐܣ. ܗ ')
        record = next(pred)
        self.assertEqual(record.prediction, '.ܗ ܣܗܐ  ܕ ܣ   ܗ ܕܗܗ ܟܕܗܣ    ܠ  ܐ .ܣܕܐܣ. ܗ ')

    def test_mm_rpred_bl_nobidi(self):
        """
        Tests baseline recognition without bidi reordering.
        """
        pred = mm_rpred(defaultdict(lambda: self.model),
                        self.overfit_line,
                        {'lines': [{'tags': {'type': 'default'},
                                    'baseline': [[0, 10], [2543, 10]],
                                    'boundary': [[0, 0], [2543, 0], [2543, 155], [0, 155]]}],
                         'script_detection': True,
                         'type': 'baselines'},
                        bidi_reordering=False)
        record = next(pred)
        self.assertEqual(record.prediction, 'ܕܗ .ܣܐܗܗ.ܐ ܗܣ ܕ   ܗܣ ܗ.ܗܝܣܗ ܣ ܗܢ ܪܗܗܕ ܐ   ܗܠ')

    def test_mm_rpred_bbox_nobidi(self):
        """
        Tests bbox recognition without bidi reordering.
        """
        pred = mm_rpred(defaultdict(lambda: self.model),
                        self.overfit_line,
                        {'boxes': [[('foobar', [0, 0, 2544, 156])],
                                   [('default', [0, 0, 2544, 156])]],
                         'text_direction': 'horizontal',
                         'script_detection': True},
                        bidi_reordering=False)
        record = next(pred)
        self.assertEqual(record.prediction,  'ܕܗܣܐܕ ܪܝ .ܡܡ ܐܠܠ ܗܠ ܐܘܗ ܟܘܗܢ ܡܡ ܐܠ')
