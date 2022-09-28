# -*- coding: utf-8 -*-
import os
import json
import pytest
import unittest

from PIL import Image
from pytest import raises
from pathlib import Path
from collections import defaultdict

from kraken.lib.models import load_any
from kraken.rpred import rpred, mm_rpred, BaselineOCRRecord, BBoxOCRRecord
from kraken.lib.exceptions import KrakenInputException

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestBBoxRecords(unittest.TestCase):
    """
    Tests the bounding box OCR record.
    """
    def setUp(self):
        with open(resources / 'records.json', 'r') as fp:
            self.box_records = json.load(fp)
            self.ltr_record = self.box_records[0]

    def test_bbox_record_cuts(self):
        """
        Make sure that the cuts of a record are converted to absolute coordinates.
        """
        record = BBoxOCRRecord(**self.ltr_record)
        self.assertEqual(record.cuts, [[[1437, 119], [1437, 256], [1449, 256], [1449, 119]],
                                       [[1484, 119], [1484, 256], [1496, 256], [1496, 119]],
                                       [[1508, 119], [1508, 256], [1520, 256], [1520, 119]],
                                       [[1568, 119], [1568, 256], [1568, 256], [1568, 119]],
                                       [[1603, 119], [1603, 256], [1603, 256], [1603, 119]],
                                       [[1615, 119], [1615, 256], [1627, 256], [1627, 119]],
                                       [[1639, 119], [1639, 256], [1639, 256], [1639, 119]],
                                       [[1663, 119], [1663, 256], [1674, 256], [1674, 119]],
                                       [[1698, 119], [1698, 256], [1698, 256], [1698, 119]],
                                       [[1722, 119], [1722, 256], [1734, 256], [1734, 119]],
                                       [[1746, 119], [1746, 256], [1758, 256], [1758, 119]],
                                       [[1793, 119], [1793, 256], [1805, 256], [1805, 119]],
                                       [[1817, 119], [1817, 256], [1829, 256], [1829, 119]],
                                       [[1853, 119], [1853, 256], [1853, 256], [1853, 119]],
                                       [[1876, 119], [1876, 256], [1888, 256], [1888, 119]],
                                       [[1924, 119], [1924, 256], [1936, 256], [1936, 119]],
                                       [[1959, 119], [1959, 256], [1971, 256], [1971, 119]],
                                       [[2007, 119], [2007, 256], [2019, 256], [2019, 119]],
                                       [[2054, 119], [2054, 256], [2054, 256], [2054, 119]],
                                       [[2078, 119], [2078, 256], [2090, 256], [2090, 119]],
                                       [[2149, 119], [2149, 256], [2149, 256], [2149, 119]],
                                       [[2161, 119], [2161, 256], [2173, 256], [2173, 119]]])

    def test_bbox_record_redisplay(self):
        """
        Test that a display order record remains in display order when
        requesting a DO record.
        """
        record = BBoxOCRRecord(**self.ltr_record, display_order=True)
        self.assertEqual(record, record.display_order())

    def test_bbox_record_relogical(self):
        """
        Test that a logical order record remains in logical order when
        requesting a LO record.
        """
        record = BBoxOCRRecord(**self.ltr_record, display_order=False)
        self.assertEqual(record, record.logical_order())

    def test_bbox_record_display(self):
        """
        test display order conversion of record.
        """
        record = BBoxOCRRecord(**self.ltr_record, display_order=False)
        re_record = record.display_order()
        self.assertEqual(re_record.prediction, 'في معجزاته عليه السلام')
        self.assertEqual(re_record[:][1], ((1437, 119), (2173, 119), (2173, 256), (1437, 256)))
        self.assertAlmostEqual(re_record[2:8][2], 0.9554762, places=4)

    def test_bbox_record_logical(self):
        """
        Test logical order conversion of record.
        """
        record = BBoxOCRRecord(**self.ltr_record, display_order=True)
        re_record = record.logical_order()
        self.assertEqual(re_record.prediction, 'في معجزاته عليه السلام')
        self.assertEqual(re_record[:][1], ((1437, 119), (2173, 119), (2173, 256), (1437, 256)))
        self.assertAlmostEqual(re_record[2:8][2], 0.9554762, places=4)

    def test_bbox_record_slicing(self):
        """
        Tests simple slicing/aggregation of elements in record.
        """
        record = BBoxOCRRecord(**self.ltr_record, display_order=True)
        pred, cut, conf = record[1:8]
        self.assertEqual(pred, 'السلا ه')
        self.assertEqual(cut, ((1484, 119), (1674, 119), (1674, 256), (1484, 256)))
        self.assertAlmostEqual(conf, 0.9259478, places=4)

    def test_bbox_record_slicing(self):
        """
        Tests complex slicing/aggregation of elements in record.
        """
        record = BBoxOCRRecord(**self.ltr_record, display_order=True)
        pred, cut, conf = record[1:5:2]
        self.assertEqual(pred, 'اس')
        self.assertEqual(cut, ((1484, 119), (1568, 119), (1568, 256), (1484, 256)))
        self.assertAlmostEqual(conf, 0.74411, places=4)

class TestBaselineRecords(unittest.TestCase):
    """
    Tests the baseline OCR record.
    """
    def setUp(self):
        self.bidi_record = ()
        self.ltr_record = ()

    def test_baseline_record_cuts(self):
        """
        Make sure that the cuts of a record are converted to absolute coordinates.
        """
        pass

    def test_baseline_record_redisplay(self):
        """
        Test that a display order record remains in display order when
        requesting a DO record.
        """
        pass

    def test_baseline_record_relogical(self):
        """
        Test that a logical order record remains in logical order when
        requesting a LO record.
        """
        pass

    def test_baseline_record_display(self):
        """
        test display order conversion of record.
        """
        pass

    def test_baseline_record_logical(self):
        """
        Test logical order conversion of record.
        """
        pass

    def test_baseline_record_slicing(self):
        """
        Tests simple slicing/aggregation of elements in record.
        """
        pass

    def test_baseline_record_slicing(self):
        """
        Tests complex slicing/aggregation of elements in record.
        """
        pass

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
