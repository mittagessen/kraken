# -*- coding: utf-8 -*-
import pickle
import unittest
import warnings
from collections import defaultdict
from pathlib import Path

from PIL import Image
from pytest import raises

from kraken.containers import (BaselineLine, BaselineOCRRecord, BBoxLine,
                               Segmentation)
from kraken.lib.models import load_any
from kraken.rpred import mm_rpred, rpred

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


class TestBBoxRecords(unittest.TestCase):
    """
    Tests the bounding box OCR record.
    """
    def setUp(self):
        # Arabic RTL (pre-constructed records, display_order=False i.e. logical order)
        with open(resources / 'arabic_bbox_records.pkl', 'rb') as fp:
            self.arabic_records = pickle.load(fp)
            self.arabic_record = self.arabic_records[0]

        # Latin LTR (pre-constructed records from Segmentation)
        with open(resources / 'box_rec.pkl', 'rb') as fp:
            seg = pickle.load(fp)
            self.latin_record = seg.lines[5]

    def test_arabic_bbox_record_cuts(self):
        """
        Tests that an Arabic bbox record has the expected number of cuts.
        """
        self.assertEqual(len(self.arabic_record.cuts), 52)

    def test_arabic_bbox_record_display_identity(self):
        """
        Test that requesting display order on a display-order record is
        an identity operation.
        """
        do = self.arabic_record.display_order()
        self.assertEqual(do, do.display_order())

    def test_arabic_bbox_record_logical_identity(self):
        """
        Test that requesting logical order on a logical-order record is
        an identity operation.
        """
        self.assertEqual(self.arabic_record, self.arabic_record.logical_order())

    def test_arabic_bbox_record_display(self):
        """
        Test display order conversion of Arabic RTL record.
        """
        do = self.arabic_record.display_order()
        self.assertEqual(do.prediction, 'مذا ذا درع نلاهو زکذل انبا ملةدیس هىک ماف نابصل ممهع')
        self.assertAlmostEqual(do[:][2], 0.7227956, places=4)

    def test_arabic_bbox_record_logical(self):
        """
        Test that logical order on a logical-order record preserves text.
        """
        lo = self.arabic_record.logical_order()
        self.assertEqual(lo.prediction, 'عهمم لصبان فام کىه سیدةلم ابنا لذکز وهالن عرد اذ اذم')

    def test_arabic_bbox_record_slicing(self):
        """
        Tests simple slicing/aggregation on Arabic bbox record.
        """
        pred, cut, conf = self.arabic_record[1:8]
        self.assertEqual(pred, 'همم لصب')
        self.assertEqual(cut, ((861, 245), (980, 245), (980, 325), (861, 325)))
        self.assertAlmostEqual(conf, 0.7444813, places=4)

    def test_arabic_bbox_record_step_slicing(self):
        """
        Tests step slicing on Arabic bbox record.
        """
        pred, cut, conf = self.arabic_record[1:5:2]
        self.assertEqual(pred, 'هم')
        self.assertEqual(cut, ((936, 245), (980, 245), (980, 325), (936, 325)))
        self.assertAlmostEqual(conf, 0.8795802, places=4)

    def test_latin_bbox_record_display_identity(self):
        """
        Test that display order on LTR record is identity.
        """
        self.assertEqual(self.latin_record, self.latin_record.display_order())

    def test_latin_bbox_record_logical_identity(self):
        """
        Test that logical order on LTR record is identity.
        """
        self.assertEqual(self.latin_record, self.latin_record.logical_order())

    def test_latin_bbox_record_slicing(self):
        """
        Tests simple slicing on Latin bbox record.
        """
        pred, cut, conf = self.latin_record[1:8]
        self.assertEqual(pred, 'i quelq')
        self.assertEqual(cut, ((321, 380), (422, 380), (422, 421), (321, 421)))
        self.assertAlmostEqual(conf, 0.9994162, places=4)

    def test_latin_bbox_record_step_slicing(self):
        """
        Tests step slicing on Latin bbox record.
        """
        pred, cut, conf = self.latin_record[1:5:2]
        self.assertEqual(pred, 'iq')
        self.assertEqual(cut, ((321, 380), (349, 380), (349, 421), (321, 421)))
        self.assertAlmostEqual(conf, 0.9995827, places=4)


class TestBaselineRecords(unittest.TestCase):
    """
    Tests the baseline OCR record with both Arabic (RTL) and Latin (LTR) data.
    """
    def setUp(self):
        # Arabic RTL records (dicts, construct with desired display_order)
        with open(resources / 'arabic_bl_records.pkl', 'rb') as fp:
            self.arabic_records = pickle.load(fp)
            self.arabic_record = self.arabic_records[0]
            self.arabic_short_record = self.arabic_records[6]

        # Latin LTR (pre-constructed records from Segmentation)
        with open(resources / 'bl_rec.pkl', 'rb') as fp:
            seg = pickle.load(fp)
            self.latin_record = seg.lines[5]

    def test_arabic_baseline_record_construction(self):
        """
        Tests that a BaselineOCRRecord can be constructed from pickled data.
        """
        record = BaselineOCRRecord(**self.arabic_record, display_order=True)
        self.assertGreater(len(record.prediction), 0)

    def test_arabic_baseline_record_display_identity(self):
        """
        Test that requesting display order on a display-order record is
        an identity operation.
        """
        record = BaselineOCRRecord(**self.arabic_record, display_order=True)
        self.assertEqual(record, record.display_order())

    def test_arabic_baseline_record_logical_identity(self):
        """
        Test that requesting logical order on a logical-order record is
        an identity operation.
        """
        record = BaselineOCRRecord(**self.arabic_record, display_order=False)
        self.assertEqual(record, record.logical_order())

    def test_arabic_baseline_record_display_to_logical(self):
        """
        Tests display->logical order conversion reorders Arabic text into
        reading order.
        """
        record = BaselineOCRRecord(**self.arabic_record, display_order=True)
        lo = record.logical_order()
        # expected text in logical (reading) order with explicit Unicode escapes
        # for combining characters (hamza above U+0654, maddah above U+0653)
        expected = ('\u0639\u0646\u062f \u0639\u062f\u0645 \u0627\u0644\u0639\u0635\u0628\u0627\u062a '
                    '\u0627\u0630\u0627 \u0644\u0645 \u064a\u0643\u0646 \u0644\u0644\u0635\u063a\u064a\u0631\u0629 '
                    '\u0627\u0654\u0645 \u0627\u0654\u064a\u0636\u0627\u064b \u0644\u0645\u0627\u0630 '
                    '\u0643\u0631. . \u0648\u0644\u0646\u0627 \u0627\u0654\u0646 \u0646\u0642\u0648\u0644 '
                    '\u0627\u0646 \u0627\u0644\u0627\u0653\u0645')
        self.assertEqual(lo.prediction, expected)
        self.assertAlmostEqual(lo[:][2], 0.9746356, places=4)

    def test_arabic_baseline_record_logical_to_display(self):
        """
        Tests logical->display order conversion. The bidi toggle is symmetric
        so both directions produce the same reordered text.
        """
        record = BaselineOCRRecord(**self.arabic_record, display_order=False)
        do = record.display_order()
        expected = ('\u0639\u0646\u062f \u0639\u062f\u0645 \u0627\u0644\u0639\u0635\u0628\u0627\u062a '
                    '\u0627\u0630\u0627 \u0644\u0645 \u064a\u0643\u0646 \u0644\u0644\u0635\u063a\u064a\u0631\u0629 '
                    '\u0627\u0654\u0645 \u0627\u0654\u064a\u0636\u0627\u064b \u0644\u0645\u0627\u0630 '
                    '\u0643\u0631. . \u0648\u0644\u0646\u0627 \u0627\u0654\u0646 \u0646\u0642\u0648\u0644 '
                    '\u0627\u0646 \u0627\u0644\u0627\u0653\u0645')
        self.assertEqual(do.prediction, expected)
        self.assertAlmostEqual(do[:][2], 0.9746356, places=4)

    def test_arabic_baseline_record_roundtrip(self):
        """
        Tests display->logical->display roundtrip produces the original record.
        """
        record = BaselineOCRRecord(**self.arabic_record, display_order=True)
        roundtripped = record.logical_order().display_order()
        self.assertEqual(roundtripped.prediction, record.prediction)

    def test_arabic_baseline_short_record_logical(self):
        """
        Tests display->logical on a short Arabic line.
        """
        record = BaselineOCRRecord(**self.arabic_short_record, display_order=True)
        lo = record.logical_order()
        self.assertEqual(lo.prediction, 'يتناولها .')

    def test_arabic_baseline_record_slicing(self):
        """
        Tests simple slicing on Arabic display order record.
        """
        record = BaselineOCRRecord(**self.arabic_record, display_order=True)
        pred, cut, conf = record[2:8]
        self.assertEqual(pred, 'الا نا')
        self.assertAlmostEqual(conf, 0.9937494, places=4)

    def test_arabic_baseline_record_step_slicing(self):
        """
        Tests step slicing on short Arabic display order record.
        """
        record = BaselineOCRRecord(**self.arabic_short_record, display_order=True)
        pred, cut, conf = record[1:5:2]
        self.assertEqual(pred, ' ه')
        self.assertAlmostEqual(conf, 0.9366213, places=4)

    def test_arabic_baseline_logical_order_slicing(self):
        """
        Tests slicing on a logical-order Arabic record.
        """
        record = BaselineOCRRecord(**self.arabic_record, display_order=True)
        lo = record.logical_order()
        pred, cut, conf = lo[2:8]
        self.assertEqual(pred, 'د عدم ')
        self.assertAlmostEqual(conf, 0.9969620, places=4)

    def test_latin_baseline_record_display_identity(self):
        """
        Test that display order on LTR record is identity.
        """
        self.assertEqual(self.latin_record, self.latin_record.display_order())

    def test_latin_baseline_record_logical_identity(self):
        """
        Test that logical order on LTR record is identity.
        """
        self.assertEqual(self.latin_record, self.latin_record.logical_order())

    def test_latin_baseline_record_slicing(self):
        """
        Tests simple slicing on Latin baseline record.
        """
        pred, cut, conf = self.latin_record[1:8]
        self.assertEqual(pred, 'i quelq')
        self.assertEqual(cut, ([320, 373], [320, 419], [424, 420], [424, 368]))
        self.assertAlmostEqual(conf, 0.9996614, places=4)

    def test_latin_baseline_record_step_slicing(self):
        """
        Tests step slicing on Latin baseline record.
        """
        pred, cut, conf = self.latin_record[1:5:2]
        self.assertEqual(pred, 'iq')
        self.assertEqual(cut, ([320, 373], [320, 419], [346, 423], [346, 375]))
        self.assertAlmostEqual(conf, 0.9998304, places=4)


class TestRecognition(unittest.TestCase):

    """
    Tests of the legacy recognition facility (rpred/mm_rpred) and associated
    routines.

    .. deprecated::
        These tests exercise the deprecated `kraken.rpred.rpred` and
        `kraken.rpred.mm_rpred` APIs as well as `kraken.lib.models.load_any`.
        All of these will be removed with kraken 8. New code should use
        `kraken.tasks.RecognitionTaskModel` instead. See `test_tasks.py` for
        tests of the replacement API.
    """
    def setUp(self):
        warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*deprecated.*kraken 8.*')
        self.im = Image.open(resources / 'bw.png')
        self.overfit_line = Image.open(resources / '000236.png')
        self.model = load_any(resources / 'overfit.mlmodel')
        self.invalid_box_seg = Segmentation(type='bbox',
                                            imagename=resources / 'bw.png',
                                            lines=[BBoxLine(id='foo',
                                                            bbox=[-1, -1, 10000, 10000])],
                                            text_direction='horizontal-lr',
                                            script_detection=False)
        self.invalid_bl_seg = Segmentation(type='baselines',
                                           imagename=resources / 'bw.png',
                                           lines=[BaselineLine(id='bar',
                                                               tags={'type': 'default'},
                                                               baseline=[[0, 0], [10000, 0]],
                                                               boundary=[[-1, -1], [-1, 10000], [10000, 10000], [10000, -1]])],
                                           text_direction='horizontal-lr',
                                           script_detection=False)

        self.simple_box_seg = Segmentation(type='bbox',
                                           imagename=resources / 'bw.png',
                                           lines=[BBoxLine(id='foo',
                                                           bbox=[0, 0, 2544, 156])],
                                           text_direction='horizontal-lr',
                                           script_detection=False)
        self.simple_bl_seg = Segmentation(type='baselines',
                                          imagename=resources / 'bw.png',
                                          lines=[BaselineLine(id='foo',
                                                              baseline=[[0, 10], [2543, 10]],
                                                              boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]])],
                                          text_direction='horizontal-lr',
                                          script_detection=False)

        self.tagged_box_seg = Segmentation(type='bbox',
                                           imagename=resources / 'bw.png',
                                           lines=[BBoxLine(id='foo',
                                                           bbox=[0, 0, 2544, 156],
                                                           tags={'type': [{'type': 'foobar'}]}),
                                                  BBoxLine(id='bar',
                                                           bbox=[0, 0, 2544, 156],
                                                           tags={'type': [{'type': 'default'}]})],
                                           text_direction='horizontal-lr',
                                           script_detection=True)
        self.tagged_bl_seg = Segmentation(type='baselines',
                                          imagename=resources / 'bw.png',
                                          lines=[BaselineLine(id='foo',
                                                              baseline=[[0, 10], [2543, 10]],
                                                              boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                                              tags={'type': [{'type': 'foobar'}]}),
                                                 BaselineLine(id='bar',
                                                              baseline=[[0, 10], [2543, 10]],
                                                              boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                                              tags={'type': [{'type': 'default'}]})],
                                          text_direction='horizontal-lr',
                                          script_detection=True)

    def test_rpred_bbox_outbounds(self):
        """
        Tests correct handling of invalid bbox line coordinates.
        """
        pred = rpred(self.model, self.im, self.invalid_box_seg, True)
        rec = next(pred)
        self.assertEqual(len(rec), 0)

    def test_rpred_bl_outbounds(self):
        """
        Tests correct handling of invalid baseline coordinates.
        """
        pred = rpred(self.model, self.im, self.invalid_bl_seg, True)
        rec = next(pred)
        self.assertEqual(len(rec), 0)

    def test_simple_bbox_rpred(self):
        """
        Tests simple recognition without tags.
        """
        pred = rpred(self.model, self.overfit_line, self.simple_box_seg, True)
        record = next(pred)
        self.assertEqual(record.prediction, 'ܡ ܘܡ ܗ ܡܕܐ ܐ ܐܐ ܡ ܗܗܐܐܐܕ')

    def test_simple_bl_rpred(self):
        """
        Tests simple recognition without tags.
        """
        pred = rpred(self.model, self.overfit_line, self.simple_bl_seg, True)
        record = next(pred)
        self.assertEqual(record.prediction, '.ܗ ܣܗܐ  ܕ ܣ   ܗ ܕܗܗ ܟܕܗܣ    ܠ  ܐ .ܣܕܐܣ. ܗ ')

    def test_mm_rpred_bbox_missing_tags(self):
        """
        Test that mm_rpred fails when tags are missing
        """
        with raises(ValueError):
            mm_rpred({('type', 'default'): self.model},
                     self.overfit_line,
                     self.simple_box_seg,
                     True)

    def test_mm_rpred_bl_missing_tags(self):
        """
        Test that mm_rpred fails when tags are missing
        """
        with raises(ValueError):
            mm_rpred({('type', 'default'): self.model},
                     self.overfit_line,
                     self.simple_bl_seg,
                     True)

    def test_mm_rpred_bbox_ignore_tags(self):
        """
        Tests mm_rpred recognition with ignore tags.
        """
        pred = mm_rpred({'default': self.model},
                        self.overfit_line,
                        self.tagged_box_seg,
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
                        self.tagged_box_seg,
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
                        self.tagged_bl_seg,
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
                        self.tagged_bl_seg,
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
                        self.simple_bl_seg,
                        bidi_reordering=False)
        record = next(pred)
        self.assertEqual(record.prediction, 'ܕܗ .ܣܐܗܗ.ܐ ܗܣ ܕ   ܗܣ ܗ.ܗܝܣܗ ܣ ܗܢ ܪܗܗܕ ܐ   ܗܠ')

    def test_mm_rpred_bbox_nobidi(self):
        """
        Tests bbox recognition without bidi reordering.
        """
        pred = mm_rpred(defaultdict(lambda: self.model),
                        self.overfit_line,
                        self.simple_box_seg,
                        bidi_reordering=False)
        record = next(pred)
        self.assertEqual(record.prediction, 'ܕܗܣܐܕ ܪܝ .ܡܡ ܐܠܠ ܗܠ ܐܘܗ ܟܘܗܢ ܡܡ ܐܠ')
