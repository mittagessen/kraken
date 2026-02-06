# -*- coding: utf-8 -*-

import unittest
import warnings
from pathlib import Path

from kraken.align import forced_align
from kraken.containers import BaselineLine, BaselineOCRRecord, Segmentation
from kraken.lib import xml
from kraken.lib.models import load_any

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestKrakenAlign(unittest.TestCase):
    """
    Tests for the legacy forced alignment module (`kraken.align.forced_align`).

    .. deprecated::
        These tests exercise the deprecated `kraken.align.forced_align` API
        which will be removed with kraken 8. New code should use
        `kraken.tasks.ForcedAlignmentTaskModel` instead. See `test_tasks.py`
        for tests of the replacement API.
    """
    def setUp(self):
        warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*deprecated.*kraken 8.*')
        self.doc = resources / '170025120000003,0074.xml'
        self.bls = xml.XMLPage(self.doc).to_container()
        self.model = load_any(resources / 'overfit.mlmodel')

    def test_forced_align_deprecation_warning(self):
        """
        Tests that forced_align emits a DeprecationWarning.
        """
        seg = Segmentation(type='baselines',
                           imagename=resources / '000236.png',
                           lines=[BaselineLine(id='foo',
                                               baseline=[[0, 10], [2543, 10]],
                                               boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                               text='ܡ')],
                           text_direction='horizontal-lr',
                           script_detection=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            forced_align(seg, self.model)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertGreater(len(dep_warnings), 0)
            self.assertIn('ForcedAlignmentTaskModel', str(dep_warnings[0].message))

    def test_forced_align_returns_segmentation(self):
        """
        Tests that forced_align returns a Segmentation object.
        """
        seg = Segmentation(type='baselines',
                           imagename=resources / '000236.png',
                           lines=[BaselineLine(id='foo',
                                               baseline=[[0, 10], [2543, 10]],
                                               boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                               text='ܡ')],
                           text_direction='horizontal-lr',
                           script_detection=False)
        result = forced_align(seg, self.model)
        self.assertIsInstance(result, Segmentation)

    def test_forced_align_record_count(self):
        """
        Tests that the number of output records matches the number of input lines.
        """
        seg = Segmentation(type='baselines',
                           imagename=resources / '000236.png',
                           lines=[BaselineLine(id='l1',
                                               baseline=[[0, 10], [2543, 10]],
                                               boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                               text='ܡ'),
                                  BaselineLine(id='l2',
                                               baseline=[[0, 10], [2543, 10]],
                                               boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                               text='ܡ')],
                           text_direction='horizontal-lr',
                           script_detection=False)
        result = forced_align(seg, self.model)
        self.assertEqual(len(result.lines), 2)

    def test_forced_align_records_are_baseline_ocr(self):
        """
        Tests that alignment produces BaselineOCRRecord instances.
        """
        seg = Segmentation(type='baselines',
                           imagename=resources / '000236.png',
                           lines=[BaselineLine(id='foo',
                                               baseline=[[0, 10], [2543, 10]],
                                               boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                               text='ܡ')],
                           text_direction='horizontal-lr',
                           script_detection=False)
        result = forced_align(seg, self.model)
        self.assertEqual(len(result.lines), 1)
        self.assertIsInstance(result.lines[0], BaselineOCRRecord)

    def test_forced_align_simple(self):
        """
        Tests alignment on a single line with encodable text produces
        character-level positions and confidences.
        """
        seg = Segmentation(type='baselines',
                           imagename=resources / '000236.png',
                           lines=[BaselineLine(id='foo',
                                               baseline=[[0, 10], [2543, 10]],
                                               boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                               text='ܡ')],
                           text_direction='horizontal-lr',
                           script_detection=False)
        result = forced_align(seg, self.model)
        record = result.lines[0]
        self.assertGreater(len(record.prediction), 0)
        self.assertGreater(len(record.cuts), 0)
        self.assertGreater(len(record.confidences), 0)

    def test_forced_align_xml_document(self):
        """
        Tests alignment on a full XML document with text that is partially
        unencodable raises ValueError when the first line cannot be encoded.
        """
        with self.assertRaises(ValueError):
            forced_align(self.bls, self.model)

    def test_forced_align_unencodable_text(self):
        """
        Tests that lines with text the model cannot encode at all raise
        ValueError because the empty label sequence cannot be aligned.
        """
        seg = Segmentation(type='baselines',
                           imagename=resources / '000236.png',
                           lines=[BaselineLine(id='foo',
                                               baseline=[[0, 10], [2543, 10]],
                                               boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                               text='ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')],
                           text_direction='horizontal-lr',
                           script_detection=False)
        with self.assertRaises(ValueError):
            forced_align(seg, self.model)

    def test_forced_align_display_order(self):
        """
        Tests that aligned records are in display order.
        """
        seg = Segmentation(type='baselines',
                           imagename=resources / '000236.png',
                           lines=[BaselineLine(id='foo',
                                               baseline=[[0, 10], [2543, 10]],
                                               boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                               text='ܡ')],
                           text_direction='horizontal-lr',
                           script_detection=False)
        result = forced_align(seg, self.model)
        record = result.lines[0]
        self.assertTrue(record._display_order)

    def test_forced_align_empty_segmentation(self):
        """
        Tests that alignment on a segmentation with no lines returns an
        empty result.
        """
        seg = Segmentation(type='baselines',
                           imagename=resources / '000236.png',
                           lines=[],
                           text_direction='horizontal-lr',
                           script_detection=False)
        result = forced_align(seg, self.model)
        self.assertIsInstance(result, Segmentation)
        self.assertEqual(len(result.lines), 0)
