# -*- coding: utf-8 -*-
"""
Tests for the new task-based inference API (kraken.tasks).

These tests cover RecognitionTaskModel, SegmentationTaskModel, and
ForcedAlignmentTaskModel which are the replacements for the deprecated
rpred/mm_rpred, blla.segment, and forced_align APIs respectively.
"""
import json
import pickle
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from pytest import raises
from difflib import SequenceMatcher

from kraken.containers import (BaselineLine, BaselineOCRRecord, BBoxLine,
                               BBoxOCRRecord, Segmentation)
from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
from kraken.models import load_models
from kraken.tasks import (RecognitionTaskModel,
                          SegmentationTaskModel,
                          ForcedAlignmentTaskModel)

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


class TestRecognitionTaskModel(unittest.TestCase):
    """
    Tests for RecognitionTaskModel which wraps recognition models for
    line-level text recognition.
    """
    def setUp(self):
        self.models = load_models(resources / 'Gallicorpora+_best.safetensors')
        self.im = Image.open(resources / 'input.webp')

        with open(resources / 'bl_rec.pkl', 'rb') as f:
            self.bl_expected = pickle.load(f)
        with open(resources / 'box_rec.pkl', 'rb') as f:
            self.box_expected = pickle.load(f)

        self.bl_seg = Segmentation(type='baselines',
                                   imagename=self.bl_expected.imagename,
                                   lines=[BaselineLine(id=r.id,
                                                       baseline=r.baseline,
                                                       boundary=r.boundary,
                                                       tags=r.tags)
                                          for r in self.bl_expected.lines],
                                   text_direction=self.bl_expected.text_direction,
                                   script_detection=self.bl_expected.script_detection)
        self.box_seg = Segmentation(type='bbox',
                                    imagename=self.box_expected.imagename,
                                    lines=[BBoxLine(id=r.id,
                                                    bbox=r.bbox,
                                                    tags=r.tags)
                                           for r in self.box_expected.lines],
                                    text_direction=self.box_expected.text_direction,
                                    script_detection=self.box_expected.script_detection)

    def test_load_model_from_path(self):
        """
        Tests loading a RecognitionTaskModel from a model file path.
        """
        task = RecognitionTaskModel.load_model(resources / 'Gallicorpora+_best.safetensors')
        self.assertIsInstance(task, RecognitionTaskModel)

    def test_instantiation_from_model_list(self):
        """
        Tests instantiating RecognitionTaskModel from a list of loaded models.
        """
        task = RecognitionTaskModel(self.models)
        self.assertIsInstance(task, RecognitionTaskModel)

    def test_reject_non_recognition_models(self):
        """
        Tests that RecognitionTaskModel raises ValueError when given models
        that don't include a recognition model.
        """
        mock_model = MagicMock()
        mock_model.model_type = ['segmentation']
        with raises(ValueError, match='No recognition model'):
            RecognitionTaskModel([mock_model])

    def test_one_channel_mode_attribute(self):
        """
        Tests that the one_channel_mode attribute is correctly propagated
        from the underlying model.
        """
        task = RecognitionTaskModel(self.models)
        self.assertIn(task.one_channel_mode, (None, '1', 'L'))

    def test_seg_type_attribute(self):
        """
        Tests that the seg_type attribute is correctly propagated from
        the underlying model.
        """
        task = RecognitionTaskModel(self.models)
        self.assertIn(task.seg_type, (None, 'bbox', 'baseline', 'baselines'))

    def test_predict_baseline(self):
        """
        Tests recognition prediction with baseline segmentation.
        """
        task = RecognitionTaskModel(self.models)
        config = RecognitionInferenceConfig()
        records = list(task.predict(self.im, self.bl_seg, config))
        self.assertEqual(len(records), 33)
        for rec in records:
            self.assertIsInstance(rec, BaselineOCRRecord)
        expected_by_id = {r.id: r.prediction for r in self.bl_expected.lines}
        for rec in records:
            self.assertTrue(SequenceMatcher(isjunk=None, a=rec.prediction, b=expected_by_id[rec.id]).ratio() > 0.9,
                            msg=f'Prediction for line {rec.id!r} differs by more than one character')

    def test_predict_bbox(self):
        """
        Tests recognition prediction with bounding box segmentation.
        """
        task = RecognitionTaskModel(self.models)
        config = RecognitionInferenceConfig()
        records = list(task.predict(self.im, self.box_seg, config))
        self.assertEqual(len(records), 29)
        for rec in records:
            self.assertIsInstance(rec, BBoxOCRRecord)
        expected_by_id = {r.id: r.prediction for r in self.box_expected.lines}
        for rec in records:
            self.assertTrue(SequenceMatcher(isjunk=None, a=rec.prediction, b=expected_by_id[rec.id]).ratio() > 0.9,
                            msg=f'Prediction for line {rec.id!r} differs by more than one character')

    def test_predict_empty_segmentation(self):
        """
        Tests that prediction on an empty segmentation yields no records.
        """
        task = RecognitionTaskModel(self.models)
        config = RecognitionInferenceConfig()
        empty_seg = Segmentation(type='baselines',
                                 imagename=resources / 'bw.png',
                                 lines=[],
                                 text_direction='horizontal-lr',
                                 script_detection=False)
        records = list(task.predict(self.im, empty_seg, config))
        self.assertEqual(len(records), 0)

    def test_predict_config_precision(self):
        """
        Tests that different precision settings are accepted.
        """
        task = RecognitionTaskModel(self.models)
        config = RecognitionInferenceConfig(precision='32-true')
        records = list(task.predict(self.im, self.bl_seg, config))
        self.assertEqual(len(records), 33)

    def test_predict_invalid_line_coords(self):
        """
        Tests that invalid line coordinates produce empty records gracefully.
        """
        task = RecognitionTaskModel(self.models)
        config = RecognitionInferenceConfig()
        invalid_seg = Segmentation(type='baselines',
                                   imagename=resources / 'bw.png',
                                   lines=[BaselineLine(id='bar',
                                                       baseline=[[0, 0], [1, 0]],
                                                       boundary=[[0, 0], [1, 0], [1, 1], [0, 1]])],
                                   text_direction='horizontal-lr',
                                   script_detection=False)
        records = list(task.predict(self.im, invalid_seg, config))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].prediction, '')


class TestRTLRecognitionTaskModel(unittest.TestCase):
    """
    Tests for RecognitionTaskModel with RTL (Arabic) text, verifying correct
    bidi reordering and display_order/logical_order switching.
    """

    def setUp(self):
        self.models = load_models(resources / 'all_arabic_scripts.safetensors')
        self.im = Image.open(resources / 'arabic.webp')
        with open(resources / 'arabic_seg.pkl', 'rb') as f:
            full_seg = pickle.load(f)
        self.arabic_seg = Segmentation(type='baselines',
                                       imagename=full_seg.imagename,
                                       lines=[full_seg.lines[0]],
                                       text_direction='horizontal-lr',
                                       script_detection=False)

    def test_predict_rtl_baseline(self):
        """
        Tests that RTL baseline recognition produces a non-empty prediction.
        """
        task = RecognitionTaskModel(self.models)
        config = RecognitionInferenceConfig()
        records = list(task.predict(self.im, self.arabic_seg, config))
        self.assertEqual(len(records), 1)
        self.assertIsInstance(records[0], BaselineOCRRecord)
        self.assertGreater(len(records[0].prediction), 0)

    def test_predict_rtl_baseline_bidi(self):
        """
        Tests that recognition with bidi_reordering=True returns logical
        (reading) order for Arabic text.
        """
        task = RecognitionTaskModel(self.models)
        config = RecognitionInferenceConfig(bidi_reordering=True)
        records = list(task.predict(self.im, self.arabic_seg, config))
        record = records[0]
        self.assertFalse(record._display_order)
        expected = ('\u0639\u0646\u062f \u0639\u062f\u0645 \u0627\u0644\u0639\u0635\u0628\u0627\u062a '
                    '\u0627\u0630\u0627 \u0644\u0645 \u064a\u0643\u0646 \u0644\u0644\u0635\u063a\u064a\u0631\u0629 '
                    '\u0627\u0654\u0645 \u0627\u0654\u064a\u0636\u0627\u064b \u0644\u0645\u0627\u0630 '
                    '\u0643\u0631. . \u0648\u0644\u0646\u0627 \u0627\u0654\u0646 \u0646\u0642\u0648\u0644 '
                    '\u0627\u0646 \u0627\u0644\u0627\u0653\u0645')
        self.assertTrue(SequenceMatcher(isjunk=None, a=record.prediction, b=expected).ratio() > 0.9,
                        msg='RTL bidi prediction differs from expected by more than one character')

    def test_predict_rtl_baseline_nobidi(self):
        """
        Tests that recognition with bidi_reordering=False returns display
        (visual left-to-right) order.
        """
        task = RecognitionTaskModel(self.models)
        config = RecognitionInferenceConfig(bidi_reordering=False)
        records = list(task.predict(self.im, self.arabic_seg, config))
        record = records[0]
        self.assertTrue(record._display_order)
        expected = ('\u0645\u0653\u0627\u0644\u0627 \u0646\u0627 \u0644\u0648\u0642\u0646 '
                    '\u0646\u0654\u0627 \u0627\u0646\u0644\u0648 . .\u0631\u0643 '
                    '\u0630\u0627\u0645\u0644 \u064b\u0627\u0636\u064a\u0654\u0627 '
                    '\u0645\u0654\u0627 \u0629\u0631\u064a\u063a\u0635\u0644\u0644 '
                    '\u0646\u0643\u064a \u0645\u0644 \u0627\u0630\u0627 '
                    '\u062a\u0627\u0628\u0635\u0639\u0644\u0627 \u0645\u062f\u0639 \u062f\u0646\u0639')
        self.assertTrue(SequenceMatcher(isjunk=None, a=record.prediction, b=expected).ratio() > 0.9,
                        msg='RTL non-bidi prediction differs from expected by more than one character')

    def test_predict_rtl_display_logical_switch(self):
        """
        Tests that the record returned by predict with bidi supports
        display_order()/logical_order() conversion roundtrip.
        """
        task = RecognitionTaskModel(self.models)
        config = RecognitionInferenceConfig(bidi_reordering=True)
        records = list(task.predict(self.im, self.arabic_seg, config))
        record = records[0]
        # record is in logical order
        do = record.display_order()
        self.assertNotEqual(do.prediction, record.prediction)
        # roundtrip back to logical
        lo = do.logical_order()
        self.assertEqual(lo.prediction, record.prediction)

    def test_predict_rtl_bidi_nobidi_differ(self):
        """
        Tests that bidi and non-bidi predictions differ for RTL text.
        """
        task = RecognitionTaskModel(self.models)
        config_bidi = RecognitionInferenceConfig(bidi_reordering=True)
        config_nobidi = RecognitionInferenceConfig(bidi_reordering=False)
        rec_bidi = list(task.predict(self.im, self.arabic_seg, config_bidi))[0]
        rec_nobidi = list(task.predict(self.im, self.arabic_seg, config_nobidi))[0]
        self.assertNotEqual(rec_bidi.prediction, rec_nobidi.prediction)


class TestSegmentationTaskModel(unittest.TestCase):
    """
    Tests for SegmentationTaskModel which wraps segmentation models for
    layout analysis and reading order determination.
    """

    def test_load_default_model(self):
        """
        Tests loading the default BLLA segmentation model.
        """
        task = SegmentationTaskModel.load_model()
        self.assertIsInstance(task, SegmentationTaskModel)

    def test_reject_non_segmentation_models(self):
        """
        Tests that SegmentationTaskModel raises ValueError when given models
        that don't include a segmentation model.
        """
        mock_model = MagicMock()
        mock_model.model_type = ['recognition']
        with raises(ValueError, match='No segmentation models'):
            SegmentationTaskModel([mock_model])

    def test_predict_returns_segmentation(self):
        """
        Tests that predict returns a Segmentation container.
        """
        task = SegmentationTaskModel.load_model()
        im = Image.open(resources / 'input.webp')
        config = SegmentationInferenceConfig()
        result = task.predict(im, config)
        self.assertIsInstance(result, Segmentation)

    def test_predict_produces_lines(self):
        """
        Tests that segmentation prediction produces lines.
        """
        task = SegmentationTaskModel.load_model()
        im = Image.open(resources / 'input.webp')
        config = SegmentationInferenceConfig()
        result = task.predict(im, config)
        self.assertGreater(len(result.lines), 0)

    def test_predict_baseline_type(self):
        """
        Tests that the default segmentation model produces baseline-type
        output.
        """
        task = SegmentationTaskModel.load_model()
        im = Image.open(resources / 'input.webp')
        config = SegmentationInferenceConfig()
        result = task.predict(im, config)
        self.assertEqual(result.type, 'baselines')
        for line in result.lines:
            self.assertIsInstance(line, BaselineLine)

    def test_merge_segmentations_single(self):
        """
        Tests that _merge_segmentations with a single segmentation returns
        it unchanged.
        """
        seg = Segmentation(type='baselines',
                           imagename='test.png',
                           text_direction='horizontal-lr',
                           script_detection=False,
                           lines=[BaselineLine(id='l1',
                                               baseline=[[0, 10], [100, 10]],
                                               boundary=[[0, 0], [100, 0], [100, 20], [0, 20]])])
        config = SegmentationInferenceConfig()
        result = SegmentationTaskModel._merge_segmentations([seg], config)
        self.assertIs(result, seg)


class TestForcedAlignmentTaskModel(unittest.TestCase):
    """
    Tests for ForcedAlignmentTaskModel which performs forced alignment of
    transcriptions with recognition model output.
    """

    def setUp(self):
        self.models = load_models(resources / 'overfit.mlmodel')
        self.im = Image.open(resources / '000236.png')
        self.seg = Segmentation(type='baselines',
                                imagename=resources / '000236.png',
                                lines=[BaselineLine(id='foo',
                                                    baseline=[[0, 10], [2543, 10]],
                                                    boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                                    text='\u0721')],
                                text_direction='horizontal-lr',
                                script_detection=False)

    def test_load_model_from_path(self):
        """
        Tests loading a ForcedAlignmentTaskModel from a model file path.
        """
        task = ForcedAlignmentTaskModel.load_model(resources / 'overfit.mlmodel')
        self.assertIsInstance(task, ForcedAlignmentTaskModel)

    def test_instantiation_from_model_list(self):
        """
        Tests instantiating ForcedAlignmentTaskModel from a list of loaded
        models.
        """
        task = ForcedAlignmentTaskModel(self.models)
        self.assertIsInstance(task, ForcedAlignmentTaskModel)

    def test_reject_non_recognition_models(self):
        """
        Tests that ForcedAlignmentTaskModel raises ValueError when given
        models that don't include a recognition model.
        """
        mock_model = MagicMock()
        mock_model.model_type = ['segmentation']
        with raises(ValueError, match='No recognition model'):
            ForcedAlignmentTaskModel([mock_model])

    def test_reject_non_vgsl_models(self):
        """
        Tests that ForcedAlignmentTaskModel raises ValueError when given
        non-TorchVGSLModel recognition models.
        """
        mock_model = MagicMock()
        mock_model.model_type = ['recognition']
        with raises(ValueError, match='only supported by TorchVGSLModel'):
            ForcedAlignmentTaskModel([mock_model])

    def test_predict_enables_logits(self):
        """
        Tests that predict automatically enables return_logits in the config.
        """
        task = ForcedAlignmentTaskModel(self.models)
        config = RecognitionInferenceConfig()
        self.assertFalse(config.return_logits)
        task.predict(self.im, self.seg, config)
        self.assertTrue(config.return_logits)
        self.assertTrue(config.return_line_image)

    def test_predict_returns_segmentation(self):
        """
        Tests that predict returns a Segmentation with aligned records.
        """
        task = ForcedAlignmentTaskModel(self.models)
        config = RecognitionInferenceConfig()
        result = task.predict(self.im, self.seg, config)
        self.assertIsInstance(result, Segmentation)
        self.assertEqual(len(result.lines), 1)

    def test_predict_record_type(self):
        """
        Tests that alignment produces BaselineOCRRecord instances.
        """
        task = ForcedAlignmentTaskModel(self.models)
        config = RecognitionInferenceConfig()
        result = task.predict(self.im, self.seg, config)
        self.assertIsInstance(result.lines[0], BaselineOCRRecord)

    def test_predict_produces_cuts_and_confidences(self):
        """
        Tests that alignment produces character-level positions and
        confidences.
        """
        task = ForcedAlignmentTaskModel(self.models)
        config = RecognitionInferenceConfig()
        result = task.predict(self.im, self.seg, config)
        record = result.lines[0]
        self.assertGreater(len(record.prediction), 0)
        self.assertGreater(len(record.cuts), 0)
        self.assertGreater(len(record.confidences), 0)

    def test_predict_record_count(self):
        """
        Tests that the number of output records matches the number of input
        lines.
        """
        task = ForcedAlignmentTaskModel(self.models)
        config = RecognitionInferenceConfig()
        seg = Segmentation(type='baselines',
                           imagename=resources / '000236.png',
                           lines=[BaselineLine(id='l1',
                                               baseline=[[0, 10], [2543, 10]],
                                               boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                               text='\u0721'),
                                  BaselineLine(id='l2',
                                               baseline=[[0, 10], [2543, 10]],
                                               boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                               text='\u0721')],
                           text_direction='horizontal-lr',
                           script_detection=False)
        result = task.predict(self.im, seg, config)
        self.assertEqual(len(result.lines), 2)

    def test_predict_display_order(self):
        """
        Tests that aligned records are in display order.
        """
        task = ForcedAlignmentTaskModel(self.models)
        config = RecognitionInferenceConfig()
        result = task.predict(self.im, self.seg, config)
        self.assertTrue(result.lines[0]._display_order)

    def test_predict_unencodable_text(self):
        """
        Tests that lines with text the model cannot encode at all raise
        ValueError.
        """
        task = ForcedAlignmentTaskModel(self.models)
        config = RecognitionInferenceConfig()
        seg = Segmentation(type='baselines',
                           imagename=resources / '000236.png',
                           lines=[BaselineLine(id='foo',
                                               baseline=[[0, 10], [2543, 10]],
                                               boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                                               text='ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')],
                           text_direction='horizontal-lr',
                           script_detection=False)
        with raises(ValueError):
            task.predict(self.im, seg, config)

    def test_predict_empty_segmentation(self):
        """
        Tests that alignment on a segmentation with no lines returns an
        empty result.
        """
        task = ForcedAlignmentTaskModel(self.models)
        config = RecognitionInferenceConfig()
        seg = Segmentation(type='baselines',
                           imagename=resources / '000236.png',
                           lines=[],
                           text_direction='horizontal-lr',
                           script_detection=False)
        result = task.predict(self.im, seg, config)
        self.assertIsInstance(result, Segmentation)
        self.assertEqual(len(result.lines), 0)
