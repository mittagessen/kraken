# -*- coding: utf-8 -*-

import unittest
import tempfile
import warnings
from unittest.mock import Mock, patch
from pathlib import Path
from traceback import print_exc
from typing import Optional, List, Union

import numpy as np
from PIL import Image

from click.testing import CliRunner

import pytest

from helpers import make_baseline_segmentation

from kraken.lib import segmentation
from kraken.lib.models import load_any
from kraken.lib.xml import XMLPage
from kraken.models import load_models
from kraken.rpred import rpred
from kraken.kraken import cli as kraken_cli
from kraken.ketos import cli as ketos_cli
import re

thisfile = Path(__file__).resolve().parent
resources = thisfile / "resources"

LEGACY_DATASET_WARNING = 'Training set contains lines extracted with the removed legacy polygon extractor'


def mock_extract_polygons():
    return Mock(side_effect=segmentation.extract_polygons)


class TestNewPolygons(unittest.TestCase):
    """
    Tests that all models, irrespective of the polygon extractor they have
    been trained with, use the same (new) polygon extraction method and that
    the legacy options are inert.
    """

    @classmethod
    def setUpClass(cls):
        cls.im = Image.open(resources / "bw.png")
        cls.old_model_path = str(resources / "overfit.mlmodel")
        cls.old_model = load_any(cls.old_model_path)
        cls.new_model_path = str(resources / "overfit_newpoly.mlmodel")
        cls.new_model = load_any(cls.new_model_path)
        cls.old_bl_model_path = str(resources / "overfit_bl.safetensors")
        cls.new_bl_model_path = str(resources / "overfit_bl_newpoly.safetensors")
        cls.segmented_img = str(resources / "170025120000003,0074-lite.xml")
        cls.color_img = resources / "input.tif"
        cls.arrow_data = str(resources / "merge_tests/base.arrow")
        cls.simple_bl_seg = make_baseline_segmentation()

    def setUp(self):
        self.runner = CliRunner()

    def _assertNotLegacy(self, extractor_mock: Mock):
        extractor_mock.assert_called()
        for cl in extractor_mock.mock_calls:
            self.assertFalse(cl[2].get('legacy', False))

    @patch("kraken.rpred.extract_polygons", new_callable=mock_extract_polygons)
    def _test_rpred(self, extractor_mock: Mock, *, model, no_legacy: bool = False):
        """
        Base recipe for testing rpred with a given model
        """
        pred = rpred(model, self.im, self.simple_bl_seg, True, no_legacy_polygons=no_legacy)
        _ = next(pred)
        self._assertNotLegacy(extractor_mock)

    @patch("kraken.models.ctc.extract_polygons", new_callable=mock_extract_polygons)
    def _test_krakencli(self, extractor_mock: Mock, *, args, no_legacy: bool = False):
        """
        Base recipe for testing kraken_cli
        """
        # Insert options after the 'ocr' subcommand: --num-line-workers 0
        # for in-process line extraction so mocks are effective, and
        # --no-legacy-polygons if requested.
        try:
            ocr_idx = args.index('ocr')
            extra = ['--num-line-workers', '0']
            if no_legacy:
                extra += ['--no-legacy-polygons']
            args = args[:ocr_idx + 1] + extra + args[ocr_idx + 1:]
        except ValueError:
            pass

        result = self.runner.invoke(kraken_cli, args)
        print("kraken", *args)

        if result.exception:
            print_exc()

        self.assertEqual(result.exit_code, 0)
        self._assertNotLegacy(extractor_mock)

    def _test_ketoscli(self, *, args, check_exit_code: Optional[Union[int, List[int]]] = 0, patching_dir="kraken.lib.dataset.recognition"):
        """
        Base recipe for testing ketos_cli
        """
        with patch(patching_dir + ".extract_polygons", new_callable=mock_extract_polygons) as extractor_mock:
            result = self.runner.invoke(ketos_cli, args)

            print("ketos", *args)
            if result.exception:
                print(result.output)
                print_exc()

            if check_exit_code is not None:
                if isinstance(check_exit_code, int):
                    check_exit_code = [check_exit_code]
                self.assertIn(result.exit_code, check_exit_code, "Command failed")

            self._assertNotLegacy(extractor_mock)

    def _find_best_model(self, model_dir):
        """Find the best model weights file in the checkpoint directory."""
        models = list(Path(model_dir).glob('best_*.*'))
        self.assertEqual(len(models), 1, f"Expected exactly one best model in {model_dir}, found {models}")
        return str(models[0])

    def _assertNewPolygonModel(self, model_dir):
        """
        Trained models are flagged as using the new extractor so older kraken
        versions pick the right one.
        """
        path = self._find_best_model(model_dir)
        self.assertIs(load_models(path)[0].use_legacy_polygons, False)
        return path

    # TESTS

    def test_extract_polygons_legacy_deprecated(self):
        """
        extract_polygons(legacy=True) warns and returns the same line images.
        """
        seg = XMLPage(self.segmented_img).to_container()
        im = Image.open(seg.imagename)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            legacy = [np.asarray(x) for x, _ in segmentation.extract_polygons(im, seg, legacy=True)]
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        new = [np.asarray(x) for x, _ in segmentation.extract_polygons(im, seg)]
        self.assertEqual(len(legacy), len(new))
        for a, b in zip(legacy, new):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.legacy
    def test_rpred_from_old_model(self):
        """
        Test rpred with old model, check that it uses the new polygon extraction method
        """
        self._test_rpred(model=self.old_model)

    @pytest.mark.legacy
    def test_rpred_from_old_model_no_legacy(self):
        """
        Test rpred with old model and the inert no_legacy_polygons argument
        """
        self._test_rpred(model=self.old_model, no_legacy=True)

    @pytest.mark.legacy
    def test_rpred_from_new_model(self):
        """
        Test rpred with new model, check that it uses new polygon extraction method
        """
        self._test_rpred(model=self.new_model)

    def test_krakencli_ocr_old_model(self):
        """
        Test kraken_cli with old model, check that it uses the new polygon extraction method
        """
        with tempfile.NamedTemporaryFile() as fp:
            self._test_krakencli(args=['-f', 'xml', '-i', self.segmented_img, fp.name, 'ocr', '-m', self.old_bl_model_path])

    def test_krakencli_ocr_old_model_no_legacy(self):
        """
        Test kraken_cli with old model and the inert --no-legacy-polygons flag
        """
        with tempfile.NamedTemporaryFile() as fp:
            self._test_krakencli(args=['-f', 'xml', '-i', self.segmented_img, fp.name, 'ocr', '-m', self.old_bl_model_path],
                                 no_legacy=True)

    def test_krakencli_ocr_new_model(self):
        """
        Test kraken_cli with new model, check that it uses new polygon extraction method
        """
        with tempfile.NamedTemporaryFile() as fp:
            self._test_krakencli(args=['-f', 'xml', '-i', self.segmented_img, fp.name, 'ocr', '-m', self.new_bl_model_path])

    @patch("kraken.models.ctc.extract_polygons", new_callable=mock_extract_polygons)
    def test_krakencli_ocr_bbox_model_default(self, extractor_mock: Mock):
        """
        Without --linetype, XML input is parsed at the model's line type: a
        bbox model gets bbox crops.
        """
        with tempfile.NamedTemporaryFile() as fp:
            result = self.runner.invoke(kraken_cli,
                                        ['-f', 'xml', '-i', self.segmented_img, fp.name,
                                         'ocr', '--num-line-workers', '0', '-m', self.old_model_path])
            if result.exception:
                print_exc()
            self.assertEqual(result.exit_code, 0)
            extractor_mock.assert_called()
            for cl in extractor_mock.mock_calls:
                self.assertEqual(cl[1][1].type, 'bbox')

    def test_ketoscli_test_old_model(self):
        """
        Test `ketos test` with old model, check that it uses the new polygon extraction method
        """
        self._test_ketoscli(args=['--workers', '0', 'test', '-m', self.old_bl_model_path, '-f', 'xml', self.segmented_img])

    def test_ketoscli_test_old_model_no_legacy(self):
        """
        Test `ketos test` with old model and the inert --no-legacy-polygons flag
        """
        self._test_ketoscli(args=['--workers', '0', 'test', '--no-legacy-polygons', '-m', self.old_bl_model_path, '-f', 'xml', self.segmented_img])

    def test_ketoscli_test_new_model(self):
        """
        Test `ketos test` with new model, check that it uses new polygon extraction method
        """
        self._test_ketoscli(args=['--workers', '0', 'test', '-m', self.new_bl_model_path, '-f', 'xml', self.segmented_img])

    def test_ketoscli_test_bbox_model_default(self):
        """
        Without --linetype, `ketos test` evaluates XML data at the model's
        line type: a bbox model uses direct bbox crops and polygon extraction
        never runs.
        """
        with patch("kraken.lib.dataset.recognition.extract_polygons", new_callable=mock_extract_polygons) as extractor_mock:
            result = self.runner.invoke(ketos_cli,
                                        ['--workers', '0', 'test', '-m', self.old_model_path,
                                         '-f', 'xml', self.segmented_img])
            if result.exception:
                print(result.output)
                print_exc()
            self.assertEqual(result.exit_code, 0)
            extractor_mock.assert_not_called()

    @pytest.mark.slow
    def test_ketoscli_train_new_model(self):
        """
        Test `ketos train` with new model, check that it uses new polygon extraction method
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['--workers', '0', 'train', '-f', 'xml', '-N', '1', '-q', 'fixed', '-o', mfp, self.segmented_img],
                check_exit_code=[0, 1],  # Model may not improve during training
            )
            self._test_krakencli(args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', self._assertNewPolygonModel(mfp)])

    @pytest.mark.slow
    def test_ketoscli_train_new_model_legacy_flag(self):
        """
        Test `ketos train` with the inert --legacy-polygons flag
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['--workers', '0', 'train', '--legacy-polygons', '-f', 'xml', '-N', '1', '-q', 'fixed', '-o', mfp, self.segmented_img],
                check_exit_code=[0, 1],  # Model may not improve during training
            )
            self._test_krakencli(args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', self._assertNewPolygonModel(mfp)])

    def test_ketoscli_train_old_model(self):
        """
        Test `ketos train` finetuning old model, check that it uses new
        polygon extraction method and flags the new model accordingly
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['--workers', '0', 'train', '-f', 'xml', '-N', '1', '-q', 'fixed', '-i', self.old_model_path, '--resize', 'union', '-o', mfp, self.segmented_img],
                check_exit_code=[0, 1],  # Model may not improve during training
            )
            self._test_krakencli(args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', self._assertNewPolygonModel(mfp)])

    def test_ketoscli_train_old_model_legacy_flag(self):
        """
        Test `ketos train` finetuning old model with the inert --legacy-polygons flag
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['--workers', '0', 'train', '--legacy-polygons', '-f', 'xml', '-N', '1', '-q', 'fixed', '-i', self.old_model_path, '--resize', 'union', '-o', mfp, self.segmented_img],
                check_exit_code=[0, 1],  # Model may not improve during training
            )
            self._test_krakencli(args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', self._assertNewPolygonModel(mfp)])

    @pytest.mark.slow
    def test_ketoscli_pretrain_new_model(self):
        """
        Test `ketos pretrain` with new model, check that it uses new polygon extraction method
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['--workers', '0', 'pretrain', '-f', 'xml', '-N', '1', '-q', 'fixed', '-o', mfp, self.segmented_img],
                check_exit_code=[0, 1],  # Model may not improve during training
            )
            self._test_krakencli(args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', self._assertNewPolygonModel(mfp)])

    @pytest.mark.slow
    def test_ketoscli_pretrain_new_model_legacy_flag(self):
        """
        Test `ketos pretrain` with the inert --legacy-polygons flag
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['--workers', '0', 'pretrain', '--legacy-polygons', '-f', 'xml', '-N', '1', '-q', 'fixed', '-o', mfp, self.segmented_img],
                check_exit_code=[0, 1],  # Model may not improve during training
            )
            self._test_krakencli(args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', self._assertNewPolygonModel(mfp)])

    def test_ketoscli_pretrain_old_model(self):
        """
        Test `ketos pretrain` with old model, check that it uses new polygon extraction method
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['--workers', '0', 'pretrain', '-f', 'xml', '-N', '1', '-q', 'fixed', '-i', self.old_model_path, '-o', mfp, self.segmented_img],
                check_exit_code=[0, 1],  # Model may not improve during training
            )
            self._test_krakencli(args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', self._assertNewPolygonModel(mfp)])

    def _assertWarnsWhenTrainingArrow(self,
                                      model: str,
                                      *dset: str,
                                      from_model: Optional[str] = None,
                                      expect_warning_msgs: List[str] = [],
                                      expect_not_warning_msgs: List[str] = []):

        args = ['-f', 'binary', '-N', '1', '-q', 'fixed', '-o', model, *dset]
        if from_model:
            args = ['-i', from_model, '--resize', 'union'] + args

        print("ketos", 'train', *args)
        run = self.runner.invoke(ketos_cli, ['train'] + args)
        output = re.sub(r'\w+\.py:\d+\n', '', run.output)
        output = re.sub(r'\s+', ' ', output)
        for warning_msg in expect_warning_msgs:
            self.assertIn(warning_msg, output, f"Expected warning '{warning_msg}' not found in output")
        for warning_msg in expect_not_warning_msgs:
            self.assertNotIn(warning_msg, output, f"Unexpected warning '{warning_msg}' found in output")

    def test_ketos_old_arrow_train(self):
        """
        Test `ketos train` on a dataset compiled with the legacy extractor,
        check that it suggests recompilation and flags the model as new
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            mfp2 = str(Path(tempdir) / "model2")

            self._assertWarnsWhenTrainingArrow(mfp, self.arrow_data, expect_warning_msgs=[LEGACY_DATASET_WARNING])
            self._assertNewPolygonModel(mfp)
            self._assertWarnsWhenTrainingArrow(mfp2, self.arrow_data, from_model=self.old_model_path, expect_warning_msgs=[LEGACY_DATASET_WARNING])
            self._assertNewPolygonModel(mfp2)

    @pytest.mark.slow
    def test_ketos_new_arrow(self):
        """
        Test `ketos compile`, check that it uses new polygon extraction method
        and that training on the dataset does not suggest recompilation
        """
        with tempfile.TemporaryDirectory() as tempdir:
            dset = str(Path(tempdir) / "dataset.arrow")
            mfp = str(Path(tempdir) / "model")

            self._test_ketoscli(
                args=['--workers', '0', 'compile', '-f', 'xml', '-o', dset, self.segmented_img],
                patching_dir="kraken.lib.arrow_dataset",
            )
            self._assertWarnsWhenTrainingArrow(mfp, dset, expect_not_warning_msgs=[LEGACY_DATASET_WARNING])

    @pytest.mark.slow
    def test_ketos_new_arrow_legacy_flag(self):
        """
        Test `ketos compile` with the inert --legacy-polygons flag
        """
        with tempfile.TemporaryDirectory() as tempdir:
            dset = str(Path(tempdir) / "dataset.arrow")
            mfp = str(Path(tempdir) / "model")

            self._test_ketoscli(
                args=['--workers', '0', 'compile', '--legacy-polygons', '-f', 'xml', '-o', dset, self.segmented_img],
                patching_dir="kraken.lib.arrow_dataset",
            )
            self._assertWarnsWhenTrainingArrow(mfp, dset, expect_not_warning_msgs=[LEGACY_DATASET_WARNING])

    def test_ketos_new_arrow_old_model(self):
        """
        Test `ketos train` finetuning an old model on a new dataset, check
        that the new model is flagged as new
        """
        with tempfile.TemporaryDirectory() as tempdir:
            dset = str(Path(tempdir) / "dataset.arrow")
            mfp = str(Path(tempdir) / "model")

            self._test_ketoscli(
                args=['--workers', '0', 'compile', '-f', 'xml', '-o', dset, self.segmented_img],
                patching_dir="kraken.lib.arrow_dataset",
            )
            self._assertWarnsWhenTrainingArrow(mfp, dset, from_model=self.old_model_path, expect_not_warning_msgs=[LEGACY_DATASET_WARNING])
            self._assertNewPolygonModel(mfp)

    @pytest.mark.slow
    def test_ketos_mixed_arrow_train(self):
        """
        Test `ketos train` on new and legacy datasets, check that it suggests
        recompilation
        """
        with tempfile.TemporaryDirectory() as tempdir:
            dset = str(Path(tempdir) / "dataset.arrow")
            mfp = str(Path(tempdir) / "model")

            self._test_ketoscli(
                args=['--workers', '0', 'compile', '-f', 'xml', '-o', dset, self.segmented_img],
                patching_dir="kraken.lib.arrow_dataset",
            )
            self._assertWarnsWhenTrainingArrow(mfp, dset, self.arrow_data, expect_warning_msgs=[LEGACY_DATASET_WARNING])
