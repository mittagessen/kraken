# -*- coding: utf-8 -*-

from contextlib import contextmanager
import unittest
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path
from traceback import print_exc
import warnings
from typing import Optional, List, Union

from PIL import Image

from click.testing import CliRunner

from kraken.containers import (
    BaselineLine,
    BaselineOCRRecord,
    BBoxLine,
    BBoxOCRRecord,
    Segmentation,
)
from kraken.lib import xml
from kraken.lib import segmentation
from kraken.lib.models import load_any
from kraken.rpred import mm_rpred, rpred
from kraken.kraken import cli as kraken_cli
from kraken.ketos import cli as ketos_cli
import re

thisfile = Path(__file__).resolve().parent
resources = thisfile / "resources"

def mock_extract_polygons():
    return Mock(side_effect=segmentation.extract_polygons)

class TestNewPolygons(unittest.TestCase):
    """
    Tests for the new polygon extraction method.
    """

    def setUp(self):
        self.im = Image.open(resources / "bw.png")
        self.old_model_path = str(resources / "overfit.mlmodel")
        self.old_model = load_any(self.old_model_path)
        self.new_model_path = str(resources / "overfit_newpoly.mlmodel")
        self.new_model = load_any(self.new_model_path)
        self.segmented_img = str(resources / "170025120000003,0074-lite.xml")
        self.runner = CliRunner()
        self.color_img = resources / "input.tif"
        self.arrow_data = str(resources / "merge_tests/base.arrow")
        self.simple_bl_seg = Segmentation(
            type="baselines",
            imagename=resources / "bw.png",
            lines=[
                BaselineLine(
                    id="foo",
                    baseline=[[0, 10], [2543, 10]],
                    boundary=[[0, 0], [2543, 0], [2543, 155], [0, 155]],
                )
            ],
            text_direction="horizontal-lr",
            script_detection=False,
        )

    ## RECIPES

    @patch("kraken.rpred.extract_polygons", new_callable=mock_extract_polygons)
    def _test_rpred(self, extractor_mock: Mock, *, model, force_no_legacy: bool=False, expect_legacy: bool):
        """
        Base recipe for testing rpred with a given model and polygon extraction method
        """
        pred = rpred(model, self.im, self.simple_bl_seg, True, no_legacy_polygons=force_no_legacy)
        _ = next(pred)

        extractor_mock.assert_called()
        for cl in extractor_mock.mock_calls:
            self.assertEqual(cl[2]["legacy"], expect_legacy)

    @patch("kraken.rpred.extract_polygons", new_callable=mock_extract_polygons)
    def _test_krakencli(self, extractor_mock: Mock, *, args, force_no_legacy: bool=False, expect_legacy: bool,):
        """
        Base recipe for testing kraken_cli with a given polygon extraction method
        """
        if force_no_legacy:
            args = ["--no-legacy-polygons"] + args

        result = self.runner.invoke(kraken_cli, args)
        print("kraken", *args)

        if result.exception:
            print_exc()

        self.assertEqual(result.exit_code, 0)
        extractor_mock.assert_called()
        for cl in extractor_mock.mock_calls:
            self.assertEqual(cl[2]["legacy"], expect_legacy)

    def _test_ketoscli(self, *, args, expect_legacy: bool, check_exit_code: Optional[Union[int, List[int]]] = 0, patching_dir="kraken.lib.dataset.recognition"):
        """
        Base recipe for testing ketos_cli with a given polygon extraction method
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

            extractor_mock.assert_called()
            for cl in extractor_mock.mock_calls:
                self.assertEqual(cl[2]["legacy"], expect_legacy)

    ## TESTS

    def test_rpred_from_old_model(self):
        """
        Test rpred with old model, check that it uses legacy polygon extraction method
        """
        self._test_rpred(model=self.old_model, force_no_legacy=False, expect_legacy=True)

    def test_rpred_from_old_model_force_new(self):
        """
        Test rpred with old model, but disabling legacy polygons
        """
        self._test_rpred(model=self.old_model, force_no_legacy=True, expect_legacy=False)

    def test_rpred_from_new_model(self):
        """
        Test rpred with new model, check that it uses new polygon extraction method
        """
        self._test_rpred(model=self.new_model, force_no_legacy=False, expect_legacy=False)


    def test_krakencli_ocr_old_model(self):
        """
        Test kraken_cli with old model, check that it uses legacy polygon extraction method
        """
        with tempfile.NamedTemporaryFile() as fp:
            self._test_krakencli(
                args=['-f', 'xml', '-i', self.segmented_img, fp.name, 'ocr', '-m', self.old_model_path],
                force_no_legacy=False,
                expect_legacy=True,
            )

    def test_krakencli_ocr_old_model_force_new(self):
        """
        Test kraken_cli with old model, check that it uses legacy polygon extraction method
        """
        with tempfile.NamedTemporaryFile() as fp:
            self._test_krakencli(
                args=['-f', 'xml', '-i', self.segmented_img, fp.name, 'ocr', '-m', self.old_model_path],
                force_no_legacy=True,
                expect_legacy=False,
            )

    def test_krakencli_ocr_new_model(self):
        """
        Test kraken_cli with new model, check that it uses new polygon extraction method
        """
        with tempfile.NamedTemporaryFile() as fp:
            self._test_krakencli(
                args=['-f', 'xml', '-i', self.segmented_img, fp.name, 'ocr', '-m', self.new_model_path],
                force_no_legacy=False,
                expect_legacy=False,
            )


    def test_ketoscli_test_old_model(self):
        """
        Test `ketos test` with old model, check that it uses legacy polygon extraction method
        """
        self._test_ketoscli(
            args=['test', '-m', self.old_model_path, '-f', 'xml', '--workers', '0', self.segmented_img],
            expect_legacy=True,
        )

    def test_ketoscli_test_old_model_force_new(self):
        """
        Test `ketos test` with old model, check that it does not use legacy polygon extraction method
        """
        self._test_ketoscli(
            args=['test', '--no-legacy-polygons', '-m', self.old_model_path, '-f', 'xml', '--workers', '0', self.segmented_img],
            expect_legacy=False,
        )

    def test_ketoscli_test_new_model(self):
        """
        Test `ketos test` with new model, check that it uses new polygon extraction method
        """
        self._test_ketoscli(
            args=['test', '-m', self.new_model_path, '-f', 'xml', '--workers', '0', self.segmented_img],
            expect_legacy=False,
        )

    @unittest.skip('fails randomly')
    def test_ketoscli_train_new_model(self):
        """
        Test `ketos train` with new model, check that it uses new polygon extraction method
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['train', '-f', 'xml', '-N', '1', '-q', 'fixed', '-o', mfp, '--workers', '0', self.segmented_img],
                expect_legacy=False,
                check_exit_code=[0, 1], # Model may not improve during training
            )

            self._test_krakencli(
                args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', mfp + "_0.mlmodel"],
                expect_legacy=False,
            )

    @unittest.skip('fails randomly')
    def test_ketoscli_train_new_model_force_legacy(self):
        """
        Test `ketos train` training new model, check that it uses legacy polygon extraction method if forced
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['train', '--legacy-polygons', '-f', 'xml', '-N', '1', '-q', 'fixed', '-o', mfp, '--workers', '0', self.segmented_img],
                expect_legacy=True,
                check_exit_code=[0, 1], # Model may not improve during training
            )

            self._test_krakencli(
                args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', mfp + "_0.mlmodel"],
                expect_legacy=True,
            )

    @unittest.skip('fails randomly')
    def test_ketoscli_train_old_model(self):
        """
        Test `ketos train` finetuning old model, check that it uses new polygon extraction method
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['train', '-f', 'xml', '-N', '1', '-q', 'fixed', '-i', self.old_model_path, '--resize', 'add', '-o', mfp, '--workers', '0', self.segmented_img],
                expect_legacy=False,
                check_exit_code=[0, 1], # Model may not improve during training
            )
            self._test_krakencli(
                args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', mfp + "_0.mlmodel"],
                expect_legacy=False,
            )

    @unittest.skip('fails randomly')
    def test_ketoscli_train_old_model_force_legacy(self):
        """
        Test `ketos train` finetuning old model, check that it uses legacy polygon extraction method if forced
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['train', '--legacy-polygons', '-f', 'xml', '-N', '1', '-q', 'fixed', '-i', self.old_model_path, '--resize', 'add', '-o', mfp, '--workers', '0', self.segmented_img],
                expect_legacy=True,
                check_exit_code=[0, 1], # Model may not improve during training
            )
            self._test_krakencli(
                args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', mfp + "_0.mlmodel"],
                expect_legacy=True,
            )

    @unittest.expectedFailure
    def test_ketoscli_pretrain_new_model(self):
        """
        Test `ketos pretrain` with new model, check that it uses new polygon extraction method
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['pretrain', '-f', 'xml', '-N', '1', '-q', 'fixed', '-o', mfp, '--workers', '0', self.segmented_img],
                expect_legacy=False,
                check_exit_code=[0, 1], # Model may not improve during training
            )
            self._test_krakencli(
                args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', mfp + "_0.mlmodel"],
                expect_legacy=False,
            )

    @unittest.expectedFailure
    def test_ketoscli_pretrain_new_model_force_legacy(self):
        """
        Test `ketos pretrain` with new model, check that it uses legacy polygon extraction method if forced
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['pretrain', '--legacy-polygons', '-f', 'xml', '-N', '1', '-q', 'fixed', '-o', mfp, '--workers', '0', self.segmented_img],
                expect_legacy=True,
                check_exit_code=[0, 1], # Model may not improve during training
            )

            self._test_krakencli(
                args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', str(mfp) + "_0.mlmodel"],
                expect_legacy=True,
            )

    @unittest.expectedFailure
    def test_ketoscli_pretrain_old_model(self):
        """
        Test `ketos pretrain` with old model, check that it uses new polygon extraction method
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            fp = str(Path(tempdir) / "test.xml")

            self._test_ketoscli(
                args=['pretrain', '-f', 'xml', '-N', '1', '-q', 'fixed', '-i', self.old_model_path, '--resize', 'add', '-o', mfp, '--workers', '0', self.segmented_img],
                expect_legacy=False,
                check_exit_code=[0, 1], # Model may not improve during training
            )

            self._test_krakencli(
                args=['-f', 'xml', '-i', self.segmented_img, fp, 'ocr', '-m', mfp + "_0.mlmodel"],
                expect_legacy=False,
            )


    def _assertWarnsWhenTrainingArrow(self,
                                      model: str,
                                      *dset: str,
                                      from_model: Optional[str] = None,
                                      force_legacy: bool = False,
                                      expect_warning_msgs: List[str] = [],
                                      expect_not_warning_msgs: List[str] = []):

        args = ['-f', 'binary', '-N', '1', '-q', 'fixed', '-o', model, *dset]
        if force_legacy:
            args = ['--legacy-polygons'] + args
        if from_model:
            args = ['-i', from_model, '--resize', 'add'] + args

        print("ketos", 'train', *args)
        run = self.runner.invoke(ketos_cli, ['train'] + args)
        output = re.sub(r'\w+\.py:\d+\n', '', run.output)
        output = re.sub(r'\s+', ' ', output)
        for warning_msg in expect_warning_msgs:
            self.assertIn(warning_msg, output, f"Expected warning '{warning_msg}' not found in output")
        for warning_msg in expect_not_warning_msgs:
            self.assertNotIn(warning_msg, output, f"Unexpected warning '{warning_msg}' found in output")

    def test_ketos_old_arrow_train_new(self):
        """
        Test `ketos train`, on old arrow dataset, check that it raises a warning about polygon extraction method only if incoherent
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            mfp2 = str(Path(tempdir) / "model2")

            self._assertWarnsWhenTrainingArrow(mfp, self.arrow_data, force_legacy=False, expect_warning_msgs=["WARNING Setting dataset legacy polygon status to True based on training set", "the new model will be flagged to use legacy"])
            self._assertWarnsWhenTrainingArrow(mfp2, self.arrow_data, force_legacy=True, expect_not_warning_msgs=["WARNING Setting dataset legacy polygon status to True based on training set", "the new model will be flagged to use legacy"])

    def test_ketos_new_arrow(self):
        """
        Test `ketos compile`, check that it uses new polygon extraction method
        """
        with tempfile.TemporaryDirectory() as tempdir:
            dset = str(Path(tempdir) / "dataset.arrow")
            mfp = str(Path(tempdir) / "model")
            mfp2 = str(Path(tempdir) / "model2")

            self._test_ketoscli(
                args=['compile', '--workers', '0', '-f', 'xml', '-o', dset, self.segmented_img],
                expect_legacy=False,
                patching_dir="kraken.lib.arrow_dataset",
            )

            self._assertWarnsWhenTrainingArrow(mfp, dset, force_legacy=False, expect_not_warning_msgs=["WARNING Setting dataset legacy polygon status to False based on training set", "the new model will be flagged to use legacy"])
            self._assertWarnsWhenTrainingArrow(mfp2, dset, force_legacy=True, expect_warning_msgs=["WARNING Setting dataset legacy polygon status to False based on training set", "the new model will be flagged to use new"])


    def test_ketos_new_arrow_force_legacy(self):
        """
        Test `ketos compile`, check that it uses old polygon extraction method
        """
        with tempfile.TemporaryDirectory() as tempdir:
            dset = str(Path(tempdir) / "dataset.arrow")
            mfp = str(Path(tempdir) / "model")
            mfp2 = str(Path(tempdir) / "model2")

            self._test_ketoscli(
                args=['compile', '--workers', '0', '--legacy-polygons', '-f', 'xml', '-o', dset, self.segmented_img],
                expect_legacy=True,
                patching_dir="kraken.lib.arrow_dataset",
            )

            self._assertWarnsWhenTrainingArrow(mfp, dset, force_legacy=False, expect_warning_msgs=["WARNING Setting dataset legacy polygon status to True based on training set", "the new model will be flagged to use legacy"])
            self._assertWarnsWhenTrainingArrow(mfp2, dset, force_legacy=True, expect_not_warning_msgs=["WARNING Setting dataset legacy polygon status to True based on training set", "the new model will be flagged to use legacy"])

    def test_ketos_old_arrow_old_model(self):
        """
        Test `ketos train`, on old arrow dataset, check that it raises a warning about polygon extraction method only if incoherent
        """
        with tempfile.TemporaryDirectory() as tempdir:
            mfp = str(Path(tempdir) / "model")
            mfp2 = str(Path(tempdir) / "model2")

            self._assertWarnsWhenTrainingArrow(mfp, self.arrow_data, from_model=self.old_model_path, force_legacy=False, expect_warning_msgs=["WARNING Setting dataset legacy polygon status to True based on training set"], expect_not_warning_msgs=["model will be flagged to use new"])
            self._assertWarnsWhenTrainingArrow(mfp2, self.arrow_data, from_model=self.old_model_path, force_legacy=True, expect_not_warning_msgs=["WARNING Setting dataset legacy polygon status to True based on training set", "model will be flagged to use new"])

    def test_ketos_new_arrow_old_model(self):
        """
        Test `ketos compile`, on new arrow dataset, check that it raises a warning about polygon extraction method only if incoherent
        """
        with tempfile.TemporaryDirectory() as tempdir:
            dset = str(Path(tempdir) / "dataset.arrow")
            mfp = str(Path(tempdir) / "model")
            mfp2 = str(Path(tempdir) / "model2")

            self._test_ketoscli(
                args=['compile', '--workers', '0', '-f', 'xml', '-o', dset, self.segmented_img],
                expect_legacy=False,
                patching_dir="kraken.lib.arrow_dataset",
            )

            self._assertWarnsWhenTrainingArrow(mfp, dset, from_model=self.old_model_path, force_legacy=False, expect_not_warning_msgs=["WARNING Setting dataset legacy polygon status to False based on training set"], expect_warning_msgs=["model will be flagged to use new"])
            self._assertWarnsWhenTrainingArrow(mfp2, dset, from_model=self.old_model_path, force_legacy=True, expect_warning_msgs=["WARNING Setting dataset legacy polygon status to False based on training set"], expect_not_warning_msgs=["model will be flagged to use new"])

    def test_ketos_mixed_arrow_train_new(self):
        """
        Test `ketos compile`, on mixed arrow dataset, check that it raises a warning about polygon extraction method only if incoherent
        """
        with tempfile.TemporaryDirectory() as tempdir:
            dset = str(Path(tempdir) / "dataset.arrow")
            mfp = str(Path(tempdir) / "model")

            self._test_ketoscli(
                args=['compile', '--workers', '0', '-f', 'xml', '-o', dset, self.segmented_img, self.arrow_data],
                expect_legacy=False,
                patching_dir="kraken.lib.arrow_dataset",
            )

            self._assertWarnsWhenTrainingArrow(mfp, dset, self.arrow_data, force_legacy=True, expect_warning_msgs=["WARNING Mixed legacy polygon", "WARNING Setting dataset legacy polygon status to False based on training set"], expect_not_warning_msgs=["model will be flagged to use legacy"])
