# -*- coding: utf-8 -*-

import unittest
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

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
        self.old_model_path = resources / "overfit.mlmodel"
        self.old_model = load_any(self.old_model_path)
        self.segmented_img = resources / "170025120000003,0074-lite.xml"
        self.runner = CliRunner()
        self.color_img = resources / "input.tif"
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

    @patch("kraken.rpred.extract_polygons", new_callable=mock_extract_polygons)
    def _test_rpred(self, extractor_mock: Mock, *, model, no_legacy_polygons: bool, expect_legacy: bool):
        """
        Base recipe for testing rpred with a given model and polygon extraction method
        """
        pred = rpred(model, self.im, self.simple_bl_seg, True, no_legacy_polygons=no_legacy_polygons)
        _ = next(pred)

        extractor_mock.assert_called()
        for cl in extractor_mock.mock_calls:
            self.assertEqual(cl[2]["legacy"], expect_legacy)

    @unittest.skip
    def test_rpred_from_old_model(self):
        """
        Test rpred with old model, check that it uses legacy polygon extraction method
        """
        self._test_rpred(model=self.old_model, no_legacy_polygons=False, expect_legacy=True)

    @unittest.skip
    def test_rpred_from_old_model_force_new(self):
        """
        Test rpred with old model, but disabling legacy polygons
        """
        self._test_rpred(model=self.old_model, no_legacy_polygons=True, expect_legacy=False)



    @patch("kraken.rpred.extract_polygons", new_callable=mock_extract_polygons)
    def _test_krakencli(self, extractor_mock: Mock, *, args, no_legacy_polygons: bool, expect_legacy: bool,):
        """
        Base recipe for testing kraken_cli with a given polygon extraction method
        """
        if no_legacy_polygons:
            args = ["--no-legacy-polygons"] + args

        result = self.runner.invoke(kraken_cli, args)
        print(result.output, result.exception)

        self.assertEqual(result.exit_code, 0)
        extractor_mock.assert_called()
        for cl in extractor_mock.mock_calls:
            # assert always called with legacy=True
            self.assertEqual(cl[2]["legacy"], expect_legacy)

    @unittest.skip
    def test_krakencli_ocr_old_model(self):
        """
        Test kraken_cli with old model, check that it uses legacy polygon extraction method
        """
        with tempfile.NamedTemporaryFile() as fp:
            self._test_krakencli(
                args=['-f', 'xml', '-i', self.segmented_img, fp.name, 'ocr', '-m', str(self.old_model_path)],
                no_legacy_polygons=False,
                expect_legacy=True,
            )

    @unittest.skip
    def test_krakencli_ocr_old_model_force_new(self):
        """
        Test kraken_cli with old model, check that it uses legacy polygon extraction method
        """
        with tempfile.NamedTemporaryFile() as fp:
            self._test_krakencli(
                args=['-f', 'xml', '-i', self.segmented_img, fp.name, 'ocr', '-m', str(self.old_model_path)],
                no_legacy_polygons=True,
                expect_legacy=False,
            )



    @patch("kraken.lib.dataset.recognition.extract_polygons", new_callable=mock_extract_polygons)
    def _test_ketoscli(self, extractor_mock: Mock, *, args, expect_legacy: bool, check_exit_code=0):
        """
        Base recipe for testing ketos_cli with a given polygon extraction method
        """
        result = self.runner.invoke(ketos_cli, args)
        print('ketos', *args)
        print(result.output, result.exception)

        if check_exit_code is not None:
            self.assertEqual(result.exit_code, check_exit_code, "Command failed")
        extractor_mock.assert_called()
        for cl in extractor_mock.mock_calls:
            # assert always called with legacy=True
            self.assertEqual(cl[2]["legacy"], expect_legacy)

    @unittest.skip
    def test_ketoscli_test_old_model(self):
        """
        Test ketos_cli with old model, check that it uses legacy polygon extraction method
        """
        self._test_ketoscli(
            args=['test', '-m', str(self.old_model_path), '-f', 'xml', '--workers', '0', str(self.segmented_img)],
            expect_legacy=True,
        )

    @unittest.skip
    def test_ketoscli_test_old_model_force_new(self):
        """
        Test ketos_cli with old model, check that it does not use legacy polygon extraction method
        """
        self._test_ketoscli(
            args=['test', '--no-legacy-polygons', '-m', str(self.old_model_path), '-f', 'xml', '--workers', '0', str(self.segmented_img)],
            expect_legacy=False,
        )

    #@unittest.skip
    def test_ketoscli_train_new_model(self):
        """
        Test ketos_cli with new model, check that it uses new polygon extraction method
        """
        with tempfile.NamedTemporaryFile() as mfp:
            self._test_ketoscli(
                args=['train', '-f', 'xml', '-N', '1', '-q', 'fixed', '-o', mfp.name, '--workers', '0', str(self.segmented_img)],
                expect_legacy=False,
            )
            with tempfile.NamedTemporaryFile() as fp:
                self._test_krakencli(
                    args=['-f', 'xml', '-i', self.segmented_img, fp.name, 'ocr', '-m', mfp.name + "_0.mlmodel"],
                    no_legacy_polygons=True,
                    expect_legacy=False,
                )

    def test_ketoscli_train_old_model(self):
        """
        Test ketos_cli with old model, check that it uses new polygon extraction method
        """
        with tempfile.NamedTemporaryFile() as mfp:
            self._test_ketoscli(
                args=['train', '-f', 'xml', '-N', '1', '-q', 'fixed', '-i', str(self.old_model_path), '--resize', 'add', '-o', mfp.name, '--workers', '0', str(self.segmented_img)],
                expect_legacy=False,
                check_exit_code=None, # Model may not improve during training
            )
            # Check that the model now expects new polygons
            with tempfile.NamedTemporaryFile() as fp:
                self._test_krakencli(
                    args=['-f', 'xml', '-i', self.segmented_img, fp.name, 'ocr', '-m', mfp.name + "_0.mlmodel"],
                    no_legacy_polygons=True,
                    expect_legacy=False,
                )