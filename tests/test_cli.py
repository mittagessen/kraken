# -*- coding: utf-8 -*-
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from PIL import Image

from kraken.kraken import cli

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


class TestCLI(unittest.TestCase):
    """
    Testing the kraken CLI (binarization subcommand).
    """

    def setUp(self):
        self.temp = tempfile.NamedTemporaryFile(delete=False)
        self.runner = CliRunner()
        self.color_img = resources / 'input.webp'
        self.bw_img = resources / 'input_bw.png'

    def tearDown(self):
        self.temp.close()
        os.unlink(self.temp.name)

    def test_binarize_color(self):
        """
        Tests binarization of color images.
        """
        with tempfile.NamedTemporaryFile() as fp:
            result = self.runner.invoke(cli, ['-i', self.color_img, fp.name, 'binarize'])
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(tuple(map(lambda x: x[1], Image.open(fp).getcolors())), (0, 255))

    def test_binarize_bw(self):
        """
        Tests binarization of b/w images.
        """
        with tempfile.NamedTemporaryFile() as fp:
            result = self.runner.invoke(cli, ['-i', self.bw_img, fp.name, 'binarize'])
            self.assertEqual(result.exit_code, 0)
            bw = np.array(Image.open(self.bw_img))
            new = np.array(Image.open(fp.name))
            self.assertTrue(np.all(bw == new))

    def test_segment_color(self):
        """
        Tests that segmentation is aborted when given color image.
        """
        with tempfile.NamedTemporaryFile() as fp:
            result = self.runner.invoke(cli, ['-r', '-i', self.color_img, fp.name, 'segment'])
            self.assertEqual(result.exit_code, 1)


class TestCLISegmentation(unittest.TestCase):
    """
    Integration tests for the kraken CLI segment subcommand (neural baseline
    segmentation).
    """

    def setUp(self):
        self.runner = CliRunner()
        self.bw_img = resources / 'input.webp'

    def test_segment_baseline_native_output(self):
        """
        Tests baseline segmentation with native JSON output.
        """
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as fp:
            try:
                result = self.runner.invoke(cli, ['-i', str(self.bw_img), fp.name, 'segment', '-bl'])
                self.assertEqual(result.exit_code, 0, msg=result.output)
                with open(fp.name, 'r') as f:
                    data = json.load(f)
                self.assertIn('type', data)
                self.assertEqual(data['type'], 'baselines')
                self.assertIn('lines', data)
                self.assertGreater(len(data['lines']), 0)
            finally:
                os.unlink(fp.name)

    def test_segment_baseline_alto_output(self):
        """
        Tests baseline segmentation with ALTO XML serialization.
        """
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as fp:
            try:
                result = self.runner.invoke(cli, ['-a', '-i', str(self.bw_img), fp.name, 'segment', '-bl'])
                self.assertEqual(result.exit_code, 0, msg=result.output)
                with open(fp.name, 'r') as f:
                    content = f.read()
                self.assertIn('alto', content.lower())
            finally:
                os.unlink(fp.name)

    def test_segment_baseline_text_direction(self):
        """
        Tests that the text direction option is accepted.
        """
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as fp:
            try:
                result = self.runner.invoke(cli, ['-i', str(self.bw_img), fp.name,
                                                  'segment', '-bl', '-d', 'horizontal-rl'])
                self.assertEqual(result.exit_code, 0, msg=result.output)
                with open(fp.name, 'r') as f:
                    data = json.load(f)
                self.assertEqual(data['text_direction'], 'horizontal-rl')
            finally:
                os.unlink(fp.name)


class TestCLIRecognition(unittest.TestCase):
    """
    Integration tests for the kraken CLI ocr subcommand.
    """

    def setUp(self):
        self.runner = CliRunner()
        self.bw_img = resources / 'bw.png'
        self.model = resources / 'overfit.mlmodel'

    def test_ocr_no_segmentation_mode(self):
        """
        Tests recognition in no-segmentation mode (whole image as one line).
        """
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as fp:
            try:
                result = self.runner.invoke(cli, ['-i', str(self.bw_img), fp.name,
                                                  'ocr', '-m', str(self.model), '-s'])
                self.assertEqual(result.exit_code, 0, msg=result.output)
                with open(fp.name, 'r') as f:
                    content = f.read()
                self.assertGreater(len(content.strip()), 0)
            finally:
                os.unlink(fp.name)

    def test_ocr_pipeline_segment_then_ocr(self):
        """
        Tests the chained segment -> ocr pipeline.
        """
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as fp:
            try:
                result = self.runner.invoke(cli, ['-i', str(self.bw_img), fp.name,
                                                  'segment', '-bl',
                                                  'ocr', '-m', str(self.model)])
                self.assertEqual(result.exit_code, 0, msg=result.output)
                with open(fp.name, 'r') as f:
                    content = f.read()
                self.assertGreater(len(content.strip()), 0)
            finally:
                os.unlink(fp.name)

    def test_ocr_alto_output(self):
        """
        Tests recognition with ALTO XML output format.
        """
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as fp:
            try:
                result = self.runner.invoke(cli, ['-a', '-i', str(self.bw_img), fp.name,
                                                  'segment', '-bl',
                                                  'ocr', '-m', str(self.model)])
                self.assertEqual(result.exit_code, 0, msg=result.output)
                with open(fp.name, 'r') as f:
                    content = f.read()
                self.assertIn('alto', content.lower())
            finally:
                os.unlink(fp.name)

    def test_ocr_hocr_output(self):
        """
        Tests recognition with hOCR output format.
        """
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as fp:
            try:
                result = self.runner.invoke(cli, ['-h', '-i', str(self.bw_img), fp.name,
                                                  'segment', '-bl',
                                                  'ocr', '-m', str(self.model)])
                self.assertEqual(result.exit_code, 0, msg=result.output)
                with open(fp.name, 'r') as f:
                    content = f.read()
                self.assertIn('ocr_page', content)
            finally:
                os.unlink(fp.name)

    def test_ocr_pagexml_output(self):
        """
        Tests recognition with PAGE XML output format.
        """
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as fp:
            try:
                result = self.runner.invoke(cli, ['-x', '-i', str(self.bw_img), fp.name,
                                                  'segment', '-bl',
                                                  'ocr', '-m', str(self.model)])
                self.assertEqual(result.exit_code, 0, msg=result.output)
                with open(fp.name, 'r') as f:
                    content = f.read()
                self.assertIn('PcGts', content)
            finally:
                os.unlink(fp.name)

    def test_ocr_missing_model_fails(self):
        """
        Tests that ocr fails gracefully when model file doesn't exist.
        """
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as fp:
            try:
                result = self.runner.invoke(cli, ['-i', str(self.bw_img), fp.name,
                                                  'ocr', '-m', 'nonexistent_model.mlmodel', '-s'])
                self.assertNotEqual(result.exit_code, 0)
            finally:
                os.unlink(fp.name)
