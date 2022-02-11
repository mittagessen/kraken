# -*- coding: utf-8 -*-
import os
import click
import unittest
import tempfile
import numpy as np

from PIL import Image
from pathlib import Path
from click.testing import CliRunner
from kraken.kraken import cli

from pytest import raises

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestCLI(unittest.TestCase):
    """
    Testing the kraken CLI
    """

    def setUp(self):
        self.temp = tempfile.NamedTemporaryFile(delete=False)
        self.runner = CliRunner()
        self.color_img = resources / 'input.tif'
        self.bw_img = resources / 'bw.png'

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

    def test_segment_color_noraise(self):
        """
        Tests that segmentation does not return 1 when given color image in noraise mode.
        """
        with tempfile.NamedTemporaryFile() as fp:
            result = self.runner.invoke(cli, ['-i', self.color_img, fp.name, 'segment'])
            self.assertEqual(result.exit_code, 0)
