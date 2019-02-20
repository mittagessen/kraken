# -*- coding: utf-8 -*-
import unittest

from nose.tools import raises

import os
import torch
import tempfile
from kraken.lib import vgsl


class TestVGSL(unittest.TestCase):
    """
    Testing VGSL module
    """
    def test_helper_train(self):
        """
        Tests train/eval mode helper methods
        """
        rnn = vgsl.TorchVGSLModel('[1,1,0,48 Lbx10 Do O1c57]')
        rnn.train()
        self.assertTrue(torch.is_grad_enabled())
        self.assertTrue(rnn.nn.training)
        rnn.eval()
        self.assertFalse(torch.is_grad_enabled())
        self.assertFalse(rnn.nn.training)

    @unittest.skip('works randomly on ci')
    def test_helper_threads(self):
        """
        Test openmp threads helper method.
        """
        rnn = vgsl.TorchVGSLModel('[1,1,0,48 Lbx10 Do O1c57]')
        rnn.set_num_threads(4)
        self.assertEqual(torch.get_num_threads(), 4)

    def test_save_model(self):
        """
        Test model serialization.
        """
        rnn = vgsl.TorchVGSLModel('[1,1,0,48 Lbx10 Do O1c57]')
        with tempfile.TemporaryDirectory() as dir:
            rnn.save_model(dir + '/foo.mlmodel')
            self.assertTrue(os.path.exists(dir + '/foo.mlmodel'))

    def test_resize(self):
        """
        Tests resizing of output layers.
        """
        rnn = vgsl.TorchVGSLModel('[1,1,0,48 Lbx10 Do O1c57]')
        rnn.resize_output(80)
        self.assertEqual(rnn.nn[-1].lin.out_features, 80)

    def test_del_resize(self):
        """
        Tests resizing of output layers with entry deletion.
        """
        rnn = vgsl.TorchVGSLModel('[1,1,0,48 Lbx10 Do O1c57]')
        rnn.resize_output(80, [2, 4, 5, 6, 7, 12, 25])
        self.assertEqual(rnn.nn[-1].lin.out_features, 80)
