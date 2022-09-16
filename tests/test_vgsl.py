# -*- coding: utf-8 -*-
import unittest
from pytest import raises

import os
import torch
import tempfile
from kraken.lib import vgsl, layers


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

    def test_append(self):
        """
        Test appending one VGSL spec to another.
        """
        rnn = vgsl.TorchVGSLModel('[1,1,0,48 Lbx10 Do O1c57]')
        rnn.append(1, '[Cr1,1,2 Gn2 Cr3,3,4]')
        self.assertEqual(rnn.spec, '[1,1,0,48 Lbx{L_0}10 Cr{C_1}1,1,2 Gn{Gn_2}2 Cr{C_3}3,3,4]')

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

    def test_nested_serial_model(self):
        """
        Test the creation of a nested serial model.
        """
        net = vgsl.TorchVGSLModel('[1,48,0,1 Cr4,2,1,4,2 ([Cr4,2,1,1,1 Do Cr3,3,2,1,1] [Cr4,2,1,1,1 Cr3,3,2,1,1 Do]) S1(1x0)1,3 Lbx2 Do0.5 Lbx2]')
        self.assertIsInstance(net.nn[1], layers.MultiParamParallel)
        for x in net.nn[1].children():
            self.assertIsInstance(x, layers.MultiParamSequential)
            self.assertEqual(len(x), 3)

    def test_parallel_model_inequal(self):
        """
        Test proper raising of ValueError when parallel layers do not have the same output shape.
        """
        with raises(ValueError):
            net = vgsl.TorchVGSLModel('[1,48,0,1 Cr4,2,1,4,2 [Cr4,2,1,1,1 (Cr4,2,1,4,2 Cr3,3,2,1,1) S1(1x0)1,3 Lbx2 Do0.5] Lbx2]')

    def test_complex_serialization(self):
        """
        Test proper serialization and deserialization of a complex model.
        """
        net = vgsl.TorchVGSLModel('[1,48,0,1 Cr4,2,1,4,2 ([Cr4,2,1,1,1 Do Cr3,3,2,1,1] [Cr4,2,1,1,1 Cr3,3,2,1,1 Do]) S1(1x0)1,3 Lbx2 Do0.5 Lbx2]')
