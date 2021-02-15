# -*- coding: utf-8 -*-
import unittest

from nose.tools import raises

import torch
from kraken.lib import layers


class TestLayers(unittest.TestCase):

    """
    Testing custom layer implementations.
    """
    def setUp(self):
        torch.set_grad_enabled(False)

    def test_maxpool(self):
        """
        Test maximum pooling layer.
        """
        mp = layers.MaxPool((3, 3), (2, 2))
        o = mp(torch.randn(1, 2, 32, 64))
        self.assertEqual(o[0].shape, (1, 2, 15, 31))

    def test_1d_dropout(self):
        """
        Test 1d dropout layer.
        """
        do = layers.Dropout(0.2, 1)
        o = do(torch.randn(1, 2, 32, 64))
        self.assertEqual(o[0].shape, (1, 2, 32, 64))

    def test_2d_dropout(self):
        """
        Test 2d dropout layer.
        """
        do = layers.Dropout(0.2, 2)
        o = do(torch.randn(1, 2, 32, 64))
        self.assertEqual(o[0].shape, (1, 2, 32, 64))

    def test_forward_rnn_layer_x(self):
        """
        Test unidirectional RNN layer in x-dimension.
        """
        rnn = layers.TransposedSummarizingRNN(10, 2, 'f', False, False)
        o = rnn(torch.randn(1, 10, 32, 64))
        self.assertEqual(o[0].shape, (1, 2, 32, 64))

    def test_forward_rnn_layer_y(self):
        """
        Test unidirectional RNN layer in y-dimension.
        """
        rnn = layers.TransposedSummarizingRNN(10, 2, 'f', True, False)
        o = rnn(torch.randn(1, 10, 32, 64))
        self.assertEqual(o[0].shape, (1, 2, 32, 64))

    def test_forward_rnn_layer_x_summarize(self):
        """
        Test unidirectional summarizing RNN layer in x-dimension.
        """
        rnn = layers.TransposedSummarizingRNN(10, 2, 'f', False, True)
        o = rnn(torch.randn(1, 10, 32, 64))
        self.assertEqual(o[0].shape, (1, 2, 32, 1))

    def test_forward_rnn_layer_y_summarize(self):
        """
        Test unidirectional summarizing RNN layer in y-dimension.
        """
        rnn = layers.TransposedSummarizingRNN(10, 2, 'f', True, True)
        o = rnn(torch.randn(1, 10, 32, 64))
        self.assertEqual(o[0].shape, (1, 2, 1, 64))

    def test_bidi_rnn_layer_x(self):
        """
        Test bidirectional RNN layer in x-dimension.
        """
        rnn = layers.TransposedSummarizingRNN(10, 2, 'b', False, False)
        o = rnn(torch.randn(1, 10, 32, 64))
        self.assertEqual(o[0].shape, (1, 4, 32, 64))

    def test_bidi_rnn_layer_y(self):
        """
        Test bidirectional RNN layer in y-dimension.
        """
        rnn = layers.TransposedSummarizingRNN(10, 2, 'b', True, False)
        o = rnn(torch.randn(1, 10, 32, 64))
        self.assertEqual(o[0].shape, (1, 4, 32, 64))

    def test_bidi_rnn_layer_x_summarize(self):
        """
        Test bidirectional summarizing RNN layer in x-dimension.
        """
        rnn = layers.TransposedSummarizingRNN(10, 2, 'b', False, True)
        o = rnn(torch.randn(1, 10, 32, 64))
        self.assertEqual(o[0].shape, (1, 4, 32, 1))

    def test_bidi_rnn_layer_y_summarize(self):
        """
        Test bidirectional summarizing RNN layer in y-dimension.
        """
        rnn = layers.TransposedSummarizingRNN(10, 2, 'b', True, True)
        o = rnn(torch.randn(1, 10, 32, 64))
        self.assertEqual(o[0].shape, (1, 4, 1, 64))

    def test_linsoftmax(self):
        """
        Test basic function of linear layer.
        """
        lin = layers.LinSoftmax(20, 10)
        o = lin(torch.randn(1, 20, 12, 24))
        self.assertEqual(o[0].shape, (1, 10, 12, 24))

    def test_linsoftmax_train(self):
        """
        Test function of linear layer in training mode (log_softmax)
        """
        lin = layers.LinSoftmax(20, 10).train()
        o = lin(torch.randn(1, 20, 12, 24))
        self.assertLess(o[0].max(), 0)

    def test_linsoftmax_test(self):
        """
        Test function of linear layer in eval mode (softmax)
        """
        lin = layers.LinSoftmax(20, 10).eval()
        o = lin(torch.randn(1, 20, 12, 24))
        self.assertGreaterEqual(o[0].min(), 0)

    def test_linsoftmax_aug(self):
        """
        Test basic function of linear layer with 1-augmentation.
        """
        lin = layers.LinSoftmax(20, 10, True)
        o = lin(torch.randn(1, 20, 12, 24))
        self.assertEqual(o[0].shape, (1, 10, 12, 24))

    def test_linsoftmax_aug_train(self):
        """
        Test function of linear layer in training mode (log_softmax) with 1-augmentation
        """
        lin = layers.LinSoftmax(20, 10, True).train()
        o = lin(torch.randn(1, 20, 12, 24))
        self.assertLess(o[0].max(), 0)

    def test_linsoftmax_aug_test(self):
        """
        Test function of linear layer in eval mode (softmax) with 1-augmentation
        """
        lin = layers.LinSoftmax(20, 10, True).eval()
        o = lin(torch.randn(1, 20, 12, 24))
        self.assertGreaterEqual(o[0].min(), 0)

    def test_actconv2d_lin(self):
        """
        Test convolutional layer without activation.
        """
        conv = layers.ActConv2D(5, 12, (3, 3), (1, 1), 'l')
        o = conv(torch.randn(1, 5, 24, 12))
        self.assertEqual(o[0].shape, (1, 12, 24, 12))

    def test_actconv2d_sigmoid(self):
        """
        Test convolutional layer with sigmoid activation.
        """
        conv = layers.ActConv2D(5, 12, (3, 3), (1, 1), 's')
        o = conv(torch.randn(1, 5, 24, 12))
        self.assertTrue(0 <= o[0].min() <= 1)
        self.assertTrue(0 <= o[0].max() <= 1)

    def test_actconv2d_tanh(self):
        """
        Test convolutional layer with tanh activation.
        """
        conv = layers.ActConv2D(5, 12, (3, 3), (1, 1), 't')
        o = conv(torch.randn(1, 5, 24, 12))
        self.assertTrue(-1 <= o[0].min() <= 1)
        self.assertTrue(-1 <= o[0].max() <= 1)

    def test_actconv2d_softmax(self):
        """
        Test convolutional layer with softmax activation.
        """
        conv = layers.ActConv2D(5, 12, (3, 3), (1, 1), 'm')
        o = conv(torch.randn(1, 5, 24, 12))
        self.assertTrue(0 <= o[0].min() <= 1)
        self.assertTrue(0 <= o[0].max() <= 1)

    def test_actconv2d_relu(self):
        """
        Test convolutional layer with relu activation.
        """
        conv = layers.ActConv2D(5, 12, (3, 3), (1, 1), 'r')
        o = conv(torch.randn(1, 5, 24, 12))
        self.assertLessEqual(0, o[0].min())
        self.assertLessEqual(0, o[0].max())

    def test_linsoftmax_resize_add(self):
        """
        Tests resizing of a fully connected layer.
        """
        lin = layers.LinSoftmax(20, 10)
        w_cp = lin.lin.weight.clone()
        b_cp = lin.lin.bias.clone()
        lin.resize(25)
        self.assertTrue(w_cp.eq(lin.lin.weight[:10, :]).all())
        self.assertTrue(b_cp.eq(lin.lin.bias[:10]).all())
        self.assertTrue(lin.lin.weight.shape[0] == 25)
        self.assertTrue(lin.lin.bias.shape[0] == 25)

    def test_linsoftmax_resize_remove(self):
        """
        Tests resizing of a fully connected layer.
        """
        lin = layers.LinSoftmax(20, 10)
        w_cp = lin.lin.weight.clone()
        b_cp = lin.lin.bias.clone()
        lin.resize(5, (1, 5, 6, 7, 9))
        self.assertTrue(w_cp[(0, 2, 3, 4, 8), :].eq(lin.lin.weight).all())
        self.assertTrue(b_cp[(0, 2, 3, 4, 8),].eq(lin.lin.bias).all())

    def test_linsoftmax_resize_both(self):
        """
        Tests resizing of a fully connected layer.
        """
        lin = layers.LinSoftmax(20, 10)
        w_cp = lin.lin.weight.clone()
        b_cp = lin.lin.bias.clone()
        lin.resize(25, (1, 5, 6, 7, 9))
        self.assertTrue(w_cp[(0, 2, 3, 4, 8), :].eq(lin.lin.weight[:5, :]).all())
        self.assertTrue(b_cp[(0, 2, 3, 4, 8),].eq(lin.lin.bias[:5]).all())
        self.assertTrue(lin.lin.weight.shape[0] == 25)
        self.assertTrue(lin.lin.bias.shape[0] == 25)

    def test_conv_resize_add(self):
        """
        Tests resizing of a convolutional output layer.
        """
        conv = layers.ActConv2D(20, 10, (1, 1), (1, 1))
        w_cp = conv.co.weight.clone()
        b_cp = conv.co.bias.clone()
        conv.resize(25)
        self.assertTrue(w_cp.eq(conv.co.weight[:10, :]).all())
        self.assertTrue(b_cp.eq(conv.co.bias[:10]).all())
        self.assertTrue(conv.co.weight.shape[0] == 25)
        self.assertTrue(conv.co.bias.shape[0] == 25)

    def test_conv_resize_remove(self):
        """
        Tests resizing of a convolutional output layer.
        """
        conv = layers.ActConv2D(20, 10, (1, 1), (1, 1))
        w_cp = conv.co.weight.clone()
        b_cp = conv.co.bias.clone()
        conv.resize(5, (1, 5, 6, 7, 9))
        self.assertTrue(w_cp[(0, 2, 3, 4, 8), :].eq(conv.co.weight).all())
        self.assertTrue(b_cp[(0, 2, 3, 4, 8),].eq(conv.co.bias).all())

    def test_conv_resize_both(self):
        """
        Tests resizing of a convolutional output layer.
        """
        conv = layers.ActConv2D(20, 10, (1, 1), (1, 1))
        w_cp = conv.co.weight.clone()
        b_cp = conv.co.bias.clone()
        conv.resize(25, (1, 5, 6, 7, 9))
        self.assertTrue(w_cp[(0, 2, 3, 4, 8), :].eq(conv.co.weight[:5, :]).all())
        self.assertTrue(b_cp[(0, 2, 3, 4, 8),].eq(conv.co.bias[:5]).all())
        self.assertTrue(conv.co.weight.shape[0] == 25)
        self.assertTrue(conv.co.bias.shape[0] == 25)
