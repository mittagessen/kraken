# -*- coding: utf-8 -*-
import unittest

import torch

from kraken.lib.vgsl import layers


class TestLayers(unittest.TestCase):
    """
    Testing VGSL custom layer implementations.
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

    def test_rnn_layer(self):
        """
        Test RNN layer output shapes over direction, axis, and summarization.
        """
        cases = [('f', False, False, (1, 2, 32, 64)),
                 ('f', True, False, (1, 2, 32, 64)),
                 ('f', False, True, (1, 2, 32, 1)),
                 ('f', True, True, (1, 2, 1, 64)),
                 ('b', False, False, (1, 4, 32, 64)),
                 ('b', True, False, (1, 4, 32, 64)),
                 ('b', False, True, (1, 4, 32, 1)),
                 ('b', True, True, (1, 4, 1, 64))]
        for direction, transpose, summarize, expected in cases:
            with self.subTest(direction=direction, transpose=transpose, summarize=summarize):
                rnn = layers.TransposedSummarizingRNN(10, 2, direction, transpose, summarize)
                o = rnn(torch.randn(1, 10, 32, 64))
                self.assertEqual(o[0].shape, expected)

    def test_linsoftmax(self):
        """
        Test basic function of linear layer.
        """
        lin = layers.LinSoftmax(20, 10)
        o = lin(torch.randn(1, 20, 12, 24))
        self.assertEqual(o[0].shape, (1, 10, 12, 24))

    def test_linsoftmax_aug(self):
        """
        Test basic function of linear layer with 1-augmentation.
        """
        lin = layers.LinSoftmax(20, 10, True)
        o = lin(torch.randn(1, 20, 12, 24))
        self.assertEqual(o[0].shape, (1, 10, 12, 24))

    def test_actconv2d_lin(self):
        """
        Test convolutional layer without activation.
        """
        conv = layers.ActConv2D(5, 12, (3, 3), (1, 1), 'l')
        o = conv(torch.randn(1, 5, 24, 12))
        self.assertEqual(o[0].shape, (1, 12, 24, 12))

    def test_actconv2d_sigmoid_logits(self):
        """
        Sigmoid output layers return unsquashed logits, identically in train
        and eval mode. Weights are fixed so a squashed output would land in
        [0, 1] while logits are exactly the bias value.
        """
        conv = layers.ActConv2D(5, 12, (3, 3), (1, 1), 's')
        torch.nn.init.zeros_(conv.co.weight)
        torch.nn.init.constant_(conv.co.bias, 10.)
        x = torch.randn(1, 5, 24, 12)
        conv.train()
        train_o = conv(x)[0]
        conv.eval()
        eval_o = conv(x)[0]
        self.assertTrue(torch.allclose(train_o, eval_o))
        self.assertGreater(train_o.min().item(), 1)

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

    def test_output_layer_resize(self):
        """
        Tests resizing of fully connected and convolutional output layers
        with pure growth, pure deletion, and both combined.
        """
        kept = (0, 2, 3, 4, 8)
        deleted = (1, 5, 6, 7, 9)
        factories = [('linsoftmax', lambda: layers.LinSoftmax(20, 10), 'lin'),
                     ('conv', lambda: layers.ActConv2D(20, 10, (1, 1), (1, 1)), 'co')]
        scenarios = [('add', 25, None),
                     ('remove', 5, deleted),
                     ('both', 25, deleted)]
        for layer_name, factory, attr in factories:
            for scenario, new_size, del_indices in scenarios:
                with self.subTest(layer=layer_name, scenario=scenario):
                    layer = factory()
                    mod = getattr(layer, attr)
                    w_cp = mod.weight.clone()
                    b_cp = mod.bias.clone()
                    if del_indices is None:
                        layer.resize(new_size)
                        self.assertTrue(w_cp.eq(mod.weight[:10, :]).all())
                        self.assertTrue(b_cp.eq(mod.bias[:10]).all())
                    else:
                        layer.resize(new_size, del_indices)
                        self.assertTrue(w_cp[kept, :].eq(mod.weight[:len(kept), :]).all())
                        self.assertTrue(b_cp[kept,].eq(mod.bias[:len(kept)]).all())
                    self.assertEqual(mod.weight.shape[0], new_size)
                    self.assertEqual(mod.bias.shape[0], new_size)
