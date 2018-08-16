# -*- coding: utf-8 -*-
import unittest

from nose.tools import raises

from kraken.lib import train
from itertools import cycle

class TestTrain(unittest.TestCase):
    """
    Testing model trainer interrupter classes
    """
    def test_early_stopping(self):
        """
        Tests early stopping interrupter.
        """
        it = train.EarlyStopping(cycle('a'), min_delta = 1, lag = 5)
        for epoch, _ in enumerate(it):
            it.update(epoch if epoch < 10 else 10)
        self.assertEqual(15, epoch)
        self.assertEqual(it.best_epoch, 11)
        self.assertEqual(it.best_loss, 10)

    def test_epoch_stopping(self):
        """
        Tests stopping after n epochs.
        """
        it = train.EpochStopping(cycle('a'), epochs = 57)
        for epoch, _ in enumerate(it):
            it.update(epoch)
        self.assertEqual(56, epoch)
        self.assertEqual(it.best_epoch, 56)
        self.assertEqual(it.best_loss, 56)
