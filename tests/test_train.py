# -*- coding: utf-8 -*-
import unittest

from nose.tools import raises

from kraken.lib.exceptions import KrakenStopTrainingException
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
        it = train.EarlyStopping(min_delta = 1, lag = 5)
        for iteration in range(16):
            self.assertTrue(it.trigger())
            it.update(iteration if iteration < 10 else 10)
        self.assertFalse(it.trigger())
        self.assertEqual(it.best_epoch, 11) # epochs are 1-indexed now
        self.assertEqual(it.best_loss, 10)

    def test_epoch_stopping(self):
        """
        Tests stopping after n epochs.
        """
        it = train.EpochStopping(epochs=57)
        for iteration in range(57):
            self.assertTrue(it.trigger())
            it.update(iteration)
        self.assertFalse(it.trigger())
        self.assertEqual(it.best_epoch, 56)
        self.assertEqual(it.best_loss, 56)
