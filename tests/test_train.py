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
        it = train.EarlyStopping(cycle('a'), min_delta = 1, lag = 5)
        with self.assertRaises(KrakenStopTrainingException):
            for iteration, _ in enumerate(it):
                it.update(iteration if iteration < 10 else 10)
        self.assertEqual(15, iteration)
        self.assertEqual(it.best_iteration, 10)
        self.assertEqual(it.best_loss, 10)

    def test_epoch_stopping(self):
        """
        Tests stopping after n epochs.
        """
        it = train.EpochStopping(cycle('a'), iterations = 57)
        with self.assertRaises(KrakenStopTrainingException):
            for iteration, _ in enumerate(it):
                it.update(iteration)
        self.assertEqual(56, iteration)
        self.assertEqual(it.best_iteration, 56)
        self.assertEqual(it.best_loss, 56)
