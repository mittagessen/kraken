# -*- coding: utf-8 -*-
import os
import pickle
import unittest
import tempfile

from pytest import raises
from pathlib import Path

import kraken.lib.lstm

from kraken.lib import models
from kraken.lib.exceptions import KrakenInvalidModelException

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestModels(unittest.TestCase):
    """
    Testing model loading routines
    """

    def setUp(self):
        self.temp = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self):
        self.temp.close()
        os.unlink(self.temp.name)

    def test_load_invalid(self):
        """
        Tests correct handling of invalid files.
        """
        with raises(KrakenInvalidModelException):
            models.load_any(self.temp.name)

    def test_load_clstm(self):
        """
        Tests loading of valid clstm files.
        """
        rnn = models.load_any(resources / 'toy.clstm')
        self.assertIsInstance(rnn, models.TorchSeqRecognizer)

    def test_load_pyrnn_no_seqrecognizer(self):
        """
        Test correct handling of non-SeqRecognizer pickles.
        """
        pickle.dump(u'Iámnõtãrécðçnízer', self.temp)
        self.temp.close()
        with raises(KrakenInvalidModelException):
            models.load_any(self.temp.name)

    def test_load_any_pyrnn_py3(self):
        """
        Test load_any doesn't load pickled models on python 3
        """
        with raises(KrakenInvalidModelException):
            rnn = models.load_any(resources / 'model.pyrnn.gz')

    def test_load_any_proto(self):
        """
        Test load_any loads protobuf models.
        """
        rnn = models.load_any(resources / 'model.pronn')
        self.assertIsInstance(rnn, kraken.lib.models.TorchSeqRecognizer)
