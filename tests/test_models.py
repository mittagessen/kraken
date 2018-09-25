# -*- coding: utf-8 -*-
import unittest
import os
import tempfile
import pickle

from nose.tools import raises

import kraken.lib.lstm

from kraken.lib import models
from kraken.lib.exceptions import KrakenInvalidModelException

thisfile = os.path.abspath(os.path.dirname(__file__))
resources = os.path.abspath(os.path.join(thisfile, 'resources'))

class TestModels(unittest.TestCase):
    """
    Testing model loading routines
    """

    def setUp(self):
        self.temp = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self):
        self.temp.close()
        os.unlink(self.temp.name)

    @raises(KrakenInvalidModelException)
    def test_load_invalid(self):
        """
        Tests correct handling of invalid files.
        """
        models.load_any(self.temp.name)

    def test_load_clstm(self):
        """
        Tests loading of valid clstm files.
        """
        rnn = models.load_any(os.path.join(resources, 'toy.clstm').encode('utf-8'))
        self.assertIsInstance(rnn, models.TorchSeqRecognizer)

    @raises(KrakenInvalidModelException)
    def test_load_pyrnn_no_seqrecognizer(self):
        """
        Test correct handling of non-SeqRecognizer pickles.
        """
        pickle.dump(u'Iámnõtãrécðçnízer', self.temp)
        self.temp.close()
        models.load_any(self.temp.name)

    @raises(KrakenInvalidModelException)
    def test_load_any_pyrnn_py3(self):
        """
        Test load_any doesn't load pickled models on python 3
        """
        rnn = models.load_any(os.path.join(resources, 'model.pyrnn.gz'))

    def test_load_any_proto(self):
        """
        Test load_any loads protobuf models.
        """
        rnn = models.load_any(os.path.join(resources, 'model.pronn'))
        self.assertIsInstance(rnn, kraken.lib.models.TorchSeqRecognizer)
