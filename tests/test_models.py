# -*- coding: utf-8 -*-
import unittest
import os
import tempfile
import pickle

from future.utils import PY2
from nose.tools import raises 

import kraken.lib.lstm

from kraken.lib import models
from kraken.lib import pyrnn_pb2
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
    def test_load_pronn_invalid(self):
        """
        Test correct handling of invalid files.
        """
        models.load_pronn(self.temp.name)
       
    @raises(KrakenInvalidModelException)
    @unittest.skipIf(not PY2, "not supported in this version")
    def test_load_pyrnn_invalid(self):
        """
        Test correct handling of non-pickle files.
        """
        self.temp.write('adfhewf')
        models.load_pyrnn(self.temp.name)

    @raises(KrakenInvalidModelException)
    def test_load_pyrnn_no_seqrecognizer(self):
        """
        Test correct handling of non-SeqRecognizer pickles.
        """
        pickle.dump(u'Iámnõtãrécðçnízer', self.temp)
        self.temp.close()
        models.load_pyrnn(self.temp.name)

    @raises(KrakenInvalidModelException)
    @unittest.skipIf(not PY2, "not supported in this version")
    def test_load_any_invalid(self):
        """
        Test load_any raises the proper exception if object is neither pickle
        nor protobuf.
        """
        models.load_any(self.temp.name)

    @unittest.skipIf(not PY2, "not supported in this version")
    def test_load_pyrnn_gz(self):
        """
        Test correct handling of gzipped models.
        """
        rnn = models.load_pyrnn(os.path.join(resources, 'model.pyrnn.gz'))
        self.assertIsInstance(rnn, kraken.lib.lstm.SeqRecognizer)

    @unittest.skipIf(not PY2, "not supported in this version")
    def test_load_pyrnn_uncompressed(self):
        """
        Test correct handling of uncompressed models.
        """
        rnn = models.load_pyrnn(os.path.join(resources, 'model.pyrnn'))
        self.assertIsInstance(rnn, kraken.lib.lstm.SeqRecognizer)

    @unittest.skipIf(not PY2, "not supported in this version")
    def test_load_pyrnn_aliasing_old(self):
        """
        Test correct aliasing of pre-ocrolib classes.
        """
        pass

    @unittest.skipIf(not PY2, "not supported in this version")
    def test_load_any_pyrnn(self):
        """
        Test load_any loads pickled models.
        """
        rnn = models.load_any(os.path.join(resources, 'model.pyrnn.gz'))
        self.assertIsInstance(rnn, kraken.lib.lstm.SeqRecognizer)

    def test_load_any_proto(self):
        """
        Test load_any loads protobuf models.
        """
        rnn = models.load_any(os.path.join(resources, 'model.pronn'))
        self.assertIsInstance(rnn, kraken.lib.lstm.SeqRecognizer)


