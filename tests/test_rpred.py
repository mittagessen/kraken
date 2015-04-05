# -*- coding: utf-8 -*-
import os
import tempfile
import pickle

from nose.tools import raises 

from kraken import rpred
from kraken.lib.exceptions import KrakenInvalidModelException

class TestRecognition(object):

    """
    Tests of the recognition facility and associated routines.
    """

    def setUp(self):
        self.temp = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self):
        self.temp.close()
        os.unlink(self.temp.name)

    @raises(KrakenInvalidModelException)
    def test_load_rnn_invalid(self):
        """
        Test correct handling of non-pickle files.
        """
        self.temp.write('agfwilfdq')
        self.temp.close()
        rpred.load_rnn(self.temp.name)

    @raises(KrakenInvalidModelException)
    def test_load_rnn_no_seqrecognizer(self):
        """
        Test correct handling of non-SeqRecognizer pickles.
        """
        pickle.dump({}, self.temp)
        self.temp.close()
        rpred.load_rnn(self.temp.name)

    def test_load_rnn_gz(self):
        """
        Test correct handling of gzipped models.
        """
        pass

    def test_load_rnn_bz2(self):
        """
        Test correct handling of bzip2 compressed models.
        """
        pass

    def test_load_rnn_uncompressed(self):
        """
        Test correct handling of uncompressed models.
        """
        pass

    def test_load_rnn_aliasing(self):
        """
        Test correct aliasing of ocrolib classes.
        """
        pass

    def test_load_rnn_aliasing_old(self):
        """
        Test correct aliasing of pre-ocrolib classes.
        """
        pass
