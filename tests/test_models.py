# -*- coding: utf-8 -*-
import os
import tempfile
import pickle
import h5py

from nose.tools import raises 

from kraken.lib import models
from kraken.lib.exceptions import KrakenInvalidModelException

class TestModels(object):

    """
    Testing model loading routines
    """

    def setUp(self):
        self.temp = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self):
        self.temp.close()
        os.unlink(self.temp.name)

    @raises(KrakenInvalidModelException)
    def test_load_hdf5_unknown(self):
        """
        Test correct handling of unknown HDF5 files.
        """
        b = h5py.File(self.temp.name, 'w')
        b.attrs['kind'] = 'not-an-rnn'
        b.close()
        models.load_hdf5(self.temp.name)

    @raises(KrakenInvalidModelException)
    def test_load_hdf5_incomplete(self):
        """
        Test correct handling of incomplete HDF5 files.
        """
        b = h5py.File(self.temp.name, 'w')
        b.attrs['kind'] = 'pyrnn-bidi'
        b.close()
        models.load_hdf5(self.temp.name)

    @raises(IOError)
    def test_load_hdf5_invalid(self):
        """
        Test correct handling of invalid HDF5 files.
        """
        models.load_hdf5(self.temp.name)
       
    @raises(KrakenInvalidModelException)
    def test_load_pyrnn_invalid(self):
        """
        Test correct handling of non-pickle files.
        """
        b = h5py.File(self.temp.name, 'w')
        b.attrs['kind'] = 'pyrnn-bidi'
        b.close()
        models.load_pyrnn(self.temp.name)

    @raises(KrakenInvalidModelException)
    def test_load_pyrnn_no_seqrecognizer(self):
        """
        Test correct handling of non-SeqRecognizer pickles.
        """
        pickle.dump(u'Iámnõtãrécðçnízer', self.temp)
        self.temp.close()
        models.load_pyrnn(self.temp.name)

    def test_load_pyrnn_gz(self):
        """
        Test correct handling of gzipped models.
        """

    def test_load_pyrnn_bz2(self):
        """
        Test correct handling of bzip2 compressed models.
        """
        pass

    def test_load_pyrnn_uncompressed(self):
        """
        Test correct handling of uncompressed models.
        """
        pass

    def test_load_pyrnn_aliasing(self):
        """
        Test correct aliasing of ocrolib classes.
        """
        pass

    def test_load_pyrnn_aliasing_old(self):
        """
        Test correct aliasing of pre-ocrolib classes.
        """
        pass
