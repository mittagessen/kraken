# -*- coding: utf-8 -*-
from nose.tools import raises 

from kraken import rpred

class TestRecognition(object):

    """
    Tests of the recognition facility and associated routines.
    """

    def setUp(self):
        self.temp = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self):
        self.temp.close()
        os.unlink(self.temp.name)
