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
