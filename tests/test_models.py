# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from pathlib import Path

from pytest import raises

from kraken.lib import models
from kraken.lib.exceptions import KrakenInvalidModelException

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


class TestModels(unittest.TestCase):
    """
    Testing legacy model loading routines (`kraken.lib.models.load_any`).

    .. deprecated::
        These tests exercise the deprecated `kraken.lib.models.load_any` API
        which will be removed with kraken 8. New code should use
        `kraken.models.load_models` instead. See `test_plugins.py` for tests
        of the replacement loading API.
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
