# -*- coding: utf-8 -*-
import shutil
import tempfile
import unittest
from pathlib import Path

from kraken import repo

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


class TestRepo(unittest.TestCase):
    """
    Testing our wrappers around HTRMoPo
    """

    def setUp(self):
        self.temp_model = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_model.name)

    def tearDown(self):
        shutil.rmtree(self.temp_model.name)

    def test_listing(self):
        """
        Tests fetching the model list.
        """
        records = repo.get_listing()
        self.assertGreater(len(records), 15)

    def test_get_description(self):
        """
        Tests fetching the description of a model.
        """
        record = repo.get_description('10.5281/zenodo.8425684')
        self.assertEqual(record.doi, '10.5281/zenodo.8425684')

    def test_prev_record_version_get_description(self):
        """
        Tests fetching the description of a model that has a superseding newer version.
        """
        record = repo.get_description('10.5281/zenodo.6657809')
        self.assertEqual(record.doi, '10.5281/zenodo.6657809')
