# -*- coding: utf-8 -*-
import socket
import unittest

import pytest

from kraken import repo


def _zenodo_reachable(timeout=2.0):
    try:
        socket.create_connection(('zenodo.org', 443), timeout=timeout).close()
        return True
    except OSError:
        return False


@pytest.mark.network
class TestRepo(unittest.TestCase):
    """
    Testing our wrappers around HTRMoPo. These tests query the live Zenodo
    repository.
    """

    @classmethod
    def setUpClass(cls):
        if not _zenodo_reachable():
            raise unittest.SkipTest('zenodo.org is not reachable')

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
