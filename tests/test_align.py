# -*- coding: utf-8 -*-

import unittest
from pathlib import Path

from kraken.align import forced_align
from kraken.lib import xml

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestKrakenAlign(unittest.TestCase):
    """
    Tests for the forced alignment module
    """
    def setUp(self):
        self.doc = resources / '170025120000003,0074.xml'
        self.bls = xml.XMLPage(self.doc).to_container()

    def test_forced_align_simple(self):
        """
        Simple alignment test.
        """
        pass
