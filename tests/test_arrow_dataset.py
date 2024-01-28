# -*- coding: utf-8 -*-

import json
import unittest
from pathlib import Path

from pytest import raises

import kraken
from kraken.lib import xml
from kraken.lib.arrow_dataset import build_binary_dataset

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestKrakenArrowCompilation(unittest.TestCase):
    """
    Tests for binary datasets
    """
    def setUp(self):
        self.xml = resources / '170025120000003,0074.xml'
        self.bls = xml.XMLPage(self.xml)
        self.box_lines = [resources / '000236.png']

    def test_build_path_dataset(self):
        pass

    def test_build_xml_dataset(self):
        pass

    def test_build_obj_dataset(self):
        pass

    def test_build_empty_dataset(self):
        pass
