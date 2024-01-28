# -*- coding: utf-8 -*-
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pytest import raises

from kraken.lib import xml

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestXMLParser(unittest.TestCase):
    """
    Tests XML (ALTO/PAGE) parsing
    """
    def setUp(self):
        self.page_doc = resources / 'cPAS-2000.xml'
        self.alto_doc = resources / 'bsb00084914_00007.xml'

    def test_page_parsing(self):
        """
        Test parsing of PAGE XML files with reading order.
        """
        doc = xml.XMLPage(self.page_doc, filetype='page')
        self.assertEqual(len(doc.get_sorted_lines()), 97)
        self.assertEqual(len([item for x in doc.regions.values() for item in x]), 4)

    def test_alto_parsing(self):
        """
        Test parsing of ALTO XML files with reading order.
        """
        doc = xml.XMLPage(self.alto_doc, filetype='alto')

    def test_auto_parsing(self):
        """
        Test parsing of PAGE and ALTO XML files with auto-format determination.
        """
        doc = xml.XMLPage(self.page_doc, filetype='xml')
        self.assertEqual(doc.filetype, 'page')
        doc = xml.XMLPage(self.alto_doc, filetype='xml')
        self.assertEqual(doc.filetype, 'alto')

    def test_failure_page_alto_parsing(self):
        """
        Test that parsing ALTO files with PAGE as format fails.
        """
        with raises(ValueError):
            xml.XMLPage(self.alto_doc, filetype='page')

    def test_failure_alto_page_parsing(self):
        """
        Test that parsing PAGE files with ALTO as format fails.
        """
        with raises(ValueError):
            xml.XMLPage(self.page_doc, filetype='alto')

