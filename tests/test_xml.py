# -*- coding: utf-8 -*-
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pytest import raises

from kraken.lib import xml
from kraken.containers import BaselineLine, BBoxLine

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestXMLParser(unittest.TestCase):
    """
    Tests XML (ALTO/PAGE) parsing
    """
    def setUp(self):
        self.alto_doc_root = resources / 'alto'
        self.page_doc_root = resources / 'page'
        self.alto_doc = self.alto_doc_root / 'bsb00084914_00007.xml'
        self.page_doc = self.page_doc_root / 'cPAS-2000.xml'
        self.reg_alto_doc = self.alto_doc_root / 'reg_test.xml'

        self.invalid_alto_docs = self.alto_doc_root / 'invalid'
        self.invalid_page_docs = self.page_doc_root / 'invalid'

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

    def test_failure_alto_invalid_image(self):
        """
        Test that parsing aborts if image file path is invalid.
        """
        with raises(ValueError):
            xml.XMLPage(self.invalid_alto_docs / 'image.xml')

    def test_failure_alto_invalid_measurement_unit(self):
        """
        Test that parsing aborts if measurement unit isn't "pixel"
        """
        with raises(ValueError):
            xml.XMLPage(self.invalid_alto_docs / 'mu.xml')

    def test_failure_alto_invalid_dims(self):
        """
        Test that parsing aborts if page dimensions aren't parseable as ints.
        """
        with raises(ValueError):
            xml.XMLPage(self.invalid_alto_docs / 'dims.xml')

    def test_alto_basedirection(self):
        """
        Test proper handling of base direction attribute, including inheritance
        from regions.
        """
        seg = xml.XMLPage(self.alto_doc).to_container()
        base_dirs = [x.base_dir for x in seg.lines]
        self.assertEqual(base_dirs, ['L', 'L', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', None, None, None, None, 'R'])

    def test_alto_language_parsing(self):
        """
        Test proper handling of language attribute, including inheritance from
        page and region.
        """
        seg = xml.XMLPage(self.alto_doc).to_container()
        languages = [x.language for x in seg.lines]
        self.assertEqual(languages, ['iai', ['deu', 'heb'], ['deu', 'heb'],
                                    ['deu', 'heb'], ['deu', 'heb'], ['deu',
                                    'heb'], ['deu', 'heb'], ['deu', 'heb'],
                                    ['deu', 'heb'], ['deu', 'heb'], ['deu',
                                    'heb'], ['deu', 'heb'], ['deu', 'heb'],
                                    ['deu', 'heb'], ['deu', 'heb'], ['deu',
                                    'heb'], ['deu', 'heb'], ['deu', 'heb'],
                                    ['deu', 'heb'], ['deu', 'heb'], ['deu',
                                    'heb'], ['deu', 'heb'], ['deu', 'heb'],
                                    'eng', ['deu', 'heb'], ['hbo'], ['hbo'],
                                    ['hbo'], ['hbo'], ['hbo']])
        reg_langs = [x.language for x in seg.regions['text']]
        self.assertEqual(reg_langs, [['deu', 'heb'], ['hbo'], ['hbo'], ['hbo'], ['hbo']])

    def test_alto_fallback_region_boundaries(self):
        """
        Test region boundary parsing hierarchy shape -> rect -> None.
        """
        doc = xml.XMLPage(self.reg_alto_doc)
        self.assertEqual(set(doc.regions.keys()), set(['text']))
        for reg, boundary in zip(doc.regions['text'], [[(812, 606), (2755, 648), (2723, 3192), (808, 3240)],
                                                       [(596, 2850), (596, 3008), (729, 3008), (729, 2850)],
                                                       None]):
            self.assertEqual(reg.boundary, boundary)

    def test_alto_tag_parsing(self):
        """
        Test correct parsing of tag references.
        """
        seg = xml.XMLPage(self.alto_doc).to_container()
        line_tags = [line.tags for line in seg.lines]
        self.assertEqual(line_tags, [{'type': 'default'}, {'type': 'default'},
                                     {'type': 'heading'}, {'type': 'default'},
                                     {'type': 'default'}, {'type': 'default'},
                                     {'type': 'default'}, {'type': 'default'},
                                     {'type': 'default'}, {'type': 'default'},
                                     {'type': 'default'}, {'label_0': 'foo', 'label_1': 'bar', 'type': 'default'},
                                     {'label_1': ['bar', 'baz'], 'type': 'default'},
                                     {'type': 'default'}, {'type': 'default'},
                                     {'type': 'default'}, {'type': 'default'},
                                     {'type': 'default'}, {'type': 'default'},
                                     {'type': 'default'}, {'type': 'default'},
                                     {'type': 'default'}, {'type': 'default'},
                                     {'type': 'default'}, {'type': 'default'},
                                     {'type': 'default'}, {'type': 'default'},
                                     {'type': 'default'}, {'type': 'default'},
                                     {'type': 'default'}])


    def test_alto_baseline_linetype(self):
        """
        Test parsing with baseline line objects.
        """
        seg = xml.XMLPage(self.alto_doc, linetype='baselines').to_container()
        self.assertEqual(len(seg.lines), 30)
        for line in seg.lines:
            self.assertIsInstance(line, BaselineLine)

    def test_alto_bbox_linetype(self):
        """
        Test parsing with bbox line objects.
        """
        seg = xml.XMLPage(self.alto_doc, linetype='bbox').to_container()
        self.assertEqual(len(seg.lines), 31)
        for line in seg.lines:
            self.assertIsInstance(line, BBoxLine)

    def test_failure_page_invalid_image(self):
        """
        Test that parsing aborts if image file path is invalid.
        """
        with raises(ValueError):
            xml.XMLPage(self.invalid_page_docs / 'image.xml')

    def test_failure_page_invalid_dims(self):
        """
        Test that parsing aborts if page dimensions aren't parseable as ints.
        """
        with raises(ValueError):
            xml.XMLPage(self.invalid_page_docs / 'dims.xml')

    def test_page_basedirection(self):
        """
        Test proper handling of base direction attribute, including inheritance
        from regions.
        """
        seg = xml.XMLPage(self.page_doc).to_container()
        base_dirs = [x.base_dir for x in seg.lines]
        self.assertEqual(base_dirs, ['R', 'L', 'L', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                     'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                     'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                     'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                     'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                     'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                     'L'])

    def test_page_language_parsing(self):
        """
        Test proper handling of language attribute, custom string and
        inheritance from page and region.
        """
        seg = xml.XMLPage(self.page_doc).to_container()
        languages = [x.language for x in seg.lines]
        self.assertEqual(languages, [['hbo'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['deu'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu']])
        reg_langs = [x.language for x in seg.regions['text']]
        self.assertEqual(reg_langs, [['hbo'], ['heb', 'deu', 'eng'], ['pol', 'deu']])

