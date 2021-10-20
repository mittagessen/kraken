# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import unittest
import json
import os

from lxml import etree
from io import StringIO
from hocr_spec import HocrValidator
from collections import Counter

from kraken import rpred
from kraken import serialization

thisfile = os.path.abspath(os.path.dirname(__file__))
resources = os.path.abspath(os.path.join(thisfile, 'resources'))

def validate_hocr(self, fp):
    fp.seek(0)

    validator = HocrValidator('standard')
    report = validator.validate(fp, parse_strict=True)
    self.assertTrue(report.is_valid())

    doc = etree.fromstring(fp.getvalue().encode('utf-8'))

    ids = [x.get('id') for x in doc.findall('.//*[@id]')]
    counts = Counter(ids)
    self.assertEqual(counts.most_common(1)[0][1], 1, msg='Duplicate IDs in hOCR output')

def validate_page(self, fp):
    doc = etree.fromstring(fp.getvalue().encode('utf-8'))

    ids = [x.get('id') for x in doc.findall('.//*[@id]')]
    counts = Counter(ids)
    self.assertEqual(counts.most_common(1)[0][1], 1, msg='Duplicate IDs in PageXML output')

    with open(os.path.join(resources, 'pagecontent.xsd')) as schema_fp:
        page_schema = etree.XMLSchema(etree.parse(schema_fp))
        page_schema.assertValid(doc)

def validate_alto(self, fp):
    doc = etree.fromstring(fp.getvalue().encode('utf-8'))

    ids = [x.get('ID') for x in doc.findall('.//*[@ID]')]
    counts = Counter(ids)
    self.assertEqual(counts.most_common(1)[0][1], 1, msg='Duplicate IDs in ALTO output')

    with open(os.path.join(resources, 'alto-4-2.xsd')) as schema_fp:
        alto_schema = etree.XMLSchema(etree.parse(schema_fp))
        alto_schema.assertValid(doc)


class TestSerializations(unittest.TestCase):
    """
    Tests for output serialization
    """
    def setUp(self):
        with open(os.path.join(resources, 'records.json'), 'r') as fp:
            self.records = [rpred.ocr_record(**x) for x in json.load(fp)]

    def test_box_vertical_hocr_serialization(self):
        """
        Test vertical line hOCR serialization
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.records, image_name='foo.png', writing_mode='vertical-lr', template='hocr'))
        validate_hocr(self, fp)

    def test_box_hocr_serialization(self):
        """
        Test hOCR serialization
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.records, image_name='foo.png', template='hocr'))
        validate_hocr(self, fp)

    def test_box_alto_serialization_validation(self):
        """
        Validates output against ALTO schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.records, image_name='foo.png', template='alto'))
        validate_alto(self, fp)

    def test_box_abbyyxml_serialization_validation(self):
        """
        Validates output against abbyyXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.records, image_name='foo.png', template='abbyyxml'))
        doc = etree.fromstring(fp.getvalue().encode('utf-8'))
        with open(os.path.join(resources, 'FineReader10-schema-v1.xml')) as schema_fp:
            abbyy_schema = etree.XMLSchema(etree.parse(schema_fp))
            abbyy_schema.assertValid(doc)

    def test_box_pagexml_serialization_validation(self):
        """
        Validates output against abbyyXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.records, image_name='foo.png', template='pagexml'))
        validate_page(self, fp)
