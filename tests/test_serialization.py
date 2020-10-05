# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import unittest
import json
import os

from lxml import etree
from io import StringIO
from hocr_spec import HocrValidator

from kraken import rpred
from kraken import serialization

thisfile = os.path.abspath(os.path.dirname(__file__))
resources = os.path.abspath(os.path.join(thisfile, 'resources'))

class TestSerializations(unittest.TestCase):
    """
    Tests for output serialization
    """
    def setUp(self):
        with open(os.path.join(resources, 'records.json'), 'r') as fp:
            self.records = [rpred.ocr_record(**x) for x in json.load(fp)]
        self.validator = HocrValidator('standard')

    def test_vertical_hocr_serialization(self):
        """
        Test vertical line hOCR serialization
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.records, image_name='foo.png', writing_mode='vertical-lr', template='hocr'))
        fp.seek(0)

        report = self.validator.validate(fp, parse_strict=True)
        self.assertTrue(report.is_valid())

    def test_hocr_serialization(self):
        """
        Test hOCR serialization
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.records, image_name='foo.png', template='hocr'))
        fp.seek(0)

        report = self.validator.validate(fp, parse_strict=True)
        self.assertTrue(report.is_valid())

    def test_alto_serialization_validation(self):
        """
        Validates output against ALTO schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.records, image_name='foo.png', template='alto'))
        doc = etree.fromstring(fp.getvalue().encode('utf-8'))
        with open(os.path.join(resources, 'alto-4-2.xsd')) as schema_fp:
            alto_schema = etree.XMLSchema(etree.parse(schema_fp))
            alto_schema.assertValid(doc)

    def test_abbyyxml_serialization_validation(self):
        """
        Validates output against abbyyXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.records, image_name='foo.png', template='abbyyxml'))
        doc = etree.fromstring(fp.getvalue().encode('utf-8'))
        with open(os.path.join(resources, 'FineReader10-schema-v1.xml')) as schema_fp:
            abbyy_schema = etree.XMLSchema(etree.parse(schema_fp))
            abbyy_schema.assertValid(doc)

    def test_pagexml_serialization_validation(self):
        """
        Validates output against abbyyXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.records, image_name='foo.png', template='pagexml'))
        doc = etree.fromstring(fp.getvalue().encode('utf-8'))
        with open(os.path.join(resources, 'pagecontent.xsd')) as schema_fp:
            abbyy_schema = etree.XMLSchema(etree.parse(schema_fp))
            abbyy_schema.assertValid(doc)
