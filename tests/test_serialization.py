# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import unittest
import json
import os

from lxml import etree
from io import StringIO
from kraken import rpred
from kraken import serialization

thisfile = os.path.abspath(os.path.dirname(__file__))
resources = os.path.abspath(os.path.join(thisfile, 'resources'))

class TestSerializations(object):
    """
    Tests for output serialization
    """
    def setUp(self):
        with open(os.path.join(resources, 'records.json'), 'rb') as fp:
            self.records = [rpred.ocr_record(**x) for x in json.load(fp)]

    def test_hocr_serialization(self):
        """
        Test hOCR serialization
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.records, image_name='foo.png', template='hocr'))

        doc = etree.fromstring(fp.getvalue())

    def test_alto_serialization_validation(self):
        """
        Validates output against ALTO schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.records, image_name='foo.png', template='alto'))
        doc = etree.fromstring(fp.getvalue())
        print(fp.getvalue()[:2000])
        with open(os.path.join(resources, 'alto-3-1.xsd')) as schema_fp:
            alto_schema = etree.XMLSchema(etree.parse(schema_fp))
            alto_schema.assertValid(doc)
