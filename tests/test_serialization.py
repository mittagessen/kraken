# -*- coding: utf-8 -*-
import json
import unittest
import tempfile
import numpy as np

from lxml import etree
from io import StringIO
from pathlib import Path
from hocr_spec import HocrValidator
from collections import Counter

from kraken import rpred, serialization
from kraken.lib import xml

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

def roundtrip(self, records, fp):
    """
    Checks that the order of lines after serialization and deserialization is
    equal to the records.
    """
    with tempfile.NamedTemporaryFile() as out:
        fp.seek(0)
        out.write(fp.getvalue().encode('utf-8'))
        doc = xml.parse_xml(out.name)['lines']
        for orig_line, parsed_line in zip(records, doc):
            self.assertSequenceEqual(np.array(orig_line.baseline).tolist(),
                                     np.array(parsed_line['baseline']).tolist(),
                                     msg='Baselines differ after serialization.')

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
    if len(counts):
        self.assertEqual(counts.most_common(1)[0][1], 1, msg='Duplicate IDs in PageXML output')

    with open(resources / 'pagecontent.xsd') as schema_fp:
        page_schema = etree.XMLSchema(etree.parse(schema_fp))
        page_schema.assertValid(doc)

def validate_alto(self, fp):
    doc = etree.fromstring(fp.getvalue().encode('utf-8'))

    ids = [x.get('ID') for x in doc.findall('.//*[@ID]')]
    counts = Counter(ids)
    self.assertEqual(counts.most_common(1)[0][1], 1, msg='Duplicate IDs in ALTO output')

    with open(resources / 'alto-4-3.xsd') as schema_fp:
        alto_schema = etree.XMLSchema(etree.parse(schema_fp))
        alto_schema.assertValid(doc)


class TestSerializations(unittest.TestCase):
    """
    Tests for output serialization
    """
    def setUp(self):
        with open(resources /'records.json', 'r') as fp:
            self.box_records = [rpred.BBoxOCRRecord(**x) for x in json.load(fp)]

        with open(resources / 'bl_records.json', 'r') as fp:
            recs = json.load(fp)
            self.bl_records = [rpred.BaselineOCRRecord(**bl) for bl in recs['lines']]
            self.bl_regions = recs['regions']

        self.metadata_steps = [{'category': 'preprocessing', 'description': 'PDF image extraction', 'settings': {}},
                               {'category': 'processing',
                                'description': 'Baseline and region segmentation',
                                'settings': {'model': 'foo.mlmodel', 'text_direction': 'horizontal-lr'}},
                               {'category': 'processing',
                                'description': 'Text line recognition',
                                'settings': {'text_direction': 'horizontal-lr',
                                             'models': 'bar.mlmodel',
                                             'pad': 16,
                                             'bidi_reordering': True}}]


    def test_box_vertical_hocr_serialization(self):
        """
        Test vertical line hOCR serialization
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.box_records, image_name='foo.png', writing_mode='vertical-lr', template='hocr'))
        validate_hocr(self, fp)

    def test_box_hocr_serialization(self):
        """
        Test hOCR serialization
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.box_records, image_name='foo.png', template='hocr'))
        validate_hocr(self, fp)

    def test_box_alto_serialization_validation(self):
        """
        Validates output against ALTO schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.box_records, image_name='foo.png', template='alto'))
        validate_alto(self, fp)

    def test_box_abbyyxml_serialization_validation(self):
        """
        Validates output against abbyyXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.box_records, image_name='foo.png', template='abbyyxml'))
        doc = etree.fromstring(fp.getvalue().encode('utf-8'))
        with open(resources / 'FineReader10-schema-v1.xml') as schema_fp:
            abbyy_schema = etree.XMLSchema(etree.parse(schema_fp))
            abbyy_schema.assertValid(doc)

    def test_box_pagexml_serialization_validation(self):
        """
        Validates output against abbyyXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.box_records, image_name='foo.png', template='pagexml'))
        validate_page(self, fp)

    def test_bl_alto_serialization_validation(self):
        """
        Validates output against ALTO schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.bl_records, image_name='foo.png', template='alto'))
        validate_alto(self, fp)
        roundtrip(self, self.bl_records, fp)

    def test_bl_abbyyxml_serialization_validation(self):
        """
        Validates output against abbyyXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.bl_records, image_name='foo.png', template='abbyyxml'))
        doc = etree.fromstring(fp.getvalue().encode('utf-8'))
        with open(resources / 'FineReader10-schema-v1.xml') as schema_fp:
            abbyy_schema = etree.XMLSchema(etree.parse(schema_fp))
            abbyy_schema.assertValid(doc)

    def test_bl_pagexml_serialization_validation(self):
        """
        Validates output against PageXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.bl_records, image_name='foo.png', template='pagexml'))
        validate_page(self, fp)
        roundtrip(self, self.bl_records, fp)

    def test_bl_region_alto_serialization_validation(self):
        """
        Validates output against ALTO schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.bl_records, image_name='foo.png', template='alto', regions=self.bl_regions))
        validate_alto(self, fp)
        roundtrip(self, self.bl_records, fp)

    def test_bl_region_abbyyxml_serialization_validation(self):
        """
        Validates output against abbyyXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.bl_records, image_name='foo.png', template='abbyyxml', regions=self.bl_regions))
        doc = etree.fromstring(fp.getvalue().encode('utf-8'))
        with open(resources / 'FineReader10-schema-v1.xml') as schema_fp:
            abbyy_schema = etree.XMLSchema(etree.parse(schema_fp))
            abbyy_schema.assertValid(doc)

    def test_bl_region_pagexml_serialization_validation(self):
        """
        Validates output against PageXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.bl_records, image_name='foo.png', template='pagexml', regions=self.bl_regions))
        validate_page(self, fp)
        roundtrip(self, self.bl_records, fp)

    def test_region_only_alto_serialization_validation(self):
        """
        Validates output without baselines (but regions) against ALTO schema
        """
        fp = StringIO()

        fp.write(serialization.serialize([], image_name='foo.png', template='alto', regions=self.bl_regions))
        validate_alto(self, fp)

    def test_region_only_abbyyxml_serialization_validation(self):
        """
        Validates output without baselines (but regions) against abbyyXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize([], image_name='foo.png', template='abbyyxml', regions=self.bl_regions))
        doc = etree.fromstring(fp.getvalue().encode('utf-8'))
        with open(resources / 'FineReader10-schema-v1.xml') as schema_fp:
            abbyy_schema = etree.XMLSchema(etree.parse(schema_fp))
            abbyy_schema.assertValid(doc)

    def test_region_only_pagexml_serialization_validation(self):
        """
        Validates output without baselines (but regions) against PageXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize([], image_name='foo.png', template='pagexml', regions=self.bl_regions))
        validate_page(self, fp)

    def test_serialize_segmentation_alto(self):
        """
        Validates output of `serialize_segmentation` against ALTO schema
        """
        fp = StringIO()

        fp.write(serialization.serialize_segmentation({'boxes': []}, image_name='foo.png', template='alto'))
        validate_alto(self, fp)

    def test_serialize_segmentation_pagexml(self):
        """
        Validates output of `serialize_segmentation` against ALTO schema
        """
        fp = StringIO()

        fp.write(serialization.serialize_segmentation({'boxes': []}, image_name='foo.png', template='pagexml'))
        validate_page(self, fp)

    def test_serialize_segmentation_alto_steps(self):
        """
        Validates output of `serialize_segmentation` with processing steps against ALTO schema
        """
        fp = StringIO()

        fp.write(serialization.serialize_segmentation({'boxes': []}, image_name='foo.png', template='alto', processing_steps=self.metadata_steps))
        validate_alto(self, fp)

    def test_serialize_segmentation_pagexml(self):
        """
        Validates output of `serialize_segmentation` with processing steps against PageXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize_segmentation({'boxes': []}, image_name='foo.png', template='pagexml', processing_steps=self.metadata_steps))
        validate_page(self, fp)

    def test_bl_region_alto_serialization_validation_steps(self):
        """
        Validates output with processing steps against ALTO schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.bl_records, image_name='foo.png', template='alto', regions=self.bl_regions, processing_steps=self.metadata_steps))
        validate_alto(self, fp)
        roundtrip(self, self.bl_records, fp)

    def test_bl_region_abbyyxml_serialization_validation_steps(self):
        """
        Validates output with processing steps against abbyyXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.bl_records, image_name='foo.png', template='abbyyxml', regions=self.bl_regions, processing_steps=self.metadata_steps))
        doc = etree.fromstring(fp.getvalue().encode('utf-8'))
        with open(resources / 'FineReader10-schema-v1.xml') as schema_fp:
            abbyy_schema = etree.XMLSchema(etree.parse(schema_fp))
            abbyy_schema.assertValid(doc)

    def test_bl_region_pagexml_serialization_validation_steps(self):
        """
        Validates output with processing steps against PageXML schema
        """
        fp = StringIO()

        fp.write(serialization.serialize(self.bl_records, image_name='foo.png', template='pagexml', regions=self.bl_regions, processing_steps=self.metadata_steps))
        validate_page(self, fp)
        roundtrip(self, self.bl_records, fp)

