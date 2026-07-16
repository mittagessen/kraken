# -*- coding: utf-8 -*-
import copy
import tempfile
import unittest
import uuid
from collections import Counter
from io import StringIO
from pathlib import Path

import numpy as np
from hocr_spec import HocrValidator
from lxml import etree

from helpers import load_segmentation

from kraken import containers, serialization
from kraken.lib import xml

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

IMAGE_SIZE = (2000, 2000)


def roundtrip(self, records, fp):
    """
    Checks that the order of lines after serialization and deserialization is
    equal to the records.
    """
    with tempfile.NamedTemporaryFile() as out:
        fp.seek(0)
        out.write(fp.getvalue().encode('utf-8'))
        doc = xml.XMLPage(out.name).to_container().lines
        for orig_line, parsed_line in zip(records, doc):
            self.assertSequenceEqual(np.array(orig_line.baseline).tolist(),
                                     np.array(parsed_line.baseline).tolist(),
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
    duplicates = [i for i, c in counts.items() if c > 1]
    self.assertEqual(duplicates, [], msg='Duplicate IDs in PageXML output')

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


def validate_abbyy(self, fp):
    doc = etree.fromstring(fp.getvalue().encode('utf-8'))

    with open(resources / 'FineReader10-schema-v1.xml') as schema_fp:
        abbyy_schema = etree.XMLSchema(etree.parse(schema_fp))
        abbyy_schema.assertValid(doc)


class TestSerializations(unittest.TestCase):
    """
    Tests for output serialization
    """
    @classmethod
    def setUpClass(cls):
        box_seg = load_segmentation('box_rec.json')
        box_records = box_seg.lines

        bl_seg = load_segmentation('bl_rec.json')
        cls.bl_records = bl_seg.lines
        bl_regions = bl_seg.regions

        cls.box_segmentation = containers.Segmentation(type='bbox',
                                                       imagename='foo.png',
                                                       text_direction='horizontal-lr',
                                                       lines=box_records,
                                                       script_detection=True,
                                                       regions={})

        bl_records_no_regions = copy.deepcopy(cls.bl_records)
        for line in bl_records_no_regions:
            line.regions = []

        cls.bl_segmentation = containers.Segmentation(type='baselines',
                                                      imagename='foo.png',
                                                      text_direction='horizontal-lr',
                                                      lines=bl_records_no_regions,
                                                      script_detection=True,
                                                      regions={})

        cls.bl_segmentation_regs = containers.Segmentation(type='baselines',
                                                           imagename='foo.png',
                                                           text_direction='horizontal-lr',
                                                           lines=cls.bl_records,
                                                           script_detection=True,
                                                           regions=bl_regions)

        cls.bl_seg_nolines_regs = containers.Segmentation(type='baselines',
                                                          imagename='foo.png',
                                                          text_direction='horizontal-lr',
                                                          script_detection=False,
                                                          lines=[],
                                                          regions=bl_regions)

        no_boundary_line = containers.BaselineLine(
            id='line_no_boundary',
            baseline=[(10, 50), (200, 50)],
            boundary=None,
            text='test text',
        )
        normal_bl_line = containers.BaselineLine(
            id='line_normal',
            baseline=[(10, 100), (200, 100)],
            boundary=[(10, 80), (200, 80), (200, 120), (10, 120)],
            text='normal text',
        )

        cls.bl_no_boundary_seg = containers.Segmentation(
            type='baselines',
            imagename='foo.png',
            text_direction='horizontal-lr',
            lines=[no_boundary_line, normal_bl_line],
            script_detection=False,
            regions={},
        )

        cls.metadata_steps = [containers.ProcessingStep(id=str(uuid.uuid4()),
                                                        category='preprocessing',
                                                        description='PDF image extraction',
                                                        settings={}),
                              containers.ProcessingStep(id=str(uuid.uuid4()),
                                                        category='processing',
                                                        description='Baseline and region segmentation',
                                                        settings={'model': 'foo.mlmodel', 'text_direction': 'horizontal-lr'}),
                              containers.ProcessingStep(id=str(uuid.uuid4()),
                                                        category='processing',
                                                        description='Text line recognition',
                                                        settings={'text_direction': 'horizontal-lr',
                                                                  'models': 'bar.mlmodel',
                                                                  'pad': 16,
                                                                  'bidi_reordering': True})]

    def _segmentation_cases(self):
        return [('box', self.box_segmentation, {}, False),
                ('bl', self.bl_segmentation, {}, True),
                ('bl_regions', self.bl_segmentation_regs, {}, True),
                ('region_only', self.bl_seg_nolines_regs, {}, False),
                ('bl_steps', self.bl_segmentation, {'processing_steps': self.metadata_steps}, True),
                ('no_boundary', self.bl_no_boundary_seg, {}, False)]

    def test_serialize_alto(self):
        """
        Validates ALTO output against the schema for each segmentation variant.
        """
        for name, seg, kwargs, check_roundtrip in self._segmentation_cases():
            with self.subTest(name):
                fp = StringIO()
                fp.write(serialization.serialize(seg, image_size=IMAGE_SIZE, template='alto', **kwargs))
                validate_alto(self, fp)
                if check_roundtrip:
                    roundtrip(self, self.bl_records, fp)

    def test_serialize_pagexml(self):
        """
        Validates PageXML output against the schema for each segmentation variant.
        """
        for name, seg, kwargs, check_roundtrip in self._segmentation_cases():
            with self.subTest(name):
                fp = StringIO()
                fp.write(serialization.serialize(seg, image_size=IMAGE_SIZE, template='pagexml', **kwargs))
                validate_page(self, fp)
                if check_roundtrip:
                    roundtrip(self, self.bl_records, fp)

    def test_serialize_abbyyxml(self):
        """
        Validates abbyyXML output against the schema for each segmentation variant.
        """
        for name, seg, kwargs, _ in self._segmentation_cases():
            with self.subTest(name):
                fp = StringIO()
                fp.write(serialization.serialize(seg, image_size=IMAGE_SIZE, template='abbyyxml', **kwargs))
                validate_abbyy(self, fp)

    def test_serialize_hocr(self):
        """
        Validates hOCR output for each segmentation variant and writing mode.
        """
        cases = [('box', self.box_segmentation, {}),
                 ('box_vertical', self.box_segmentation, {'writing_mode': 'vertical-lr'}),
                 ('no_boundary', self.bl_no_boundary_seg, {})]
        for name, seg, kwargs in cases:
            with self.subTest(name):
                fp = StringIO()
                fp.write(serialization.serialize(seg, image_size=IMAGE_SIZE, template='hocr', **kwargs))
                validate_hocr(self, fp)
