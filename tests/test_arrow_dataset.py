# -*- coding: utf-8 -*-

import json
import unittest
import tempfile
import pyarrow as pa

from pathlib import Path
from pytest import raises

import kraken
from kraken.lib import xml
from kraken.lib.arrow_dataset import build_binary_dataset

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

def _validate_ds(self, path, num_lines, num_empty_lines, ds_type):
    with pa.memory_map(path, 'rb') as source:
        ds_table = pa.ipc.open_file(source).read_all()
        raw_metadata = ds_table.schema.metadata
        if not raw_metadata or b'lines' not in raw_metadata:
            raise ValueError(f'{file} does not contain a valid metadata record.')
        metadata = json.loads(raw_metadata[b'lines'])
    self.assertEqual(metadata['type'],
                    ds_type,
                    f'Unexpected dataset type (expected: {ds_type}, found: {metadata["type"]}')
    self.assertEqual(metadata['counts']['all'],
                     num_lines,
                     'Unexpected number of lines in dataset metadata '
                     f'(expected: {num_lines}, found: {metadata["counts"]["all"]}')
    self.assertEqual(len(ds_table),
                     num_lines,
                     'Unexpected number of rows in arrow table '
                     f'(expected: {num_lines}, found: {metadata["counts"]["all"]}')

    real_empty_lines = len([line for line in ds_table.column('lines') if not str(line[0])])
    self.assertEqual(real_empty_lines,
                     num_empty_lines,
                     'Unexpected number of empty lines in dataset '
                     f'(expected: {num_empty_lines}, found: {real_empty_lines}')


class TestKrakenArrowCompilation(unittest.TestCase):
    """
    Tests for binary datasets
    """
    def setUp(self):
        self.xml = resources / '170025120000003,0074-lite.xml'
        self.seg = xml.XMLPage(self.xml).to_container()
        self.box_lines = [resources / '000236.png']

    def test_build_path_dataset(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            build_binary_dataset(files=4*self.box_lines,
                                 output_file=tmp_file.name,
                                 format_type='path')
            _validate_ds(self, tmp_file.name, 4, 0, 'kraken_recognition_bbox')

    def test_build_xml_dataset(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            build_binary_dataset(files=[self.xml],
                                 output_file=tmp_file.name,
                                 format_type='xml')
            _validate_ds(self, tmp_file.name, 4, 0, 'kraken_recognition_baseline')

    def test_build_seg_dataset(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            build_binary_dataset(files=[self.seg],
                                 output_file=tmp_file.name,
                                 format_type=None)
            _validate_ds(self, tmp_file.name, 4, 0, 'kraken_recognition_baseline')

    def test_forced_type_dataset(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            build_binary_dataset(files=4*self.box_lines,
                                 output_file=tmp_file.name,
                                 format_type='path',
                                 force_type='kraken_recognition_baseline')
            _validate_ds(self, tmp_file.name, 4, 0, 'kraken_recognition_baseline')

    def test_build_empty_dataset(self):
        """
        Test that empty lines are retained in compiled dataset.
        """
        with tempfile.NamedTemporaryFile() as tmp_file:
            build_binary_dataset(files=[self.xml],
                                 output_file=tmp_file.name,
                                 format_type='xml',
                                 skip_empty_lines=False)
            _validate_ds(self, tmp_file.name, 5, 1, 'kraken_recognition_baseline')
