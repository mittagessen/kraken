# -*- coding: utf-8 -*-

import unittest
import json

import kraken

from pytest import raises
from pathlib import Path

from kraken.lib import xml
from kraken.lib.train import KrakenTrainer, RecognitionModel, SegmentationModel
from kraken.lib.exceptions import KrakenInputException

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestKrakenTrainer(unittest.TestCase):
    """
    Tests for KrakenTrainer class
    """
    def setUp(self):
        self.xml = resources / '170025120000003,0074.xml'
        self.bls = xml.parse_page(self.xml)
        self.box_lines = [resources / '000236.png']
        self.model = resources / 'model_small.mlmodel'

    def test_krakentrainer_rec_box_load_fail(self):
        training_data = self.box_lines
        evaluation_data = self.box_lines
        module = RecognitionModel(format_type='path',
                                  model=self.model,
                                  training_data=training_data,
                                  evaluation_data=evaluation_data,
                                  resize='fail')
        with raises(KrakenInputException):
            module.setup()

    def test_krakentrainer_rec_bl_load_fail(self):
        """
        Tests that the proper exception is raised when loading model not fitting the dataset.
        """
        training_data = [self.xml]
        evaluation_data = [self.xml]
        module = RecognitionModel(format_type='xml',
                                  model=self.model,
                                  training_data=training_data,
                                  evaluation_data=evaluation_data,
                                  resize='fail')
        with raises(KrakenInputException):
            module.setup()

    def test_krakentrainer_rec_box_load_add(self):
        """
        Tests that adaptation works in add mode.
        """
        training_data = self.box_lines
        evaluation_data = self.box_lines
        module = RecognitionModel(format_type='path',
                                  model=self.model,
                                  training_data=training_data,
                                  evaluation_data=evaluation_data,
                                  resize='add')
        module.setup()
        self.assertEqual(module.nn.seg_type, 'bbox')
        self.assertIsInstance(module.train_set.dataset, kraken.lib.dataset.GroundTruthDataset)
        trainer = KrakenTrainer(max_steps=1)
        self.assertEqual(module.nn.named_spec[-1].split("c")[-1], '19')

    def test_krakentrainer_rec_box_load_both(self):
        """
        Tests that adaptation works in both mode.
        """
        training_data = self.box_lines
        evaluation_data = self.box_lines
        module = RecognitionModel(format_type='path',
                                  model=self.model,
                                  training_data=training_data,
                                  evaluation_data=evaluation_data,
                                  resize='both')
        module.setup()
        self.assertEqual(module.nn.seg_type, 'bbox')
        self.assertIsInstance(module.train_set.dataset, kraken.lib.dataset.GroundTruthDataset)
        trainer = KrakenTrainer(max_steps=1)
        self.assertEqual(module.nn.named_spec[-1].split("c")[-1], '16')

    def test_krakentrainer_rec_box_append(self):
        """
        Tests that appending new layers onto a loaded model works.
        """
        training_data = self.box_lines
        evaluation_data = self.box_lines
        module = RecognitionModel(format_type='path',
                                  model=self.model,
                                  append=1,
                                  spec='[Cr4,4,32]',
                                  training_data=training_data,
                                  evaluation_data=evaluation_data)
        module.setup()
        self.assertEqual(module.nn.seg_type, 'bbox')
        self.assertIsInstance(module.train_set.dataset, kraken.lib.dataset.GroundTruthDataset)
        self.assertTrue(module.nn.spec.startswith('[1,48,0,1 Cr{C_0}4,2,1,4,2 Cr{C_1}4,4,32 O{O_2}'))
        trainer = KrakenTrainer(max_steps=1)

    def test_krakentrainer_rec_bl_load(self):
        training_data = [self.xml]
        evaluation_data = [self.xml]
        module = RecognitionModel(format_type='xml',
                                  model=self.model,
                                  training_data=training_data,
                                  evaluation_data=evaluation_data,
                                  resize='fail')
        with raises(KrakenInputException):
            module.setup()

    def test_krakentrainer_rec_bl_load_add(self):
        training_data = [self.xml]
        evaluation_data = [self.xml]
        module = RecognitionModel(format_type='xml',
                                  model=self.model,
                                  training_data=training_data,
                                  evaluation_data=evaluation_data,
                                  resize='add')
        module.setup()
        self.assertEqual(module.nn.seg_type, 'baselines')
        self.assertIsInstance(module.train_set.dataset, kraken.lib.dataset.PolygonGTDataset)
        trainer = KrakenTrainer(max_steps=1)
        self.assertEqual(module.nn.named_spec[-1].split("c")[-1], '60')

    def test_krakentrainer_rec_bl_load_both(self):
        training_data = [self.xml]
        evaluation_data = [self.xml]
        module = RecognitionModel(format_type='xml',
                                  model=self.model,
                                  training_data=training_data,
                                  evaluation_data=evaluation_data,
                                  resize='both')
        module.setup()
        self.assertEqual(module.nn.seg_type, 'baselines')
        self.assertIsInstance(module.train_set.dataset, kraken.lib.dataset.PolygonGTDataset)
        trainer = KrakenTrainer(max_steps=1)
        self.assertEqual(module.nn.named_spec[-1].split("c")[-1], '60')

    def test_krakentrainer_rec_bl_append(self):
        training_data = [self.xml]
        evaluation_data = [self.xml]
        module = RecognitionModel(format_type='xml',
                                  model=self.model,
                                  append=1,
                                  spec='[Cr4,4,32]',
                                  training_data=training_data,
                                  evaluation_data=evaluation_data)
        module.setup()
        self.assertEqual(module.nn.seg_type, 'baselines')
        self.assertIsInstance(module.train_set.dataset, kraken.lib.dataset.PolygonGTDataset)
        self.assertTrue(module.nn.spec.startswith('[1,48,0,1 Cr{C_0}4,2,1,4,2 Cr{C_1}4,4,32 O{O_2}'))
        trainer = KrakenTrainer(max_steps=1)

    def test_krakentrainer_rec_box_path(self):
        """
        Tests recognition trainer constructor with legacy path training data.
        """
        training_data = self.box_lines
        evaluation_data = self.box_lines
        module = RecognitionModel(format_type='path',
                                  training_data=training_data,
                                  evaluation_data=evaluation_data)
        module.setup()
        self.assertEqual(module.nn.seg_type, 'bbox')
        self.assertIsInstance(module.train_set.dataset, kraken.lib.dataset.GroundTruthDataset)
        trainer = KrakenTrainer(max_steps=1)

    def test_krakentrainer_rec_bl_xml(self):
        """
        Tests recognition trainer constructor with XML training data.
        """
        training_data = [self.xml]
        evaluation_data = [self.xml]
        module = RecognitionModel(format_type='xml',
                                  training_data=training_data,
                                  evaluation_data=evaluation_data)
        module.setup()
        self.assertEqual(module.nn.seg_type, 'baselines')
        self.assertIsInstance(module.train_set.dataset, kraken.lib.dataset.PolygonGTDataset)
        self.assertEqual(len(module.train_set.dataset), 44)
        self.assertEqual(len(module.val_set.dataset), 44)
        trainer = KrakenTrainer(max_steps=1)

    def test_krakentrainer_rec_bl_dict(self):
        """
        Tests recognition trainer constructor with dictionary style training data.
        """
        training_data = [{'image': resources / 'bw.png', 'text': 'foo', 'baseline': [[10, 10], [300, 10]], 'boundary': [[10, 5], [300, 5], [300, 15], [10, 15]]}]
        evaluation_data = [{'image': resources / 'bw.png', 'text': 'foo', 'baseline': [[10, 10], [300, 10]], 'boundary': [[10, 5], [300, 5], [300, 15], [10, 15]]}]
        module = RecognitionModel(format_type=None,
                                  training_data=training_data,
                                  evaluation_data=evaluation_data)
        module.setup()
        self.assertEqual(module.nn.seg_type, 'baselines')
        self.assertIsInstance(module.train_set.dataset, kraken.lib.dataset.PolygonGTDataset)
        trainer = KrakenTrainer(max_steps=1)
