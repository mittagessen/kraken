# -*- coding: utf-8 -*-

import json
import unittest
from pathlib import Path

from pytest import raises

import kraken
from kraken.lib import xml
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.train import KrakenTrainer, RecognitionModel, SegmentationModel

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestKrakenTrainer(unittest.TestCase):
    """
    Tests for KrakenTrainer class
    """
    def setUp(self):
        self.xml = resources / '170025120000003,0074.xml'
        self.bls = xml.XMLPage(self.xml)
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

    def test_krakentrainer_rec_box_load_union(self):
        """
        Tests that adaptation works in `union` mode.

        The dataset brings 15 new characters. Input model had 3. There is one spec. char to account for.
        """
        training_data = self.box_lines
        evaluation_data = self.box_lines
        module = RecognitionModel(format_type='path',
                                  model=self.model,
                                  training_data=training_data,
                                  evaluation_data=evaluation_data,
                                  resize='union')
        module.setup("fit")
        self.assertEqual(module.nn.seg_type, 'bbox')
        self.assertIsInstance(module.train_set.dataset, kraken.lib.dataset.GroundTruthDataset)
        trainer = KrakenTrainer(max_steps=1)
        self.assertEqual(module.nn.named_spec[-1].split("c")[-1], '19')

    def test_krakentrainer_rec_box_load_new(self):
        """
        Tests that adaptation works in `new` mode.
        """
        training_data = self.box_lines
        evaluation_data = self.box_lines
        module = RecognitionModel(format_type='path',
                                  model=self.model,
                                  training_data=training_data,
                                  evaluation_data=evaluation_data,
                                  resize='new')
        module.setup("fit")
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

    def test_krakentrainer_rec_bl_load_union(self):
        training_data = [self.xml]
        evaluation_data = [self.xml]
        module = RecognitionModel(format_type='xml',
                                  model=self.model,
                                  training_data=training_data,
                                  evaluation_data=evaluation_data,
                                  resize='union')
        module.setup()
        self.assertEqual(module.nn.seg_type, 'baselines')
        self.assertIsInstance(module.train_set.dataset, kraken.lib.dataset.PolygonGTDataset)
        trainer = KrakenTrainer(max_steps=1)
        self.assertEqual(module.nn.named_spec[-1].split("c")[-1], '60')

    def test_krakentrainer_rec_bl_load_new(self):
        training_data = [self.xml]
        evaluation_data = [self.xml]
        module = RecognitionModel(format_type='xml',
                                  model=self.model,
                                  training_data=training_data,
                                  evaluation_data=evaluation_data,
                                  resize='new')
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

    def test_krakentrainer_rec_bl_augment(self):
        """
        Test that augmentation is added if specified.
        """
        training_data = [self.xml]
        evaluation_data = [self.xml]
        module = RecognitionModel(format_type='xml',
                                  training_data=training_data,
                                  evaluation_data=evaluation_data)
        module.setup()
        self.assertEqual(module.train_set.dataset.aug, None)

        module = RecognitionModel({'augment': True},
                                  format_type='xml',
                                  training_data=training_data,
                                  evaluation_data=evaluation_data)
        module.setup()
        self.assertIsInstance(module.train_set.dataset.aug, kraken.lib.dataset.recognition.DefaultAugmenter)

    def test_krakentrainer_rec_box_augment(self):
        """
        Test that augmentation is added if specified.
        """
        training_data = self.box_lines
        evaluation_data = self.box_lines
        module = RecognitionModel(format_type='path',
                                  training_data=training_data,
                                  evaluation_data=evaluation_data)
        module.setup()
        self.assertEqual(module.train_set.dataset.aug, None)

        module = RecognitionModel({'augment': True},
                                  format_type='path',
                                  training_data=training_data,
                                  evaluation_data=evaluation_data)
        module.setup()
        self.assertIsInstance(module.train_set.dataset.aug, kraken.lib.dataset.recognition.DefaultAugmenter)
