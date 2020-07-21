# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import unittest
import json
import os

import kraken

from os import path

from kraken.lib import xml
from kraken.lib.train import KrakenTrainer

thisfile = os.path.abspath(os.path.dirname(__file__))
resources = os.path.abspath(os.path.join(thisfile, 'resources'))

class TestKrakenTrainer(unittest.TestCase):
    """
    Tests for KrakenTrainer class
    """
    def setUp(self):
        self.xml = path.join(resources, '170025120000003,0074.xml')
        self.bls = xml.parse_page(self.xml)
        self.box_lines = [path.join(resources, '000236.png')]
        self.model = path.join(resources, 'model_small.mlmodel')

    def test_krakentrainer_rec_box_load(self):
        training_data = self.box_lines
        evaluation_data = self.box_lines
        trainer = KrakenTrainer.recognition_train_gen(format_type='path',
                                                      load=self.model,
                                                      training_data=training_data,
                                                      evaluation_data=evaluation_data)
        self.assertEqual(trainer.model.seg_type, 'bbox')
        self.assertIsInstance(trainer.train_set.dataset, kraken.lib.dataset.GroundTruthDataset)

    def test_krakentrainer_rec_box_append(self):
        training_data = self.box_lines
        evaluation_data = self.box_lines
        trainer = KrakenTrainer.recognition_train_gen(format_type='path',
                                                      load=self.model,
                                                      append=1,
                                                      spec='[Cr4,4,32]',
                                                      training_data=training_data,
                                                      evaluation_data=evaluation_data)
        self.assertEqual(trainer.model.seg_type, 'bbox')
        self.assertIsInstance(trainer.train_set.dataset, kraken.lib.dataset.GroundTruthDataset)
        self.assertTrue(trainer.model.spec.startswith('[1,48,0,1 Cr{C_0}4,2,1,4,2 Cr{C_1}4,4,32 O{O_2}'))

    def test_krakentrainer_rec_bl_load(self):
        training_data = [self.xml]
        evaluation_data = [self.xml]
        trainer = KrakenTrainer.recognition_train_gen(format_type='xml',
                                                      load=self.model,
                                                      training_data=training_data,
                                                      evaluation_data=evaluation_data)
        self.assertEqual(trainer.model.seg_type, 'baselines')
        self.assertIsInstance(trainer.train_set.dataset, kraken.lib.dataset.PolygonGTDataset)

    def test_krakentrainer_rec_bl_append(self):
        training_data = [self.xml]
        evaluation_data = [self.xml]
        trainer = KrakenTrainer.recognition_train_gen(format_type='xml',
                                                      load=self.model,
                                                      append=1,
                                                      spec='[Cr4,4,32]',
                                                      training_data=training_data,
                                                      evaluation_data=evaluation_data)
        self.assertEqual(trainer.model.seg_type, 'baselines')
        self.assertIsInstance(trainer.train_set.dataset, kraken.lib.dataset.PolygonGTDataset)
        self.assertTrue(trainer.model.spec.startswith('[1,48,0,1 Cr{C_0}4,2,1,4,2 Cr{C_1}4,4,32 O{O_2}'))

    def test_krakentrainer_rec_box_path(self):
        """
        Tests recognition trainer constructor with legacy path training data.
        """
        training_data = self.box_lines
        evaluation_data = self.box_lines
        trainer = KrakenTrainer.recognition_train_gen(format_type='path',
                                                      training_data=training_data,
                                                      evaluation_data=evaluation_data)
        self.assertEqual(trainer.model.seg_type, 'bbox')
        self.assertIsInstance(trainer.train_set.dataset, kraken.lib.dataset.GroundTruthDataset)

    def test_krakentrainer_rec_bl_xml(self):
        """
        Tests recognition trainer constructor with XML training data.
        """
        training_data = [self.xml]
        evaluation_data = [self.xml]
        trainer = KrakenTrainer.recognition_train_gen(format_type='xml',
                                                      training_data=training_data,
                                                      evaluation_data=evaluation_data)
        self.assertEqual(trainer.model.seg_type, 'baselines')
        self.assertIsInstance(trainer.train_set.dataset, kraken.lib.dataset.PolygonGTDataset)
        self.assertEqual(len(trainer.train_set.dataset), 44)
        self.assertEqual(len(trainer.val_set.dataset), 44)

    def test_krakentrainer_rec_bl_dict(self):
        """
        Tests recognition trainer constructor with dictionary style training data.
        """
        training_data = [{'image': path.join(resources, 'bw.png'), 'text': 'foo', 'baseline': [[10, 10], [300, 10]], 'boundary': [[10, 5], [300, 5], [300, 15], [10, 15]]}]
        evaluation_data = [{'image': path.join(resources, 'bw.png'), 'text': 'foo', 'baseline': [[10, 10], [300, 10]], 'boundary': [[10, 5], [300, 5], [300, 15], [10, 15]]}]
        trainer = KrakenTrainer.recognition_train_gen(format_type=None,
                                                      training_data=training_data,
                                                      evaluation_data=evaluation_data)
        self.assertEqual(trainer.model.seg_type, 'baselines')
        self.assertIsInstance(trainer.train_set.dataset, kraken.lib.dataset.PolygonGTDataset)
