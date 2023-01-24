# -*- coding: utf-8 -*-
import unittest

from pathlib import Path
from pytest import raises

from PIL import Image
from kraken.lib.dataset import ImageInputTransforms, BaselineSet

from kraken.lib.util import is_bitonal
from kraken.lib.exceptions import KrakenInputException

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

def check_output(self, config, im, output_tensor):
    if config['height'] != 0:
        self.assertEqual(config['height'], output_tensor.shape[1])
    if config['width'] != 0:
        self.assertEqual(config['width'], output_tensor.shape[2])
    if config['force_binarization'] or is_bitonal(im):
        self.assertEqual(len(output_tensor.int().unique()), 2)
    if config['channels'] == 3:
        self.assertEqual(output_tensor.shape[0], 3)

class TestBaselineSet(unittest.TestCase):
    """
    Tests for the BaselineSet segmentation dataset class
    """
    def setUp(self):
        self.doc = resources / '170025120000003,0074.xml'
        self.transforms = ImageInputTransforms(batch=1,
                                               height=200,
                                               width=100,
                                               channels=1,
                                               pad=0)

    def test_baselineset_simple_xml(self):
        """
        Tests simple BaselineSet instantiation
        """
        ds = BaselineSet(imgs=[self.doc, self.doc],
                         im_transforms=self.transforms,
                         mode='xml')

        sample = ds[0]
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_classes, 10)
        self.assertEqual(sample['image'].shape, (1, 200, 100))
        self.assertEqual(sample['target'].shape, (ds.num_classes, 200, 100))

    def test_baselineset_simple_valid_baselines(self):
        """
        Test baseline whitelisting in BaselineSet
        """
        # filter out $pac and $pag baseline classes
        ds = BaselineSet(imgs=[self.doc, self.doc],
                         im_transforms=self.transforms,
                         valid_baselines=['$par', '$tip'],
                         mode='xml')

        sample = ds[0]
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_classes, 8)
        self.assertEqual(set(ds.class_mapping['baselines'].keys()), set(('$tip', '$par')))
        self.assertNotIn('$pac', ds.class_mapping['baselines'])
        self.assertNotIn('$pag', ds.class_mapping['baselines'])
        self.assertEqual(sample['image'].shape, (1, 200, 100))
        self.assertEqual(sample['target'].shape, (ds.num_classes, 200, 100))

    def test_baselineset_simple_valid_regions(self):
        """
        Test region whitelisting in BaselineSet
        """
        # filter out $tip and $par regions
        ds = BaselineSet(imgs=[self.doc, self.doc],
                         im_transforms=self.transforms,
                         valid_regions=['$pag', '$pac'],
                         mode='xml')

        sample = ds[0]
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_classes, 8)
        self.assertEqual(set(ds.class_mapping['regions'].keys()), set(('$pag', '$pac')))
        self.assertNotIn('$par', ds.class_mapping['regions'])
        self.assertNotIn('$tip', ds.class_mapping['regions'])
        self.assertEqual(sample['image'].shape, (1, 200, 100))
        self.assertEqual(sample['target'].shape, (ds.num_classes, 200, 100))

    def test_baselineset_simple_merge_baselines(self):
        """
        Test baseline merging in BaselineSet
        """
        # merge $par into $tip
        ds = BaselineSet(imgs=[self.doc, self.doc],
                         im_transforms=self.transforms,
                         merge_baselines={'$par': '$tip'},
                         mode='xml')

        sample = ds[0]
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_classes, 9)
        self.assertEqual(set(ds.class_mapping['baselines'].keys()), set(('$tip', '$pag', '$pac')))
        self.assertEqual(len(ds.targets[0]['baselines']['$tip']), 18)
        self.assertNotIn('$par', ds.class_mapping['baselines'])
        self.assertEqual(sample['image'].shape, (1, 200, 100))
        self.assertEqual(sample['target'].shape, (ds.num_classes, 200, 100))

    def test_baselineset_merge_after_valid_baselines(self):
        """
        Test that filtering with valid_baselines occurs before merging.
        """
        # merge $par and $pac into $tip but discard $par before
        ds = BaselineSet(imgs=[self.doc, self.doc],
                         im_transforms=self.transforms,
                         valid_baselines=('$tip', '$pac'),
                         merge_baselines={'$par': '$tip', '$pac': '$tip'},
                         mode='xml')

        sample = ds[0]
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_classes, 7)
        self.assertEqual(set(ds.class_mapping['baselines'].keys()), set(('$tip',)))
        self.assertEqual(len(ds.targets[0]['baselines']['$tip']), 26)
        self.assertNotIn('$par', ds.class_mapping['baselines'])
        self.assertEqual(sample['image'].shape, (1, 200, 100))
        self.assertEqual(sample['target'].shape, (ds.num_classes, 200, 100))

    def test_baselineset_merge_after_valid_regions(self):
        """
        Test that filtering with valid_regions occurs before merging.
        """
        # merge $par and $pac into $tip but discard $par before
        ds = BaselineSet(imgs=[self.doc, self.doc],
                         im_transforms=self.transforms,
                         valid_regions=('$tip', '$pac'),
                         merge_regions={'$par': '$tip', '$pac': '$tip'},
                         mode='xml')

        sample = ds[0]
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_classes, 7)
        self.assertEqual(set(ds.class_mapping['regions'].keys()), set(('$tip',)))
        self.assertEqual(len(ds.targets[0]['regions']['$tip']), 2)
        self.assertNotIn('$par', ds.class_mapping['regions'])
        self.assertEqual(sample['image'].shape, (1, 200, 100))
        self.assertEqual(sample['target'].shape, (ds.num_classes, 200, 100))


class TestInputTransforms(unittest.TestCase):
    """
    Tests for ImageInputTransforms class
    """

    def setUp(self):
        self.im = Image.open(resources / '000236.png')

        self.simple_inst = {'batch': 1,
                            'height': 48,
                            'width': 0,
                            'channels': 1,
                            'pad': (16, 0),
                            'valid_norm': False,
                            'force_binarization': False}

        self.simple_inst_norm = {'batch': 1,
                                 'height': 48,
                                 'width': 0,
                                 'channels': 1,
                                 'pad': (16, 0),
                                 'valid_norm': True,
                                 'force_binarization': False}

        self.simple_inst_rgb = {'batch': 1,
                                'height': 48,
                                'width': 0,
                                'channels': 3,
                                'pad': (16, 0),
                                'valid_norm': False,
                                'force_binarization': False}

        self.simple_inst_norm_rgb = {'batch': 1,
                                     'height': 48,
                                     'width': 0,
                                     'channels': 3,
                                     'pad': (16, 0),
                                     'valid_norm': True,
                                     'force_binarization': False}

        self.channel_height_inst = {'batch': 1,
                                    'height': 1,
                                    'width': 0,
                                    'channels': 72,
                                    'pad': (16, 0),
                                    'valid_norm': False,
                                    'force_binarization': False}

        self.invalid_channels = {'batch': 1,
                                 'height': 48,
                                 'width': 0,
                                 'channels': 4,
                                 'pad': (16, 0),
                                 'valid_norm': False,
                                 'force_binarization': False}

    def test_imageinputtransforms_simple(self):
        """
        Simple ImageInputTransforms instantiation.
        """
        tf = ImageInputTransforms(**self.simple_inst)
        for k, v in self.simple_inst.items():
            self.assertEqual(getattr(tf, k), v)
        self.assertFalse(tf.centerline_norm)
        check_output(self, self.simple_inst, self.im, tf(self.im))

    def test_imageinputtransforms_simple_rgb(self):
        """
        Simple RGB ImageInputTransforms instantiation.
        """
        tf = ImageInputTransforms(**self.simple_inst_rgb)
        for k, v in self.simple_inst_rgb.items():
            self.assertEqual(getattr(tf, k), v)
        self.assertFalse(tf.centerline_norm)
        check_output(self, self.simple_inst_rgb, self.im, tf(self.im))

    def test_imageinputtransforms_norm_rgb(self):
        """
        RGB ImageInputTransforms instantiation with centerline normalization
        valid (but not enabled).
        """
        tf = ImageInputTransforms(**self.simple_inst_norm_rgb)
        for k, v in self.simple_inst_norm_rgb.items():
            self.assertEqual(getattr(tf, k), v)
        self.assertFalse(tf.centerline_norm)
        check_output(self, self.simple_inst_norm_rgb, self.im, tf(self.im))

    def test_imageinputtransforms_simple_norm(self):
        """
        ImageInputTransforms instantiation with centerline normalization valid.
        """
        tf = ImageInputTransforms(**self.simple_inst_norm)
        for k, v in self.simple_inst_norm.items():
            self.assertEqual(getattr(tf, k), v)
        self.assertTrue(tf.centerline_norm)
        check_output(self, self.simple_inst_norm, self.im, tf(self.im))

    def test_imageinputtransforms_channel_height(self):
        """
        ImageInputTransforms with height in channel dimension
        """
        tf = ImageInputTransforms(**self.channel_height_inst)
        for k, v in self.channel_height_inst.items():
            if k == 'channels':
                self.assertEqual(1, tf.channels)
            elif k == 'height':
                self.assertEqual(self.channel_height_inst['channels'], tf.height)
            else:
                self.assertEqual(getattr(tf, k), v)
        self.assertFalse(tf.centerline_norm)
        self.channel_height_inst['height'] = self.channel_height_inst['channels']
        self.channel_height_inst['channels'] = 1
        check_output(self, self.channel_height_inst, self.im, tf(self.im))

    def test_imageinputtransforms_invalid_channels(self):
        """
        ImageInputTransforms instantiation with invalid number of channels
        """
        with raises(KrakenInputException):
            tf = ImageInputTransforms(**self.invalid_channels)

