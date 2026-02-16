# -*- coding: utf-8 -*-
import unittest
from pathlib import Path

from PIL import Image
from pytest import raises

from kraken.lib import xml
from kraken.lib.dataset import BaselineSet, ImageInputTransforms
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.util import is_bitonal

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
        self.doc = xml.XMLPage(resources / '170025120000003,0074.xml').to_container()
        self.transforms = ImageInputTransforms(batch=1,
                                               height=200,
                                               width=100,
                                               channels=1,
                                               pad=0)

    def test_baselineset_simple_xml(self):
        """
        Tests simple BaselineSet instantiation with all baseline and region types.
        """
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$pag': 2, '$pac': 3, '$tip': 4, '$par': 5},
            'regions': {'$pag': 6, '$pac': 7, '$tip': 8, '$par': 9},
        }
        ds = BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)
        ds.add(self.doc)
        ds.add(self.doc)

        sample = ds[0]
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_classes, 10)
        self.assertEqual(sample['image'].shape, (1, 200, 100))
        self.assertEqual(sample['target'].shape, (ds.num_classes, 200, 100))

    def test_baselineset_filter_baselines(self):
        """
        Test that only baselines present in class_mapping are included.
        """
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$tip': 2, '$par': 3},
            'regions': {'$pag': 4, '$pac': 5, '$tip': 6, '$par': 7},
        }
        ds = BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)
        ds.add(self.doc)
        ds.add(self.doc)

        sample = ds[0]
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_classes, 8)
        self.assertEqual(set(ds.class_mapping['baselines'].keys()), {'$tip', '$par'})
        self.assertNotIn('$pac', ds.class_mapping['baselines'])
        self.assertNotIn('$pag', ds.class_mapping['baselines'])
        self.assertEqual(sample['image'].shape, (1, 200, 100))
        self.assertEqual(sample['target'].shape, (ds.num_classes, 200, 100))

    def test_baselineset_filter_regions(self):
        """
        Test that only regions present in class_mapping are included.
        """
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$pag': 2, '$pac': 3, '$tip': 4, '$par': 5},
            'regions': {'$pag': 6, '$pac': 7},
        }
        ds = BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)
        ds.add(self.doc)
        ds.add(self.doc)

        sample = ds[0]
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_classes, 8)
        self.assertEqual(set(ds.class_mapping['regions'].keys()), {'$pag', '$pac'})
        self.assertNotIn('$par', ds.class_mapping['regions'])
        self.assertNotIn('$tip', ds.class_mapping['regions'])
        self.assertEqual(sample['image'].shape, (1, 200, 100))
        self.assertEqual(sample['target'].shape, (ds.num_classes, 200, 100))

    def test_baselineset_merge_baselines(self):
        """
        Test baseline merging via duplicate index values in class_mapping.
        """
        # $par and $tip share index 4 (merged), $pag=2, $pac=3
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$pag': 2, '$pac': 3, '$par': 4, '$tip': 4},
            'regions': {'$pag': 5, '$pac': 6, '$tip': 7, '$par': 8},
        }
        ds = BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)
        ds.add(self.doc)
        ds.add(self.doc)

        sample = ds[0]
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_classes, 9)
        # merged baselines should have 17 ($par) + 1 ($tip) = 18 lines at index 4
        self.assertEqual(len(ds.targets[0]['baselines'][4]), 18)
        self.assertEqual(sample['image'].shape, (1, 200, 100))
        self.assertEqual(sample['target'].shape, (ds.num_classes, 200, 100))

    def test_baselineset_merge_and_filter_baselines(self):
        """
        Test merging and filtering baselines simultaneously.
        """
        # only $tip and $pac in mapping, both merged to index 2
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$tip': 2, '$pac': 2},
            'regions': {'$pag': 3, '$pac': 4, '$tip': 5, '$par': 6},
        }
        ds = BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)
        ds.add(self.doc)
        ds.add(self.doc)

        sample = ds[0]
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_classes, 7)
        # merged baselines: $tip (1) + $pac (25) = 26 lines at index 2
        self.assertEqual(len(ds.targets[0]['baselines'][2]), 26)
        self.assertNotIn('$par', ds.class_mapping['baselines'])
        self.assertNotIn('$pag', ds.class_mapping['baselines'])
        self.assertEqual(sample['image'].shape, (1, 200, 100))
        self.assertEqual(sample['target'].shape, (ds.num_classes, 200, 100))

    def test_baselineset_merge_and_filter_regions(self):
        """
        Test merging and filtering regions simultaneously.
        """
        # only $tip and $pac in region mapping, both merged to index 6
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$pag': 2, '$pac': 3, '$tip': 4, '$par': 5},
            'regions': {'$tip': 6, '$pac': 6},
        }
        ds = BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)
        ds.add(self.doc)
        ds.add(self.doc)

        sample = ds[0]
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_classes, 7)
        # merged regions: $tip (1) + $pac (1) = 2 regions at index 6
        self.assertEqual(len(ds.targets[0]['regions'][6]), 2)
        self.assertNotIn('$par', ds.class_mapping['regions'])
        self.assertNotIn('$pag', ds.class_mapping['regions'])
        self.assertEqual(sample['image'].shape, (1, 200, 100))
        self.assertEqual(sample['target'].shape, (ds.num_classes, 200, 100))

    def test_baselineset_canonical_class_mapping_no_merging(self):
        """
        When no classes are merged, canonical_class_mapping equals class_mapping.
        """
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$pag': 2, '$pac': 3, '$tip': 4, '$par': 5},
            'regions': {'$pag': 6, '$pac': 7, '$tip': 8, '$par': 9},
        }
        ds = BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)
        self.assertEqual(ds.canonical_class_mapping, class_mapping)

    def test_baselineset_canonical_class_mapping_with_merging(self):
        """
        When classes are merged, canonical mapping keeps only the first by insertion order.
        """
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$pag': 2, '$pac': 3, '$par': 4, '$tip': 4},
            'regions': {'$pag': 5, '$pac': 6, '$tip': 7, '$par': 7},
        }
        ds = BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)
        canonical = ds.canonical_class_mapping
        # $par is first at index 4, $tip should be dropped
        self.assertEqual(canonical['baselines'], {'$pag': 2, '$pac': 3, '$par': 4})
        # $tip is first at index 7, $par should be dropped
        self.assertEqual(canonical['regions'], {'$pag': 5, '$pac': 6, '$tip': 7})
        # aux unchanged
        self.assertEqual(canonical['aux'], {'_start_separator': 0, '_end_separator': 1})

    def test_baselineset_merged_classes_empty(self):
        """
        When no merging, merged_classes sections are all empty dicts.
        """
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$pag': 2, '$pac': 3, '$tip': 4, '$par': 5},
            'regions': {'$pag': 6, '$pac': 7, '$tip': 8, '$par': 9},
        }
        ds = BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)
        merged = ds.merged_classes
        self.assertEqual(merged['aux'], {})
        self.assertEqual(merged['baselines'], {})
        self.assertEqual(merged['regions'], {})

    def test_baselineset_merged_classes_with_merging(self):
        """
        Reports correct aliases when classes are merged.
        """
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$pag': 2, '$pac': 3, '$par': 4, '$tip': 4},
            'regions': {'$pag': 5, '$pac': 6, '$tip': 7, '$par': 7},
        }
        ds = BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)
        merged = ds.merged_classes
        # $par is canonical for index 4, $tip is the alias
        self.assertEqual(merged['baselines'], {'$par': ['$tip']})
        # $tip is canonical for index 7, $par is the alias
        self.assertEqual(merged['regions'], {'$tip': ['$par']})
        self.assertEqual(merged['aux'], {})

    def test_baselineset_empty_baselines_and_regions(self):
        """
        Test aux-only class_mapping with empty baselines and regions.
        """
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {},
            'regions': {},
        }
        ds = BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)
        ds.add(self.doc)

        self.assertEqual(len(ds), 1)
        self.assertEqual(ds.num_classes, 2)
        self.assertEqual(ds.targets[0]['baselines'], {})
        self.assertEqual(ds.targets[0]['regions'], {})

    def test_baselineset_invalid_missing_aux(self):
        """
        Test that missing 'aux' key raises ValueError.
        """
        class_mapping = {
            'baselines': {'$par': 2},
            'regions': {'$par': 3},
        }
        with raises(ValueError):
            BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)

    def test_baselineset_invalid_missing_baselines(self):
        """
        Test that missing 'baselines' key raises ValueError.
        """
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'regions': {'$par': 2},
        }
        with raises(ValueError):
            BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)

    def test_baselineset_invalid_missing_regions(self):
        """
        Test that missing 'regions' key raises ValueError.
        """
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$par': 2},
        }
        with raises(ValueError):
            BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)

    def test_baselineset_invalid_missing_start_separator(self):
        """
        Test that missing '_start_separator' in aux raises ValueError.
        """
        class_mapping = {
            'aux': {'_end_separator': 1},
            'baselines': {'$par': 2},
            'regions': {'$par': 3},
        }
        with raises(ValueError):
            BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)

    def test_baselineset_invalid_missing_end_separator(self):
        """
        Test that missing '_end_separator' in aux raises ValueError.
        """
        class_mapping = {
            'aux': {'_start_separator': 0},
            'baselines': {'$par': 2},
            'regions': {'$par': 3},
        }
        with raises(ValueError):
            BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)

    def test_baselineset_invalid_negative_value(self):
        """
        Test that negative index values raise ValueError.
        """
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$par': -1},
            'regions': {'$par': 2},
        }
        with raises(ValueError):
            BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)

    def test_baselineset_invalid_non_integer_value(self):
        """
        Test that non-integer index values raise ValueError.
        """
        class_mapping = {
            'aux': {'_start_separator': 0, '_end_separator': 1},
            'baselines': {'$par': 2.5},
            'regions': {'$par': 3},
        }
        with raises(ValueError):
            BaselineSet(class_mapping=class_mapping, im_transforms=self.transforms)


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
        out = tf(self.im)
        # In channel-height mode the target height is moved into the channel
        # dimension, resulting in a Cx1xW tensor.
        self.assertEqual(self.channel_height_inst['channels'], out.shape[0])
        self.assertEqual(self.channel_height_inst['height'], out.shape[1])
        check_output(self, self.channel_height_inst, self.im, out)

    def test_imageinputtransforms_invalid_channels(self):
        """
        ImageInputTransforms instantiation with invalid number of channels
        """
        with raises(KrakenInputException):
            ImageInputTransforms(**self.invalid_channels)
