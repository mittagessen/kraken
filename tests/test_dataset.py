# -*- coding: utf-8 -*-
import unittest

from pathlib import Path
from pytest import raises

from PIL import Image
from kraken.lib.dataset import ImageInputTransforms

from kraken.lib.exceptions import KrakenInputException

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

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
                            'pad': 16,
                            'valid_norm': False,
                            'force_binarization': False}

        self.simple_inst_norm = {'batch': 1,
                                 'height': 48,
                                 'width': 0,
                                 'channels': 1,
                                 'pad': 16,
                                 'valid_norm': True,
                                 'force_binarization': False}

        self.simple_inst_rgb = {'batch': 1,
                                'height': 48,
                                'width': 0,
                                'channels': 3,
                                'pad': 16,
                                'valid_norm': False,
                                'force_binarization': False}

        self.simple_inst_norm_rgb = {'batch': 1,
                                     'height': 48,
                                     'width': 0,
                                     'channels': 3,
                                     'pad': 16,
                                     'valid_norm': True,
                                     'force_binarization': False}

        self.channel_height_inst = {'batch': 1,
                                    'height': 1,
                                    'width': 0,
                                    'channels': 72,
                                    'pad': 16,
                                    'valid_norm': False,
                                    'force_binarization': False}

        self.invalid_channels = {'batch': 1,
                                 'height': 48,
                                 'width': 0,
                                 'channels': 4,
                                 'pad': 16,
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

    def test_imageinputtransforms_simple_rgb(self):
        """
        Simple RGB ImageInputTransforms instantiation.
        """
        tf = ImageInputTransforms(**self.simple_inst_rgb)
        for k, v in self.simple_inst_rgb.items():
            self.assertEqual(getattr(tf, k), v)
        self.assertFalse(tf.centerline_norm)

    def test_imageinputtransforms_norm_rgb(self):
        """
        RGB ImageInputTransforms instantiation with centerline normalization
        valid (but not enabled).
        """
        tf = ImageInputTransforms(**self.simple_inst_norm_rgb)
        for k, v in self.simple_inst_norm_rgb.items():
            self.assertEqual(getattr(tf, k), v)
        self.assertFalse(tf.centerline_norm)

    def test_imageinputtransforms_simple_norm(self):
        """
        ImageInputTransforms instantiation with centerline normalization valid.
        """
        tf = ImageInputTransforms(**self.simple_inst_norm)
        for k, v in self.simple_inst_norm.items():
            self.assertEqual(getattr(tf, k), v)
        self.assertTrue(tf.centerline_norm)

    def test_imageinputtransforms_channel_height(self):
        """
        ImageInputTransforms with height in channel dimension
        """
        tf = ImageInputTransforms(**self.channel_height_inst)
        for k, v in self.channel_height_inst.items():
            if k == 'channels':
                self.assertEqual(self.channel_height_inst['height'], tf.channels)
            elif k == 'height':
                self.assertEqual(self.channel_height_inst['channels'], tf.height)
            else:
                self.assertEqual(getattr(tf, k), v)
        self.assertFalse(tf.centerline_norm)

    def test_imageinputtransforms_invalid_channels(self):
        """
        ImageInputTransforms instantiation with invalid number of channels
        """
        with raises(KrakenInputException):
            tf = ImageInputTransforms(**self.invalid_channels)

