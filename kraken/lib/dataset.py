#
# Copyright 2015 Benjamin Kiessling
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

# -*- coding: utf-8 -*-
"""
Utility functions for training VGSL networks.
"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

from future import standard_library
standard_library.install_aliases()
from builtins import str

import os
import click
import numpy as np
import unicodedata
import bidi.algorithm as bd

from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset

from kraken import rpred
from kraken.lib import models
from kraken.lib.codec import PytorchCodec
from kraken.lib.util import pil2array, array2pil
from kraken.lib.lineest import CenterNormalizer

__all__ = ['GroundTruthDataset', 'compute_error']

def _fast_levenshtein(seq1, seq2):

    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        oneago, thisrow = thisrow, [0] * len(seq2) + [x + 1]
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len(seq2) - 1]


def compute_error(model, test_set):
    """
    Computes detailed error report from a model and a list of line image-text
    pairs.

    Args:
        model (kraken.lib.models.ClstmSeqRecognizer): Model used for
                                                      recognition
        test_set (list): List of tuples (image, text) for testing

    Returns:
        A tuple with total number of characters and edit distance across the
        whole test set.
    """
    total_chars = 0
    error = 0
    for im, text in test_set:
        pred = model.predict_string(im)
        total_chars += len(text)
        error += _fast_levenshtein(pred, text)
    return total_chars, error


class GroundTruthDataset(Dataset):
    """
    Dataset for ground truth used during training.

    All data is cached in memory.

    Attributes:
        training_set (list): List of tuples (image, text) for training
        test_set (list): List of tuples (image, text) for testing
        alphabet (str): Sorted string of all code points found in the ground
                        truth
    """
    def __init__(self, split=lambda x: os.path.splitext(x)[0],
                 suffix='.gt.txt', mode='1', scale=48, normalization=None,
                 reorder=True, pad=16, format=(0, 1, 2)):
        """
        Reads a list of image-text pairs and creates a ground truth set.

        Args:
            split (func): Function for generating the base name without
                          extensions from paths
            suffix (str): Suffix to attach to image base name for text
                          retrieval
            mode (str): Image color space (either RGB, L or 1)
            scale (int, tuple): Target height or (width, height) of dewarped
                                line images. Vertical-only scaling is through
                                CenterLineNormalizer, resizing with Lanczos
                                interpolation. Set to 0 to disable.
            codec (kraken.codec.PytorchCodec): Codec used to translate code
                                               points. If not given one will be
                                               constructed.
            normalization (str): Unicode normalization for gt
            reorder (bool): Whether to rearrange code points in "display"/LTR
                            order
            pad (int): Padding to add to images left and right
            format (tuple): defines the order of dimensions (0=channels, 1=height,
                          2=width) of samples.
        """
        self.split = lambda x: split(x) + self.suffix
        self.suffix = suffix
        self._images = []
        self._gt = []
        self.alphabet = Counter()
        self.text_transforms = []
        self.transforms = []
        # built text transformations
        if normalization:
            self.text_transforms.append(lambda x: unicodedata.normalize(normalization, x))
        if reorder:
            self.text_transforms.append(bd.get_display)

        # first built image transforms
        if scale:
            if isinstance(scale, int):
                lnorm = CenterNormalizer(scale)
                self.transforms.append(transforms.Lambda(lambda x: x.convert('L')))
                self.transforms.append(transforms.Lambda(lambda x: rpred.dewarp(lnorm, x)))
                self.transforms.append(transforms.Lambda(lambda x: x.convert('L')))
            elif isinstance(scale, tuple):
                self.transforms.append(transforms.Resize(scale, Image.LANCZOS))
        if pad:
            self.transforms.append(transforms.Pad((pad, 0), fill=255))
        self.transforms.append(transforms.ToTensor())
        # invert
        self.transforms.append(transforms.Lambda(lambda x: x.max() - x))
        self.transforms.append(transforms.Lambda(lambda x: x.permute(*format)))
        self.transforms = transforms.Compose(self.transforms)

    def add(self, image):
        """
        Adds a line-image-text pair to the dataset.
        """
        with click.open_file(self.split(image), 'r', encoding='utf-8') as fp:
            gt = fp.read().strip('\n\r')
            for func in self.text_transforms:
                gt = func(gt)
            self.alphabet.update(gt)
        im = Image.open(image)
        im = self.transforms(im)
        self._images.append(im)
        self._gt.append(gt)

    def encode(self, codec=None):
        """
        Adds a codec to the dataset and encodes all text lines.

        Has to be run before sampling from the dataset.
        """
        if codec:
            self.codec = codec
        else:
            self.codec = PytorchCodec(''.join(self.alphabet.keys()))
        self.training_set = []
        for im, gt in zip(self._images, self._gt):
            self.training_set.append((im, self.codec.encode(gt)))

    def __getitem__(self, index):
        return self.training_set[index]

    def __len__(self):
        return len(self.training_set)

