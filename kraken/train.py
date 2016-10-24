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
Utility functions for training CLSTM neural networks.
"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

from future import standard_library
standard_library.install_aliases()
from builtins import str

import os
import click
import numpy as np
import bidi.algorithm as bd

from PIL import Image

from kraken import rpred
from kraken.lib import lstm
from kraken.lib import models
from kraken.lib.util import pil2array, array2pil
from kraken.lib.lineest import CenterNormalizer

class GroundTruthContainer(object):
    """
    Container for ground truth used during training.

    Attributes:
        training_set (list): List of tuples (image, text) for training
        test_set (list): List of tuples (image, text) for testing
        alphabet (str): Sorted string of all codepoint found in the ground
                        truth
    """
    def __init__(self, images=None, split=lambda(x): os.path.splitext(x)[0],
                 suffix='.gt.txt', normalization=None, reorder=True,
                 partition=0.9, pad=16):
        """
        Reads a list of image-text pairs and creates a ground truth set.

        Args:
            images (list): List of file paths of line images
            split (func): Function for generating the base name without
                          extensions from paths
            suffix (str): Suffix to attach to image base name for text
                          retrieval
            normalization (str): Unicode normalization for gt
            reorder (bool): Whether to rearrange code points in "display"/LTR
                            order
            partition (float): Ground truth data partition ratio between
                               train/test set.
            pad (int): Padding to add to images left and right
        """
        self.lnorm = CenterNormalizer()
        self.training_set = []
        self.test_set = []
        self.alphabet = set()

        if not images:
            return
        for line in images:
            self.add(line, split, suffix, normalization, reorder, pad)

        self.repartition(partition)
   
        self.alphabet = sorted(set(''.join(t for _, t in self.training_set)))


    def add(self, image, split=lambda(x): os.path.splitext(x)[0],
                 suffix='.gt.txt', normalization=None, reorder=True,
                 pad=16):
        """
        Adds a single image to the training set.
        """
        with click.open_file(split(image) + suffix, 'r', encoding='utf-8') as fp:
            gt = fp.read()
            if normalization:
                gt = unicodedata.normalize(normalization, gt)
            if reorder:
                gt = bd.get_display(gt)

            im = Image.open(image)
            im = rpred.dewarp(self.lnorm, im)
            im = pil2array(im)
            im = lstm.prepare_line(im, pad)
            self.training_set.append((im, gt))

    def repartition(self, partition=0.9):
        """
        Repartitions the training/test sets.

        Args:
            partition (float): Ground truth data partition ratio between
                               training/test sets.
        """
        self.training_set = self.training_set + self.test_set
        idx = np.random.choice(len(self.training_set), int(len(self.training_set) * partition), replace=False)
        tmp_set = [self.training_set[x] for x in idx]
        [self.training_set.pop(x) for x in sorted(idx, reverse=True)]
        self.test_set = self.training_set
        self.training_set = tmp_set

    def sample(self):
        """
        Samples a line image-text pair from the training set.

        Returns:
            A tuple (line, text) with line being a numpy.array run through
            kraken.lib.lstm.prepare_line.
        """
        return self.training_set[np.random.choice(len(self.training_set))]
