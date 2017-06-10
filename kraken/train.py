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
import unicodedata
import bidi.algorithm as bd

from PIL import Image
from collections import Counter

from kraken import rpred
from kraken.lib import lstm
from kraken.lib import models
from kraken.lib.util import pil2array, array2pil
from kraken.lib.lineest import CenterNormalizer

__all__ = ['GroundTruthContainer']

def _fast_levenshtein(seq1, seq2):

    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    for x in xrange(len(seq1)):
        oneago, thisrow = thisrow, [0] * len(seq2) + [x + 1]
        for y in xrange(len(seq2)):
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
        test_set (list): List of tuples (imae, text) for testing

    Returns:
        A tuple with total number of characters and edit distance across the
        whole test set.
    """
    total_chars = 0
    error = 0
    for im, text in test_set:
        pred = model.predictString(im)
        total_chars += len(text)
        error += _fast_levenshtein(pred, text)
    return total_chars, error


class GroundTruthContainer(object):
    """
    Container for ground truth used during training.

    Attributes:
        training_set (list): List of tuples (image, text) for training
        test_set (list): List of tuples (image, text) for testing
        alphabet (str): Sorted string of all codepoint found in the ground
                        truth
    """
    def __init__(self, images=None, split=lambda x: os.path.splitext(x)[0],
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
        self.training_alphabet = Counter()
        self.test_alphabet = Counter()

        if not images:
            return
        for line in images:
            self.add(line, split, suffix, normalization, reorder, pad)

        self.repartition(partition)

    def add(self, image, split=lambda x: os.path.splitext(x)[0],
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

        self.training_alphabet = Counter(''.join(t for _, t in self.training_set))
        self.test_alphabet = Counter(''.join(t for _, t in self.test_set))

    def sample(self):
        """
        Samples a line image-text pair from the training set.

        Returns:
            A tuple (line, text) with line being a numpy.array run through
            kraken.lib.lstm.prepare_line.
        """
        return self.training_set[np.random.choice(len(self.training_set))]
