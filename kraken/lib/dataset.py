# -*- coding: utf-8 -*-
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
"""
Utility functions for data loading and training of VGSL networks.
"""
import os
import click
import unicodedata
import numpy as np
import bidi.algorithm as bd

from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset

from kraken.lib.codec import PytorchCodec
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.lineest import CenterNormalizer, dewarp

__all__ = ['GroundTruthDataset', 'compute_error', 'generate_input_transforms']

import logging

logger = logging.getLogger(__name__)


def generate_input_transforms(batch, height, width, channels, pad):
    """
    Generates a torchvision transformation converting a PIL.Image into a
    tensor usable in a network forward pass.

    Args:
        batch (int): mini-batch size
        height (int): height of input image in pixels
        width (int): width of input image in pixels
        channels (int): color channels of input
        pad (int): Amount of padding on horizontal ends of image

    Returns:
        A torchvision transformation composition converting the input image to
        the appropriate tensor.
    """
    if height == 1 and width == 0 and channels > 3:
        perm = (1, 0, 2)
        scale = channels
        mode = 'L'
    # arbitrary (or fixed) height and width and channels 1 or 3 => needs a
    # summarizing network (or a not yet implemented scale operation) to move
    # height to the channel dimension.
    elif height > 1 and width == 0 and channels in (1, 3):
        perm  = (0, 1, 2)
        scale = height
        mode = 'RGB' if channels == 3 else 'L'
    # fixed height and width image => bicubic scaling of the input image, disable padding
    elif height > 0 and width > 0 and channels in (1, 3):
        perm = (0, 1, 2)
        pad = 0
        scale = (height, width)
        mode = 'RGB' if channels == 3 else 'L'
    elif height == 0 and width == 0 and channels in (1, 3):
        perm = (0, 1, 2)
        pad = 0
        scale = 0
        mode = 'RGB' if channels == 3 else 'L'
    else:
        raise KrakenInputException('Invalid input spec (variable height and fixed width not supported)')

    out_transforms = []
    out_transforms.append(transforms.Lambda(lambda x: x.convert(mode)))
    if scale:
        if isinstance(scale, int):
            if mode not in ['1', 'L']:
                raise KrakenInputException('Invalid mode {} for line dewarping'.format(mode))
            lnorm = CenterNormalizer(scale)
            out_transforms.append(transforms.Lambda(lambda x: dewarp(lnorm, x)))
            out_transforms.append(transforms.Lambda(lambda x: x.convert(mode)))
        elif isinstance(scale, tuple):
            out_transforms.append(transforms.Resize(scale, Image.LANCZOS))
    if pad:
        out_transforms.append(transforms.Pad((pad, 0), fill=255))
    out_transforms.append(transforms.ToTensor())
    # invert
    out_transforms.append(transforms.Lambda(lambda x: x.max() - x))
    out_transforms.append(transforms.Lambda(lambda x: x.permute(*perm)))
    return transforms.Compose(out_transforms)


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
                 suffix='.gt.txt', normalization=None, reorder=True,
                 im_transforms=None, preload=True):
        """
        Reads a list of image-text pairs and creates a ground truth set.

        Args:
            split (func): Function for generating the base name without
                          extensions from paths
            suffix (str): Suffix to attach to image base name for text
                          retrieval
            mode (str): Image color space. Either RGB (color) or L
                        (grayscale/bw). Only L is compatible with vertical
                        scaling/dewarping.
            scale (int, tuple): Target height or (width, height) of dewarped
                                line images. Vertical-only scaling is through
                                CenterLineNormalizer, resizing with Lanczos
                                interpolation. Set to 0 to disable.
            normalization (str): Unicode normalization for gt
            reorder (bool): Whether to rearrange code points in "display"/LTR
                            order
            im_transforms (func): Function taking an PIL.Image and returning a
                                  tensor suitable for forward passes.
            preload (bool): Enables preloading and preprocessing of image files.
        """
        self.split = lambda x: split(x) + self.suffix
        self.suffix = suffix
        self._images = []
        self._gt = []
        self.alphabet = Counter()
        self.text_transforms = []
        self.transforms = im_transforms
        self.preload = preload
        # built text transformations
        if normalization:
            self.text_transforms.append(lambda x: unicodedata.normalize(normalization, x))
        if reorder:
            self.text_transforms.append(bd.get_display)

    def add(self, image):
        """
        Adds a line-image-text pair to the dataset.

        Args:
            image (str): Input image path
        """
        with click.open_file(self.split(image), 'r', encoding='utf-8') as fp:
            gt = fp.read().strip('\n\r')
            for func in self.text_transforms:
                gt = func(gt)
        if self.preload:
            im = Image.open(image)
            try:
                im = self.transforms(im)
            except ValueError as e:
                raise KrakenInputException('Image transforms failed on {}'.format(image))
            self._images.append(im)
        else:
            self._images.append(image)
        self._gt.append(gt)
        self.alphabet.update(gt)

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
        if self.preload:
            return self.training_set[index]
        else:
            item = self.training_set[index]
            try:
                logger.debug('Attempting to load {}'.format(item[0]))
                return (self.transforms(Image.open(item[0])), item[1])
            except Exception:
                idx = np.random.randint(0, len(self.training_set))
                logger.debug('Failed. Replacing with sample {}'.format(idx))
                return self[np.random.randint(0, len(self.training_set))]

    def __len__(self):
        return len(self.training_set)
