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
Named functions for all the transforms that were lambdas in the past to
facilitate pickling.
"""
import regex
import unicodedata
import bidi.algorithm as bd

from os import path
from PIL import Image

from kraken.binarization import nlbin
from kraken.lib.lineest import dewarp


def pil_to_mode(im, mode):
    return im.convert(mode)


def pil_to_bin(im):
    return nlbin(im)


def dummy(x):
    return x


def pil_dewarp(im, lnorm):
    return dewarp(lnorm, im)


def pil_fixed_resize(im, scale):
    return _fixed_resize(im, scale, Image.LANCZOS)


def tensor_invert(im):
    return im.max() - im


def tensor_permute(im, perm):
    return im.permute(*perm)


def _fixed_resize(img, size, interpolation=Image.LANCZOS):
    """
    Doesn't do the annoying runtime scale dimension switching the default
    pytorch transform does.

    Args:
        img (PIL.Image): image to resize
        size (tuple): Tuple (height, width)
    """
    w, h = img.size
    oh, ow = size
    if oh == 0:
        oh = int(h * ow/w)
    elif ow == 0:
        ow = int(w * oh/h)
    img = img.resize((ow, oh), interpolation)
    return img


def text_normalize(text, normalization):
    return unicodedata.normalize(normalization, text)


def text_whitespace_normalize(text):
    return regex.sub(r'\s', ' ', text).strip()


def text_reorder(text, base_dir=None):
    return bd.get_display(text, base_dir=base_dir)


def default_split(x):
    return path.splitext(x)[0]


def suffix_split(x, split, suffix):
    return split(x) + suffix
