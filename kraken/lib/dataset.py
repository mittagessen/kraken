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
import json
import regex
import torch
import unicodedata
import numpy as np
import pkg_resources
import bidi.algorithm as bd
import torchvision.transforms.functional as tf

from os import path
from PIL import Image, ImageDraw
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Sequence, Callable, Optional, Any, Union, cast

from kraken.lib.xml import parse_alto, parse_page
from kraken.lib.codec import PytorchCodec
from kraken.lib.models import TorchSeqRecognizer
from kraken.lib.segmentation import extract_polygons
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.lineest import CenterNormalizer, dewarp

__all__ = ['BaselineSet', 'GroundTruthDataset', 'compute_error', 'generate_input_transforms']

import logging

logger = logging.getLogger(__name__)


def generate_input_transforms(batch: int, height: int, width: int, channels: int, pad: int, valid_norm: bool = True) -> transforms.Compose:
    """
    Generates a torchvision transformation converting a PIL.Image into a
    tensor usable in a network forward pass.

    Args:
        batch (int): mini-batch size
        height (int): height of input image in pixels
        width (int): width of input image in pixels
        channels (int): color channels of input
        pad (int): Amount of padding on horizontal ends of image
        valid_norm (bool): Enables/disables baseline normalization as a valid
                           preprocessing step. If disabled we will fall back to
                           standard scaling.

    Returns:
        A torchvision transformation composition converting the input image to
        the appropriate tensor.
    """
    scale = (height, width) # type: Tuple[int, int]
    center_norm = False
    mode = 'RGB' if channels == 3 else 'L'
    if height == 1 and width == 0 and channels > 3:
        perm = (1, 0, 2)
        scale = (channels, 0)
        if valid_norm:
            center_norm = True
        mode = 'L'
    elif height > 1 and width == 0 and channels in (1, 3):
        perm = (0, 1, 2)
        if valid_norm and channels == 1:
            center_norm = True
    elif height == 0 and width > 1 and channels in (1, 3):
        perm = (0, 1, 2)
    # fixed height and width image => bicubic scaling of the input image, disable padding
    elif height > 0 and width > 0 and channels in (1, 3):
        perm = (0, 1, 2)
        pad = 0
    elif height == 0 and width == 0 and channels in (1, 3):
        perm = (0, 1, 2)
        pad = 0
    else:
        raise KrakenInputException('Invalid input spec {}, {}, {}, {}, {}'.format(batch,
                                                                                  height,
                                                                                  width,
                                                                                  channels,
                                                                                  pad))

    out_transforms = []
    out_transforms.append(transforms.Lambda(lambda x: x.convert(mode)))

    if scale != (0, 0):
        if center_norm:
            lnorm = CenterNormalizer(scale[0])
            out_transforms.append(transforms.Lambda(lambda x: dewarp(lnorm, x)))
            out_transforms.append(transforms.Lambda(lambda x: x.convert(mode)))
        else:
            out_transforms.append(transforms.Lambda(lambda x: _fixed_resize(x, scale, Image.LANCZOS)))
    if pad:
        out_transforms.append(transforms.Pad((pad, 0), fill=255))
    out_transforms.append(transforms.ToTensor())
    # invert
    out_transforms.append(transforms.Lambda(lambda x: x.max() - x))
    out_transforms.append(transforms.Lambda(lambda x: x.permute(*perm)))
    return transforms.Compose(out_transforms)

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
    return img.resize((ow, oh), interpolation)

def _fast_levenshtein(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:

    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    rows = [thisrow]
    for x in range(len(seq1)):
        oneago, thisrow = thisrow, [0] * len(seq2) + [x + 1]
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
        rows.append(thisrow)
    return thisrow[len(seq2) - 1]


def global_align(seq1: Sequence[Any], seq2: Sequence[Any]) -> Tuple[int, List[str], List[str]]:
    """
    Computes a global alignment of two strings.

    Args:
        seq1 (Sequence[Any]):
        seq2 (Sequence[Any]):

    Returns a tuple (distance, list(algn1), list(algn2))
    """
    # calculate cost and direction matrix
    cost = [[0] * (len(seq2) + 1) for x in range(len(seq1) + 1)]
    for i in range(1, len(cost)):
        cost[i][0] = i
    for i in range(1, len(cost[0])):
        cost[0][i] = i
    direction = [[(0, 0)] * (len(seq2) + 1) for x in range(len(seq1) + 1)]
    direction[0] = [(0, x) for x in range(-1, len(seq2))]
    for i in range(-1, len(direction) - 1):
        direction[i + 1][0] = (i, 0)
    for i in range(1, len(cost)):
        for j in range(1, len(cost[0])):
            delcost = ((i - 1, j), cost[i - 1][j] + 1)
            addcost = ((i, j - 1), cost[i][j - 1] + 1)
            subcost = ((i - 1, j - 1), cost[i - 1][j - 1] + (seq1[i - 1] != seq2[j - 1]))
            best = min(delcost, addcost, subcost, key=lambda x: x[1])
            cost[i][j] = best[1]
            direction[i][j] = best[0]
    d = cost[-1][-1]
    # backtrace
    algn1: List[Any] = []
    algn2: List[Any] = []
    i = len(direction) - 1
    j = len(direction[0]) - 1
    while direction[i][j] != (-1, 0):
        k, l = direction[i][j]
        if k == i - 1 and l == j - 1:
            algn1.insert(0, seq1[i - 1])
            algn2.insert(0, seq2[j - 1])
        elif k < i:
            algn1.insert(0, seq1[i - 1])
            algn2.insert(0, '')
        elif l < j:
            algn1.insert(0, '')
            algn2.insert(0, seq2[j - 1])
        i, j = k, l
    return d, algn1, algn2


def compute_confusions(algn1: Sequence[str], algn2: Sequence[str]):
    """
    Compute confusion matrices from two globally aligned strings.

    Args:
        align1 (Sequence[str]): sequence 1
        align2 (Sequence[str]): sequence 2

    Returns:
        A tuple (counts, scripts, ins, dels, subs) with `counts` being per-character
        confusions, `scripts` per-script counts, `ins` a dict with per script
        insertions, `del` an integer of the number of deletions, `subs` per
        script substitutions.
    """
    counts: Dict[Tuple[str, str], int] = Counter()
    with pkg_resources.resource_stream(__name__, 'scripts.json') as fp:
        script_map = json.load(fp)

    def _get_script(c):
        for s, e, n in script_map:
            if ord(c) == s or (e and s <= ord(c) <= e):
                return n
        return 'Unknown'

    scripts: Dict[Tuple[str, str], int] = Counter()
    ins: Dict[Tuple[str, str], int] = Counter()
    dels: int = 0
    subs: Dict[Tuple[str, str], int] = Counter()
    for u,v in zip(algn1, algn2):
        counts[(u, v)] += 1
    for k, v in counts.items():
        if k[0] == '':
            dels += v
        else:
            script = _get_script(k[0])
            scripts[script] += v
            if k[1] == '':
                ins[script] += v
            elif k[0] != k[1]:
                subs[script] += v
    return counts, scripts, ins, dels, subs

def compute_error(model: TorchSeqRecognizer, validation_set: Sequence[Tuple[str, str]]) -> Tuple[int, int]:
    """
    Computes error report from a model and a list of line image-text pairs.

    Args:
        model (kraken.lib.models.TorchSeqRecognizer): Model used for recognition
        validation_set (list): List of tuples (image, text) for validation

    Returns:
        A tuple with total number of characters and edit distance across the
        whole validation set.
    """
    total_chars = 0
    error = 0
    for im, text in validation_set:
        pred = model.predict_string(im)
        total_chars += len(text)
        error += _fast_levenshtein(pred, text)
    return total_chars, error


def preparse_xml_data(filenames, format_type):
    """
    Loads training data from a set of xml files.

    Extracts line information from Page/ALTO xml files for training purposes.

    Args:
        filenames (list): List of XML files.
        format_type (str): Either `page` or `alto`

    Returns:
        A list of dicts {'text': text, 'baseline': [[x0, y0], ...], 'boundary':
        [[x0, y0], ...], 'image': PIL.Image}.
    """
    training_pairs = []
    if format_type == 'alto':
        parse_fn = parse_alto
    elif format_type == 'page':
        parse_fn = parse_page
    else:
        raise Exception('invalid format {} for preparse_xml_data'.format(format_type))
    for fn in filenames:
        try:
            data = parse_fn(fn)
        except KrakenInputException as e:
            logger.warning(e)
            continue
        try:
            Image.open(data['image'])
        except FileNotFoundError as e:
            logger.warning('Could not open file {} in {}'.format(e.filename, fn))
            continue
        for line in data['lines']:
            training_pairs.append({'image': data['image'], **line})
    return training_pairs


class PolygonGTDataset(Dataset):
    """
    Dataset for training a line recognition model from polygonal/baseline data.
    """
    def __init__(self,
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 reorder: bool = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 preload: bool = True) -> None:
        self._images = []  # type:  Union[List[Image], List[torch.Tensor]]
        self._gt = []  # type:  List[str]
        self.alphabet = Counter()  # type: Counter
        self.text_transforms = []  # type: List[Callable[[str], str]]
        self.transforms = im_transforms
        self.preload = preload
        # built text transformations
        if normalization:
            self.text_transforms.append(lambda x: unicodedata.normalize(cast(str, normalization), x))
        if whitespace_normalization:
            self.text_transforms.append(lambda x: regex.sub('\s', ' ', x).strip())
        if reorder:
            self.text_transforms.append(bd.get_display)

    def add(self, image: Union[str, Image.Image], text: str, baseline: List[Tuple[int, int]], boundary: List[Tuple[int, int]]):
        """
        Adds a line to the dataset.

        Args:
            im (path): Path to the whole page image
            text (str): Transcription of the line.
            baseline (list): A list of coordinates [[x0, y0], ..., [xn, yn]].
            boundary (list): A polygon mask for the line.
        """
        for func in self.text_transforms:
            text = func(text)
            if not text:
                raise KrakenInputException('Text line is empty after transformations')
        if self.preload:
            if not isinstance(image, Image.Image):
                im = Image.open(image)
            im, _ = next(extract_polygons(im, {'type': 'baselines', 'lines': [{'baseline': baseline, 'boundary': boundary}]}))
            try:
                im = self.transforms(im)
            except ValueError as e:
                raise KrakenInputException('Image transforms failed on {}'.format(image))
            self._images.append(im)
        else:
            self._images.append((image, baseline, boundary))
        self._gt.append(text)
        self.alphabet.update(text)

    def encode(self, codec: Optional[PytorchCodec] = None) -> None:
        """
        Adds a codec to the dataset and encodes all text lines.

        Has to be run before sampling from the dataset.
        """
        if codec:
            self.codec = codec
        else:
            self.codec = PytorchCodec(''.join(self.alphabet.keys()))
        self.training_set = []  # type: List[Tuple[Union[Image, torch.Tensor], torch.Tensor]]
        for im, gt in zip(self._images, self._gt):
            self.training_set.append((im, self.codec.encode(gt)))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.preload:
            return self.training_set[index]
        else:
            item = self.training_set[index]
            try:
                logger.debug('Attempting to load {}'.format(item[0]))
                im = item[0][0]
                if not isinstance(im, Image.Image):
                    im = Image.open(im)
                im, _ = next(extract_polygons(im, {'type': 'baselines', 'lines': [{'baseline': item[0][1], 'boundary': item[0][2]}]}))
                return (self.transforms(im), item[1])
            except Exception:
                idx = np.random.randint(0, len(self.training_set))
                logger.debug('Failed. Replacing with sample {}'.format(idx))
                return self[np.random.randint(0, len(self.training_set))]

    def __len__(self) -> int:
        return len(self.training_set)


class GroundTruthDataset(Dataset):
    """
    Dataset for training a line recognition model.

    All data is cached in memory.
    """
    def __init__(self, split: Callable[[str], str] = lambda x: path.splitext(x)[0],
                 suffix: str = '.gt.txt',
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 reorder: bool = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 preload: bool = True) -> None:
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
            whitespace_normalization (str): Normalizes unicode whitespace and
                                            strips whitespace.
            reorder (bool): Whether to rearrange code points in "display"/LTR
                            order
            im_transforms (func): Function taking an PIL.Image and returning a
                                  tensor suitable for forward passes.
            preload (bool): Enables preloading and preprocessing of image files.
        """
        self.suffix = suffix
        self.split = lambda x: split(x) + self.suffix
        self._images = []  # type:  Union[List[Image], List[torch.Tensor]]
        self._gt = []  # type:  List[str]
        self.alphabet = Counter()  # type: Counter
        self.text_transforms = []  # type: List[Callable[[str], str]]
        self.transforms = im_transforms
        self.preload = preload
        # built text transformations
        if normalization:
            self.text_transforms.append(lambda x: unicodedata.normalize(cast(str, normalization), x))
        if whitespace_normalization:
            self.text_transforms.append(lambda x: regex.sub('\s', ' ', x).strip())
        if reorder:
            self.text_transforms.append(bd.get_display)

    def add(self, image: Union[str, Image.Image]) -> None:
        """
        Adds a line-image-text pair to the dataset.

        Args:
            image (str): Input image path
        """
        with open(self.split(image), 'r', encoding='utf-8') as fp:
            gt = fp.read().strip('\n\r')
            for func in self.text_transforms:
                gt = func(gt)
            if not gt:
                raise KrakenInputException('Text line is empty ({})'.format(fp.name))
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

    def add_loaded(self, image: Image.Image, gt: str) -> None:
        """
        Adds an already loaded line-image-text pair to the dataset.

        Args:
            image (PIL.Image.Image): Line image
            gt (str): Text contained in the line image
        """
        if self.preload:
            try:
                im = self.transforms(image)
            except ValueError as e:
                raise KrakenInputException('Image transforms failed on {}'.format(image))
            self._images.append(im)
        else:
            self._images.append(image)
        for func in self.text_transforms:
            gt = func(gt)
        self._gt.append(gt)
        self.alphabet.update(gt)

    def encode(self, codec: Optional[PytorchCodec] = None) -> None:
        """
        Adds a codec to the dataset and encodes all text lines.

        Has to be run before sampling from the dataset.
        """
        if codec:
            self.codec = codec
        else:
            self.codec = PytorchCodec(''.join(self.alphabet.keys()))
        self.training_set = []  # type: List[Tuple[Union[Image, torch.Tensor], torch.Tensor]]
        for im, gt in zip(self._images, self._gt):
            self.training_set.append((im, self.codec.encode(gt)))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.preload:
            return self.training_set[index]
        else:
            item = self.training_set[index]
            try:
                logger.debug('Attempting to load {}'.format(item[0]))
                im = item[0]
                if not isinstance(im, Image.Image):
                    im = Image.open(im)
                return (self.transforms(im), item[1])
            except Exception:
                idx = np.random.randint(0, len(self.training_set))
                logger.debug('Failed. Replacing with sample {}'.format(idx))
                return self[np.random.randint(0, len(self.training_set))]

    def __len__(self) -> int:
        return len(self.training_set)


class BaselineSet(Dataset):
    """
    Dataset for training a baseline recognition model.
    """
    def __init__(self, imgs: Sequence[str] = None,
                 suffix: str = '.path',
                 line_width: int = 4,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 mode: str = 'path'):
        """
        Reads a list of image-json pairs and creates a data set.

        Args:
            imgs (list):
            suffix (int): Suffix to attach to image base name to load JSON
                          files from.
            line_width (int): Height of the baseline in the scaled input.
            target_size (tuple): Target size of the image as a (height, width) tuple.
            mode (str): Either path, alto, page, or None. In alto and page mode
                        the baseline paths and image data is retrieved from an
                        ALTO/PageXML file. In `None` mode data is iteratively
                        added through the `add` method.
        """
        super().__init__()
        self.mode = mode
        if mode in ['alto', 'page']:
            if mode == 'alto':
                fn = parse_alto
            else:
                fn = parse_page
            im_paths = []
            self.targets = []
            for img in imgs:
                data = fn(img)
                im_paths.append(data['image'])
                self.targets.append([line['baseline'] for line in data['lines']])
            imgs = im_paths
        elif mode == 'path':
            pass
        elif mode is None:
            imgs = []
            self.targets = []
        else:
            raise Exception('invalid dataset mode')
        self.imgs = imgs
        self.line_width = line_width
        self.im_transforms = im_transforms

    def add(self, image: Union[str, Image.Image], baselines: List[List[Tuple[int, int]]]):
        """
        Adds a line to the dataset.

        Args:
            im (path): Path to the whole page image
            baseline (list): A list of lists of coordinates [[x0, y0], ..., [xn, yn]]].
        """
        if self.mode:
            raise Exception('The `add` method is incompatible with dataset mode {}'.format(self.mode))
        self.imgs.append(image)
        self.targets.append(baselines)

    def __getitem__(self, idx):
        im = self.imgs[idx]
        if self.mode != 'path':
            target = self.targets[idx]
        else:
            with open('{}.path'.format(path.splitext(im)[0]), 'r') as fp:
                target = json.load(fp)
        if not isinstance(im, Image.Image):
            try:
                logger.debug('Attempting to load {}'.format(im))
                im = Image.open(im)
                return self.transform(im, target)
            except Exception:
                idx = np.random.randint(0, len(self.imgs))
                logger.debug('Failed. Replacing with sample {}'.format(idx))
                return self[np.random.randint(0, len(self.imgs))]
        return self.transform(im, target)

    @staticmethod
    def _get_ortho_line(lineseg, point):
        lineseg = np.array(lineseg)
        norm_vec = lineseg[1,...] - lineseg[0,...]
        norm_vec_len = np.sqrt(np.sum(norm_vec**2))
        unit_vec = norm_vec / norm_vec_len
        ortho_vec = unit_vec[::-1] * ((1,-1), (-1,1))
        return (ortho_vec * 10 + point).astype('int').tolist()

    def transform(self, image, target):
        orig_size = image.size
        image = self.im_transforms(image)
        scale = image.shape[2]/orig_size[0]
        t = Image.new('L', image.shape[:0:-1])
        line_mask = ImageDraw.Draw(t)
        s = Image.new('L', image.shape[:0:-1])
        separator_mask = ImageDraw.Draw(s)
        lines = []
        for line in target:
            l = []
            for point in line:
                l.append((int(point[0]*scale), int(point[1]*scale)))
            line_mask.line(l, fill=255, width=self.line_width)
            sep_1 = [tuple(x) for x in self._get_ortho_line(l[:2], l[0])]
            separator_mask.line(sep_1, fill=255, width=self.line_width)
            sep_2 = [tuple(x) for x in self._get_ortho_line(l[-2:], l[-1])]
            separator_mask.line(sep_2, fill=255, width=self.line_width)
        del line_mask
        del separator_mask
        target = np.array(t)
        separator = np.array(s)
        target = tf.to_tensor(Image.fromarray(target)).long()
        separator = tf.to_tensor(Image.fromarray(separator)).long()
        target[separator != 0] = 2
        # squeeze away channel dimension for NLLLoss
        target = target.squeeze()
        return image, target

    def __len__(self):
        return len(self.imgs)

