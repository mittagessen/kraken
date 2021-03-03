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
import traceback
import unicodedata
import numpy as np
import pkg_resources
import bidi.algorithm as bd
import shapely.geometry as geom
import torch.nn.functional as F
import torchvision.transforms.functional as tf

from os import path
from shapely.ops import split, snap
from PIL import Image, ImageDraw
from itertools import groupby
from collections import Counter, defaultdict
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Iterable, Sequence, Callable, Optional, Any, Union, cast

from skimage.draw import polygon

from kraken.lib.xml import parse_alto, parse_page, parse_xml
from kraken.lib.util import is_bitonal
from kraken.lib.codec import PytorchCodec
from kraken.lib.models import TorchSeqRecognizer
from kraken.lib.segmentation import extract_polygons, calculate_polygonal_environment
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.lineest import CenterNormalizer, dewarp

__all__ = ['BaselineSet', 'PolygonGTDataset', 'GroundTruthDataset', 'compute_error', 'generate_input_transforms', 'preparse_xml_data']

import logging

logger = logging.getLogger(__name__)


def generate_input_transforms(batch: int, height: int, width: int, channels: int, pad: int, valid_norm: bool = True, force_binarization=False) -> transforms.Compose:
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
        force_binarization (bool): Forces binarization of input images using
                                   the nlbin algorithm.

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
    if mode != 'L' and force_binarization:
        raise KrakenInputException('Invalid input spec {}, {}, {}, {} in'
                                   ' combination with forced binarization.'.format(batch,
                                                                                   height,
                                                                                   width,
                                                                                   channels,
                                                                                   pad))

    out_transforms = []
    out_transforms.append(transforms.Lambda(lambda x: x.convert(mode)))

    if force_binarization:
        out_transforms.append(transforms.Lambda(lambda x: nlbin(im)))
    # dummy transforms to ensure we can determine color mode of input material
    # from first two transforms. It's stupid but it works.
    out_transforms.append(transforms.Lambda(lambda x: x))
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
    img = img.resize((ow, oh), interpolation)
    return img


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

def compute_error(model: TorchSeqRecognizer, validation_set: Iterable[Dict[str, torch.Tensor]]) -> Tuple[int, int]:
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
    for batch in validation_set:
        preds = model.predict_string(batch['image'], batch['seq_lens'])
        total_chars += batch['target_lens'].sum()
        for pred, text in zip(preds, batch['target']):
            error += _fast_levenshtein(pred, text)
    return total_chars, error


def preparse_xml_data(filenames, format_type='xml', repolygonize=False):
    """
    Loads training data from a set of xml files.

    Extracts line information from Page/ALTO xml files for training of
    recognition models.

    Args:
        filenames (list): List of XML files.
        format_type (str): Either `page`, `alto` or `xml` for
                           autodetermination.
        repolygonize (bool): (Re-)calculates polygon information using the
                             kraken algorithm.

    Returns:
        A list of dicts {'text': text, 'baseline': [[x0, y0], ...], 'boundary':
        [[x0, y0], ...], 'image': PIL.Image}.
    """
    training_pairs = []
    if format_type == 'xml':
        parse_fn = parse_xml
    elif format_type == 'alto':
        parse_fn = parse_alto
    elif format_type == 'page':
        parse_fn = parse_page
    else:
        raise Exception(f'invalid format {format_type} for preparse_xml_data')
    for fn in filenames:
        try:
            data = parse_fn(fn)
        except KrakenInputException as e:
            logger.warning(e)
            continue
        try:
            with open(data['image'], 'rb') as fp:
                Image.open(fp)
        except FileNotFoundError as e:
            logger.warning(f'Could not open file {e.filename} in {fn}')
            continue
        if repolygonize:
            logger.info('repolygonizing {} lines in {}'.format(len(data['lines']), data['image']))
            data['lines'] = _repolygonize(data['image'], data['lines'])
        for line in data['lines']:
            training_pairs.append({'image': data['image'], **line})
    return training_pairs


def _repolygonize(im: Image.Image, lines):
    """
    Helper function taking an output of the lib.xml parse_* functions and
    recalculating the contained polygonization.

    Args:
        im (Image.Image): Input image
        lines (list): List of dicts [{'boundary': [[x0, y0], ...], 'baseline': [[x0, y0], ...], 'text': 'abcvsd'}, {...]

    Returns:
        A data structure `lines` with a changed polygonization.
    """
    im = Image.open(im).convert('L')
    polygons = calculate_polygonal_environment(im, [x['baseline'] for x in lines])
    return [{'boundary': polygon, 'baseline': orig['baseline'], 'text': orig['text'], 'script': orig['script']} for orig, polygon in zip(lines, polygons)]


def collate_sequences(batch):
    """
    Sorts and pads sequences.
    """
    sorted_batch = sorted(batch, key=lambda x: x['image'].shape[2], reverse=True)
    seqs = [x['image'] for x in sorted_batch]
    seq_lens = torch.LongTensor([seq.shape[2] for seq in seqs])
    max_len = seqs[0].shape[2]
    seqs = torch.stack([F.pad(seq, pad=(0, max_len-seq.shape[2])) for seq in seqs])
    if isinstance(sorted_batch[0]['target'], str):
        labels = [x['target'] for x in sorted_batch]
    else:
        labels = torch.cat([x['target'] for x in sorted_batch]).long()
    label_lens = torch.LongTensor([len(x['target']) for x in sorted_batch])
    return {'image': seqs, 'target': labels, 'seq_lens': seq_lens, 'target_lens': label_lens}


class InfiniteDataLoader(DataLoader):
    """
    Version of DataLoader that auto-reinitializes the iterator once it is
    exhausted.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iter = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            sample = next(self.dataset_iter)
        except StopIteration:
            self.dataset_iter = super().__iter__()
            sample = next(self.dataset_iter)
        return sample


class PolygonGTDataset(Dataset):
    """
    Dataset for training a line recognition model from polygonal/baseline data.
    """
    def __init__(self,
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 reorder: bool = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 preload: bool = True,
                 augmentation: bool = False) -> None:
        self._images = []  # type:  Union[List[Image], List[torch.Tensor]]
        self._gt = []  # type:  List[str]
        self.alphabet = Counter()  # type: Counter
        self.text_transforms = []  # type: List[Callable[[str], str]]
        # split image transforms into two. one part giving the final PIL image
        # before conversion to a tensor and the actual tensor conversion part.
        self.head_transforms = transforms.Compose(im_transforms.transforms[:2])
        self.tail_transforms = transforms.Compose(im_transforms.transforms[2:])
        self.transforms = im_transforms
        self.preload = preload
        self.aug = None

        self.seg_type = 'baselines'
        # built text transformations
        if normalization:
            self.text_transforms.append(lambda x: unicodedata.normalize(cast(str, normalization), x))
        if whitespace_normalization:
            self.text_transforms.append(lambda x: regex.sub('\s', ' ', x).strip())
        if reorder:
            self.text_transforms.append(bd.get_display)
        if augmentation:
            from albumentations import (
                Compose, ToFloat, FromFloat, Flip, OneOf, MotionBlur, MedianBlur, Blur,
                ShiftScaleRotate, OpticalDistortion, ElasticTransform, RandomBrightnessContrast,
                )

            self.aug = Compose([
                                ToFloat(),
                                OneOf([
                                    MotionBlur(p=0.2),
                                    MedianBlur(blur_limit=3, p=0.1),
                                    Blur(blur_limit=3, p=0.1),
                                ], p=0.2),
                                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=3, p=0.2),
                                OneOf([
                                    OpticalDistortion(p=0.3),
                                    ElasticTransform(p=0.1),
                                ], p=0.2),
                               ], p=0.5)

        self.im_mode = '1'

    def add(self, image: Union[str, Image.Image], text: str, baseline: List[Tuple[int, int]], boundary: List[Tuple[int, int]], *args, **kwargs):
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
        if not baseline:
            raise KrakenInputException('No baseline given for line')
        if not boundary:
            raise KrakenInputException('No boundary given for line')
        if self.preload:
            if not isinstance(image, Image.Image):
                im = Image.open(image)
            try:
                im, _ = next(extract_polygons(im, {'type': 'baselines', 'lines': [{'baseline': baseline, 'boundary': boundary}]}))
            except IndexError:
                raise KrakenInputException('Patch extraction failed for baseline')
            try:
                im = self.head_transforms(im)
                if not is_bitonal(im):
                    self.im_mode = im.mode
                im = self.tail_transforms(im)
            except ValueError:
                raise KrakenInputException(f'Image transforms failed on {image}')
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

    def no_encode(self) -> None:
        """
        Creates an unencoded dataset.
        """
        self.training_set = []  # type: List[Tuple[Union[Image, torch.Tensor], str]]
        for im, gt in zip(self._images, self._gt):
            self.training_set.append((im, gt))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.preload:
            x, y = self.training_set[index]
            if self.aug:
                x = x.permute((1, 2, 0)).numpy()
                o = self.aug(image=x)
                x = torch.tensor(o['image'].transpose(2, 0, 1))
            return {'image': x, 'target': y}
        else:
            item = self.training_set[index]
            try:
                logger.debug(f'Attempting to load {item[0]}')
                im = item[0][0]
                if not isinstance(im, Image.Image):
                    im = Image.open(im)
                im, _ = next(extract_polygons(im, {'type': 'baselines', 'lines': [{'baseline': item[0][1], 'boundary': item[0][2]}]}))
                im = self.head_transforms(im)
                if not is_bitonal(im):
                    self.im_mode = im.mode
                im = self.tail_transforms(im)
                if self.aug:
                    im = im.permute((1, 2, 0)).numpy()
                    o = self.aug(image=im)
                    im = torch.tensor(o['image'].transpose(2, 0, 1))
                return {'image': im, 'target': item[1]}
            except Exception:
                idx = np.random.randint(0, len(self.training_set))
                logger.debug(traceback.format_exc())
                logger.info(f'Failed. Replacing with sample {idx}')
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
                 preload: bool = True,
                 augmentation: bool = False) -> None:
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
        # split image transforms into two. one part giving the final PIL image
        # before conversion to a tensor and the actual tensor conversion part.
        self.head_transforms = transforms.Compose(im_transforms.transforms[:2])
        self.tail_transforms = transforms.Compose(im_transforms.transforms[2:])
        self.aug = None

        self.preload = preload
        self.seg_type = 'bbox'
        # built text transformations
        if normalization:
            self.text_transforms.append(lambda x: unicodedata.normalize(cast(str, normalization), x))
        if whitespace_normalization:
            self.text_transforms.append(lambda x: regex.sub('\s', ' ', x).strip())
        if reorder:
            self.text_transforms.append(bd.get_display)
        if augmentation:
            from albumentations import (
                Compose, ToFloat, FromFloat, Flip, OneOf, MotionBlur, MedianBlur, Blur,
                ShiftScaleRotate, OpticalDistortion, ElasticTransform, RandomBrightnessContrast,
                )

            self.aug = Compose([
                                ToFloat(),
                                OneOf([
                                    MotionBlur(p=0.2),
                                    MedianBlur(blur_limit=3, p=0.1),
                                    Blur(blur_limit=3, p=0.1),
                                ], p=0.2),
                                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                                OneOf([
                                    OpticalDistortion(p=0.3),
                                    ElasticTransform(p=0.1),
                                ], p=0.2),
                               ], p=0.5)

        self.im_mode = '1'

    def add(self, image: Union[str, Image.Image], *args, **kwargs) -> None:
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
                raise KrakenInputException(f'Text line is empty ({fp.name})')
        if self.preload:
            try:
                im = Image.open(image)
                im = self.head_transforms(im)
                if not is_bitonal(im):
                    self.im_mode = im.mode
                im = self.tail_transforms(im)
            except ValueError:
                raise KrakenInputException(f'Image transforms failed on {image}')
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
                im = self.head_transforms(im)
                if not is_bitonal(im):
                    self.im_mode = im.mode
                im = self.tail_transforms(im)
            except ValueError:
                raise KrakenInputException(f'Image transforms failed on {image}')
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

    def no_encode(self) -> None:
        """
        Creates an unencoded dataset.
        """
        self.training_set = []  # type: List[Tuple[Union[Image, torch.Tensor], str]]
        for im, gt in zip(self._images, self._gt):
            self.training_set.append((im, gt))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.preload:
            x, y = self.training_set[index]
            if self.aug:
                im = x.permute((1, 2, 0)).numpy()
                o = self.aug(image=im)
                im = torch.tensor(o['image'].transpose(2, 0, 1))
                return {'image': im, 'target': y}
            return {'image': x, 'target': y}
        else:
            item = self.training_set[index]
            try:
                logger.debug(f'Attempting to load {item[0]}')
                im = item[0]
                if not isinstance(im, Image.Image):
                    im = Image.open(im)
                im = self.head_transforms(im)
                if not is_bitonal(im):
                    self.im_mode = im.mode
                im = self.tail_transforms(im)
                if self.aug:
                    im = im.permute((1, 2, 0)).numpy()
                    o = self.aug(image=im)
                    im = torch.tensor(o['image'].transpose(2, 0, 1))
                return {'image': im, 'target': item[1]}
            except Exception:
                idx = np.random.randint(0, len(self.training_set))
                logger.debug(traceback.format_exc())
                logger.info(f'Failed. Replacing with sample {idx}')
                return self[np.random.randint(0, len(self.training_set))]

    def __len__(self) -> int:
        return len(self.training_set)


class BaselineSet(Dataset):
    """
    Dataset for training a baseline/region segmentation model.
    """
    def __init__(self, imgs: Sequence[str] = None,
                 suffix: str = '.path',
                 line_width: int = 4,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 mode: str = 'path',
                 augmentation: bool = False,
                 valid_baselines: Sequence[str] = None,
                 merge_baselines: Dict[str, Sequence[str]] = None,
                 valid_regions: Sequence[str] = None,
                 merge_regions: Dict[str, Sequence[str]] = None):
        """
        Reads a list of image-json pairs and creates a data set.

        Args:
            imgs (list):
            suffix (int): Suffix to attach to image base name to load JSON
                          files from.
            line_width (int): Height of the baseline in the scaled input.
            target_size (tuple): Target size of the image as a (height, width) tuple.
            mode (str): Either path, alto, page, xml, or None. In alto, page,
                        and xml mode the baseline paths and image data is
                        retrieved from an ALTO/PageXML file. In `None` mode
                        data is iteratively added through the `add` method.
            augmentation (bool): Enable/disable augmentation.
            valid_baselines (list): Sequence of valid baseline identifiers. If
                                    `None` all are valid.
            merge_baselines (dict): Sequence of baseline identifiers to merge.
                                    Note that merging occurs after entities not
                                    in valid_* have been discarded.
            valid_regions (list): Sequence of valid region identifiers. If
                                  `None` all are valid.
            merge_regions (dict): Sequence of region identifiers to merge.
                                  Note that merging occurs after entities not
                                  in valid_* have been discarded.
        """
        super().__init__()
        self.mode = mode
        self.im_mode = '1'
        self.aug = None
        self.targets = []
        # n-th entry contains semantic of n-th class
        self.class_mapping = {'aux': {'_start_separator': 0, '_end_separator': 1}, 'baselines': {}, 'regions': {}}
        self.class_stats = {'baselines': defaultdict(int), 'regions': defaultdict(int)}
        self.num_classes = 2
        self.mbl_dict = merge_baselines if merge_baselines is not None else {}
        self.mreg_dict = merge_regions if merge_regions is not None else {}
        self.valid_baselines = valid_baselines
        self.valid_regions = valid_regions
        if mode in ['alto', 'page', 'xml']:
            if mode == 'alto':
                fn = parse_alto
            elif mode == 'page':
                fn = parse_page
            elif mode == 'xml':
                fn = parse_xml
            im_paths = []
            self.targets = []
            for img in imgs:
                try:
                    data = fn(img)
                    im_paths.append(data['image'])
                    lines = defaultdict(list)
                    for line in data['lines']:
                        if valid_baselines is None or line['script'] in valid_baselines:
                            lines[self.mbl_dict.get(line['script'], line['script'])].append(line['baseline'])
                            self.class_stats['baselines'][self.mbl_dict.get(line['script'], line['script'])] += 1
                    regions = defaultdict(list)
                    for k, v in data['regions'].items():
                        if valid_regions is None or k in valid_regions:
                            regions[self.mreg_dict.get(k, k)].extend(v)
                            self.class_stats['regions'][self.mreg_dict.get(k, k)] += len(v)
                    data['regions'] = regions
                    self.targets.append({'baselines': lines, 'regions': data['regions']})
                except KrakenInputException as e:
                    logger.warning(e)
                    continue
            # get line types
            imgs = im_paths
            # calculate class mapping
            line_types = set()
            region_types = set()
            for page in self.targets:
                for line_type in page['baselines'].keys():
                    line_types.add(line_type)
                for reg_type in page['regions'].keys():
                    region_types.add(reg_type)
            idx = -1
            for idx, line_type in enumerate(line_types):
                self.class_mapping['baselines'][line_type] = idx + self.num_classes
            self.num_classes += idx + 1
            idx = -1
            for idx, reg_type in enumerate(region_types):
                self.class_mapping['regions'][reg_type] = idx + self.num_classes
            self.num_classes += idx + 1
        elif mode == 'path':
            pass
        elif mode is None:
            imgs = []
        else:
            raise Exception('invalid dataset mode')
        if augmentation:
            from albumentations import (
                Compose, ToFloat, FromFloat, RandomRotate90, Flip, OneOf, MotionBlur, MedianBlur, Blur,
                ShiftScaleRotate, OpticalDistortion, ElasticTransform, RandomBrightnessContrast,
                HueSaturationValue,
                )

            self.aug = Compose([
                                ToFloat(),
                                RandomRotate90(),
                                Flip(),
                                OneOf([
                                    MotionBlur(p=0.2),
                                    MedianBlur(blur_limit=3, p=0.1),
                                    Blur(blur_limit=3, p=0.1),
                                ], p=0.2),
                                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                                OneOf([
                                    OpticalDistortion(p=0.3),
                                    ElasticTransform(p=0.1),
                                ], p=0.2),
                                HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
                               ], p=0.5)
        self.imgs = imgs
        self.line_width = line_width
        # split image transforms into two. one part giving the final PIL image
        # before conversion to a tensor and the actual tensor conversion part.
        self.head_transforms = transforms.Compose(im_transforms.transforms[:2])
        self.tail_transforms = transforms.Compose(im_transforms.transforms[2:])
        self.seg_type = None

    def add(self,
            image: Union[str, Image.Image],
            baselines: List[List[List[Tuple[int, int]]]] = None,
            regions: Dict[str, List[List[Tuple[int, int]]]] = None,
            *args,
            **kwargs):
        """
        Adds a page to the dataset.

        Args:
            im (path): Path to the whole page image
            baseline (dict): A list containing dicts with a list of coordinates
                             and script types [{'baseline': [[x0, y0], ...,
                             [xn, yn]], 'script': 'script_type'}, ...]
            regions (dict): A dict containing list of lists of coordinates {'region_type_0': [[x0, y0], ..., [xn, yn]]], 'region_type_1': ...}.
        """
        if self.mode:
            raise Exception(f'The `add` method is incompatible with dataset mode {self.mode}')
        baselines_ = defaultdict(list)
        for line in baselines:
            line_type = self.mbl_dict.get(line['script'], line['script'])
            if self.valid_baselines is None or line['script'] in self.valid_baselines:
                baselines_[line_type].append(line['baseline'])
                self.class_stats['baselines'][line_type] += 1

                if line_type not in self.class_mapping['baselines']:
                    self.num_classes += 1
                    self.class_mapping['baselines'][line_type] = self.num_classes - 1

        regions_ = defaultdict(list)
        for k, v in regions.items():
            reg_type = self.mreg_dict.get(k, k)
            if self.valid_regions is None or reg_type in self.valid_regions:
                regions_[reg_type].extend(v)
                self.class_stats['baselines'][reg_type] += len(v)
                if reg_type not in self.class_mapping['regions']:
                    self.num_classes += 1
                    self.class_mapping['regions'][reg_type] = self.num_classes - 1

        self.targets.append({'baselines': baselines_, 'regions': regions_})
        self.imgs.append(image)

    def __getitem__(self, idx):
        im = self.imgs[idx]
        if self.mode != 'path':
            target = self.targets[idx]
        else:
            with open('{}.path'.format(path.splitext(im)[0]), 'r') as fp:
                target = json.load(fp)
        if not isinstance(im, Image.Image):
            try:
                logger.debug(f'Attempting to load {im}')
                im = Image.open(im)
                im, target = self.transform(im, target)
                return {'image': im, 'target': target}
            except Exception:
                idx = np.random.randint(0, len(self.imgs))
                logger.debug(traceback.format_exc())
                logger.info(f'Failed. Replacing with sample {idx}')
                return self[np.random.randint(0, len(self.imgs))]
        im, target = self.transform(im, target)
        return {'image': im, 'target': target}

    @staticmethod
    def _get_ortho_line(lineseg, point, line_width, offset):
        lineseg = np.array(lineseg)
        norm_vec = lineseg[1,...] - lineseg[0,...]
        norm_vec_len = np.sqrt(np.sum(norm_vec**2))
        unit_vec = norm_vec / norm_vec_len
        ortho_vec = unit_vec[::-1] * ((1,-1), (-1,1))
        if offset == 'l':
            point -= unit_vec * line_width
        else:
            point += unit_vec * line_width
        return (ortho_vec * 10 + point).astype('int').tolist()

    def transform(self, image, target):
        orig_size = image.size
        image = self.head_transforms(image)
        if not is_bitonal(image):
            self.im_mode = image.mode
        image = self.tail_transforms(image)
        scale = image.shape[2]/orig_size[0]
        t = torch.zeros((self.num_classes,) + image.shape[1:])
        start_sep_cls = self.class_mapping['aux']['_start_separator']
        end_sep_cls = self.class_mapping['aux']['_end_separator']

        for key, lines in target['baselines'].items():
            try:
                cls_idx = self.class_mapping['baselines'][key]
            except KeyError:
                # skip lines of classes not present in the training set
                continue
            for line in lines:
                # buffer out line to desired width
                line = [k for k, g in groupby(line)]
                line = np.array(line)*scale
                shp_line = geom.LineString(line)
                split_offset = min(5, shp_line.length/2)
                line_pol = np.array(shp_line.buffer(-self.line_width, cap_style=2, single_sided=True).boundary, dtype=np.int)
                rr, cc = polygon(line_pol[:,1], line_pol[:,0], shape=image.shape[1:])
                t[cls_idx, rr, cc] = 1
                split_pt = shp_line.interpolate(split_offset).buffer(0.001)
                # top
                start_sep = np.array((split(shp_line, split_pt)[0].parallel_offset(0.5*self.line_width, side='right').buffer(1.5*self.line_width, cap_style=3).boundary), dtype=np.int)
                rr_s, cc_s = polygon(start_sep[:,1], start_sep[:,0], shape=image.shape[1:])
                t[start_sep_cls, rr_s, cc_s] = 1
                t[start_sep_cls, rr, cc] = 0
                split_pt = shp_line.interpolate(-split_offset).buffer(0.001)
                # top
                end_sep = np.array((split(shp_line, split_pt)[-1].parallel_offset(0.5*self.line_width, side='right').buffer(1.5*self.line_width, cap_style=3).boundary), dtype=np.int)
                rr_s, cc_s = polygon(end_sep[:,1], end_sep[:,0], shape=image.shape[1:])
                t[end_sep_cls, rr_s, cc_s] = 1
                t[end_sep_cls, rr, cc] = 0
        for key, regions in target['regions'].items():
            try:
                cls_idx = self.class_mapping['regions'][key]
            except KeyError:
                # skip regions of classes not present in the training set
                continue
            for region in regions:
                region = np.array(region)*scale
                rr, cc = polygon(region[:,1], region[:,0], shape=image.shape[1:])
                t[cls_idx, rr, cc] = 1
        target = t
        if self.aug:
            image = image.permute(1, 2, 0).numpy()
            target = target.permute(1, 2, 0).numpy()
            o = self.aug(image=image, mask=target)
            image = torch.tensor(o['image']).permute(2, 0, 1)
            target = torch.tensor(o['mask']).permute(2, 0, 1)
        return image, target

    def __len__(self):
        return len(self.imgs)
