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
import io
import json
import torch
import pathlib
import warnings
import traceback
import numpy as np
import pyarrow as pa
import pkg_resources
import shapely.geometry as geom
import torch.nn.functional as F

from os import path
from PIL import Image
from functools import partial
from shapely.ops import split
from itertools import groupby
from torchvision import transforms
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Iterable, Sequence, Callable, Optional, Any, Union

from skimage.draw import polygon

from kraken.lib.xml import parse_alto, parse_page, parse_xml, preparse_xml_data

from kraken.lib.util import is_bitonal
from kraken.lib.codec import PytorchCodec
from kraken.lib.models import TorchSeqRecognizer
from kraken.lib.segmentation import extract_polygons
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.lineest import CenterNormalizer

from kraken.lib import functional_im_transforms as F_t

__all__ = ['ArrowIPCRecognitionDataset',
           'BaselineSet',
           'PolygonGTDataset',
           'GroundTruthDataset',
           'ImageInputTransforms',
           'compute_error',
           'preparse_xml_data']

import logging

logger = logging.getLogger(__name__)


class ImageInputTransforms(transforms.Compose):
    def __init__(self,
                 batch: int,
                 height: int,
                 width: int,
                 channels: int,
                 pad: int,
                 valid_norm: bool = True,
                 force_binarization: bool = False) -> None:
        """
        Container for image input transforms for recognition and segmentation
        networks.

        Args:
            batch: mini-batch size
            height: height of input image in pixels
            width: width of input image in pixels
            channels: color channels of input
            pad: Amount of padding on horizontal ends of image
            valid_norm: Enables/disables baseline normalization as a valid
                        preprocessing step. If disabled we will fall back to
                        standard scaling.
            force_binarization: Forces binarization of input images using the
                                nlbin algorithm.

        """
        super().__init__(None)

        self._scale = (height, width)  # type: Tuple[int, int]
        self._pad = pad
        self._valid_norm = valid_norm
        self._force_binarization = force_binarization
        self._batch = batch
        self._channels = channels

        self._create_transforms()

    def _create_transforms(self) -> None:
        height = self._scale[0]
        width = self._scale[1]
        self._center_norm = False
        self._mode = 'RGB' if self._channels == 3 else 'L'
        if height == 1 and width == 0 and self._channels > 3:
            perm = (1, 0, 2)
            self._scale = (self._channels, 0)
            if self._valid_norm:
                self._center_norm = True
            self._mode = 'L'
        elif height > 1 and width == 0 and self._channels in (1, 3):
            perm = (0, 1, 2)
            if self._valid_norm and self._channels == 1:
                self._center_norm = True
        elif height == 0 and width > 1 and self._channels in (1, 3):
            perm = (0, 1, 2)
        # fixed height and width image => bicubic scaling of the input image, disable padding
        elif height > 0 and width > 0 and self._channels in (1, 3):
            perm = (0, 1, 2)
            self._pad = 0
        elif height == 0 and width == 0 and self._channels in (1, 3):
            perm = (0, 1, 2)
            self._pad = 0
        else:
            raise KrakenInputException(f'Invalid input spec {self._batch}, {height}, {width}, {self._channels}, {self._pad}.')

        if self._mode != 'L' and self._force_binarization:
            raise KrakenInputException(f'Invalid input spec {self._batch}, {height}, {width}, {self._channels}, {self._pad} in '
                                       'combination with forced binarization.')

        self.transforms = []
        self.transforms.append(transforms.Lambda(partial(F_t.pil_to_mode, mode=self._mode)))

        if self._force_binarization:
            self.transforms.append(transforms.Lambda(F_t.pil_to_bin))
        if self._scale != (0, 0):
            if self._center_norm:
                lnorm = CenterNormalizer(self._scale[0])
                self.transforms.append(transforms.Lambda(partial(F_t.pil_dewarp, lnorm=lnorm)))
                self.transforms.append(transforms.Lambda(partial(F_t.pil_to_mode, mode=self._mode)))
            else:
                self.transforms.append(transforms.Lambda(partial(F_t.pil_fixed_resize, scale=self._scale)))
        if self._pad:
            self.transforms.append(transforms.Pad((self._pad, 0), fill=255))
        self.transforms.append(transforms.ToTensor())
        # invert
        self.transforms.append(transforms.Lambda(F_t.tensor_invert))
        self.transforms.append(transforms.Lambda(partial(F_t.tensor_permute, perm=perm)))

    @property
    def batch(self) -> int:
        """
        Batch size attribute. Ignored.
        """
        return self._batch

    @batch.setter
    def batch(self, batch: int) -> None:
        self._batch = batch

    @property
    def channels(self) -> int:
        """
        Channels attribute. Can be either 1 (binary/grayscale), 3 (RGB).
        """
        if self._channels not in [1, 3] and self._scale[0] == self._channels:
            return 1
        else:
            return self._channels

    @channels.setter
    def channels(self, channels: int) -> None:
        self._channels = channels
        self._create_transforms()

    @property
    def height(self) -> int:
        """
        Desired output image height. If set to 0, image will be rescaled
        proportionally with width, if 1 and `channels` is larger than 3 output
        will be grayscale and of the height set with the channels attribute.
        """
        if self._scale == (1, 0) and self.channels > 3:
            return self._channels
        else:
            return self._scale[0]

    @height.setter
    def height(self, height: int) -> None:
        self._scale = (height, self.scale[1])
        self._create_transforms()

    @property
    def width(self) -> int:
        """
        Desired output image width. If set to 0, image will be rescaled
        proportionally with height.
        """
        return self._scale[1]

    @width.setter
    def width(self, width: int) -> None:
        self._scale = (self._scale[0], width)
        self._create_transforms()

    @property
    def mode(self) -> str:
        """
        Imaginary PIL.Image.Image mode of the output tensor. Possible values
        are RGB, L, and 1.
        """
        return self._mode if not self.force_binarization else '1'

    @property
    def scale(self) -> Tuple[int, int]:
        """
        Desired output shape (height, width) of the image. If any value is set
        to 0, image will be rescaled proportionally with height, width, if 1
        and `channels` is larger than 3 output will be grayscale and of the
        height set with the channels attribute.
        """
        if self._scale == (1, 0) and self.channels > 3:
            return (self._channels, self._scale[1])
        else:
            return self._scale

    @scale.setter
    def scale(self, scale: Tuple[int, int]) -> None:
        self._scale = scale
        self._create_transforms()

    @property
    def pad(self) -> int:
        """
        Amount of padding around left/right end of image.
        """
        return self._pad

    @pad.setter
    def pad(self, pad: int) -> None:
        self._pad = pad
        self._create_transforms()

    @property
    def valid_norm(self) -> bool:
        """
        Switch allowing/disallowing centerline normalization. Even if enabled
        won't be applied to 3-channel images.
        """
        return self._valid_norm

    @valid_norm.setter
    def valid_norm(self, valid_norm: bool) -> None:
        self._valid_norm = valid_norm
        self._create_transforms()

    @property
    def centerline_norm(self) -> bool:
        """
        Attribute indicating if centerline normalization will be applied to
        input images.
        """
        return self._center_norm

    @property
    def force_binarization(self) -> bool:
        """
        Switch enabling/disabling forced binarization.
        """
        return self._force_binarization

    @force_binarization.setter
    def force_binarization(self, force_binarization: bool) -> None:
        self._force_binarization = force_binarization
        self._create_transforms()


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
        k, m = direction[i][j]
        if k == i - 1 and m == j - 1:
            algn1.insert(0, seq1[i - 1])
            algn2.insert(0, seq2[j - 1])
        elif k < i:
            algn1.insert(0, seq1[i - 1])
            algn2.insert(0, '')
        elif m < j:
            algn1.insert(0, '')
            algn2.insert(0, seq2[j - 1])
        i, j = k, m
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
    for u, v in zip(algn1, algn2):
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


def compute_error(model: TorchSeqRecognizer, sample: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    Computes error report from a model and a list of line image-text pairs.

    Args:
        model (kraken.lib.models.TorchSeqRecognizer): Model used for recognition
        validation_set (list): List of tuples (image, text) for validation

    Returns:
        A tuple with total number of characters and edit distance across the
        whole validation set.
    """
    pred = model.predict_string(sample['image'], sample['seq_lens'])
    text = sample['target']
    if isinstance(text, torch.Tensor):
        text = ''.join(x[0] for x in model.codec.decode([(x, 0, 0, 0) for x in text]))
    return int(sample['target_lens'].sum()), _fast_levenshtein(pred, text)


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


class ArrowIPCRecognitionDataset(Dataset):
    """
    Dataset for training a recognition model from a precompiled dataset in
    Arrow IPC format.
    """
    def __init__(self,
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 reorder: Union[bool, str] = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 augmentation: bool = False,
                 preload: bool = False,
                 split_filter: Optional[str] = None) -> None:
        """
        Creates a dataset for a polygonal (baseline) transcription model.

        Args:
            normalization: Unicode normalization for gt
            whitespace_normalization: Normalizes unicode whitespace and strips
                                      whitespace.
            reorder: Whether to rearrange code points in "display"/LTR order.
                     Set to L|R to change the default text direction.
            im_transforms: Function taking an PIL.Image and returning a tensor
                           suitable for forward passes.
            augmentation: Enables preloading and preprocessing of image files.
            preload: Ignored.
            split_filter: Enables filtering of the dataset according to mask
                          values in the set split. If set to `None` all rows
                          are sampled, if set to `train`, `validation`, or
                          `test` only rows with the appropriate flag set in the
                          file will be considered.
        """
        self.alphabet = Counter()  # type: Counter
        self.text_transforms = []  # type: List[Callable[[str], str]]
        self.transforms = im_transforms
        self.aug = None
        self.split_filter = None
        self._num_lines = 0
        self.arrow_table = None
        self.codec = None

        self.seg_type = 'baselines'
        # built text transformations
        if normalization:
            self.text_transforms.append(partial(F_t.text_normalize, normalization=normalization))
        if whitespace_normalization:
            self.text_transforms.append(F_t.text_whitespace_normalize)
        if reorder:
            if reorder in ('L', 'R'):
                self.text_transforms.append(partial(F_t.text_reorder, base_dir=reorder))
            else:
                self.text_transforms.append(F_t.text_reorder)
        if augmentation:
            from albumentations import (
                Compose, ToFloat, OneOf, MotionBlur, MedianBlur, Blur,
                ShiftScaleRotate, OpticalDistortion, ElasticTransform,
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
        if split_filter:
            if split_filter in ['train', 'validation', 'test']:
                self.split_filter = split_filter
            else:
                raise ValueError(f'split_filter has to be one of [train, validation, test] (is {split_filter}).')
        self.im_mode = self.transforms.mode

    def add(self, file: Union[str, pathlib.Path]) -> None:
        """
        Adds an Arrow IPC file to the dataset.

        Args:
            file: Location of the precompiled dataset file.
        """
        # extract metadata and update alphabet
        with pa.memory_map(file, 'rb') as source:
            ds_table = pa.ipc.open_file(source).read_all()
            raw_metadata = ds_table.schema.metadata
            if not raw_metadata or b'lines' not in raw_metadata:
                raise ValueError(f'{file} does not contain a valid metadata record.')
            metadata = json.loads(raw_metadata[b'lines'])
        if metadata['type'] != 'kraken_recognition_baseline':
            raise ValueError(f'Unknown type {metadata["type"]} of dataset.')
        if self.split_filter and metadata['counts'][self.split_filter] == 0:
            logger.warning(f'No explicit split for "{self.split_filter}" in dataset {file} (with splits {metadata["counts"].items()}).')
            return
        if metadata['im_mode'] > self.im_mode and self.transforms.mode >= metadata['im_mode']:
            logger.info(f'Upgrading "im_mode" from {self.im_mode} to {metadata["im_mode"]}.')
            self.im_mode = metadata['im_mode']
        self.alphabet.update(metadata['alphabet'])
        num_lines = metadata['counts'][self.split_filter] if self.split_filter else metadata['counts']['all']
        if not self.arrow_table:
            self.arrow_table = ds_table
        else:
            self.arrow_table = pa.concat_tables([self.arrow_table, ds_table])
        self._num_lines += num_lines

    def encode(self, codec: Optional[PytorchCodec] = None) -> None:
        """
        Adds a codec to the dataset (but does NOT actually encode the text!).
        """
        if codec:
            self.codec = codec
        else:
            self.codec = PytorchCodec(''.join(self.alphabet.keys()))

    def no_encode(self) -> None:
        """
        Creates an unencoded dataset.
        """
        pass

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split_filter:
            table = self.arrow_table.filter(self.arrow_table.column(self.split_filter))
        else:
            table = self.arrow_table
        sample = table.column('lines')[index].as_py()
        logger.debug(f'Loading sample {index}')
        im = Image.open(io.BytesIO(sample['im']))
        im = self.transforms(im)
        if self.aug:
            im = im.permute((1, 2, 0)).numpy()
            o = self.aug(image=im)
            im = torch.tensor(o['image'].transpose(2, 0, 1))
        return {'image': im, 'target': self.codec.encode(sample['text']) if self.codec is not None else sample['text']}

    def __len__(self) -> int:
        return self._num_lines


class PolygonGTDataset(Dataset):
    """
    Dataset for training a line recognition model from polygonal/baseline data.
    """
    def __init__(self,
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 reorder: Union[bool, str] = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 preload: bool = True,
                 augmentation: bool = False) -> None:
        """
        Creates a dataset for a polygonal (baseline) transcription model.

        Args:
            normalization (str): Unicode normalization for gt
            whitespace_normalization (str): Normalizes unicode whitespace and
                                            strips whitespace.
            reorder (bool|str): Whether to rearrange code points in "display"/LTR
                                order. Set to L|R to change the default text
                                direction.
            im_transforms (func): Function taking an PIL.Image and returning a
                                  tensor suitable for forward passes.
            preload (bool): Enables preloading and preprocessing of image files.
            augmentation (bool): Enables augmentation.
        """
        self._images = []  # type:  Union[List[Image], List[torch.Tensor]]
        self._gt = []  # type:  List[str]
        self.alphabet = Counter()  # type: Counter
        self.text_transforms = []  # type: List[Callable[[str], str]]
        self.transforms = im_transforms
        if preload:
            warnings.warn('Preloading is deprecated and will be removed in the '
                          'next major release of kraken. Use precompiled datasets '
                          'instead.', PendingDeprecationWarning, stacklevel=2)
        self.preload = preload
        self.aug = None

        self.seg_type = 'baselines'
        # built text transformations
        if normalization:
            self.text_transforms.append(partial(F_t.text_normalize, normalization=normalization))
        if whitespace_normalization:
            self.text_transforms.append(F_t.text_whitespace_normalize)
        if reorder:
            if reorder in ('L', 'R'):
                self.text_transforms.append(partial(F_t.text_reorder, base_dir=reorder))
            else:
                self.text_transforms.append(F_t.text_reorder)
        if augmentation:
            from albumentations import (
                Compose, ToFloat, OneOf, MotionBlur, MedianBlur, Blur,
                ShiftScaleRotate, OpticalDistortion, ElasticTransform,
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

    def add(self, *args, **kwargs):
        """
        Adds a line to the dataset.

        Args:
            im (path): Path to the whole page image
            text (str): Transcription of the line.
            baseline (list): A list of coordinates [[x0, y0], ..., [xn, yn]].
            boundary (list): A polygon mask for the line.
        """
        if 'preparse' not in kwargs or not kwargs['preparse']:
            kwargs = self.parse(*args, **kwargs)
        if kwargs['preload']:
            if kwargs['im_mode'] > self.im_mode:
                logger.info(f'Upgrading "im_mode" from {self.im_mode} to {kwargs["im_mode"]}.')
                self.im_mode = kwargs['im_mode']
            self._images.append(kwargs['image'])
        else:
            self._images.append((kwargs['image'], kwargs['baseline'], kwargs['boundary']))
        self._gt.append(kwargs['text'])
        self.alphabet.update(kwargs['text'])

    def parse(self,
              image: Union[str, Image.Image],
              text: str,
              baseline: List[Tuple[int, int]],
              boundary: List[Tuple[int, int]],
              *args,
              **kwargs):
        """
        Parses a sample for the dataset and returns it.

        This function is mainly uses for parallelized loading of training data.

        Args:
            im (path): Path to the whole page image
            text (str): Transcription of the line.
            baseline (list): A list of coordinates [[x0, y0], ..., [xn, yn]].
            boundary (list): A polygon mask for the line.
        """
        orig_text = text
        for func in self.text_transforms:
            text = func(text)
        if not text:
            raise KrakenInputException(f'Text line "{orig_text}" is empty after transformations')
        if not baseline:
            raise KrakenInputException('No baseline given for line')
        if not boundary:
            raise KrakenInputException('No boundary given for line')
        if self.preload:
            if not isinstance(image, Image.Image):
                im = Image.open(image)
            try:
                im, _ = next(extract_polygons(im, {'type': 'baselines',
                                                   'lines': [{'baseline': baseline, 'boundary': boundary}]}))
            except IndexError:
                raise KrakenInputException('Patch extraction failed for baseline')
            try:
                im = self.transforms(im)
                if im.shape[0] == 3:
                    im_mode = 'RGB'
                elif im.shape[0] == 1:
                    im_mode = 'L'
                if is_bitonal(im):
                    im_mode = '1'
            except ValueError:
                raise KrakenInputException(f'Image transforms failed on {image}')

            return {'text': text,
                    'image': im,
                    'baseline': baseline,
                    'boundary': boundary,
                    'im_mode': im_mode,
                    'preload': True,
                    'preparse': True}
        else:
            return {'text': text,
                    'image': image,
                    'baseline': baseline,
                    'boundary': boundary,
                    'preload': False,
                    'preparse': True}

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
                im, _ = next(extract_polygons(im, {'type': 'baselines',
                                                   'lines': [{'baseline': item[0][1], 'boundary': item[0][2]}]}))
                im = self.transforms(im)
                if im.shape[0] == 3:
                    im_mode = 'RGB'
                elif im.shape[0] == 1:
                    im_mode = 'L'
                if is_bitonal(im):
                    im_mode = '1'

                if im_mode > self.im_mode:
                    logger.info(f'Upgrading "im_mode" from {self.im_mode} to {im_mode}')
                    self.im_mode = im_mode
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
        return len(self._images)


class GroundTruthDataset(Dataset):
    """
    Dataset for training a line recognition model.

    All data is cached in memory.
    """
    def __init__(self, split: Callable[[str], str] = F_t.default_split,
                 suffix: str = '.gt.txt',
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 reorder: Union[bool, str] = True,
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
            reorder (bool|str): Whether to rearrange code points in "display"/LTR
                                order. Set to L|R to change the default text
                                direction.
            im_transforms (func): Function taking an PIL.Image and returning a
                                  tensor suitable for forward passes.
            preload (bool): Enables preloading and preprocessing of image files.
            augmentation (bool): Enables augmentation.
        """
        self.suffix = suffix
        self.split = partial(F_t.suffix_split, split=split, suffix=suffix)
        self._images = []  # type:  Union[List[Image], List[torch.Tensor]]
        self._gt = []  # type:  List[str]
        self.alphabet = Counter()  # type: Counter
        self.text_transforms = []  # type: List[Callable[[str], str]]
        self.transforms = im_transforms
        self.aug = None

        self.preload = preload
        if preload:
            warnings.warn('Preloading is deprecated and will be removed in the '
                          'next major release of kraken. Use precompiled datasets '
                          'instead.', PendingDeprecationWarning, stacklevel=2)

        self.seg_type = 'bbox'
        # built text transformations
        if normalization:
            self.text_transforms.append(partial(F_t.text_normalize, normalization=normalization))
        if whitespace_normalization:
            self.text_transforms.append(F_t.text_whitespace_normalize)
        if reorder:
            if reorder in ('L', 'R'):
                self.text_transforms.append(partial(F_t.text_reorder, base_dir=reorder))
            else:
                self.text_transforms.append(F_t.text_reorder)
        if augmentation:
            from albumentations import (
                Compose, ToFloat, OneOf, MotionBlur, MedianBlur, Blur,
                ShiftScaleRotate, OpticalDistortion, ElasticTransform,
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

    def add(self, *args, **kwargs) -> None:
        """
        Adds a line-image-text pair to the dataset.

        Args:
            image (str): Input image path
        """
        if 'preparse' not in kwargs or not kwargs['preparse']:
            kwargs = self.parse(*args, **kwargs)
        if kwargs['preload'] and kwargs['im_mode'] > self.im_mode:
            logger.info(f'upgrading "im_mode" from {self.im_mode} to {kwargs["im_mode"]}.')
            self.im_mode = kwargs['im_mode']
        self._images.append(kwargs['image'])
        self._gt.append(kwargs['text'])
        self.alphabet.update(kwargs['text'])

    def parse(self, image: Union[str, Image.Image], *args, **kwargs) -> Dict:
        """
        Parses a sample for this dataset.

        This is mostly used to parallelize populating the dataset.

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
                im = self.transforms(im)
                if im.shape[0] == 3:
                    im_mode = 'RGB'
                elif im.shape[0] == 1:
                    im_mode = 'L'
                if is_bitonal(im):
                    im_mode = '1'
            except ValueError:
                raise KrakenInputException(f'Image transforms failed on {image}')
            return {'image': im, 'text': gt, 'im_mode': im_mode, 'preload': True, 'preparse': True}
        else:
            return {'image': image, 'text': gt, 'preload': False, 'preparse': True}

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
                im = self.transforms(im)
                if im.shape[0] == 3:
                    im_mode = 'RGB'
                elif im.shape[0] == 1:
                    im_mode = 'L'
                if is_bitonal(im):
                    im_mode = '1'
                if im_mode > self.im_mode:
                    logger.info(f'Upgrading "im_mode" from {self.im_mode} to {im_mode}')
                    self.im_mode = im_mode
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
        return len(self._images)


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
                Compose, ToFloat, RandomRotate90, Flip, OneOf, MotionBlur, MedianBlur, Blur,
                ShiftScaleRotate, OpticalDistortion, ElasticTransform,
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
        self.transforms = im_transforms
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
            regions (dict): A dict containing list of lists of coordinates
                            {'region_type_0': [[x0, y0], ..., [xn, yn]]],
                            'region_type_1': ...}.
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
        norm_vec = lineseg[1, ...] - lineseg[0, ...]
        norm_vec_len = np.sqrt(np.sum(norm_vec**2))
        unit_vec = norm_vec / norm_vec_len
        ortho_vec = unit_vec[::-1] * ((1, -1), (-1, 1))
        if offset == 'l':
            point -= unit_vec * line_width
        else:
            point += unit_vec * line_width
        return (ortho_vec * 10 + point).astype('int').tolist()

    def transform(self, image, target):
        orig_size = image.size
        image = self.transforms(image)
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
                line_pol = np.array(shp_line.buffer(self.line_width/2, cap_style=2).boundary, dtype=int)
                rr, cc = polygon(line_pol[:, 1], line_pol[:, 0], shape=image.shape[1:])
                t[cls_idx, rr, cc] = 1
                split_pt = shp_line.interpolate(split_offset).buffer(0.001)
                # top
                start_sep = np.array((split(shp_line, split_pt)[0].buffer(self.line_width,
                                                                          cap_style=3).boundary), dtype=int)
                rr_s, cc_s = polygon(start_sep[:, 1], start_sep[:, 0], shape=image.shape[1:])
                t[start_sep_cls, rr_s, cc_s] = 1
                t[start_sep_cls, rr, cc] = 0
                split_pt = shp_line.interpolate(-split_offset).buffer(0.001)
                # top
                end_sep = np.array((split(shp_line, split_pt)[-1].buffer(self.line_width,
                                                                         cap_style=3).boundary), dtype=int)
                rr_s, cc_s = polygon(end_sep[:, 1], end_sep[:, 0], shape=image.shape[1:])
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
                rr, cc = polygon(region[:, 1], region[:, 0], shape=image.shape[1:])
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
