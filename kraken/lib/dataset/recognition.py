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
import numpy as np
import pyarrow as pa
import traceback
import dataclasses
import multiprocessing as mp

from collections import Counter
from functools import partial
from typing import (TYPE_CHECKING, Any, Callable, List, Literal, Optional,
                    Tuple, Union)

from PIL import Image
from ctypes import c_char
from torchvision import transforms
from torch.utils.data import Dataset

from kraken.containers import BaselineLine, BBoxLine, Segmentation
from kraken.lib import functional_im_transforms as F_t
from kraken.lib.codec import PytorchCodec
from kraken.lib.exceptions import KrakenEncodeException, KrakenInputException
from kraken.lib.segmentation import extract_polygons
from kraken.lib.util import is_bitonal

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['DefaultAugmenter',
           'ArrowIPCRecognitionDataset',
           'PolygonGTDataset',
           'GroundTruthDataset']

import logging

logger = logging.getLogger(__name__)


class DefaultAugmenter():
    def __init__(self):
        import cv2
        cv2.setNumThreads(0)
        from albumentations import (Blur, Compose, ElasticTransform,
                                    MedianBlur, MotionBlur, OneOf,
                                    OpticalDistortion, PixelDropout,
                                    ShiftScaleRotate, ToFloat)

        self._transforms = Compose([
                                    ToFloat(),
                                    PixelDropout(p=0.2),
                                    OneOf([
                                        MotionBlur(p=0.2),
                                        MedianBlur(blur_limit=3, p=0.1),
                                        Blur(blur_limit=3, p=0.1),
                                    ], p=0.2),
                                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=1, p=0.2),
                                    OneOf([
                                        OpticalDistortion(p=0.3),
                                        ElasticTransform(alpha=64, sigma=25, alpha_affine=0.25, p=0.1),
                                    ], p=0.2),
                                   ], p=0.5)

    def __call__(self, image):
        return self._transforms(image=image)


class ArrowIPCRecognitionDataset(Dataset):
    """
    Dataset for training a recognition model from a precompiled dataset in
    Arrow IPC format.
    """
    def __init__(self,
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 skip_empty_lines: bool = True,
                 reorder: Union[bool, Literal['L', 'R']] = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 augmentation: bool = False,
                 split_filter: Optional[str] = None) -> None:
        """
        Creates a dataset for a polygonal (baseline) transcription model.

        Args:
            normalization: Unicode normalization for gt
            whitespace_normalization: Normalizes unicode whitespace and strips
                                      whitespace.
            skip_empty_lines: Whether to return samples without text.
            reorder: Whether to rearrange code points in "display"/LTR order.
                     Set to L|R to change the default text direction.
            im_transforms: Function taking an PIL.Image and returning a tensor
                           suitable for forward passes.
            augmentation: Enables augmentation.
            split_filter: Enables filtering of the dataset according to mask
                          values in the set split. If set to `None` all rows
                          are sampled, if set to `train`, `validation`, or
                          `test` only rows with the appropriate flag set in the
                          file will be considered.
        """
        self.alphabet: Counter = Counter()
        self.text_transforms: List[Callable[[str], str]] = []
        self.failed_samples = set()
        self.transforms = im_transforms
        self.aug = None
        self._split_filter = split_filter
        self._num_lines = 0
        self.arrow_table = None
        self.codec = None
        self.skip_empty_lines = skip_empty_lines
        self.legacy_polygons_status = None

        self.seg_type = None
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
            self.aug = DefaultAugmenter()

        self.im_mode = self.transforms.mode

    def add(self, file: Union[str, 'PathLike']) -> None:
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
        if metadata['type'] == 'kraken_recognition_baseline':
            if not self.seg_type:
                self.seg_type = 'baselines'
            if self.seg_type != 'baselines':
                raise ValueError(f'File {file} has incompatible type {metadata["type"]} for dataset with type {self.seg_type}.')
        elif metadata['type'] == 'kraken_recognition_bbox':
            if not self.seg_type:
                self.seg_type = 'bbox'
            if self.seg_type != 'bbox':
                raise ValueError(f'File {file} has incompatible type {metadata["type"]} for dataset with type {self.seg_type}.')
        else:
            raise ValueError(f'Unknown type {metadata["type"]} of dataset.')
        if self._split_filter and metadata['counts'][self._split_filter] == 0:
            logger.warning(f'No explicit split for "{self._split_filter}" in dataset {file} (with splits {metadata["counts"].items()}).')
            return
        if metadata['im_mode'] > self.im_mode and self.transforms.mode >= metadata['im_mode']:
            logger.info(f'Upgrading "im_mode" from {self.im_mode} to {metadata["im_mode"]}.')
            self.im_mode = metadata['im_mode']
        # centerline normalize raw bbox dataset
        if self.seg_type == 'bbox' and metadata['image_type'] == 'raw':
            self.transforms.valid_norm = True

        legacy_polygons = metadata.get('legacy_polygons', True)
        if self.legacy_polygons_status is None:
            self.legacy_polygons_status = legacy_polygons
        elif self.legacy_polygons_status != legacy_polygons:
            self.legacy_polygons_status = "mixed"

        self.alphabet.update(metadata['alphabet'])
        num_lines = metadata['counts'][self._split_filter] if self._split_filter else metadata['counts']['all']
        if self._split_filter:
            ds_table = ds_table.filter(ds_table.column(self._split_filter))
        if self.skip_empty_lines:
            logger.debug('Getting indices of empty lines after text transformation.')
            self.skip_empty_lines = False
            mask = np.ones(len(ds_table), dtype=bool)
            for index in range(len(ds_table)):
                try:
                    self._apply_text_transform(ds_table.column('lines')[index].as_py(),)
                except KrakenInputException:
                    mask[index] = False
                    continue
            num_lines = np.count_nonzero(mask)
            logger.debug(f'Filtering out {np.count_nonzero(~mask)} empty lines')
            if np.any(~mask):
                ds_table = ds_table.filter(pa.array(mask))
            self.skip_empty_lines = True
        if not self.arrow_table:
            self.arrow_table = ds_table
        else:
            self.arrow_table = pa.concat_tables([self.arrow_table, ds_table])
        self._num_lines += num_lines

    def rebuild_alphabet(self):
        """
        Recomputes the alphabet depending on the given text transformation.
        """
        self.alphabet = Counter()
        for index in range(len(self)):
            try:
                text = self._apply_text_transform(self.arrow_table.column('lines')[index].as_py(),)
                self.alphabet.update(text)
            except KrakenInputException:
                continue

    def _apply_text_transform(self, sample) -> str:
        """
        Applies text transform to a sample.
        """
        text = sample['text']
        for func in self.text_transforms:
            text = func(text)
        if not text:
            logger.debug(f'Text line "{sample["text"]}" is empty after transformations')
            if not self.skip_empty_lines:
                raise KrakenInputException('empty text line')
        return text

    def encode(self, codec: Optional[PytorchCodec] = None) -> None:
        """
        Adds a codec to the dataset.
        """
        if codec:
            self.codec = codec
            logger.info(f'Trying to encode dataset with codec {codec}')
            for index in range(self._num_lines):
                try:
                    text = self._apply_text_transform(
                        self.arrow_table.column('lines')[index].as_py(),
                    )
                    self.codec.encode(text)
                except KrakenEncodeException as e:
                    raise e
                except KrakenInputException:
                    pass
        else:
            self.codec = PytorchCodec(''.join(self.alphabet.keys()))

    def no_encode(self) -> None:
        """
        Creates an unencoded dataset.
        """
        pass

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            sample = self.arrow_table.column('lines')[index].as_py()
            logger.debug(f'Loading sample {index}')
            im = Image.open(io.BytesIO(sample['im']))
            im = self.transforms(im)
            if self.aug:
                im = im.permute((1, 2, 0)).numpy()
                o = self.aug(image=im)
                im = torch.tensor(o['image'].transpose(2, 0, 1))
            text = self._apply_text_transform(sample)
        except Exception:
            self.failed_samples.add(index)
            idx = np.random.randint(0, len(self))
            logger.debug(traceback.format_exc())
            logger.info(f'Failed. Replacing with sample {idx}')
            return self[idx]

        return {'image': im, 'target': self.codec.encode(text) if self.codec is not None else text}

    def __len__(self) -> int:
        return self._num_lines


class PolygonGTDataset(Dataset):
    """
    Dataset for training a line recognition model from polygonal/baseline data.
    """
    def __init__(self,
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 skip_empty_lines: bool = True,
                 reorder: Union[bool, Literal['L', 'R']] = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 augmentation: bool = False,
                 legacy_polygons: bool = False) -> None:
        """
        Creates a dataset for a polygonal (baseline) transcription model.

        Args:
            normalization: Unicode normalization for gt
            whitespace_normalization: Normalizes unicode whitespace and strips
                                      whitespace.
            skip_empty_lines: Whether to return samples without text.
            reorder: Whether to rearrange code points in "display"/LTR order.
                     Set to L|R to change the default text direction.
            im_transforms: Function taking an PIL.Image and returning a tensor
                           suitable for forward passes.
            augmentation: Enables augmentation.
        """
        self._images: Union[List[Image.Image], List[torch.Tensor]] = []
        self._gt: List[str] = []
        self.alphabet: Counter = Counter()
        self.text_transforms: List[Callable[[str], str]] = []
        self.transforms = im_transforms
        self.aug = None
        self.skip_empty_lines = skip_empty_lines
        self.failed_samples = set()
        self.legacy_polygons = legacy_polygons

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
            self.aug = DefaultAugmenter()

        self._im_mode = mp.Value(c_char, b'1')

    def add(self,
            line: Optional[BaselineLine] = None,
            page: Optional[Segmentation] = None):
        """
        Adds an individual line or all lines on a page to the dataset.

        Args:
            line: BaselineLine container object of a line.
            page: Segmentation container object for a page.
        """
        if line:
            self.add_line(line)
        if page:
            self.add_page(page)
        if not (line or page):
            raise ValueError('Neither line nor page data provided in dataset builder')

    def add_page(self, page: Segmentation):
        """
        Adds all lines on a page to the dataset.

        Invalid lines will be skipped and a warning will be printed.

        Args:
            page: Segmentation container object for a page.
        """
        if page.type != 'baselines':
            raise ValueError(f'Invalid segmentation of type {page.type} (expected "baselines")')
        for line in page.lines:
            try:
                self.add_line(dataclasses.replace(line, imagename=page.imagename))
            except ValueError as e:
                logger.warning(e)

    def add_line(self, line: BaselineLine):
        """
        Adds a line to the dataset.

        Args:
            line: BaselineLine container object for a line.

        Raises:
            ValueError if the transcription of the line is empty after
            transformation or either baseline or bounding polygon are missing.
        """
        if line.type != 'baselines':
            raise ValueError(f'Invalid line of type {line.type} (expected "baselines")')

        text = line.text
        for func in self.text_transforms:
            text = func(text)
        if not text and self.skip_empty_lines:
            raise ValueError(f'Text line "{line.text}" is empty after transformations')
        if not line.baseline:
            raise ValueError('No baseline given for line')
        if not line.boundary:
            raise ValueError('No boundary given for line')

        self._images.append((line.imagename, line.baseline, line.boundary))
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
        self.training_set: List[Tuple[Union[Image.Image, torch.Tensor], torch.Tensor]] = []
        for im, gt in zip(self._images, self._gt):
            self.training_set.append((im, self.codec.encode(gt)))

    def no_encode(self) -> None:
        """
        Creates an unencoded dataset.
        """
        self.training_set: List[Tuple[Union[Image.Image, torch.Tensor], str]] = []
        for im, gt in zip(self._images, self._gt):
            self.training_set.append((im, gt))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.training_set[index]
        try:
            logger.debug(f'Attempting to load {item[0]}')
            im = item[0][0]
            if not isinstance(im, Image.Image):
                im = Image.open(im)
            im, _ = next(extract_polygons(im,
                                          Segmentation(type='baselines',
                                                       imagename=item[0][0],
                                                       text_direction='horizontal-lr',
                                                       lines=[BaselineLine('id_0',
                                                                           baseline=item[0][1],
                                                                           boundary=item[0][2])],
                                                       script_detection=True,
                                                       regions={},
                                                       line_orders=[]),
                                          legacy=self.legacy_polygons))
            im = self.transforms(im)
            if im.shape[0] == 3:
                im_mode = b'R'
            elif im.shape[0] == 1:
                im_mode = b'L'
            if is_bitonal(im):
                im_mode = b'1'

            with self._im_mode.get_lock():
                if im_mode > self._im_mode.value:
                    logger.info(f'Upgrading "im_mode" from {self._im_mode.value} to {im_mode}')
                    self._im_mode.value = im_mode
            if self.aug:
                im = im.permute((1, 2, 0)).numpy()
                o = self.aug(image=im)
                im = torch.tensor(o['image'].transpose(2, 0, 1))
            return {'image': im, 'target': item[1]}
        except Exception:
            self.failed_samples.add(index)
            idx = np.random.randint(0, len(self.training_set))
            logger.debug(traceback.format_exc())
            logger.info(f'Failed. Replacing with sample {idx}')
            return self[idx]

    def __len__(self) -> int:
        return len(self._images)

    @property
    def im_mode(self):
        return {b'1': '1',
                b'L': 'L',
                b'R': 'RGB'}[self._im_mode.value]


class GroundTruthDataset(Dataset):
    """
    Dataset for training a line recognition model.

    All data is cached in memory.
    """
    def __init__(self,
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 skip_empty_lines: bool = True,
                 reorder: Union[bool, str] = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 augmentation: bool = False) -> None:
        """
        Reads a list of image-text pairs and creates a ground truth set.

        Args:
            mode: Image color space. Either RGB (color) or L
                  (grayscale/bw). Only L is compatible with vertical
                  scaling/dewarping.
            scale: Target height or (width, height) of dewarped
                   line images. Vertical-only scaling is through
                   CenterLineNormalizer, resizing with Lanczos
                   interpolation. Set to 0 to disable.
            normalization: Unicode normalization for gt
            whitespace_normalization: Normalizes unicode whitespace and
                                      strips whitespace.
            skip_empty_lines: Whether to return samples without text.
            reorder: Whether to rearrange code points in "display"/LTR
                     order. Set to L|R to change the default text
                     direction.
            im_transforms: Function taking an PIL.Image and returning a
                           tensor suitable for forward passes.
            augmentation: Enables augmentation.
        """
        self._images = []  # type:  Union[List[Image], List[torch.Tensor]]
        self._gt = []  # type:  List[str]
        self.alphabet = Counter()  # type: Counter
        self.text_transforms = []  # type: List[Callable[[str], str]]
        self.transforms = im_transforms
        self.skip_empty_lines = skip_empty_lines
        self.aug = None
        self.failed_samples = set()

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
            self.aug = DefaultAugmenter()

        self._im_mode = mp.Value(c_char, b'1')

    def add(self,
            line: Optional[BBoxLine] = None,
            page: Optional[Segmentation] = None):
        """
        Adds an individual line or all lines on a page to the dataset.

        Args:
            line: BBoxLine container object of a line.
            page: Segmentation container object for a page.
        """
        if line:
            self.add_line(line)
        if page:
            self.add_page(page)
        if not (line or page):
            raise ValueError('Neither line nor page data provided in dataset builder')

    def add_page(self, page: Segmentation):
        """
        Adds all lines on a page to the dataset.

        Invalid lines will be skipped and a warning will be printed.

        Args:
            page: Segmentation container object for a page.
        """
        if page.type != 'bbox':
            raise ValueError(f'Invalid segmentation of type {page.type} (expected "bbox")')
        for line in page.lines:
            try:
                self.add_line(dataclasses.replace(line, imagename=page.imagename))
            except ValueError as e:
                logger.warning(e)

    def add_line(self, line: BBoxLine):
        """
        Adds a line to the dataset.

        Args:
            line: BBoxLine container object for a line.

        Raises:
            ValueError if the transcription of the line is empty after
            transformation or either baseline or bounding polygon are missing.
        """
        if line.type != 'bbox':
            raise ValueError(f'Invalid line of type {line.type} (expected "bbox")')

        text = line.text
        for func in self.text_transforms:
            text = func(text)
        if not text and self.skip_empty_lines:
            raise ValueError(f'Text line "{line.text}" is empty after transformations')
        if not line.bbox:
            raise ValueError('No bounding box given for line')

        self._images.append((line.imagename, line.bbox))
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
        self.training_set: List[Tuple[Union[Image.Image, torch.Tensor], torch.Tensor]] = []
        for im, gt in zip(self._images, self._gt):
            self.training_set.append((im, self.codec.encode(gt)))

    def no_encode(self) -> None:
        """
        Creates an unencoded dataset.
        """
        self.training_set: List[Tuple[Union[Image.Image, torch.Tensor], str]] = []
        for im, gt in zip(self._images, self._gt):
            self.training_set.append((im, gt))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.training_set[index]
        try:
            logger.debug(f'Attempting to load {item[0]}')
            im, bbox = item[0]
            flat_box = [x for point in bbox for x in point]
            xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
            ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
            im = Image.open(im)
            im = im.crop((xmin, ymin, xmax, ymax))
            im = self.transforms(im)
            if im.shape[0] == 3:
                im_mode = b'R'
            elif im.shape[0] == 1:
                im_mode = b'L'
            if is_bitonal(im):
                im_mode = b'1'
            with self._im_mode.get_lock():
                if im_mode > self._im_mode.value:
                    logger.info(f'Upgrading "im_mode" from {self._im_mode.value} to {im_mode}')
                    self._im_mode.value = im_mode
            if self.aug:
                im = im.permute((1, 2, 0)).numpy()
                o = self.aug(image=im)
                im = torch.tensor(o['image'].transpose(2, 0, 1))
            return {'image': im, 'target': item[1]}
        except Exception:
            raise
            self.failed_samples.add(index)
            idx = np.random.randint(0, len(self.training_set))
            logger.debug(traceback.format_exc())
            logger.info(f'Failed. Replacing with sample {idx}')
            return self[idx]

    def __len__(self) -> int:
        return len(self._images)

    @property
    def im_mode(self):
        return {b'1': '1',
                b'L': 'L',
                b'R': 'RGB'}[self._im_mode.value]
