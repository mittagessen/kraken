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
import traceback
import numpy as np
import pyarrow as pa

from PIL import Image
from os import PathLike
from functools import partial
from torchvision import transforms
from collections import Counter
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Callable, Optional, Any, Union

from kraken.lib.util import is_bitonal
from kraken.lib.codec import PytorchCodec
from kraken.lib.segmentation import extract_polygons
from kraken.lib.exceptions import KrakenInputException, KrakenEncodeException

from kraken.lib import functional_im_transforms as F_t

__all__ = ['ArrowIPCRecognitionDataset',
           'PolygonGTDataset',
           'GroundTruthDataset']

import logging

logger = logging.getLogger(__name__)


class ArrowIPCRecognitionDataset(Dataset):
    """
    Dataset for training a recognition model from a precompiled dataset in
    Arrow IPC format.
    """
    def __init__(self,
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 skip_empty_lines: bool = True,
                 reorder: Union[bool, str] = True,
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
        self.alphabet = Counter()  # type: Counter
        self.text_transforms = []  # type: List[Callable[[str], str]]
        self.failed_samples = set()
        self.transforms = im_transforms
        self.aug = None
        self._split_filter = split_filter
        self._num_lines = 0
        self.arrow_table = None
        self.codec = None
        self.skip_empty_lines = skip_empty_lines

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

        self.im_mode = self.transforms.mode

    def add(self, file: Union[str, PathLike]) -> None:
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
                    text = self._apply_text_transform(ds_table.column('lines')[index].as_py(),)
                except KrakenInputException:
                    mask[index] = False
                    continue
            num_lines = np.count_nonzero(mask)
            logger.debug(f'Filtering out {np.count_nonzero(~mask)} empty lines')
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
                 reorder: Union[bool, str] = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 augmentation: bool = False) -> None:
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
        self._images = []  # type:  Union[List[Image], List[torch.Tensor]]
        self._gt = []  # type:  List[str]
        self.alphabet = Counter()  # type: Counter
        self.text_transforms = []  # type: List[Callable[[str], str]]
        self.transforms = im_transforms
        self.aug = None
        self.skip_empty_lines = skip_empty_lines
        self.failed_samples = set()

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
        self._images.append((kwargs['image'], kwargs['baseline'], kwargs['boundary']))
        self._gt.append(kwargs['text'])
        self.alphabet.update(kwargs['text'])

    def parse(self,
              image: Union[PathLike, str, Image.Image],
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
        if not text and self.skip_empty_lines:
            raise KrakenInputException(f'Text line "{orig_text}" is empty after transformations')
        if not baseline:
            raise KrakenInputException('No baseline given for line')
        if not boundary:
            raise KrakenInputException('No boundary given for line')
        return {'text': text,
                'image': image,
                'baseline': baseline,
                'boundary': boundary,
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
            self.failed_samples.add(index)
            idx = np.random.randint(0, len(self.training_set))
            logger.debug(traceback.format_exc())
            logger.info(f'Failed. Replacing with sample {idx}')
            return self[idx]

    def __len__(self) -> int:
        return len(self._images)


class GroundTruthDataset(Dataset):
    """
    Dataset for training a line recognition model.

    All data is cached in memory.
    """
    def __init__(self, split: Callable[[Union[PathLike, str]], str] = F_t.default_split,
                 suffix: str = '.gt.txt',
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 skip_empty_lines: bool = True,
                 reorder: Union[bool, str] = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 augmentation: bool = False) -> None:
        """
        Reads a list of image-text pairs and creates a ground truth set.

        Args:
            split: Function for generating the base name without
                   extensions from paths
            suffix: Suffix to attach to image base name for text
                    retrieval
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
        self.suffix = suffix
        self.split = partial(F_t.suffix_split, split=split, suffix=suffix)
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
        self._images.append(kwargs['image'])
        self._gt.append(kwargs['text'])
        self.alphabet.update(kwargs['text'])

    def parse(self, image: Union[PathLike, str, Image.Image], *args, **kwargs) -> Dict:
        """
        Parses a sample for this dataset.

        This is mostly used to parallelize populating the dataset.

        Args:
            image (str): Input image path
        """
        with open(self.split(image), 'r', encoding='utf-8') as fp:
            text = fp.read().strip('\n\r')
            for func in self.text_transforms:
                text = func(text)
            if not text and self.skip_empty_lines:
                raise KrakenInputException(f'Text line is empty ({fp.name})')
        return {'image': image, 'text': text, 'preparse': True}

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
            self.failed_samples.add(index)
            idx = np.random.randint(0, len(self.training_set))
            logger.debug(traceback.format_exc())
            logger.info(f'Failed. Replacing with sample {idx}')
            return self[idx]

    def __len__(self) -> int:
        return len(self._images)
