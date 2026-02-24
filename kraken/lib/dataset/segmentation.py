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
import traceback
import multiprocessing as mp
from collections import defaultdict
from ctypes import c_char
from itertools import groupby
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import shapely.geometry as geom
import torch
import torch.nn.functional as F
from PIL import Image
from shapely.ops import split
from skimage.draw import polygon
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from kraken.lib.dataset.utils import _get_type
from kraken.lib.segmentation import scale_regions
from kraken.lib.util import is_bitonal, open_image

if TYPE_CHECKING:
    from kraken.containers import Segmentation


__all__ = ['BaselineSet']

import logging

logger = logging.getLogger(__name__)


class SegmentationAugmenter():
    def __init__(self) -> None:
        self._blur = v2.RandomChoice([
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])
        self._affine = v2.RandomAffine(degrees=45,
                                       translate=(0.0625, 0.0625),
                                       scale=(0.8, 1.2),
                                       shear=(-5.0, 5.0),
                                       interpolation=InterpolationMode.BILINEAR,
                                       fill=0.0)
        self._perspective = v2.RandomPerspective(distortion_scale=0.2,
                                                 p=1.0,
                                                 interpolation=InterpolationMode.BILINEAR,
                                                 fill=0.0)
        self._color = v2.ColorJitter(brightness=0.1,
                                     contrast=0.1,
                                     saturation=0.1,
                                     hue=0.05)
        self._augment = v2.RandomApply([v2.Compose([
            v2.RandomApply([self._blur], p=0.2),
            v2.RandomApply([self._affine], p=0.2),
            v2.RandomApply([self._perspective], p=0.2),
            v2.RandomApply([self._color], p=0.3),
        ])], p=0.5)

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = tv_tensors.Mask(target)
        image, mask = self._augment(image, mask)
        return image.clamp(0.0, 1.0), mask.as_subclass(torch.Tensor)


class BaselineSet(Dataset):
    """
    Dataset for training a baseline/region segmentation model.
    """
    def __init__(self,
                 class_mapping: dict[str, dict[str, int]],
                 line_width: int = 4,
                 padding: tuple[int, int, int, int] = (0, 0, 0, 0),
                 im_transforms: Callable[[Any], torch.Tensor] = v2.Identity(),
                 augmentation: bool = False) -> None:
        """
        Creates a dataset for a text-line and region segmentation model.

        Args:
            line_width: Height of the baseline in the scaled input.
            padding: Tuple of ints containing the left/right, top/bottom
                     padding of the input images.
            augmentation: Enable/disable augmentation.
            class_mapping: class map.
        """
        super().__init__()
        # validate class_mapping structure
        required_keys = {'aux', 'baselines', 'regions'}
        if set(class_mapping.keys()) != required_keys:
            raise ValueError(f'class_mapping must have exactly keys {required_keys}, got {set(class_mapping.keys())}')
        for req in ('_start_separator', '_end_separator'):
            if req not in class_mapping['aux']:
                raise ValueError(f"class_mapping['aux'] must contain '{req}'")
        for section, sub_dict in class_mapping.items():
            for key, val in sub_dict.items():
                if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                    raise ValueError(f'class_mapping[{section!r}][{key!r}] must be a non-negative integer, got {val!r}')
        self.imgs = []
        self.pad = padding
        self.targets = []
        self.class_mapping = class_mapping

        # keep track of samples that failed to load
        self.failed_samples = set()
        self.class_stats = {'baselines': defaultdict(int), 'regions': defaultdict(int)}
        self.aug = None

        if augmentation:
            self.aug = SegmentationAugmenter()

        self.line_width = line_width
        self.transforms = im_transforms
        self.seg_type = None
        self._im_mode = mp.Value(c_char, b'1')

    @property
    def num_classes(self):
        return max(v for d in self.class_mapping.values() for v in d.values()) + 1

    @property
    def canonical_class_mapping(self) -> dict[str, dict[str, int]]:
        """Returns a one-to-one class mapping (one string per label index).

        For merged classes (multiple strings -> same index), the first string
        by insertion order is kept as the canonical name.
        """
        result = {}
        for section, sub_dict in self.class_mapping.items():
            seen_indices = set()
            canonical = {}
            for key, idx in sub_dict.items():
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    canonical[key] = idx
            result[section] = canonical
        return result

    @property
    def merged_classes(self) -> dict[str, dict[str, list[str]]]:
        """Returns merged class info: {section: {canonical_name: [aliases]}}.

        Only includes entries where multiple strings map to the same index.
        """
        result = {}
        for section, sub_dict in self.class_mapping.items():
            idx_to_names: dict[int, list[str]] = defaultdict(list)
            for key, idx in sub_dict.items():
                idx_to_names[idx].append(key)
            merged = {}
            for idx, names in idx_to_names.items():
                if len(names) > 1:
                    merged[names[0]] = names[1:]
            result[section] = merged
        return result

    def add(self, doc: 'Segmentation'):
        """
        Adds a page to the dataset.

        Args:
            doc: A Segmentation container class.
        """
        if doc.type != 'baselines':
            raise ValueError(f'{doc} is of type {doc.type}. Expected "baselines".')

        baselines_ = defaultdict(list)
        for line in doc.lines:
            tag = _get_type(line.tags)
            try:
                idx = self.class_mapping['baselines'][tag]
                baselines_[idx].append(line.baseline)
                self.class_stats['baselines'][tag] += 1
            except KeyError:
                continue

        regions_ = defaultdict(list)
        for k, v in doc.regions.items():
            try:
                idx = self.class_mapping['regions'][k]
                v = [x for x in v if x.boundary]
                regions_[idx].extend(v)
                self.class_stats['regions'][k] += len(v)
            except KeyError:
                continue
        self.targets.append({'baselines': baselines_, 'regions': regions_})
        self.imgs.append(doc.imagename)

    def _update_im_mode(self, im):
        if im.shape[0] == 3:
            im_mode = b'R'
        elif im.shape[0] == 1:
            im_mode = b'L'
        else:
            im_mode = b'R'
        if is_bitonal(im):
            im_mode = b'1'
        with self._im_mode.get_lock():
            if im_mode > self._im_mode.value:
                logger.info(f'Upgrading "im_mode" from {self._im_mode.value} to {im_mode}')
                self._im_mode.value = im_mode

    def __getitem__(self, idx):
        if len(self.failed_samples) == len(self):
            raise ValueError(f'All {len(self)} samples in dataset invalid.')
        im = self.imgs[idx]
        target = self.targets[idx]
        if not isinstance(im, Image.Image):
            try:
                logger.debug(f'Attempting to load {im}')
                im = open_image(im)
                im, target, baselines = self.transform(im, target)
                self._update_im_mode(im)
                return {'image': im, 'target': target, 'baselines': baselines}
            except Exception:
                self.failed_samples.add(idx)
                idx = np.random.randint(0, len(self.imgs))
                logger.debug(traceback.format_exc())
                logger.info(f'Failed. Replacing with sample {idx}')
                return self[idx]
        im, target, baselines = self.transform(im, target)
        self._update_im_mode(im)
        return {'image': im, 'target': target, 'baselines': baselines}

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
        scale = (image.shape[2] - 2 * self.pad[1]) / orig_size[0]
        t = torch.zeros((self.num_classes,) + tuple(np.subtract(image.shape[1:], (2 * self.pad[1], 2 * self.pad[0]))))
        start_sep_cls = self.class_mapping['aux']['_start_separator']
        end_sep_cls = self.class_mapping['aux']['_end_separator']

        scaled_baselines = defaultdict(list)
        for cls_idx, lines in target['baselines'].items():
            for line in lines:
                # buffer out line to desired width
                line = [k for k, g in groupby(line)]
                line = np.array(line) * scale
                scaled_baselines[cls_idx].append(line.tolist())
                shp_line = geom.LineString(line)
                split_offset = min(5, shp_line.length / 2)
                line_pol = np.array(shp_line.buffer(self.line_width / 2, cap_style=2).boundary.coords, dtype=int)
                rr, cc = polygon(line_pol[:, 1], line_pol[:, 0], shape=image.shape[1:])
                t[cls_idx, rr, cc] = 1
                split_pt = shp_line.interpolate(split_offset).buffer(0.001)
                # top
                start_sep = np.array((split(shp_line, split_pt).geoms[0].buffer(self.line_width,
                                                                                cap_style=3).boundary.coords), dtype=int)
                rr_s, cc_s = polygon(start_sep[:, 1], start_sep[:, 0], shape=image.shape[1:])
                t[start_sep_cls, rr_s, cc_s] = 1
                t[start_sep_cls, rr, cc] = 0
                split_pt = shp_line.interpolate(-split_offset).buffer(0.001)
                # top
                end_sep = np.array((split(shp_line, split_pt).geoms[-1].buffer(self.line_width,
                                                                               cap_style=3).boundary.coords), dtype=int)
                rr_s, cc_s = polygon(end_sep[:, 1], end_sep[:, 0], shape=image.shape[1:])
                t[end_sep_cls, rr_s, cc_s] = 1
                t[end_sep_cls, rr, cc] = 0
        for cls_idx, regions in target['regions'].items():
            for region in regions:
                region = np.array(scale_regions([region.boundary], scale)[0])
                rr, cc = polygon(region[:, 1], region[:, 0], shape=image.shape[1:])
                t[cls_idx, rr, cc] = 1
        target = F.pad(t, self.pad)
        if self.aug:
            image, target = self.aug(image, target)
        return image, target, dict(scaled_baselines)

    def __len__(self):
        return len(self.imgs)

    @property
    def im_mode(self):
        return {b'1': '1',
                b'L': 'L',
                b'R': 'RGB'}[self._im_mode.value]
