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
import torch
import traceback
import numpy as np
import torch.nn.functional as F
import shapely.geometry as geom

from os import path, PathLike
from PIL import Image
from shapely.ops import split
from itertools import groupby
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Sequence, Callable, Any, Union, Literal, Optional

from skimage.draw import polygon

from kraken.lib.xml import parse_alto, parse_page, parse_xml

from kraken.lib.exceptions import KrakenInputException

__all__ = ['BaselineSet']

import logging

logger = logging.getLogger(__name__)


class BaselineSet(Dataset):
    """
    Dataset for training a baseline/region segmentation model.
    """
    def __init__(self, imgs: Sequence[Union[PathLike, str]] = None,
                 suffix: str = '.path',
                 line_width: int = 4,
                 padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 mode: Optional[Literal['path', 'alto', 'page', 'xml']] = 'path',
                 augmentation: bool = False,
                 valid_baselines: Sequence[str] = None,
                 merge_baselines: Dict[str, Sequence[str]] = None,
                 valid_regions: Sequence[str] = None,
                 merge_regions: Dict[str, Sequence[str]] = None):
        """
        Reads a list of image-json pairs and creates a data set.

        Args:
            imgs:
            suffix: Suffix to attach to image base name to load JSON files
                    from.
            line_width: Height of the baseline in the scaled input.
            padding: Tuple of ints containing the left/right, top/bottom
                     padding of the input images.
            target_size: Target size of the image as a (height, width) tuple.
            mode: Either path, alto, page, xml, or None. In alto, page, and xml
                  mode the baseline paths and image data is retrieved from an
                  ALTO/PageXML file. In `None` mode data is iteratively added
                  through the `add` method.
            augmentation: Enable/disable augmentation.
            valid_baselines: Sequence of valid baseline identifiers. If `None`
                             all are valid.
            merge_baselines: Sequence of baseline identifiers to merge.  Note
                             that merging occurs after entities not in valid_*
                             have been discarded.
            valid_regions: Sequence of valid region identifiers. If `None` all
                           are valid.
            merge_regions: Sequence of region identifiers to merge. Note that
                           merging occurs after entities not in valid_* have
                           been discarded.
        """
        super().__init__()
        self.mode = mode
        self.im_mode = '1'
        self.pad = padding
        self.aug = None
        self.targets = []
        # n-th entry contains semantic of n-th class
        self.class_mapping = {'aux': {'_start_separator': 0, '_end_separator': 1}, 'baselines': {}, 'regions': {}}
        # keep track of samples that failed to load
        self.failed_samples = set()
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
                        if valid_baselines is None or set(line['tags'].values()).intersection(valid_baselines):
                            tags = set(line['tags'].values()).intersection(valid_baselines) if valid_baselines else line['tags'].values()
                            for tag in tags:
                                lines[self.mbl_dict.get(tag, tag)].append(line['baseline'])
                                self.class_stats['baselines'][self.mbl_dict.get(tag, tag)] += 1
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
            image: Union[PathLike, str, Image.Image],
            baselines: List[List[List[Tuple[int, int]]]] = None,
            regions: Dict[str, List[List[Tuple[int, int]]]] = None,
            *args,
            **kwargs):
        """
        Adds a page to the dataset.

        Args:
            im: Path to the whole page image
            baseline: A list containing dicts with a list of coordinates
                      and tags [{'baseline': [[x0, y0], ...,
                      [xn, yn]], 'tags': ('script_type',)}, ...]
            regions: A dict containing list of lists of coordinates
                     {'region_type_0': [[x0, y0], ..., [xn, yn]]],
                     'region_type_1': ...}.
        """
        if self.mode:
            raise Exception(f'The `add` method is incompatible with dataset mode {self.mode}')
        baselines_ = defaultdict(list)
        for line in baselines:
            if self.valid_baselines is None or set(line['tags'].values()).intersection(self.valid_baselines):
                tags = set(line['tags'].values()).intersection(self.valid_baselines) if self.valid_baselines else line['tags'].values()
                for tag in tags:
                    baselines_[tag].append(line['baseline'])
                    self.class_stats['baselines'][tag] += 1

                    if tag not in self.class_mapping['baselines']:
                        self.num_classes += 1
                        self.class_mapping['baselines'][tag] = self.num_classes - 1

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
                self.failed_samples.add(idx)
                idx = np.random.randint(0, len(self.imgs))
                logger.debug(traceback.format_exc())
                logger.info(f'Failed. Replacing with sample {idx}')
                return self[idx]
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
        scale = (image.shape[2] - 2*self.pad[1])/orig_size[0]
        t = torch.zeros((self.num_classes,) + tuple(np.subtract(image.shape[1:], (2*self.pad[1], 2*self.pad[0]))))
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
                line_pol = np.array(shp_line.buffer(self.line_width/2, cap_style=2).boundary.coords, dtype=int)
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
        target = F.pad(t, self.pad)
        if self.aug:
            image = image.permute(1, 2, 0).numpy()
            target = target.permute(1, 2, 0).numpy()
            o = self.aug(image=image, mask=target)
            image = torch.tensor(o['image']).permute(2, 0, 1)
            target = torch.tensor(o['mask']).permute(2, 0, 1)
        return image, target

    def __len__(self):
        return len(self.imgs)
