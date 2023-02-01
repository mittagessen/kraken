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
import numbers
import pkg_resources
import torch.nn.functional as F

from functools import partial
from torchvision import transforms
from collections import Counter
from typing import Dict, List, Tuple, Sequence, Any, Union

from kraken.lib.models import TorchSeqRecognizer
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.lineest import CenterNormalizer

from kraken.lib import functional_im_transforms as F_t

__all__ = ['ImageInputTransforms',
           'collate_sequences']

import logging

logger = logging.getLogger(__name__)


class ImageInputTransforms(transforms.Compose):
    def __init__(self,
                 batch: int,
                 height: int,
                 width: int,
                 channels: int,
                 pad: Union[int, Tuple[int, int], Tuple[int, int, int, int]],
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
        self._valid_norm = valid_norm
        self._force_binarization = force_binarization
        self._batch = batch
        self._channels = channels
        self.pad = pad

        self._create_transforms()

    def _create_transforms(self) -> None:
        height = self._scale[0]
        width = self._scale[1]
        self._center_norm = False
        self._mode = 'RGB' if self._channels == 3 else 'L'
        if height == 1 and width == 0 and self._channels > 3:
            perm = (1, 0, 2)
            self._scale = (self._channels, 0)
            self._channels = 1
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
            self.transforms.append(transforms.Pad(self._pad, fill=255))
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
    def pad(self, pad: Union[int, Tuple[int, int], Tuple[int, int, int, int]]) -> None:
        if not isinstance(pad, (numbers.Number, tuple, list)):
            raise TypeError('Got inappropriate padding arg')
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
