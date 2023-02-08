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
import torch
import regex
import unicodedata
import bidi.algorithm as bd

from os import PathLike
from pathlib import Path
from PIL import Image
from PIL.Image import Resampling

from typing import Tuple, Optional, Callable, Any, Union

from kraken.binarization import nlbin
from kraken.lib.lineest import dewarp, CenterNormalizer


def pil_to_mode(im: Image.Image, mode: str) -> Image.Image:
    return im.convert(mode)


def pil_to_bin(im: Image.Image) -> Image.Image:
    return nlbin(im)


def dummy(x: Any) -> Any:
    return x


def pil_dewarp(im: Image.Image, lnorm: CenterNormalizer) -> Image.Image:
    return dewarp(lnorm, im)


def pil_fixed_resize(im: Image.Image, scale: Tuple[int, int]) -> Image.Image:
    return _fixed_resize(im, scale, Resampling.LANCZOS)


def tensor_invert(im: torch.Tensor) -> torch.Tensor:
    return im.max() - im


def tensor_permute(im: torch.Tensor, perm: Tuple[int, ...]) -> torch.Tensor:
    return im.permute(*perm)


def _fixed_resize(img: Image.Image, size: Tuple[int, int], interpolation: int = Resampling.LANCZOS):
    """
    Doesn't do the annoying runtime scale dimension switching the default
    pytorch transform does.

    Args:
        img (PIL.Image.Image): image to resize
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


def text_normalize(text: str, normalization: str) -> str:
    return unicodedata.normalize(normalization, text)


def text_whitespace_normalize(text: str) -> str:
    return regex.sub(r'\s', ' ', text).strip()


def text_reorder(text: str, base_dir: Optional[str] = None) -> str:
    return bd.get_display(text, base_dir=base_dir)


def default_split(x: Union[PathLike, str]) -> str:
    x = Path(x)
    while x.suffixes:
        x = x.with_suffix('')
    return str(x)


def suffix_split(x: Union[PathLike, str], split: Callable[[Union[PathLike, str]], str], suffix: str) -> str:
    return split(x) + suffix
