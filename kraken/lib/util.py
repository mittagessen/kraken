"""
Ocropus's magic PIL-numpy array conversion routines. They express slightly
different behavior from PIL.Image.toarray().
"""
import unicodedata
import uuid
from typing import TYPE_CHECKING, Callable, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image

from kraken.containers import BBoxLine
from kraken.lib import functional_im_transforms as F_t
from kraken.lib.exceptions import KrakenInputException

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['pil2array', 'array2pil', 'is_bitonal', 'make_printable', 'get_im_str', 'parse_gt_path']


def pil2array(im: Image.Image, alpha: int = 0) -> np.ndarray:
    if im.mode == '1':
        return np.array(im.convert('L'))
    return np.array(im)


def array2pil(a: np.ndarray) -> Image.Image:
    if a.dtype == np.dtype("B"):
        if a.ndim == 2:
            return Image.frombytes("L", (a.shape[1], a.shape[0]),
                                   a.tobytes())
        elif a.ndim == 3:
            return Image.frombytes("RGB", (a.shape[1], a.shape[0]),
                                   a.tobytes())
        else:
            raise Exception("bad image rank")
    elif a.dtype == np.dtype('float32'):
        return Image.frombytes("F", (a.shape[1], a.shape[0]), a.tobytes())
    else:
        raise Exception("unknown image type")


def is_bitonal(im: Union[Image.Image, torch.Tensor]) -> bool:
    """
    Tests a PIL image or torch tensor for bitonality.

    Args:
        im: Image to test

    Returns:
        True if the image contains only two different color values. False
        otherwise.
    """
    if isinstance(im, Image.Image):
        return im.getcolors(2) is not None and len(im.getcolors(2)) == 2
    elif isinstance(im, torch.Tensor):
        return len(im.unique()) == 2


def get_im_str(im: Image.Image) -> str:
    return im.filename if hasattr(im, 'filename') else str(im)


def is_printable(char: str) -> bool:
    """
    Determines if a chode point is printable/visible when printed.

    Args:
        char (str): Input code point.

    Returns:
        True if printable, False otherwise.
    """
    letters = ('LC', 'Ll', 'Lm', 'Lo', 'Lt', 'Lu')
    numbers = ('Nd', 'Nl', 'No')
    punctuation = ('Pc', 'Pd', 'Pe', 'Pf', 'Pi', 'Po', 'Ps')
    symbol = ('Sc', 'Sk', 'Sm', 'So')
    printable = letters + numbers + punctuation + symbol

    return unicodedata.category(char) in printable


def make_printable(char: str) -> str:
    """
    Takes a Unicode code point and return a printable representation of it.

    Args:
        char (str): Input code point

    Returns:
        Either the original code point, the name of the code point if it is a
        combining mark, whitespace etc., or the hex code if it is a control
        symbol.
    """
    if not char or is_printable(char):
        return char
    elif unicodedata.category(char) in ('Cc', 'Cs', 'Co'):
        return '0x{:x}'.format(ord(char))
    else:
        return unicodedata.name(char)


def parse_gt_path(path: Union[str, 'PathLike'],
                  suffix: str = '.gt.txt',
                  split: Callable[[Union['PathLike', str]], str] = F_t.default_split,
                  skip_empty_lines: bool = True,
                  base_dir: Optional[Literal['L', 'R']] = None,
                  text_direction: Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'] = 'horizontal-lr') -> BBoxLine:
    """
    Returns a BBoxLine from a image/text file pair.

    Args:
        path: Path to image file
        suffix: Suffix of the corresponding ground truth text file to image
                file in `path`.
        split: Suffix stripping function.
        skip_empty_lines: Whether to raise an exception if ground truth is
                          empty or text file is missing.
        base_dir: Unicode BiDi algorithm base direction
        text_direction: Orientation of the line box.
    """
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception as e:
        raise KrakenInputException(e)

    gt = ''
    try:
        with open(F_t.suffix_split(path, split=split, suffix=suffix), 'r', encoding='utf-8') as fp:
            gt = fp.read().strip('\n\r')
    except FileNotFoundError:
        if not skip_empty_lines:
            raise KrakenInputException(f'No text file found for ground truth line {path}.')

    if not gt and skip_empty_lines:
        raise KrakenInputException(f'No text for ground truth line {path}.')

    return BBoxLine(id=uuid.uuid4(),
                    bbox=((0, 0), (w, 0), (w, h), (0, h)),
                    text=gt,
                    base_dir=base_dir,
                    imagename=path,
                    text_direction=text_direction)
