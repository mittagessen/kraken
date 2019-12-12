"""
Ocropus's magic PIL-numpy array conversion routines. They express slightly
different behavior from PIL.Image.toarray().
"""
import unicodedata
import numpy as np

from PIL import Image

__all__ = ['pil2array', 'array2pil', 'is_bitonal', 'make_printable', 'get_im_str']


def pil2array(im: Image.Image, alpha: int = 0) -> np.array:
    if im.mode == '1':
        return np.array(im.convert('L'))
    return np.array(im)


def array2pil(a: np.array) -> Image:
    if a.dtype == np.dtype("B"):
        if a.ndim == 2:
            return Image.frombytes("L", (a.shape[1], a.shape[0]),
                                   a.tostring())
        elif a.ndim == 3:
            return Image.frombytes("RGB", (a.shape[1], a.shape[0]),
                                   a.tostring())
        else:
            raise Exception("bad image rank")
    elif a.dtype == np.dtype('float32'):
        return Image.frombytes("F", (a.shape[1], a.shape[0]), a.tostring())
    else:
        raise Exception("unknown image type")


def is_bitonal(im: Image.Image) -> bool:
    """
    Tests a PIL.Image for bitonality.

    Args:
        im (PIL.Image.Image): Image to test

    Returns:
        True if the image contains only two different color values. False
        otherwise.
    """
    return im.getcolors(2) is not None and len(im.getcolors(2)) == 2


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
