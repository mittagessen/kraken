"""
Ocropus's magic PIL-numpy array conversion routines. They express slightly
different behavior from PIL.Image.toarray().
"""
import numpy as np

from PIL import Image

__all__ = ['pil2array', 'array2pil']


def pil2array(im: Image, alpha: int = 0) -> np.array:
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


def is_bitonal(im: Image) -> bool:
    """
    Tests a PIL.Image for bitonality.

    Args:
        im (PIL.Image): Image to test

    Returns:
        True if the image contains only two different color values. False
        otherwise.
    """
    return im.getcolors(2) is not None


def get_im_str(im: Image) -> str:
    return im.filename if hasattr(im, 'filename') else str(im)
