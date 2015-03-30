"""
Ocropus's magic PIL-numpy array conversion routines. They express slightly
different behavior from PIL.Image.toarray() but the reason is rather
mysterious.

"""

import numpy as np

from PIL import Image


def pil2array(im, alpha=0):
    if im.mode == "L":
        a = np.fromstring(im.tostring(), 'B')
        a.shape = im.size[1], im.size[0]
        return a
    if im.mode == "RGB":
        a = np.fromstring(im.tostring(), 'B')
        a.shape = im.size[1], im.size[0], 3
        return a
    if im.mode == "RGBA":
        a = np.fromstring(im.tostring(), 'B')
        a.shape = im.size[1], im.size[0], 4
        if not alpha:
            a = a[:, :, :3]
        return a
    return pil2array(im.convert("L"))


def array2pil(a):
    if a.dtype == np.dtype("B"):
        if a.ndim == 2:
            return Image.fromstring("L", (a.shape[1], a.shape[0]),
                                    a.tostring())
        elif a.ndim == 3:
            return Image.fromstring("RGB", (a.shape[1], a.shape[0]),
                                    a.tostring())
        else:
            raise Exception("bad image rank")
    elif a.dtype == np.dtype('float32'):
        return Image.fromstring("F", (a.shape[1], a.shape[0]), a.tostring())
    else:
        raise Exception("unknown image type")
