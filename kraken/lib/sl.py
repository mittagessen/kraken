import numpy as np


def dim0(s):
    """Dimension of the slice list for dimension 0."""
    return s[0].stop-s[0].start


def dim1(s):
    """Dimension of the slice list for dimension 1."""
    return s[1].stop-s[1].start


def area(a):
    """Return the area of the slice list (ignores anything past a[:2]."""
    return np.prod([max(x.stop-x.start, 0) for x in a[:2]])


def width(s):
    return s[1].stop-s[1].start


def height(s):
    return s[0].stop-s[0].start


def aspect(a):
    return height(a)*1.0/width(a)


def xcenter(s):
    return np.mean([s[1].stop, s[1].start])


def ycenter(s):
    return np.mean([s[0].stop, s[0].start])


def center(s):
    return (ycenter(s), xcenter(s))
