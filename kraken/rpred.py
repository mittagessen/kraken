from __future__ import absolute_import, division

import cPickle
import gzip
import bz2
import sys
import numpy as np
import kraken.lib.lstm
import kraken.lib.lineest

from kraken.lib import lstm
from kraken.lib.util import pil2array


def load_rnn(fname):
    """
    Loads a pickled lstm rnn.

    Args:
        fname (unicode): Path to the pickle object

    Returns:
        Unpickled object

    Raises:

    """

    def find_global(mname, cname):
        aliases = {
            'lstm.lstm': kraken.lib.lstm,
            'ocrolib.lstm': kraken.lib.lstm,
            'ocrolib.lineest': kraken.lib.lineest,
        }
        if mname in aliases:
            return getattr(aliases[mname], cname)
        return getattr(sys.modules[mname], cname)

    of = open
    if fname.endswith(u'.gz'):
        of = gzip.open
    elif fname.endswith(u'.bz2'):
        of = bz2.BZ2File
    with of(fname, 'rb') as fp:
        unpickler = cPickle.Unpickler(fp)
        unpickler.find_global = find_global
        return unpickler.load()


def extract_boxes(im, bounds):
    """
    Yields the subimages of image im defined in the list of bounding boxes in
    bounds preserving order.

    Args:
        im (PIL.Image): Input image
        bounds (list): A list of tuples (x1, y1, x2, y2)

    Yields:
        (PIL.Image) the extracted subimage
    """
    for box in bounds:
        yield im.crop(box), box


def rpred(network, im, bounds, pad=16, stats=None):
    """
    Uses a RNN to recognize text

    Args:
        network (kraken.lib.lstm.SegRecognizer): A SegRecognizer object
        im (PIL.Image): Image to extract text from
        bounds (iterable): An iterable returning a tuple defining the absolute
                           coordinates (x0, y0, x1, y1) of a text line in the
                           Image.
        pad (int): Extra blank padding to the left and right of text line
        stats (bool): Switch to enable statistics calculation

    Returns:
        A generator returning a tuple containing the recognized text (0),
        absolute character positions in the image (1), and miscellaneous
        statistics if enabled (2).
    """

    lnorm = getattr(network, 'lnorm', None)

    for box, coords in extract_boxes(im, bounds):
        if stats:
            stats = []
        line = pil2array(box)
        raw_line = line.copy()
        # dewarp line
        temp = np.amax(line)-line
        temp = temp*1.0/np.amax(temp)
        lnorm.measure(temp)
        line = lnorm.normalize(line, cval=np.amax(line))
        line = lstm.prepare_line(line, pad)
        pred = network.predictString(line)

        # calculate recognized LSTM locations of characters
        scale = len(raw_line.T)/(len(network.outputs)-2 * pad)
        result = lstm.translate_back(network.outputs, pos=1)
        pos = [(coords[0], coords[1], coords[0], coords[3])]
        for r, c in result:
            conf = network.outputs[r, c]
            pos.append((pos[-1][2], coords[1],
                        coords[0] + int((r-pad) * scale),
                        coords[3]))
            if stats:
                stats.append(network.outputs[r, c])
        yield pred, pos[1:], stats
