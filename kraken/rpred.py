from __future__ import absolute_import

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
        yield im.crop(box)


def rpred(network, im, bounds, pad=16):
    scale = 0
    lnorm = getattr(network, 'lnorm', None)

    for box in extract_boxes(im, bounds):
        line = pil2array(box)
        temp = np.amax(line)-line
        temp = temp*1.0/np.amax(temp)
        before_x = len(line[0])
        lnorm.measure(temp)
        line = lnorm.normalize(line, cval=np.amax(line))
        scale = before_x / float(len(line[0]))

        line = lstm.prepare_line(line, pad)
        sequence = network.predictSequence(line)
        sequence_char_only = [c for (r, c) in sequence]
        sequence_x = [int(x*scale) for (x, c) in sequence if c != 0]
        sequence_x_csv = ""
        for x in sequence_x:
            sequence_x_csv += str(x) + ","
        if (len(sequence_x_csv) > 0):
            sequence_x_csv = sequence_x_csv[:-1]
        pred = network.l2s(sequence_char_only)
        yield (pred,)
