from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object

import numpy as np

from kraken.lib import lstm
from kraken.lib.util import pil2array, array2pil
from kraken.lib.lineest import CenterNormalizer
from kraken.lib.exceptions import KrakenInputException


class ocr_record(object):
    """
    A record object containing the recognition result of a single line
    """
    def __init__(self, prediction, cuts, confidences):
        self.prediction = prediction
        self.cuts = cuts
        self.confidences = confidences

    def __len__(self):
        return len(self.prediction)

    def __str__(self):
        return self.prediction

    def __iter__(self):
        self.idx = -1
        return self

    def __next__(self):
        if self.idx + 1 < len(self):
            self.idx += 1
            return (self.prediction[self.idx], self.cuts[self.idx],
                    self.confidences[self.idx])
        else:
            raise StopIteration

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key >= len(self):
                raise IndexError('Index (%d) is out of range' % key)
            return (self.prediction[key], self.cuts[key],
                    self.confidences[key])
        else:
            raise TypeError('Invalid argument type')


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
        if (box < (0, 0, 0, 0) or box[::2] > (im.size[0], im.size[0]) or
           box[1::2] > (im.size[1], im.size[1])):
            raise KrakenInputException('Line outside of image bounds')
        yield im.crop(box), box


def dewarp(normalizer, im):
    """
    Dewarps an image of a line using a kraken.lib.lineest.CenterNormalizer
    instance.

    Args:
        normalizer (kraken.lib.lineest.CenterNormalizer): A line normalizer
                                                          instance
        im (PIL.Image): Image to dewarp

    Returns:
        PIL.Image containing the dewarped image.
    """
    line = pil2array(im)
    temp = np.amax(line)-line
    temp = temp*1.0/np.amax(temp)
    normalizer.measure(temp)
    line = normalizer.normalize(line, cval=np.amax(line))
    return array2pil(line)


def rpred(network, im, bounds, pad=16, line_normalization=True):
    """
    Uses a RNN to recognize text

    Args:
        network (kraken.lib.lstm.SegRecognizer): A SegRecognizer object
        im (PIL.Image): Image to extract text from
        bounds (iterable): An iterable returning a tuple defining the absolute
                           coordinates (x0, y0, x1, y1) of a text line in the
                           Image.
        pad (int): Extra blank padding to the left and right of text line
        line_normalization (bool): Dewarp line using the line estimator
                                   contained in the network. If no normalizer
                                   is available one using the default
                                   parameters is created. By aware that you may
                                   have to scale lines manually to the target
                                   line height if disabled.
    Yields:
        A tuple containing the recognized text (0), absolute character
        positions in the image (1), and confidence values for each
        character(2).
    """

    lnorm = getattr(network, 'lnorm', CenterNormalizer())

    for box, coords in extract_boxes(im, bounds):
        raw_line = pil2array(box)
        if line_normalization:
            # fail gracefully and return no recognition result in case the
            # input line can not be normalized.
            try:
                box = dewarp(lnorm, box)
            except ValueError:
                yield ocr_record('', [], [])
                continue
        line = pil2array(box)
        line = lstm.prepare_line(line, pad)
        pred = network.predictString(line)

        # calculate recognized LSTM locations of characters
        scale = len(raw_line.T)/(len(network.outputs)-2 * pad)
        result = lstm.translate_back(network.outputs, pos=1)
        pos = [(coords[0], coords[1], coords[0], coords[3])]
        conf = [network.outputs[r, c] for r, c in result if c != 0]
        cuts = [int((r-pad)*scale) for (r, c) in result if c != 0]
        if len(cuts) != len(pred):
            raise KrakenInputException('character cuts and result not of same length!')
        # append last offset to end of line
        cuts.append(coords[2] - coords[0])
        pos = []
        for i, d in enumerate(cuts):
            try:
                pos.append((coords[0] + d, coords[1], coords[0] + cuts[i+1], coords[3]))
            except:
                break
        yield ocr_record(pred, pos, conf)
