
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from builtins import map
from builtins import zip
from builtins import str
from builtins import object

import dominate
import logging
import regex
from dominate.tags import div, span, meta, br
from itertools import count

logger = logging.getLogger(__name__)

class micro_hocr(object):
    """
    A simple class encapsulating hOCR attributes
    """
    def __init__(self):
        self.output = u''

    def __str__(self):
        return self.output

    def add(self, *args):
        if self.output:
            self.output += u'; '
        for arg in args:
            if isinstance(arg, str):
                self.output += arg + ' '
            elif isinstance(arg, tuple):
                self.output += u','.join([str(v) for v in arg]) + u' '
            else:
                self.output += str(arg) + u' '
        self.output = self.output.strip()


def max_bbox(boxes):
    """
    Calculates the minimal bounding box containing all boxes contained in an
    iterator.

    Args:
        boxes (iterator): An iterator returning tuples of the format (x0, y0,
                          x1, y1)
    Returns:
        A box covering all bounding boxes in the input argument
    """
    sbox = list(map(sorted, list(zip(*boxes))))
    return (sbox[0][0], sbox[1][0], sbox[2][-1], sbox[3][-1])


def delta(root=(0, 0, 0, 0), coordinates=None):
    """Calculates the running delta from a root coordinate according to the
    hOCR standard.

    It uses a root bounding box (x0, y0, x1, y1) and calculates the delta from
    the points (min(x0, x1), min(y0, y1)) and (min(x0, x1), max(y0, y1)) for
    the first and second pair of values in a delta (dx0, dy0, dx1, dy1)
    respectively.

    Args:
        coordinates (list): List of tuples of length 4 containing absolute
                            coordinates for character bounding boxes.

    Returns:
        A tuple dx0, dy0, dx1, dy1
    """
    for box in coordinates:
        yield (min(box[0], box[2]) - min(root[0], root[2]),
               min(box[1], box[3]) - min(root[1], root[3]),
               max(box[0], box[2]) - min(root[0], root[2]),
               max(box[1], box[3]) - max(root[1], root[3]))
        root = box


def hocr(records, image_name=u'', image_size=(0, 0), line_bbox=True,
         split_words=True, word_bbox=True, char_cuts=True,
         confidence_vals=True):
    """
    Merges a list of predictions and their corresponding character positions
    into an hOCR document.

    Args:
        records (iterable): List of kraken.rpred.ocr_record
        image_name (unicode): Name of the source image
        image_size (tuple): Dimensions of the source image
        line_bbox (bool): Enable writing of line bounding boxes
        split_words (bool): Split recognized line into words at
                            non-alphanumeric characters
        word_bbox (bool): Enable writing of word bounding boxes (only with
                          split_words)
        char_cuts (bool): Enable writing of character cuts (only with
                          line_bbox)
        confidence_vals (bool): Enable writing of confidence values for
                                recognition results
    """

    doc = dominate.document()

    with doc.head:
        meta(name='ocr-system', content='kraken')
        meta(name='ocr-capabilities', content='ocr_page ocr_line ocrx_word')
        meta(charset='utf-8')

    with doc:
        hocr_title = micro_hocr()
        if image_size > (1, 1):
            hocr_title.add('bbox', 0, 0, *[str(s) for s in image_size])
        if image_name:
            hocr_title.add(u'image', image_name)
        with div(cls='ocr_page', title=str(hocr_title)):
            for idx, record in enumerate(records):
                logger.debug('Adding record {} - {} to hocr'.format(idx, record.prediction))
                with span(cls='ocr_line', id='line_' + str(idx)) as line_span:
                    line_title = micro_hocr()
                    if line_bbox:
                        line_title.add('bbox', *max_bbox(record.cuts))
                    if char_cuts:
                        line_title.add('cuts',
                                       *list(delta(max_bbox(record.cuts),
                                                   record.cuts)))
                    # only add if field contains text to avoid unseemly empty
                    # titles
                    if str(line_title):
                        line_span['title'] = str(line_title)
                    if split_words:
                        logger.debug('Splitting record into words')
                        splits = regex.split(u'(\w+)', record.prediction)
                        line_offset = 0
                        # operate on pairs of non-word character strings and
                        # words. The former are encoded in ocrx_cinfo classes
                        # while the latter is adorned with ocrx_word classes.
                        for w_idx, non_word, word in zip(count(), splits[0::2],
                                                         splits[1::2]):
                            # add non word blocks only if they contain actual
                            # text
                            if non_word:
                                nw_span = span(non_word, cls='ocrx_block',
                                               id='block_' + str(idx) + '_' +
                                               str(w_idx))
                                nw = micro_hocr()
                                if word_bbox:
                                    nw.add('bbox',
                                           *max_bbox(record.cuts[line_offset:line_offset
                                                                 +
                                                                 len(non_word)]))
                                if confidence_vals:
                                    nw.add('x_conf', *[str(int(100*v)) for v in
                                                       record.confidences[line_offset:line_offset
                                                                          +
                                                                          len(non_word)]])
                                if str(nw):
                                    nw_span['title'] = str(nw)
                                line_offset += len(non_word)
                            w_span = span(word, cls='ocrx_word', id='word_' +
                                          str(idx) + '_' + str(w_idx))
                            w = micro_hocr()
                            if word_bbox:
                                w.add('bbox',
                                      *max_bbox(record.cuts[line_offset:line_offset
                                                            + len(word)]))
                            if confidence_vals:
                                w.add('x_conf', *[str(int(100*v)) for v in
                                                  record.confidences[line_offset:line_offset
                                                                     +
                                                                     len(word)]])
                            if str(w):
                                w_span['title'] = str(w)
                            line_offset += len(word)
                    else:
                        line_span.add(record.prediction)
                br()
    return doc
