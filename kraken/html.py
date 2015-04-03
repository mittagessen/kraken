
from __future__ import absolute_import, print_function

import dominate
import regex
from dominate.tags import div, span, meta, br
from itertools import count


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
    sbox = map(sorted, zip(*boxes))
    return (sbox[0][0], sbox[1][0], sbox[2][-1], sbox[3][-1])


def hocr(predictions, positions, image_name=u'', image_size=(0, 0),
         line_bbox=True, split_words=True, word_bbox=True, char_cuts=True,
         confidence_vals=True):
    """
    Merges a list of predictions and their corresponding character positions
    into an hOCR document.

    Args:
        predictions (iterator): A list of unicode objects containing
                                predictions of a single line
        positions (iterator): A list of tuples containing the cuts of each
                              character
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
        page_title = []
        if image_size > (1, 1):
            page_title.append(u'bbox 0 0 ' + u' '.join([str(s) for s in
                                                        image_size]))
        if image_name:
            page_title.append(u'image ' + image_name)
        with div(cls='ocr_page', title=u'; '.join(page_title)):
            for idx, line, pos in zip(count(), predictions, positions):
                with span(cls='ocr_line', id='line_' + str(idx), title='bbox '
                          + u' '.join([str(s) for s in max_bbox(pos)])):
                    # this splits according to unicode alphanumeric character
                    # classes.
                    splits = regex.split(u'(\w+)', line)
                    line_offset = 0
                    for w_idx, non_word, word in zip(count(), splits[0::2],
                                                     splits[1::2]):
                        if non_word:
                            span(non_word, title='bbox ' +
                                 u' '.join([str(s) for s in
                                            max_bbox(pos[line_offset:line_offset+len(non_word)])]))
                            line_offset += len(non_word)
                        span(word, cls='ocrx_word', id='word_' + str(idx) + '_'
                             + str(w_idx), title='bbox ' +
                             u' '.join([str(s) for s in
                                        max_bbox(pos[line_offset:line_offset+len(word)])]))
                        line_offset += len(word)
                br()
    return doc
