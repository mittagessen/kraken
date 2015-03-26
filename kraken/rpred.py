from __future__ import absolute_import

import cPickle
import gzip
import bz2

from ocrolib import lstm
from PIL import Image

def load_lstm(fname):
    """
    Loads a pickled lstm rnn.

    Args:
        fname (unicode): Path to the pickle object

    Returns:
        Unpickled object

    Raises:

    """

    of = open
    if fname.endswith(u'.gz'):
        of = gzip.open
    elif fname.endswith(u'.bz2'):
        of = bz2.BZ2File
    with of(fname, 'rb') as fp:
        unpickler = cPickle.Unpickler(fp)
        return unpickler.load()

def extract_boxes(im, bounds):
    """
    Yields the subimages of image im defined in the list of bounding boxes in
    bounds preserving order.

    Args:
        im (PIL.Image): Input image
        bounds (list): A list of tuples (x1, y1, x2, y2)

    Yields:
        (PIL.Image) the extract subimage
    """

def dewarp():
    temp = amax(line)-line
    temp = temp*1.0/amax(temp)
    before_x = len(line[0])
    lnorm.measure(temp)
    line = lnorm.normalize(line,cval=amax(line))
    scale = before_x / float(len(line[0]))

def rpred():
    scale = 0
    #line = ocrolib.read_image_gray(fname)


    line = lstm.prepare_line(line,args.pad)
    sequence = network.predictSequence(line)
    sequence_char_only = [c for (r,c) in sequence]
    sequence_x = [int(x*scale)  for (x,c) in sequence if c != 0]
    sequence_x_csv = ""
    for x in sequence_x:
        sequence_x_csv += str(x) + ","
    if (len(sequence_x_csv) > 0):
        sequence_x_csv = sequence_x_csv[:-1]
    pred = network.l2s(sequence_char_only)
    print "sequence xs: ", sequence_x 
    print "pred len: ", len(pred)

    ocrolib.write_text(base+".txt",pred)
    ocrolib.write_text(base+".csv",sequence_x_csv)

    result = [x for x in result if x is not None]
