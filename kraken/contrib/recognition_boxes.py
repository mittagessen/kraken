#!/usr/bin/env python
"""
Draws transparent character bounding boxes over images giving a legacy
segmenter model.
"""

import os
import sys

from PIL import Image, ImageDraw

from kraken.pageseg import segment
from kraken.binarization import nlbin
from kraken.rpred import rpred
from itertools import cycle
from kraken.lib import models

cmap = cycle([(230, 25, 75, 127),
              (60, 180, 75, 127),
              (255, 225, 25, 127),
              (0, 130, 200, 127),
              (245, 130, 48, 127),
              (145, 30, 180, 127),
              (70, 240, 240, 127)])

net = models.load_any(sys.argv[1])

for fname in sys.argv[2:]:
    im = Image.open(fname)
    print(fname)
    im = nlbin(im)
    res = segment(im, maxcolseps=0)
    pred = rpred(net, im, res)
    im = im.convert('RGBA')
    tmp = Image.new('RGBA', im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    for line in pred:
        for box in line.cuts:
            draw.rectangle(box, fill=next(cmap))
    im = Image.alpha_composite(im, tmp)
    im.save('high_{}'.format(os.path.basename(fname)))
