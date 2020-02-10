#!/usr/bin/env python
"""
Draws a transparent overlay of baseline segmenter output over a list of image
files.
"""
import os
import sys

from PIL import Image, ImageDraw

from kraken.blla import segment
from kraken.rpred import rpred
from itertools import cycle
from kraken.lib.vgsl import TorchVGSLModel

cmap = cycle([(230, 25, 75, 127),
              (60, 180, 75, 127)])

bmap = (0, 130, 200, 255)

net = TorchVGSLModel.load_model(sys.argv[1])

for fname in sys.argv[2:]:
    im = Image.open(fname)
    print(fname)
    res = segment(im, model=net)
    im = im.convert('RGBA')
    tmp = Image.new('RGBA', im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    for idx, line in enumerate(res['lines']):
        c = next(cmap)
        draw.polygon([tuple(x) for x in line['boundary']], fill=c, outline=c[:3])
        draw.line([tuple(x) for x in line['baseline']], fill=bmap, width=2, joint='curve')
        draw.text(line['baseline'][0], str(idx), fill=(0, 0, 0, 255))
    im = Image.alpha_composite(im, tmp)
    im.save('high_{}.png'.format(os.path.basename(fname)))
