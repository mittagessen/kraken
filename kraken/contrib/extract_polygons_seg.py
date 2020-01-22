#! /usr/bin/env python
"""
A small script extracting lines defined in polygon and baseline format from
segmentation json files.
"""

import sys
import json
from PIL import Image
from os.path import splitext

from kraken.lib import dataset, segmentation

img = Image.open(sys.argv[1])
bounds = json.load(open(sys.argv[2]))
for idx, (im, box) in enumerate(segmentation.extract_polygons(img, bounds)):
    print('.', end='', flush=True)
    im.save('{}.{}.jpg'.format(splitext(sys.argv[1])[0], idx))
