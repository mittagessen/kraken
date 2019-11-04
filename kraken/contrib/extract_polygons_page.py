#! /usr/bin/env python
"""
A small script extracting lines defined in polygon and baseline format from
PageXML files.
"""

import sys
from PIL import Image
from os.path import splitext

from kraken.lib import dataset, segmentation

xmls = sys.argv[1:]

data = dataset.preparse_xml_data(xmls, 'page')
bounds = {'type': 'baselines', 'lines': [{'boundary': t['boundary'], 'baseline': t['baseline']} for t in data]}
for line in data:
    for idx, (im, box) in enumerate(segmentation.extract_polygons(Image.open(line['image']), bounds)):
        print('.', end='', flush=True)
        im.save('{}.{}.jpg'.format(splitext(line['image'])[0], idx))
