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

for doc in xmls:
    print(doc)
    data = dataset.preparse_xml_data([doc], 'alto')
    if len(data) > 0:
        bounds = {'type': 'baselines', 'lines': [{'boundary': t['boundary'], 'baseline': t['baseline'], 'text': t['text']} for t in data]}
        for idx, (im, box) in enumerate(segmentation.extract_polygons(Image.open(data[0]['image']), bounds)):
            print('.', end='', flush=True)
            im.save('{}.{}.jpg'.format(splitext(data[0]['image'])[0], idx))
            with open('{}.{}.gt.txt'.format(splitext(data[0]['image'])[0], idx), 'w') as fp:
                fp.write(box['text'])
