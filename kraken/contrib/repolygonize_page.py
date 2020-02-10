#!/usr/bin/env python
"""
Reads in a bunch of PAGEXML documents and repolygonizes the lines contained with
the kraken polygonizer.
"""
import os
import numpy as np
import sys
from lxml import etree
from os.path import splitext

from kraken.lib import xml
from kraken import serialization, rpred
from kraken.lib.dataset import _fixed_resize
from PIL import Image
<<<<<<< Updated upstream
from kraken.lib.segmentation import calculate_polygonal_environment, scale_polygonal_lines
=======
from kraken.lib.segmentation import calculate_polygonal_environment
from kraken.lib.dataset import _fixed_resize
>>>>>>> Stashed changes

for fname in sys.argv[1:]:
    print(fname)
    seg = xml.parse_page(fname)
    im = Image.open(seg['image']).convert('L')
    scal_im = _fixed_resize(im, (1200, 0))
<<<<<<< Updated upstream
    scale = np.divide(im.size, scal_im.size)
    scal_bls = scale_polygonal_lines([(x['baseline'], x['boundary']) for x in seg['lines']], 1/scale)
    scal_bls = [x[0] for x in scal_bls]
    o = calculate_polygonal_environment(scal_im, scal_bls)
    o = [x[1] for x in scale_polygonal_lines([([0,1], x) for x in o], scale)]
=======
    scal_bls = scale_polygonal_lines(seg['lines']
    bls = [x['baseline'] for x in seg['lines']]

    o = calculate_polygonal_environment(scal_im, bls)
>>>>>>> Stashed changes
    with open(fname, 'rb') as fp:
        doc = etree.parse(fp)
        lines = doc.findall('.//{*}TextLine')
        idx = 0
        for line in lines:
            pol = line.find('./{*}Shape/{*}Polygon')
            if pol is not None:
                pol.attrib['POINTS'] = ' '.join([str(coord) for pt in o[idx] for coord in pt])
                idx += 1
        with open(splitext(fname)[0] + '_rewrite.xml', 'wb') as fp:
            doc.write(fp)
