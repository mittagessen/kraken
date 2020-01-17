#!/usr/bin/env python
"""
Reads in a bunch of ALTO documents and repolygonizes the lines contained with
the kraken polygonizer.
"""
import os
import sys
from lxml import etree
from os.path import splitext

from kraken.lib import xml
from kraken import serialization, rpred


for fname in sys.argv[1:]:
    print(fname)
    xml.parse_alto(fname)
    im = Image.open(seg['image']).convert('L')
    bls = [x['baseline'] for x in seg['lines']]
    o = calculate_polygonal_environment(im, bls)
    with open(filename, 'rb') as fp:
        doc = etree.parse(fp)
        lines = doc.findall('.//{*}TextLine')
        idx = 0
        for line in lines:
            pol = line.find('./{*}Shape/{*}Polygon')
            if pol is not None:
                pol.attrib['POINTS'] = [coord for pt in o[idx] for coord in pt]
                idx += 1
    with open(splitext(filename)[0] + '_rewrite.xml', 'wb') as fp:
        doc.write(fp)
