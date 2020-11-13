#!/usr/bin/env python
"""
Reads in a bunch of ALTO documents and repolygonizes the lines contained with
the kraken polygonizer.
"""

import click

@click.command()
@click.option('-f', '--format-type', type=click.Choice(['alto', 'page', 'xml']), default='xml',
              help='Sets the input document format. In ALTO and PageXML mode all'
              'data is extracted from xml files containing both baselines, polygons, and a'
              'link to source images.')
@click.argument('files', nargs=-1)
def cli(format_type, files):
    """
    A small script repolygonizing line boundaries in ALTO or PageXML files.
    """
    if len(files) == 0:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    import os
    import numpy as np
    import sys
    from lxml import etree
    from os.path import splitext

    from kraken.lib import xml
    from kraken import serialization, rpred
    from kraken.lib.dataset import _fixed_resize
    from PIL import Image
    from kraken.lib.segmentation import calculate_polygonal_environment, scale_polygonal_lines

    def _repl_alto(fname, polygons):
        with open(fname, 'rb') as fp:
            doc = etree.parse(fp)
            lines = doc.findall('.//{*}TextLine')
            idx = 0
            for line in lines:
                pol = line.find('./{*}Shape/{*}Polygon')
                if pol is not None:
                    pol.attrib['POINTS'] = ' '.join([str(coord) for pt in polygons[idx] for coord in pt])
                    idx += 1
            with open(splitext(fname)[0] + '_rewrite.xml', 'wb') as fp:
                doc.write(fp, encoding='UTF-8', xml_declaration=True)

    def _repl_page(fname, polygons):
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
                doc.write(fp, encoding='UTF-8', xml_declaration=True)

    if format_type == 'xml':
        parse_fn = xml.parse_xml
    elif format_type == 'page':
        parse_fn = xml.parse_page
    else:
        parse_fn = xml.parse_alto

    for doc in files:
        click.echo(f'Processing {doc} ', nl=False)
        seg = parse_fn(doc)
        im = Image.open(seg['image']).convert('L')
        l = []
        for x in seg['lines']:
            bl = x['baseline'] if x['baseline'] is not None else [0, 0]
            l.append(bl)
        o = calculate_polygonal_environment(im, l, scale=(1800, 0))
        print(o)

if __name__ == '__main__':
    cli()
