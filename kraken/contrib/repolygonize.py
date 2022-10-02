#!/usr/bin/env python
"""
Reads in a bunch of ALTO documents and repolygonizes the lines contained with
the kraken polygonizer.
"""
import click


@click.command()
@click.option('-f', '--format-type', type=click.Choice(['alto', 'page']), default='alto',
              help='Sets the input document format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines, polygons, and a '
              'link to source images.')
@click.option('-tl', '--topline', 'topline', show_default=True, flag_value='topline',
              help='Switch for the baseline location in the scripts. '
                   'Set to topline if the data is annotated with a hanging baseline, as is '
                   'common with Hebrew, Bengali, Devanagari, etc. Set to '
                   ' centerline for scripts annotated with a central line.')
@click.option('-cl', '--centerline', 'topline', flag_value='centerline')
@click.option('-bl', '--baseline', 'topline', flag_value='baseline', default='baseline')
@click.argument('files', nargs=-1)
def cli(format_type, topline, files):
    """
    A small script repolygonizing line boundaries in ALTO or PageXML files.
    """
    if len(files) == 0:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    from lxml import etree
    from os.path import splitext

    from itertools import groupby
    from kraken.lib import xml
    from PIL import Image
    from kraken.lib.segmentation import calculate_polygonal_environment

    def _repl_alto(fname, polygons):
        with open(fname, 'rb') as fp:
            doc = etree.parse(fp)
            lines = doc.findall('.//{*}TextLine')
            idx = 0
            for line in lines:
                if line.get('BASELINE') is None:
                    continue
                pol = line.find('./{*}Shape/{*}Polygon')
                if pol is not None:
                    if polygons[idx] is not None:
                        pol.attrib['POINTS'] = ' '.join([str(coord) for pt in polygons[idx] for coord in pt])
                    else:
                        pol.attrib['POINTS'] = ''
                idx += 1
            with open(splitext(fname)[0] + '_rewrite.xml', 'wb') as fp:
                doc.write(fp, encoding='UTF-8', xml_declaration=True)

    def _parse_page_coords(coords):
        points = [x for x in coords.split(' ')]
        points = [int(c) for point in points for c in point.split(',')]
        pts = zip(points[::2], points[1::2])
        return [k for k, g in groupby(pts)]

    def _repl_page(fname, polygons):
        with open(fname, 'rb') as fp:
            doc = etree.parse(fp)
            lines = doc.findall('.//{*}TextLine')
            idx = 0
            for line in lines:
                base = line.find('./{*}Baseline')
                if base is not None and not base.get('points').isspace() and len(base.get('points')):
                    try:
                        _parse_page_coords(base.get('points'))
                    except Exception:
                        continue
                else:
                    continue
                pol = line.find('./{*}Coords')
                if pol is not None:
                    if polygons[idx] is not None:
                        pol.attrib['points'] = ' '.join([','.join([str(x) for x in pt]) for pt in polygons[idx]])
                    else:
                        pol.attrib['points'] = ''
                idx += 1
            with open(splitext(fname)[0] + '_rewrite.xml', 'wb') as fp:
                doc.write(fp, encoding='UTF-8', xml_declaration=True)

    if format_type == 'page':
        parse_fn = xml.parse_page
        repl_fn = _repl_page
    else:
        parse_fn = xml.parse_alto
        repl_fn = _repl_alto

    topline = {'topline': True,
               'baseline': False,
               'centerline': None}[topline]

    for doc in files:
        click.echo(f'Processing {doc} ')
        seg = parse_fn(doc)
        im = Image.open(seg['image']).convert('L')
        baselines = []
        for x in seg['lines']:
            bl = x['baseline'] if x['baseline'] is not None else [0, 0]
            baselines.append(bl)
        o = calculate_polygonal_environment(im, baselines, scale=(1800, 0), topline=topline)
        repl_fn(doc, o)


if __name__ == '__main__':
    cli()
