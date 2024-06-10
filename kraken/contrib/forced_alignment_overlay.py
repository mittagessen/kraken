#!/usr/bin/env python
"""
Draws a transparent overlay of the forced alignment output over the input
image.
"""
import os
import re
import unicodedata
from itertools import cycle
from unicodedata import normalize

import click
from lxml import etree

cmap = cycle([(230, 25, 75, 127),
              (60, 180, 75, 127),
              (255, 225, 25, 127),
              (0, 130, 200, 127),
              (245, 130, 48, 127),
              (145, 30, 180, 127),
              (70, 240, 240, 127)])


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = unicodedata.normalize('NFKD', value)
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    value = re.sub(r'[-\s]+', '-', value)
    return value


def _repl_alto(fname, cuts):
    with open(fname, 'rb') as fp:
        doc = etree.parse(fp)
        lines = doc.findall('.//{*}TextLine')
        char_idx = 0
        for line, line_cuts in zip(lines, cuts.lines):
            idx = 0
            for el in line:
                if el.tag.endswith('Shape'):
                    continue
                elif el.tag.endswith('SP'):
                    idx += 1
                elif el.tag.endswith('String'):
                    str_len = len(el.get('CONTENT'))
                    # clear out all
                    for chld in el:
                        if chld.tag.endswith('Glyph'):
                            el.remove(chld)
                    for char in zip(line_cuts.prediction[idx:str_len],
                                    line_cuts.cuts[idx:str_len],
                                    line_cuts.confidences[idx:str_len]):
                        glyph = etree.SubElement(el, 'Glyph')
                        glyph.set('ID', f'char_{char_idx}')
                        char_idx += 1
                        glyph.set('CONTENT', char[0])
                        glyph.set('GC', str(char[2]))
                        pol = etree.SubElement(etree.SubElement(glyph, 'Shape'), 'Polygon')
                        pol.set('POINTS', ' '.join([str(coord) for pt in char[1] for coord in pt]))
                    idx += str_len
        with open(f'{os.path.basename(fname)}_algn.xml', 'wb') as fp:
            doc.write(fp, encoding='UTF-8', xml_declaration=True)


def _repl_page(fname, cuts):
    with open(fname, 'rb') as fp:
        doc = etree.parse(fp)
        lines = doc.findall('.//{*}TextLine')
        for line, line_cuts in zip(lines, cuts.lines):
            glyphs = line.findall('../{*}Glyph/{*}Coords')
            for glyph, cut in zip(glyphs, line_cuts):
                glyph.attrib['points'] = ' '.join([','.join([str(x) for x in pt]) for pt in cut])
        with open(f'{os.path.basename(fname)}_algn.xml', 'wb') as fp:
            doc.write(fp, encoding='UTF-8', xml_declaration=True)


@click.command()
@click.option('-f', '--format-type', type=click.Choice(['alto', 'page']), default='page',
              help='Sets the input document format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines, polygons, and a '
              'link to source images.')
@click.option('-i', '--model', default=None, show_default=True, type=click.Path(exists=True),
              help='Transcription model to use.')
@click.option('-u', '--normalization', show_default=True, type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              default=None,
              help='Ground truth normalization')
@click.option('-o', '--output', type=click.Choice(['xml', 'overlay']),
              show_default=True, default='overlay', help='Output mode. Either page or '
              'alto for xml output, overlay for image overlays.')
@click.argument('files', nargs=-1)
def cli(format_type, model, normalization, output, files):
    """
    A script producing overlays of lines and regions from either ALTO or
    PageXML files or run a model to do the same.
    """
    if len(files) == 0:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    from PIL import Image, ImageDraw

    from kraken import align
    from kraken.lib import models
    from kraken.lib.xml import XMLPage

    if format_type == 'alto':
        repl_fn = _repl_alto
    else:
        repl_fn = _repl_page

    click.echo(f'Loading model {model}')
    net = models.load_any(model)

    for doc in files:
        click.echo(f'Processing {doc} ', nl=False)
        data = XMLPage(doc)
        im = Image.open(data.imagename).convert('RGBA')
        result = align.forced_align(data.to_container(), net)
        if normalization:
            for line in data._lines:
                line["text"] = normalize(normalization, line["text"])
        im = Image.open(data.imagename).convert('RGBA')
        result = align.forced_align(data.to_container(), net)
        if output == 'overlay':
            tmp = Image.new('RGBA', im.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(tmp)
            for record in result.lines:
                for pol in record.cuts:
                    c = next(cmap)
                    draw.polygon([tuple(x) for x in pol], fill=c, outline=c[:3])
            base_image = Image.alpha_composite(im, tmp)
            base_image.save(f'high_{os.path.basename(doc)}_algn.png')
        else:
            repl_fn(doc, result)
        click.secho('\u2713', fg='green')


if __name__ == '__main__':
    cli()
