#!/usr/bin/env python
"""
Draws a transparent overlay of the forced alignment output over the input
image. Needs OpenFST bindings installed.
"""
import re
import os
import click
import unicodedata
from itertools import cycle
from collections import defaultdict

from PIL import Image, ImageDraw

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


@click.command()
@click.option('-f', '--format-type', type=click.Choice(['xml', 'alto', 'page']), default='xml',
              help='Sets the input document format. In ALTO and PageXML mode all'
              'data is extracted from xml files containing both baselines, polygons, and a'
              'link to source images.')
@click.option('-i', '--model', default=None, show_default=True, type=click.Path(exists=True),
              help='Transcription model to use.')
@click.argument('files', nargs=-1)
def cli(format_type, model, files):
    """
    A script producing overlays of lines and regions from either ALTO or
    PageXML files or run a model to do the same.
    """
    if len(files) == 0:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    from PIL import Image, ImageDraw

    from kraken.lib import models, xml
    from kraken import align

    if format_type == 'xml':
        fn = xml.parse_xml
    elif format_type == 'alto':
        fn = xml.parse_palto
    else:
        fn = xml.parse_page
    click.echo(f'Loading model {model}')
    net = models.load_any(model)

    for doc in files:
        click.echo(f'Processing {doc} ', nl=False)
        data = fn(doc)
        im = Image.open(data['image']).convert('RGBA')
        records = align.forced_align(data, net)
        tmp = Image.new('RGBA', im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(tmp)
        for record in records:
            for pol in record.cuts:
                c = next(cmap)
                draw.polygon([tuple(x) for x in pol], fill=c, outline=c[:3])
        base_image = Image.alpha_composite(im, tmp)
        base_image.save(f'high_{os.path.basename(doc)}_algn.png')
        click.secho('\u2713', fg='green')

if __name__ == '__main__':
    cli()
