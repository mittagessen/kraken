#!/usr/bin/env python
"""
Draws a transparent overlay of baseline segmenter output over a list of image
files.
"""
import os
import click
from itertools import cycle
from collections import defaultdict

cmap = cycle([(230, 25, 75, 127),
              (60, 180, 75, 127)])

bmap = (0, 130, 200, 255)

@click.command()
@click.option('-f', '--format-type', type=click.Choice(['xml', 'alto', 'page']), default='xml',
              help='Sets the input document format. In ALTO and PageXML mode all'
              'data is extracted from xml files containing both baselines, polygons, and a'
              'link to source images.')
@click.option('-i', '--model', default=None, show_default=True, type=click.Path(exists=True),
              help='Baseline detection model to use. Overrides format type and expects image files as input.')
@click.option('--repolygonize/--no-repolygonize', show_default=True,
              default=False, help='Repolygonizes line data in ALTO/PageXML'
              'files. This ensures that the trained model is compatible with the'
              'segmenter in kraken even if the original image files either do'
              'not contain anything but transcriptions and baseline information'
              'or the polygon data was created using a different method. Will'
              'be ignored in `path` mode. Note, that this option will be slow'
              'and will not scale input images to the same size as the segmenter'
              'does.')
@click.argument('files', nargs=-1)
def cli(format_type, model, repolygonize, files):
    """
    A script producing overlays of lines and regions from either ALTO or
    PageXML files or run a model to do the same.
    """
    if len(files) == 0:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    from PIL import Image, ImageDraw

    from kraken.lib import vgsl, xml
    from kraken import blla

    if model is None:
        if format_type == 'xml':
            fn = xml.parse_xml
        elif format_type == 'alto':
            fn = xml.parse_palto
        else:
            fn = xml.parse_page
        for doc in files:
            click.echo(f'Processing {doc} ', nl=False)
            data = fn(doc)
            # reorder lines by type
            lines = defaultdict(list)
            for  line in data['lines']:
                lines[line['script']].append(line)
            im = Image.open(data['image']).convert('RGBA')
            for t, ls in lines.items():
                tmp = Image.new('RGBA', im.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(tmp)
                for idx, line in enumerate(ls):
                    c = next(cmap)
                    if line['boundary']:
                        draw.polygon([tuple(x) for x in line['boundary']], fill=c, outline=c[:3])
                    if line['baseline']:
                        draw.line([tuple(x) for x in line['baseline']], fill=bmap, width=2, joint='curve')
                    draw.text(line['baseline'][0], str(idx), fill=(0, 0, 0, 255))
                base_image = Image.alpha_composite(im, tmp)
                base_image.save(f'high_{os.path.basename(doc)}_lines_{t}.png')
            for t, regs in data['regions'].items():
                tmp = Image.new('RGBA', im.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(tmp)
                for reg in regs:
                    c = next(cmap)
                    draw.polygon(reg, fill=c, outline=c[:3])
                base_image = Image.alpha_composite(im, tmp)
                base_image.save(f'high_{os.path.basename(doc)}_regions_{t}.png')
            click.secho('\u2713', fg='green')
    else:
        net = vgsl.TorchVGSLModel.load_model(model)
        for doc in files:
            click.echo(f'Processing {doc} ', nl=False)
            im = Image.open(doc)
            res = blla.segment(im, model=net)
            # reorder lines by type
            lines = defaultdict(list)
            for  line in res['lines']:
                lines[line['script']].append(line)
            im = im.convert('RGBA')
            for t, ls in lines.items():
                tmp = Image.new('RGBA', im.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(tmp)
                for idx, line in enumerate(ls):
                    c = next(cmap)
                    draw.polygon([tuple(x) for x in line['boundary']], fill=c, outline=c[:3])
                    draw.line([tuple(x) for x in line['baseline']], fill=bmap, width=2, joint='curve')
                    draw.text(line['baseline'][0], str(idx), fill=(0, 0, 0, 255))
                base_image = Image.alpha_composite(im, tmp)
                base_image.save(f'high_{os.path.basename(doc)}_lines_{t}.png')
            for t, regs in res['regions'].items():
                tmp = Image.new('RGBA', im.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(tmp)
                for reg in regs:
                    c = next(cmap)
                    draw.polygon([tuple(x) for x in reg], fill=c, outline=c[:3])
                base_image = Image.alpha_composite(im, tmp)
                base_image.save(f'high_{os.path.basename(doc)}_regions_{t}.png')
            click.secho('\u2713', fg='green')

if __name__ == '__main__':
    cli()
