#
# Copyright 2022 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
kraken.ketos.transcription
~~~~~~~~~~~~~~~~~~~~~~~~~~

Legacy command line drivers for recognition training data annotation.
"""
import os
import uuid
import click
import logging
import unicodedata

from typing import IO, Any, cast
from bidi.algorithm import get_display

from kraken.lib.progress import KrakenProgressBar
from .util import message

logging.captureWarnings(True)
logger = logging.getLogger('kraken')


@click.command('extract', deprecated=True)
@click.pass_context
@click.option('-b', '--binarize/--no-binarize', show_default=True, default=True,
              help='Binarize color/grayscale images')
@click.option('-u', '--normalization', show_default=True,
              type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']), default=None,
              help='Normalize ground truth')
@click.option('-s', '--normalize-whitespace/--no-normalize-whitespace',
              show_default=True, default=True, help='Normalizes unicode whitespace')
@click.option('-n', '--reorder/--no-reorder', default=False, show_default=True,
              help='Reorder transcribed lines to display order')
@click.option('-r', '--rotate/--no-rotate', default=True, show_default=True,
              help='Skip rotation of vertical lines')
@click.option('-o', '--output', type=click.Path(), default='training', show_default=True,
              help='Output directory')
@click.option('--format',
              default='{idx:06d}',
              show_default=True,
              help='Format for extractor output. valid fields are `src` (source file), `idx` (line number), and `uuid` (v4 uuid)')
@click.argument('transcriptions', nargs=-1, type=click.File(lazy=True))
def extract(ctx, binarize, normalization, normalize_whitespace, reorder,
            rotate, output, format, transcriptions):
    """
    Extracts image-text pairs from a transcription environment created using
    ``ketos transcribe``.
    """
    import regex
    import base64

    from io import BytesIO
    from PIL import Image
    from lxml import html, etree

    from kraken import binarization

    try:
        os.mkdir(output)
    except Exception:
        pass

    text_transforms = []
    if normalization:
        text_transforms.append(lambda x: unicodedata.normalize(normalization, x))
    if normalize_whitespace:
        text_transforms.append(lambda x: regex.sub(r'\s', ' ', x))
    if reorder:
        text_transforms.append(get_display)

    idx = 0
    manifest = []

    with KrakenProgressBar() as progress:
        read_task = progress.add_task('Reading transcriptions', total=len(transcriptions), visible=True if not ctx.meta['verbose'] else False)

        for fp in transcriptions:
            logger.info('Reading {}'.format(fp.name))
            doc = html.parse(fp)
            etree.strip_tags(doc, etree.Comment)
            td = doc.find(".//meta[@itemprop='text_direction']")
            if td is None:
                td = 'horizontal-lr'
            else:
                td = td.attrib['content']

            im = None
            dest_dict = {'output': output, 'idx': 0, 'src': fp.name, 'uuid': str(uuid.uuid4())}
            for section in doc.xpath('//section'):
                img = section.xpath('.//img')[0].get('src')
                fd = BytesIO(base64.b64decode(img.split(',')[1]))
                im = Image.open(fd)
                if not im:
                    logger.info('Skipping {} because image not found'.format(fp.name))
                    break
                if binarize:
                    im = binarization.nlbin(im)
                for line in section.iter('li'):
                    if line.get('contenteditable') and (not u''.join(line.itertext()).isspace() and u''.join(line.itertext())):
                        dest_dict['idx'] = idx
                        dest_dict['uuid'] = str(uuid.uuid4())
                        logger.debug('Writing line {:06d}'.format(idx))
                        l_img = im.crop([int(x) for x in line.get('data-bbox').split(',')])
                        if rotate and td.startswith('vertical'):
                            im.rotate(90, expand=True)
                        l_img.save(('{output}/' + format + '.png').format(**dest_dict))
                        manifest.append((format + '.png').format(**dest_dict))
                        text = u''.join(line.itertext()).strip()
                        for func in text_transforms:
                            text = func(text)
                        with open(('{output}/' + format + '.gt.txt').format(**dest_dict), 'wb') as t:
                            t.write(text.encode('utf-8'))
                        idx += 1

            progress.update(read_task, advance=1)

    logger.info('Extracted {} lines'.format(idx))
    with open('{}/manifest.txt'.format(output), 'w') as fp:
        fp.write('\n'.join(manifest))


@click.command('transcribe', deprecated=True)
@click.pass_context
@click.option('-d', '--text-direction', default='horizontal-lr',
              type=click.Choice(['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction', show_default=True)
@click.option('--scale', default=None, type=click.FLOAT)
@click.option('--bw/--orig', default=True, show_default=True,
              help="Put nonbinarized images in output")
@click.option('-m', '--maxcolseps', default=2, type=click.INT, show_default=True)
@click.option('-b/-w', '--black_colseps/--white_colseps', default=False, show_default=True)
@click.option('-f', '--font', default='',
              help='Font family to use')
@click.option('-fs', '--font-style', default=None,
              help='Font style to use')
@click.option('-p', '--prefill', default=None,
              help='Use given model for prefill mode.')
@click.option('--pad', show_default=True, type=(int, int), default=(0, 0),
              help='Left and right padding around lines')
@click.option('-l', '--lines', type=click.Path(exists=True), show_default=True,
              help='JSON file containing line coordinates')
@click.option('-o', '--output', type=click.File(mode='wb'), default='transcription.html',
              help='Output file', show_default=True)
@click.argument('images', nargs=-1, type=click.File(mode='rb', lazy=True))
def transcription(ctx, text_direction, scale, bw, maxcolseps,
                  black_colseps, font, font_style, prefill, pad, lines, output,
                  images):
    """
    Creates transcription environments for ground truth generation.
    """
    import json

    from PIL import Image

    from kraken import rpred
    from kraken import pageseg
    from kraken import transcribe
    from kraken import binarization

    from kraken.lib import models

    ti = transcribe.TranscriptionInterface(font, font_style)

    if len(images) > 1 and lines:
        raise click.UsageError('--lines option is incompatible with multiple image files')

    if prefill:
        logger.info('Loading model {}'.format(prefill))
        message('Loading ANN', nl=False)
        prefill = models.load_any(prefill)
        message('\u2713', fg='green')

    with KrakenProgressBar() as progress:
        read_task = progress.add_task('Reading images', total=len(images), visible=True if not ctx.meta['verbose'] else False)
        for fp in images:
            logger.info('Reading {}'.format(fp.name))
            im = Image.open(fp)
            if im.mode not in ['1', 'L', 'P', 'RGB']:
                logger.warning('Input {} is in {} color mode. Converting to RGB'.format(fp.name, im.mode))
                im = im.convert('RGB')
            logger.info('Binarizing page')
            im_bin = binarization.nlbin(im)
            im_bin = im_bin.convert('1')
            logger.info('Segmenting page')
            if not lines:
                res = pageseg.segment(im_bin, text_direction, scale, maxcolseps, black_colseps, pad=pad)
            else:
                with click.open_file(lines, 'r') as fp:
                    try:
                        fp = cast(IO[Any], fp)
                        res = json.load(fp)
                    except ValueError as e:
                        raise click.UsageError('{} invalid segmentation: {}'.format(lines, str(e)))
            if prefill:
                it = rpred.rpred(prefill, im_bin, res.copy())
                preds = []
                logger.info('Recognizing')
                for pred in it:
                    logger.debug('{}'.format(pred.prediction))
                    preds.append(pred)
                ti.add_page(im, res, records=preds)
            else:
                ti.add_page(im, res)
            fp.close()
            progress.update(read_task, advance=1)

    logger.info('Writing transcription to {}'.format(output.name))
    message('Writing output ', nl=False)
    ti.write(output)
    message('\u2713', fg='green')
