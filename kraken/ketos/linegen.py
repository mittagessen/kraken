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
kraken.ketos.linegen
~~~~~~~~~~~~~~~~~~~~

Command line driver for synthetic recognition training data generation.
"""
import click


@click.command('linegen', deprecated=True)
@click.pass_context
@click.option('-f', '--font', default='sans',
              help='Font family to render texts in.')
@click.option('-n', '--maxlines', type=click.INT, default=0,
              help='Maximum number of lines to generate')
@click.option('-e', '--encoding', default='utf-8',
              help='Decode text files with given codec.')
@click.option('-u', '--normalization',
              type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']), default=None,
              help='Normalize ground truth')
@click.option('-ur', '--renormalize',
              type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']), default=None,
              help='Renormalize text for rendering purposes.')
@click.option('--reorder/--no-reorder', default=False, help='Reorder code points to display order')
@click.option('-fs', '--font-size', type=click.INT, default=32,
              help='Font size to render texts in.')
@click.option('-fw', '--font-weight', type=click.INT, default=400,
              help='Font weight to render texts in.')
@click.option('-l', '--language',
              help='RFC-3066 language tag for language-dependent font shaping')
@click.option('-ll', '--max-length', type=click.INT, default=None,
              help="Discard lines above length (in Unicode codepoints).")
@click.option('--strip/--no-strip', help="Remove whitespace from start and end "
              "of lines.")
@click.option('-D', '--disable-degradation', is_flag=True, help='Dont degrade '
              'output lines.')
@click.option('-a', '--alpha', type=click.FLOAT, default=1.5,
              help="Mean of folded normal distribution for sampling foreground pixel flip probability")
@click.option('-b', '--beta', type=click.FLOAT, default=1.5,
              help="Mean of folded normal distribution for sampling background pixel flip probability")
@click.option('-d', '--distort', type=click.FLOAT, default=1.0,
              help='Mean of folded normal distribution to take distortion values from')
@click.option('-ds', '--distortion-sigma', type=click.FLOAT, default=20.0,
              help='Mean of folded normal distribution to take standard deviations for the '
              'Gaussian kernel from')
@click.option('--legacy/--no-legacy', default=False,
              help='Use ocropy-style degradations')
@click.option('-o', '--output', type=click.Path(), default='training_data',
              help='Output directory')
@click.argument('text', nargs=-1, type=click.Path(exists=True))
def line_generator(ctx, font, maxlines, encoding, normalization, renormalize,
                   reorder, font_size, font_weight, language, max_length, strip,
                   disable_degradation, alpha, beta, distort, distortion_sigma,
                   legacy, output, text):
    """
    Generates artificial text line training data.
    """
    import os
    import errno
    import logging
    import numpy as np
    import unicodedata

    from typing import Set
    from bidi.algorithm import get_display

    from kraken.lib.progress import KrakenProgressBar
    from kraken.lib.exceptions import KrakenCairoSurfaceException

    from .util import message

    from kraken import linegen
    from kraken.lib.util import make_printable

    logging.captureWarnings(True)
    logger = logging.getLogger('kraken')

    lines: Set[str] = set()
    if not text:
        return
    with KrakenProgressBar() as progress:
        read_task = progress.add_task('Reading texts', total=len(text), visible=True if not ctx.meta['verbose'] else False)
        for t in text:
            with click.open_file(t, encoding=encoding) as fp:
                logger.info('Reading {}'.format(t))
                for line in fp:
                    lines.add(line.rstrip('\r\n'))
            progress.update(read_task, advance=1)

    if normalization:
        lines = set([unicodedata.normalize(normalization, line) for line in lines])
    if strip:
        lines = set([line.strip() for line in lines])
    if max_length:
        lines = set([line for line in lines if len(line) < max_length])
    logger.info('Read {} lines'.format(len(lines)))
    message('Read {} unique lines'.format(len(lines)))
    if maxlines and maxlines < len(lines):
        message('Sampling {} lines\t'.format(maxlines), nl=False)
        llist = list(lines)
        lines = set(llist[idx] for idx in np.random.randint(0, len(llist), maxlines))
        message('\u2713', fg='green')
    try:
        os.makedirs(output)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # calculate the alphabet and print it for verification purposes
    alphabet: Set[str] = set()
    for line in lines:
        alphabet.update(line)
    chars = []
    combining = []
    for char in sorted(alphabet):
        k = make_printable(char)
        if k != char:
            combining.append(k)
        else:
            chars.append(k)
    message('Σ (len: {})'.format(len(alphabet)))
    message('Symbols: {}'.format(''.join(chars)))
    if combining:
        message('Combining Characters: {}'.format(', '.join(combining)))
    lg = linegen.LineGenerator(font, font_size, font_weight, language)
    with KrakenProgressBar() as progress:
        gen_task = progress.add_task('Writing images', total=len(lines), visible=True if not ctx.meta['verbose'] else False)
        for idx, line in enumerate(lines):
            logger.info(line)
            try:
                if renormalize:
                    im = lg.render_line(unicodedata.normalize(renormalize, line))
                else:
                    im = lg.render_line(line)
            except KrakenCairoSurfaceException as e:
                logger.info('{}: {} {}'.format(e.message, e.width, e.height))
                continue
            if not disable_degradation and not legacy:
                im = linegen.degrade_line(im, alpha=alpha, beta=beta)
                im = linegen.distort_line(im, abs(np.random.normal(distort)), abs(np.random.normal(distortion_sigma)))
            elif legacy:
                im = linegen.ocropy_degrade(im)
            im.save('{}/{:06d}.png'.format(output, idx))
            with open('{}/{:06d}.gt.txt'.format(output, idx), 'wb') as fp:
                if reorder:
                    fp.write(get_display(line).encode('utf-8'))
                else:
                    fp.write(line.encode('utf-8'))
            progress.update(gen_task, advance=1)
