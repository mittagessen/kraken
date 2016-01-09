# -*- coding: utf-8 -*-
#
# Copyright 2015 Benjamin Kiessling
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

from __future__ import absolute_import, division, print_function
from future import standard_library
standard_library.install_aliases()

import os
import time
import click
import errno
import unicodedata
import numpy as np

from PIL import Image
from itertools import cycle
from kraken import linegen

from kraken.lib.exceptions import KrakenCairoSurfaceException

APP_NAME = 'kraken'
          
spinner = cycle([u'⣾', u'⣽', u'⣻', u'⢿', u'⡿', u'⣟', u'⣯', u'⣷'])
              
def spin(msg):
    click.echo(u'\r\033[?25l{}\t{}'.format(msg, next(spinner)), nl=False)


@click.group()
@click.option('-v', '--verbose', default=0, count=True)
def cli(verbose):
    ctx = click.get_current_context()
    ctx.meta['verbose'] = verbose

@cli.command('transcrib')
@click.pass_context
def transcription(ctx):
    pass

@cli.command('linegen')
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
@click.option('-fs', '--font-size', type=click.INT, default=32,
              help='Font size to render texts in.')
@click.option('-l', '--language', 
              help='RFC-3066 language tag for language-dependent font shaping')
@click.option('-m', '--mean', type=click.FLOAT, default=0.0,
              help='Mean of distribution to take means for gaussian noise '
              'from.')
@click.option('-s', '--sigma', type=click.FLOAT, default=0.001,
              help='Mean of distribution to take standard deviation values for '
              'Gaussian noise from.')
@click.option('-r', '--density', type=click.FLOAT, default=0.002,
              help='Mean of distribution to take density values for S&P noise '
              'from.')
@click.option('-d', '--distort', type=click.FLOAT, default=1.0,
              help='Mean of distribution to take distortion values from')
@click.option('-ds', '--distortion_sigma', type=click.FLOAT, default=20.0,
              help='Mean of distribution to take standard deviations for the '
              'Gaussian kernel from')
@click.option('-o', '--output', type=click.Path(), default='training_data',
              help='Output directory')
@click.argument('text', nargs=-1, type=click.Path(exists=True))
def line_generator(ctx, font, maxlines, encoding, normalization, renormalize,
                   font_size, language, mean, sigma, density, distort,
                   distortion_sigma, output, text):
    """
    Generates artificial text line training data.
    """
    lines = set()
    if not text:
        return
    st_time = time.time()
    for t in text:
        with click.open_file(t, encoding=encoding) as fp:
            if ctx.meta['verbose'] > 0:
                click.echo(u'[{:2.4f}] Reading {}'.format(time.time() - st_time, t))
            else:
                spin('Reading texts')
            lines.update(fp.readlines())
    if normalization:
        lines = set([unicodedata.normalize(normalization, line) for line in lines])
    if ctx.meta['verbose'] > 0:
        click.echo(u'[{:2.4f}] Read {} lines'.format(time.time() - st_time, len(lines)))
    else:
        click.secho(u'\b\u2713', fg='green', nl=False)
        click.echo('\033[?25h\n', nl=False)
        click.echo('Read {} unique lines'.format(len(lines)))
    if maxlines and maxlines < len(lines):
        click.echo('Sampling {} lines\t'.format(maxlines), nl=False)
        lines = list(lines)
        lines = [lines[idx] for idx in np.random.randint(0, len(lines), maxlines)]
        click.secho(u'\u2713', fg='green')
    try:
        os.makedirs(output)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    lines = [line.strip() for line in lines]

    # calculate the alphabet and print it for verification purposes
    alphabet = set()
    for line in lines:
        alphabet.update(line)
    chars = []
    combining = []
    for char in sorted(alphabet):
        if unicodedata.combining(char):
            combining.append(unicodedata.name(char))
        else:
            chars.append(char)
    click.echo(u'Σ (len: {})'.format(len(alphabet)))
    click.echo(u'Symbols: {}'.format(''.join(chars)))
    if combining:
        click.echo(u'Combining Characters: {}'.format(', '.join(combining)))
    lg = linegen.LineGenerator(font, font_size, language)
    for idx, line in enumerate(lines):
        if ctx.meta['verbose'] > 0:
            click.echo(u'[{:2.4f}] {}'.format(time.time() - st_time, line))
        else:
            spin('Writing images')
        try:
            if renormalize:
                im = lg.render_line(unicodedata.normalize(renormalize, line))
            else:
                im = lg.render_line(line)
        except KrakenCairoSurfaceException as e:
            if ctx.meta['verbose'] > 0:
                click.echo('[{:2.4f}] {}: {} {}'.format(time.time() - st_time, e.message, e.width, e.height))
            else:
                click.secho(u'\b\u2717', fg='red')
                click.echo('{}: {} {}'.format(e.message, e.width, e.height))
            continue
        im = linegen.distort_line(im, distort, distortion_sigma)
        im = linegen.degrade_line(im, mean, sigma, density)
        im.save('{}/{:06d}.png'.format(output, idx))
        with open('{}/{:06d}.gt.txt'.format(output, idx), 'wb') as fp:
            fp.write(line.encode('utf-8'))
    if ctx.meta['verbose'] == 0:
        click.secho(u'\b\u2713', fg='green', nl=False)
        click.echo('\033[?25h\n', nl=False)

if __name__ == '__main__':
    cli()
