# -*- coding: utf-8 -*-

from __future__ import absolute_import

import click
import csv
import os
import urllib2

from PIL import Image
from urlparse import urljoin

from kraken import binarization
from kraken import pageseg
from kraken import rpred
from kraken import html
from kraken.lib import models

APP_NAME = 'kraken'
MODEL_URL = 'http://www.tmbdev.net/ocropy/'
DEFAULT_MODEL = 'en-default.pyrnn.gz'
LEGACY_MODEL_DIR = '/usr/local/share/ocropus'


@click.group()
def cli():
    pass


@click.command('binarize')
@click.option('--threshold', default=0.5, type=click.FLOAT)
@click.option('--zoom', default=0.5, type=click.FLOAT)
@click.option('--escale', default=1.0, type=click.FLOAT)
@click.option('--border', default=0.1, type=click.FLOAT)
@click.option('--perc', default=80, type=click.IntRange(1, 100))
@click.option('--range', default=20, type=click.INT)
@click.option('--low', default=5, type=click.IntRange(1, 100))
@click.option('--high', default=90, type=click.IntRange(1, 100))
@click.argument('input', type=click.File(mode='rb'))
@click.argument('output', type=click.File(mode='wb'))
@click.pass_context
def binarize(ctx, threshold, zoom, escale, border, perc, range, low, high,
             input, output):
    im = Image.open(input)
    res = binarization.nlbin(im, threshold, zoom, escale, border, perc, range,
                             low, high)
    res.save(output)


@click.command('segment')
@click.option('--scale', default=None, type=click.FLOAT)
@click.option('-b/-w', '--black_colseps/--white_colseps', default=False)
@click.argument('input', type=click.File(mode='rb'))
@click.argument('output', type=click.File(mode='wb'), required=False)
@click.pass_context
def segment(ctx, scale, black_colseps, input, output):
    try:
        im = Image.open(input)
    except IOError as e:
        click.BadParameter(e.message)
    res = pageseg.segment(im, scale, black_colseps)
    for box in res:
        click.echo(u','.join([str(c) for c in box]), file=output)


@click.command('ocr')
@click.pass_context
@click.option('-m', '--model', default=DEFAULT_MODEL, help='Path to an '
              'recognition model')
@click.option('-p', '--pad', type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('-h/-t', '--hocr/--text', default=False, help='Switch between '
              'hOCR and plain text output')
@click.option('-l', '--lines', type=click.File(mode='rb'), required=True,
              help='JSON file containing line coordinates')
@click.option('--enable-autoconversion/--disable-autoconversion', 'conv',
              default=True, help='Automatically convert pyrnn models zu HDF5')
@click.argument('input', type=click.File(mode='rb'))
@click.argument('output', type=click.File(mode='w', encoding='utf-8'),
                required=False)
def ocr(ctx, model, pad, hocr, lines, conv, input, output):
    # first we try to find the model in the absolue path, then ~/.kraken, then
    # LEGACY_MODEL_DIR
    search = [model,
              os.path.join(click.get_app_dir(APP_NAME, force_posix=True), model),
              os.path.join(LEGACY_MODEL_DIR, model)]
    # if automatic conversion is enabled we look for an converted model in
    # ~/.kraken
    if conv:
        search.insert(0, os.path.join(click.get_app_dir(APP_NAME,
                       force_posix=True),
                       os.path.basename(os.path.splitext(model)[0]) +
                       '.hdf5'))
    for loc in search:
        if os.path.isfile(loc):
            location = loc
            break
    if not location:
        raise click.BadParameter('No model found')
    click.echo('Loading RNN\t', nl=False)
    rnn = models.load_any(location)
    if rnn:
        click.secho(u'\u2713', fg='green')
    else:
        click.secho(u'\u2717', fg='red')
        ctx.exit(1)
    # convert input model to HDF5
    if conv and rnn.kind == 'pyrnn':
        name, _ = os.path.splitext(os.path.basename(model))
        op = os.path.join(click.get_app_dir(APP_NAME, force_posix=True), name + '.hdf5')
        models.pyrnn_to_hdf5(rnn, op)
    try:
        im = Image.open(input)
    except IOError as e:
        click.BadParameter(e.message)
    lc = len(lines.readlines())
    lines.seek(0)
    with click.progressbar(csv.reader(lines), lc, label='Reading line bounds',
                           fill_char=click.style('#', fg='green'),) as b:
        bounds = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in b]

    it = rpred.rpred(rnn, im, bounds, pad)
    with click.progressbar(it, len(bounds),
                           label='Recognizing lines',
                           fill_char=click.style('#', fg='green')) as pred:
        records = []
        for rec in pred:
            records.append(rec)
        if hocr:
            click.echo(html.hocr(records, input.name, im.size), file=output,
                       nl=False)
        else:
            click.echo(u'\n'.join([unicode(s) for s in records]), file=output,
                       nl=False)


@click.command('download')
def download():
    default_model = urllib2.urlopen(urljoin(MODEL_URL, DEFAULT_MODEL))
    try:
        os.makedirs(click.get_app_dir(APP_NAME, force_posix=True))
    except OSError:
        pass
    # overwrite next function for iterator to return 8192 octets instead of
    # line
    default_model.next = lambda: default_model.read(8192)
    fs = int(default_model.info().getheaders("Content-Length")[0])
    with open(os.path.join(click.get_app_dir(APP_NAME, force_posix=True),
                           DEFAULT_MODEL), 'wb') as fp:
        with click.progressbar(default_model, fs/256,
                               label='Downloading default model',
                               fill_char=click.style('#', fg='green')) as dl:
            for buf in dl:
                if not buffer:
                    raise StopIteration()
                fp.write(buf)


cli.add_command(binarize)
cli.add_command(segment)
cli.add_command(ocr)
cli.add_command(download)
