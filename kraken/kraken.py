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

import click
import csv
import os
import tempfile
import requests
import time

from PIL import Image
from click import open_file
from itertools import cycle
from collections import namedtuple
from functools import partial
from multiprocessing import Queue, Pool, cpu_count
from kraken import binarization
from kraken import pageseg
from kraken import rpred
from kraken import html
from kraken import repo
from kraken.lib import models

APP_NAME = 'kraken'
MODEL_URL = 'http://l.unchti.me/'
DEFAULT_MODEL = 'en-default.pronn'
LEGACY_MODEL_DIR = '/usr/local/share/ocropus'

spinner = cycle([u'⣾', u'⣽', u'⣻', u'⢿', u'⡿', u'⣟', u'⣯', u'⣷'])

def spin(msg):
    click.echo(u'\r\033[?25l{}\t{}'.format(msg, next(spinner)), nl=False)


def binarizer(threshold, zoom, escale, border, perc, range, low, high, base_image, input, output):
    try:
        im = Image.open(input)
    except IOError as e:
        raise click.BadParameter(str(e))
    click.echo('Binarizing\t', nl=False)
    try:
        res = binarization.nlbin(im, threshold, zoom, escale, border, perc, range,
                                 low, high)
        res.save(output, format='png')
    except:
        click.secho(u'\u2717', fg='red')
        raise
    click.secho(u'\u2713', fg='green')


def segmenter(scale, black_colseps, base_image, input, output):
    try:
        im = Image.open(input)
    except IOError as e:
        raise click.BadParameter(str(e))
    click.echo('Segmenting\t', nl=False)
    try:
        res = pageseg.segment(im, scale, black_colseps)
    except:
        click.secho(u'\u2717', fg='red')
        raise
    with open_file(output, 'w') as fp:
        for box in res:
            fp.write(u'{},{},{},{}\n'.format(*box))
    click.secho(u'\u2713', fg='green')


def recognizer(model, pad, base_image, input, output, lines):
    try:
        im = Image.open(base_image)
    except IOError as e:
        raise click.BadParameter(str(e))

    ctx = click.get_current_context()

    if not lines:
        lines = input
    with open_file(lines, 'r') as fp:
        bounds = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2
                  in csv.reader(fp)]
        it = rpred.rpred(model, im, bounds, pad)
    preds = []

    st_time = time.time()
    for pred in it:
        if ctx.meta['verbose'] > 0:
            click.echo(u'[{:2.4f}] {}'.format(time.time() - st_time, pred.prediction))
        else:
            spin('Processing')
        preds.append(pred)
    if ctx.meta['verbose'] > 0:
        click.echo(u'Execution time: {}s'.format(time.time() - st_time))
    else:
        click.secho(u'\b\u2713', fg='green', nl=False)
        click.echo('\033[?25h\n', nl=False)
    
    ctx = click.get_current_context()
    with open_file(output, 'w', encoding='utf-8') as fp:
        click.echo('Writing recognition results for {}\t'.format(base_image), nl=False)
        if ctx.meta['mode'] == 'hocr':
            fp.write(unicode(html.hocr(preds, base_image)))
        else:
            fp.write(u'\n'.join(s.prediction for s in preds))
        click.secho(u'\u2713', fg='green')



@click.group(chain=True, invoke_without_command=True)
@click.option('-i', '--input', type=(click.Path(exists=True),
                                     click.Path(writable=True)), multiple=True)
@click.option('-c', '--concurrency', default=cpu_count(), type=click.INT)
@click.option('-v', '--verbose', default=0, count=True)
def cli(input, concurrency, verbose):
    ctx = click.get_current_context()
    ctx.meta['verbose'] = verbose


@cli.resultcallback()
def process_pipeline(subcommands, input, concurrency, verbose):
    if not len(subcommands):
        subcommands = [binarize.callback(),
                       segment.callback(),
                       ocr.callback()]
    for io_pair in input:
        try:
            base_image = io_pair[0]
            fc = [io_pair[0]] + [tempfile.mkstemp()[1] for cmd in subcommands[1:]] + [io_pair[1]]
            for task, input, output in zip(subcommands, fc, fc[1:]):
                task(base_image=base_image, input=input, output=output)
                base_image = input
        finally:
            for f in fc[1:-1]:
                os.unlink(f)


@cli.command('binarize')
@click.option('--threshold', default=0.5, type=click.FLOAT)
@click.option('--zoom', default=0.5, type=click.FLOAT)
@click.option('--escale', default=1.0, type=click.FLOAT)
@click.option('--border', default=0.1, type=click.FLOAT)
@click.option('--perc', default=80, type=click.IntRange(1, 100))
@click.option('--range', default=20, type=click.INT)
@click.option('--low', default=5, type=click.IntRange(1, 100))
@click.option('--high', default=90, type=click.IntRange(1, 100))
def binarize(threshold, zoom, escale, border, perc, range, low, high):
    """
    Binarizes page images.
    """
    return partial(binarizer, threshold, zoom, escale, border, perc, range, low, high)


@cli.command('segment')
@click.option('--scale', default=None, type=click.FLOAT)
@click.option('-b/-w', '--black_colseps/--white_colseps', default=False)
def segment(scale=None, black_colseps=False):
    """
    Segments page images into text lines.
    """
    return partial(segmenter, scale, black_colseps)


@cli.command('ocr')
@click.pass_context
@click.option('-m', '--model', default=DEFAULT_MODEL, help='Path to an '
              'recognition model')
@click.option('-p', '--pad', type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('-h/-t', '--hocr/--text', default=False, help='Switch between '
              'hOCR and plain text output')
@click.option('-l', '--lines', type=click.Path(exists=True),
              help='JSON file containing line coordinates')
@click.option('--enable-autoconversion/--disable-autoconversion', 'conv',
              default=True, help='Automatically convert pyrnn models zu HDF5')
def ocr(ctx, model, pad, hocr, lines, conv):
    """
    Recognizes text in line images.
    """
    # we do the locating and loading of the model here to spare us the overhead
    # in each worker.

    # first we try to find the model in the absolue path, then ~/.kraken, then
    # LEGACY_MODEL_DIR
    search = [model,
              os.path.join(click.get_app_dir(APP_NAME), model),
              os.path.join(LEGACY_MODEL_DIR, model)]
    # if automatic conversion is enabled we look for an converted model in
    # ~/.kraken
    if conv is True:
        search.insert(0, os.path.join(click.get_app_dir(APP_NAME),
                      os.path.basename(os.path.splitext(model)[0]) + '.hdf5'))
    location = None
    for loc in search:
        if os.path.isfile(loc):
            location = loc
            break
    if not location:
        raise click.BadParameter('No model found')
    click.echo('Loading RNN\t', nl=False)
    try:
        rnn = models.load_any(location)
    except:
        click.secho(u'\u2717', fg='red')
        raise
        ctx.exit(1)
    click.secho(u'\u2713', fg='green')

    # convert input model to protobuf
    if conv and rnn.kind == 'pyrnn':
        name, _ = os.path.splitext(os.path.basename(model))
        op = os.path.join(click.get_app_dir(APP_NAME), name + '.pronn')
        try:
            os.makedirs(click.get_app_dir(APP_NAME))
        except OSError:
            pass
        models.pyrnn_to_pronn(rnn, op)

    # set output mode
    if hocr:
        ctx.meta['mode'] = 'hocr'
    else:
        ctx.meta['mode'] = 'text'
    return partial(recognizer, model=rnn, pad=pad, lines=lines)



@cli.command('show')
@click.pass_context
@click.argument('model_id')
def show(ctx, model_id):
    """
    Retrieves model metadata from the repository.
    """
    desc = repo.get_description(model_id)
    click.echo('name: {}\n\n{}\n\nauthor: {} ({})\n{}'.format(desc['name'],
                                                              desc['summary'],
                                                              desc['author'],
                                                              desc['author-email'],
                                                              desc['url']))
    ctx.exit(0)

@cli.command('list')
@click.pass_context
def list(ctx):
    """
    Lists repositories in the repository.
    """
    model_list = repo.get_listing(partial(spin, 'Retrieving model list'))
    click.secho(u'\b\u2713', fg='green', nl=False)
    click.echo('\033[?25h\n', nl=False)
    for m in model_list:
        click.echo('{} ({}) - {}'.format(m, model_list[m]['type'], model_list[m]['summary']))
    ctx.exit(0)

@cli.command('get')
@click.pass_context
@click.argument('model_id')
def get(ctx, model_id):
    """
    Retrieves a model from the repository.
    """
    repo.get_model(model_id, click.get_app_dir(APP_NAME),
                   partial(spin, 'Retrieving model'))
    click.secho(u'\b\u2713', fg='green', nl=False)
    click.echo('\033[?25h\n', nl=False)
    ctx.exit(0)

if __name__ == '__main__':
    cli()
