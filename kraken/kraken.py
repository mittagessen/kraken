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
from future.utils import PY2
standard_library.install_aliases()
from builtins import str

import os
import json
import click
import time
import tempfile
import warnings
import logging
import unicodedata

from PIL import Image
from click import open_file
from itertools import cycle
from functools import partial
from collections import defaultdict

from kraken import repo
from kraken import rpred
from kraken import pageseg
from kraken.lib import log
from kraken.lib import models
from kraken import binarization
from kraken import serialization

warnings.simplefilter('ignore', UserWarning)

logger = logging.getLogger('kraken')

APP_NAME = 'kraken'
DEFAULT_MODEL = ['en-default.pronn']
LEGACY_MODEL_DIR = '/usr/local/share/ocropus'

spinner = cycle([u'⣾', u'⣽', u'⣻', u'⢿', u'⡿', u'⣟', u'⣯', u'⣷'])

def have_clstm():
    try:
        import clstm
    except ImportError:
        return False
    return True

def spin(msg):
    if logger.getEffectiveLevel() >= 30:
        click.echo(u'\r\033[?25l{}\t{}'.format(msg, next(spinner)), nl=False)

def message(msg, **styles):
    if logger.getEffectiveLevel() >= 30:
        click.secho(msg, **styles)

def binarizer(threshold, zoom, escale, border, perc, range, low, high, base_image, input, output):
    try:
        im = Image.open(input)
    except IOError as e:
        raise click.BadParameter(str(e))
    message('Binarizing\t', nl=False)
    try:
        res = binarization.nlbin(im, threshold, zoom, escale, border, perc, range,
                                 low, high)
        res.save(output, format='png')
    except:
        message(u'\u2717', fg='red')
        raise
    message(u'\u2713', fg='green')


def segmenter(text_direction, script_detect, scale, maxcolseps, black_colseps, base_image, input, output):
    try:
        im = Image.open(input)
    except IOError as e:
        raise click.BadParameter(str(e))
    message('Segmenting\t', nl=False)
    try:
        res = pageseg.segment(im, text_direction, scale, maxcolseps, black_colseps)
        if script_detect:
            res = pageseg.detect_scripts(im, res)
    except:
        message(u'\u2717', fg='red')
        raise
    with open_file(output, 'w') as fp:
        json.dump(res, fp)
    message(u'\u2713', fg='green')


def recognizer(model, pad, bidi_reordering, script_ignore, base_image, input, output, lines):
    try:
        im = Image.open(base_image)
    except IOError as e:
        raise click.BadParameter(str(e))

    ctx = click.get_current_context()

    scripts = None

    st_time = time.time()

    if not lines:
        lines = input
    with open_file(lines, 'r') as fp:
        bounds = json.load(fp)
        # script detection
        if bounds['script_detection']:
            scripts = set()
            for l in bounds['boxes']:
                for t in l:
                    scripts.add(t[0])
            it = rpred.mm_rpred(model, im, bounds, pad,
                                bidi_reordering=bidi_reordering,
                                script_ignore=script_ignore)
        else:
            it = rpred.rpred(model['default'], im, bounds, pad,
                             bidi_reordering=bidi_reordering)

    preds = []

    for pred in it:
        spin('Processing')
        preds.append(pred)
    message(u'\b\u2713', fg='green', nl=False)
    message('\033[?25h\n', nl=False)

    ctx = click.get_current_context()
    with open_file(output, 'w', encoding='utf-8') as fp:
        message(u'Writing recognition results for {}\t'.format(base_image), nl=False)
        if PY2:
            output = output.decode('utf-8')
        logger.info(u'Serializing as {} into {}'.format(ctx.meta['mode'], output))
        if ctx.meta['mode'] != 'text':
            fp.write(serialization.serialize(preds, base_image,
                     Image.open(base_image).size, ctx.meta['text_direction'],
                     scripts, ctx.meta['mode']))
        else:
            fp.write(u'\n'.join(s.prediction for s in preds))
        if not ctx.meta['verbose']:
            message(u'\u2713', fg='green')


@click.group(chain=True)
@click.version_option()
@click.option('-i', '--input', type=(click.Path(exists=True),
                                     click.Path(writable=True)), multiple=True)
@click.option('-v', '--verbose', default=0, count=True)
def cli(input, verbose):
    ctx = click.get_current_context()
    log.set_logger(logger, level=30-10*verbose)
    ctx.meta['verbose'] = verbose


@cli.resultcallback()
def process_pipeline(subcommands, input, verbose):
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
@click.option('-d', '--text-direction', default='horizontal-lr',
              type=click.Choice(['horizontal-lr', 'horizontal-rl',
                                 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction')
@click.option('-s/-n', '--script-detect/--no-script-detect', default=have_clstm(),
              help='Enable script detection on segmenter output')
@click.option('--scale', default=None, type=click.FLOAT)
@click.option('-m', '--maxcolseps', default=2, type=click.INT)
@click.option('-b/-w', '--black_colseps/--white_colseps', default=False)
def segment(text_direction, script_detect, scale, maxcolseps, black_colseps):
    """
    Segments page images into text lines.
    """
    return partial(segmenter, text_direction, script_detect, scale, maxcolseps, black_colseps)


def validate_mm(ctx, param, value):
    model_dict = {}
    if len(value) == 1 and len(value[0].split(':')) == 1:
        return {'default': value[0]}
    model_dict['ignore'] = []
    try:
        for m in value:
            k, v =  m.split(':')
            if v == 'ignore':
                model_dict['ignore'].append(k)
            else:
                model_dict[k] = os.path.expanduser(v)
    except:
        raise click.BadParameter('Mappings must be in format script:model')
    return model_dict

@cli.command('ocr')
@click.pass_context
@click.option('-m', '--model', default=DEFAULT_MODEL, multiple=True, callback=validate_mm,
              help='Path to an recognition model or mapping of the form '
              '$script1:$model1. Add multiple mappings to run multi-model '
              'recognition based on detected scripts. Use the default keyword '
              'for adding a catch-all model. Recognition on scripts can be '
              'ignored with the model value ignore.')
@click.option('-p', '--pad', type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('-n', '--reorder/--no-reorder', default=True,
              help='Reorder code points to logical order')
@click.option('-h', '--hocr', 'serialization', help='Switch between hOCR, '
              'ALTO, and plain text output', flag_value='hocr')
@click.option('-a', '--alto', 'serialization', flag_value='alto')
@click.option('-t', '--text', 'serialization', flag_value='text', default=True)
@click.option('-d', '--text-direction', default='horizontal-tb',
              type=click.Choice(['horizontal-tb', 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction in serialization output')
@click.option('-l', '--lines', type=click.Path(exists=True),
              help='JSON file containing line coordinates')
@click.option('--enable-autoconversion/--disable-autoconversion', 'conv',
              default=True, help='Automatically convert pyrnn models to protobuf')
def ocr(ctx, model, pad, reorder, serialization, text_direction, lines, conv):
    """
    Recognizes text in line images.
    """
    # we do the locating and loading of the model here to spare us the overhead
    # in each worker.

    # first we try to find the model in the absolue path, then ~/.kraken, then
    # LEGACY_MODEL_DIR
    nm = {}
    ign_scripts = model.pop('ignore')
    for k, v in model.items():
        search = [v,
                  os.path.join(click.get_app_dir(APP_NAME), v),
                  os.path.join(LEGACY_MODEL_DIR, v)]
        # if automatic conversion is enabled we look for an converted model in
        # ~/.kraken
        if conv is True:
            search.insert(0, os.path.join(click.get_app_dir(APP_NAME),
                          os.path.basename(os.path.splitext(v)[0]) + '.pronn'))
        location = None
        for loc in search:
            if os.path.isfile(loc):
                location = loc
                break
        if not location:
            raise click.BadParameter('No model for {} found'.format(k))
        message('Loading RNN {}\t'.format(k), nl=False)
        try:
            rnn = models.load_any(location)
            nm[k] = rnn
        except:
            message(u'\u2717', fg='red')
            raise
            ctx.exit(1)
        message(u'\u2713', fg='green')

        # convert input model to protobuf
        if conv and rnn.kind == 'pyrnn':
            name, _ = os.path.splitext(os.path.basename(v))
            op = os.path.join(click.get_app_dir(APP_NAME), name + u'.pronn')
            try:
                os.makedirs(click.get_app_dir(APP_NAME))
            except OSError:
                pass
            models.pyrnn_to_pronn(rnn, op)

    if 'default' in nm:
        nn = defaultdict(lambda: nm['default'])
        nn.update(nm)
        nm = nn
    # set output mode
    ctx.meta['mode'] = serialization
    ctx.meta['text_direction'] = text_direction
    return partial(recognizer, model=nm, pad=pad, bidi_reordering=reorder, script_ignore=ign_scripts, lines=lines)


@cli.command('show')
@click.pass_context
@click.argument('model_id')
def show(ctx, model_id):
    """
    Retrieves model metadata from the repository.
    """
    desc = repo.get_description(model_id)

    chars = []
    combining = []
    for char in sorted(desc['graphemes']):
        if unicodedata.combining(char):
            combining.append(unicodedata.name(char))
        else:
            chars.append(char)
    message(u'name: {}\n\n{}\n\n{}\nscripts: {}\nalphabet: {} {}\nlicense: {}\nauthor: {} ({})\n{}'.format(desc['name'],
                                                                                                 desc['summary'],
                                                                                                 desc['description'],
                                                                                                 ' '.join(desc['script']),
                                                                                                 ''.join(chars),
                                                                                                 ', '.join(combining),
                                                                                                 desc['license'],
                                                                                                 desc['author'],
                                                                                                 desc['author-email'],
                                                                                                 desc['url']))
    ctx.exit(0)


@cli.command('list')
@click.pass_context
def list(ctx):
    """
    Lists models in the repository.
    """
    model_list = repo.get_listing(partial(spin, 'Retrieving model list'))
    message(u'\b\u2713', fg='green', nl=False)
    message('\033[?25h\n', nl=False)
    for m in model_list:
        message('{} ({}) - {}'.format(m, model_list[m]['type'], model_list[m]['summary']))
    ctx.exit(0)


@cli.command('get')
@click.pass_context
@click.argument('model_id')
def get(ctx, model_id):
    """
    Retrieves a model from the repository.
    """
    try:
        os.makedirs(click.get_app_dir(APP_NAME))
    except OSError:
        pass

    repo.get_model(model_id, click.get_app_dir(APP_NAME),
                   partial(spin, 'Retrieving model'))
    message(u'\b\u2713', fg='green', nl=False)
    message('\033[?25h\n', nl=False)
    ctx.exit(0)


if __name__ == '__main__':
    cli()
