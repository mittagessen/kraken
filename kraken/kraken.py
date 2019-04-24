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
"""
kraken.kraken
~~~~~~~~~~~~~

Command line drivers for recognition functionality.
"""
import os
import warnings
import logging

from typing import Dict, Union, List, cast, Any, IO
from functools import partial
from PIL import Image

import click
from click import open_file

from kraken.lib import log

warnings.simplefilter('ignore', UserWarning)

logger = logging.getLogger('kraken')

APP_NAME = 'kraken'
DEFAULT_MODEL = ['en-default.mlmodel']
LEGACY_MODEL_DIR = '/usr/local/share/ocropus'


def message(msg: str, **styles) -> None:
    if logger.getEffectiveLevel() >= 30:
        click.secho(msg, **styles)


def binarizer(threshold, zoom, escale, border, perc, range, low, high, base_image, input, output) -> None:
    from kraken import binarization

    try:
        im = Image.open(input)
    except IOError as e:
        raise click.BadParameter(str(e))
    message('Binarizing\t', nl=False)
    try:
        res = binarization.nlbin(im, threshold, zoom, escale, border, perc, range,
                                 low, high)
        form = None
        ext = os.path.splitext(output)[1]
        if ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '']:
            form = 'png'
            if ext:
                logger.warning('jpeg does not support 1bpp images. Forcing to png.')
        res.save(output, format=form)
    except Exception:
        message('\u2717', fg='red')
        raise
    message('\u2713', fg='green')


def segmenter(text_direction, script_detect, allowed_scripts, scale,
              maxcolseps, black_colseps, remove_hlines, pad, mask, base_image, input,
              output) -> None:
    import json

    from kraken import pageseg

    try:
        im = Image.open(input)
    except IOError as e:
        raise click.BadParameter(str(e))
    if mask:
        try:
            mask = Image.open(mask)
        except IOError as e:
            raise click.BadParameter(str(e))
    message('Segmenting\t', nl=False)
    try:
        res = pageseg.segment(im, text_direction, scale, maxcolseps, black_colseps, no_hlines=remove_hlines, pad=pad, mask=mask)
        if script_detect:
            res = pageseg.detect_scripts(im, res, valid_scripts=allowed_scripts)
    except Exception:
        message('\u2717', fg='red')
        raise
    with open_file(output, 'w') as fp:
        fp = cast(IO[Any], fp)
        json.dump(res, fp)
    message('\u2713', fg='green')


def recognizer(model, pad, no_segmentation, bidi_reordering, script_ignore, base_image, input, output, lines) -> None:

    import json
    import tempfile

    from kraken import rpred

    try:
        im = Image.open(base_image)
    except IOError as e:
        raise click.BadParameter(str(e))

    ctx = click.get_current_context()

    # input may either be output from the segmenter then it is a JSON file or
    # be an image file when running the OCR subcommand alone. might still come
    # from some other subcommand though.
    scripts = set()
    if not lines and base_image != input:
        lines = input
    if not lines:
        if no_segmentation:
            lines = tempfile.NamedTemporaryFile(mode='w', delete=False)
            logger.info('Running in no_segmentation mode. Creating temporary segmentation {}.'.format(lines.name))
            json.dump({'script_detection': False,
                       'text_direction': 'horizontal-lr',
                       'boxes': [(0, 0) + im.size]}, lines)
            lines.close()
            lines = lines.name
        else:
            raise click.UsageError('No line segmentation given. Add one with `-l` or run `segment` first.')
    elif no_segmentation:
        logger.warning('no_segmentation mode enabled but segmentation defined. Ignoring --no-segmentation option.')

    with open_file(lines, 'r') as fp:
        try:
            fp = cast(IO[Any], fp)
            bounds = json.load(fp)
        except ValueError as e:
            raise click.UsageError('{} invalid segmentation: {}'.format(lines, str(e)))
        # script detection
        if bounds['script_detection']:
            for l in bounds['boxes']:
                for t in l:
                    scripts.add(t[0])
            it = rpred.mm_rpred(model, im, bounds, pad,
                                bidi_reordering=bidi_reordering,
                                script_ignore=script_ignore)
        else:
            it = rpred.rpred(model['default'], im, bounds, pad,
                             bidi_reordering=bidi_reordering)

    if not lines and no_segmentation:
        logger.debug('Removing temporary segmentation file.')
        os.unlink(lines.name)

    preds = []

    with log.progressbar(it, label='Processing', length=len(bounds['boxes'])) as bar:
        for pred in bar:
            preds.append(pred)

    ctx = click.get_current_context()
    with open_file(output, 'w', encoding='utf-8') as fp:
        fp = cast(IO[Any], fp)
        message('Writing recognition results for {}\t'.format(base_image), nl=False)
        logger.info('Serializing as {} into {}'.format(ctx.meta['mode'], output))
        if ctx.meta['mode'] != 'text':
            from kraken import serialization
            fp.write(serialization.serialize(preds, base_image,
                                             Image.open(base_image).size,
                                             ctx.meta['text_direction'],
                                             scripts,
                                             ctx.meta['mode']))
        else:
            fp.write('\n'.join(s.prediction for s in preds))
        message('\u2713', fg='green')


@click.group(chain=True)
@click.version_option()
@click.option('-i', '--input', type=(click.Path(exists=True),  # type: ignore
                                     click.Path(writable=True)), multiple=True,
                                     help='Input-output file pairs. Each input file (first argument) is mapped to one '
                                          'output file (second argument), e.g. `-i input.png output.txt`')
@click.option('-I', '--batch-input', multiple=True, help='Glob expression to add multiple files at once.')
@click.option('-o', '--suffix', help='Suffix for output files from batch inputs.')
@click.option('-v', '--verbose', default=0, count=True, show_default=True)
@click.option('-d', '--device', default='cpu', show_default=True, help='Select device to use (cpu, cuda:0, cuda:1, ...)')
def cli(input, batch_input, suffix, verbose, device):
    """
    Base command for recognition functionality.

    Inputs are defined as one or more pairs `-i input_file output_file`
    followed by one or more chainable processing commands. Likewise, verbosity
    is set on all subcommands with the `-v` switch.
    """
    ctx = click.get_current_context()
    ctx.meta['device'] = device
    log.set_logger(logger, level=30-min(10*verbose, 20))


@cli.resultcallback()
def process_pipeline(subcommands, input, batch_input, suffix, **args):
    """
    Helper function calling the partials returned by each subcommand and
    placing their respective outputs in temporary files.
    """
    import glob
    import tempfile

    input = list(input)
    if batch_input and suffix:
        for batch_expr in batch_input:
            for in_file in glob.glob(batch_expr, recursive=True):
                input.append((in_file, '{}{}'.format(os.path.splitext(in_file)[0], suffix)))

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
@click.option('--threshold', show_default=True, default=0.5, type=click.FLOAT)
@click.option('--zoom', show_default=True, default=0.5, type=click.FLOAT)
@click.option('--escale', show_default=True, default=1.0, type=click.FLOAT)
@click.option('--border', show_default=True, default=0.1, type=click.FLOAT)
@click.option('--perc', show_default=True, default=80, type=click.IntRange(1, 100))
@click.option('--range', show_default=True, default=20, type=click.INT)
@click.option('--low', show_default=True, default=5, type=click.IntRange(1, 100))
@click.option('--high', show_default=True, default=90, type=click.IntRange(1, 100))
def binarize(threshold, zoom, escale, border, perc, range, low, high):
    """
    Binarizes page images.
    """
    return partial(binarizer, threshold, zoom, escale, border, perc, range, low, high)


@cli.command('segment')
@click.option('-d', '--text-direction', default='horizontal-lr',
              show_default=True,
              type=click.Choice(['horizontal-lr', 'horizontal-rl',
                                 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction')
@click.option('-s/-n', '--script-detect/--no-script-detect', default=False,
              show_default=True,
              help='Enable script detection on segmenter output')
@click.option('-a', '--allowed-scripts', default=None, multiple=True,
              show_default=True,
              help='List of allowed scripts in script detection output. Ignored if disabled.')
@click.option('--scale', show_default=True, default=None, type=click.FLOAT)
@click.option('-m', '--maxcolseps', show_default=True, default=2, type=click.INT)
@click.option('-b/-w', '--black-colseps/--white_colseps', show_default=True, default=False)
@click.option('-r/-l', '--remove_hlines/--hlines', show_default=True, default=True)
@click.option('-p', '--pad', show_default=True, type=(int, int), default=(0, 0),
              help='Left and right padding around lines')
@click.option('-m', '--mask', show_default=True, default=None,
              type=click.File(mode='rb', lazy=True), help='Segmentation mask '
              'suppressing page areas for line detection. 0-valued image '
              'regions are ignored for segmentation purposes. Disables column '
              'detection.')
def segment(text_direction, script_detect, allowed_scripts, scale, maxcolseps,
            black_colseps, remove_hlines, pad, mask):
    """
    Segments page images into text lines.
    """
    return partial(segmenter, text_direction, script_detect, allowed_scripts,
                   scale, maxcolseps, black_colseps, remove_hlines, pad, mask)


def _validate_mm(ctx, param, value):
    model_dict = {'ignore': []}  # type: Dict[str, Union[str, List[str]]]
    if len(value) == 1 and len(value[0].split(':')) == 1:
        model_dict['default'] = value[0]
        return model_dict
    try:
        for m in value:
            k, v = m.split(':')
            if v == 'ignore':
                model_dict['ignore'].append(k)  # type: ignore
            else:
                model_dict[k] = os.path.expanduser(v)
    except Exception as e:
        raise click.BadParameter('Mappings must be in format script:model')
    return model_dict


@cli.command('ocr')
@click.pass_context
@click.option('-m', '--model', default=DEFAULT_MODEL, multiple=True,
              show_default=True, callback=_validate_mm,
              help='Path to an recognition model or mapping of the form '
              '$script1:$model1. Add multiple mappings to run multi-model '
              'recognition based on detected scripts. Use the default keyword '
              'for adding a catch-all model. Recognition on scripts can be '
              'ignored with the model value ignore.')
@click.option('-p', '--pad', show_default=True, type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('-n', '--reorder/--no-reorder', show_default=True, default=True,
              help='Reorder code points to logical order')
@click.option('-s', '--no-segmentation', default=False, show_default=True, is_flag=True,
              help='Enables non-segmentation mode treating each input image as a whole line.')
@click.option('-h', '--hocr', 'serializer', help='Switch between hOCR, '
              'ALTO, and plain text output', flag_value='hocr')
@click.option('-a', '--alto', 'serializer', flag_value='alto')
@click.option('-y', '--abbyy', 'serializer', flag_value='abbyyxml')
@click.option('-t', '--text', 'serializer', flag_value='text', default=True,
              show_default=True)
@click.option('-d', '--text-direction', default='horizontal-tb',
              show_default=True,
              type=click.Choice(['horizontal-tb', 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction in serialization output')
@click.option('-l', '--lines', type=click.Path(exists=True), show_default=True,
              help='JSON file containing line coordinates')
@click.option('--threads', default=1, show_default=True,
              help='Number of threads to use for OpenMP parallelization.')
def ocr(ctx, model, pad, reorder, no_segmentation, serializer, text_direction, lines, threads):
    """
    Recognizes text in line images.
    """
    from kraken.lib import models

    # first we try to find the model in the absolue path, then ~/.kraken, then
    # LEGACY_MODEL_DIR
    nm = {}  # type: Dict[str, models.TorchSeqRecognizer]
    ign_scripts = model.pop('ignore')
    for k, v in model.items():
        search = [v,
                  os.path.join(click.get_app_dir(APP_NAME), v),
                  os.path.join(LEGACY_MODEL_DIR, v)]
        location = None
        for loc in search:
            if os.path.isfile(loc):
                location = loc
                break
        if not location:
            raise click.BadParameter('No model {} for {} found'.format(v, k))
        message('Loading RNN {}\t'.format(k), nl=False)
        try:
            rnn = models.load_any(location, device=ctx.meta['device'])
            nm[k] = rnn
        except Exception:
            message('\u2717', fg='red')
            raise
            ctx.exit(1)
        message('\u2713', fg='green')

    if 'default' in nm:
        from collections import defaultdict

        nn = defaultdict(lambda: nm['default'])  # type: Dict[str, models.TorchSeqRecognizer]
        nn.update(nm)
        nm = nn
    # thread count is global so setting it once is sufficient
    nn[k].nn.set_num_threads(threads)

    # set output mode
    ctx.meta['mode'] = serializer
    ctx.meta['text_direction'] = text_direction
    return partial(recognizer,
                   model=nm,
                   pad=pad,
                   no_segmentation=no_segmentation,
                   bidi_reordering=reorder,
                   script_ignore=ign_scripts,
                   lines=lines)


@cli.command('show')
@click.pass_context
@click.argument('model_id')
def show(ctx, model_id):
    """
    Retrieves model metadata from the repository.
    """
    import unicodedata

    from kraken import repo
    from kraken.lib.util import make_printable, is_printable

    desc = repo.get_description(model_id)

    chars = []
    combining = []
    for char in sorted(desc['graphemes']):
        if not is_printable(char):
            combining.append(make_printable(char))
        else:
            chars.append(char)
    message('name: {}\n\n{}\n\n{}\nscripts: {}\nalphabet: {} {}\naccuracy: {:.2f}%\nlicense: {}\nauthor(s): {}\ndate: {}'.format(model_id,
                                                                                                                                 desc['summary'],
                                                                                                                                 desc['description'],
                                                                                                                                 ' '.join(desc['script']),
                                                                                                                                 ''.join(chars),
                                                                                                                                 ', '.join(combining),
                                                                                                                                 desc['accuracy'],
                                                                                                                                 desc['license']['id'],
                                                                                                                                 '; '.join(x['name'] for x in desc['creators']),
                                                                                                                                 desc['publication_date']))
    ctx.exit(0)


@cli.command('list')
@click.pass_context
def list_models(ctx):
    """
    Lists models in the repository.
    """
    from kraken import repo

    message('Retrieving model list ', nl=False)
    model_list = repo.get_listing(partial(message, '.', nl=False))
    message('\b\u2713', fg='green', nl=False)
    message('\033[?25h\n', nl=False)
    for id, metadata in model_list.items():
        message('{} ({}) - {}'.format(id, ', '.join(metadata['type']), metadata['summary']))
    ctx.exit(0)


@cli.command('get')
@click.pass_context
@click.argument('model_id')
def get(ctx, model_id):
    """
    Retrieves a model from the repository.
    """
    from kraken import repo

    try:
        os.makedirs(click.get_app_dir(APP_NAME))
    except OSError:
        pass

    message('Retrieving model ', nl=False)
    filename = repo.get_model(model_id, click.get_app_dir(APP_NAME),
                              partial(message, '.', nl=False))
    message('\b\u2713', fg='green', nl=False)
    message('\033[?25h')
    message('Model name: {}'.format(filename))
    ctx.exit(0)


if __name__ == '__main__':
    cli()
