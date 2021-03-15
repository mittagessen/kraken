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
import pkg_resources

from typing import Dict, Union, List, cast, Any, IO, Callable
from functools import partial
from itertools import cycle
from PIL import Image

import click
from click import open_file

from kraken.lib import log

warnings.simplefilter('ignore', UserWarning)

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

APP_NAME = 'kraken'
SEGMENTATION_DEFAULT_MODEL = pkg_resources.resource_filename(__name__, 'blla.mlmodel')
DEFAULT_MODEL = ['en-default.mlmodel']
LEGACY_MODEL_DIR = '/usr/local/share/ocropus'


def message(msg: str, **styles) -> None:
    if logger.getEffectiveLevel() >= 30:
        click.secho(msg, **styles)


def get_input_parser(type_str: str) -> Callable[[str], Dict[str, Any]]:
    if type_str == 'alto':
        from kraken.lib.xml import parse_alto
        return parse_alto
    elif type_str == 'page':
        from kraken.lib.xml import parse_page
        return parse_page
    elif type_str == 'xml':
        from kraken.lib.xml import parse_xml
        return parse_xml
    elif type_str == 'image':
        return Image.open


# chainable functions of functional components (binarization/segmentation/recognition)

def binarizer(threshold, zoom, escale, border, perc, range, low, high, input, output) -> None:
    from kraken import binarization

    ctx = click.get_current_context()
    if ctx.meta['first_process']:
        if ctx.meta['input_format_type'] != 'image':
            input = get_input_parser(ctx.meta['input_format_type'])(input)['image']
        ctx.meta['first_process'] = False
    else:
        raise click.UsageError('Binarization has to be the initial process.')

    try:
        im = Image.open(input)
    except IOError as e:
        raise click.BadParameter(str(e))
    message('Binarizing\t', nl=False)
    try:
        res = binarization.nlbin(im, threshold, zoom, escale, border, perc, range,
                                 low, high)
        if ctx.meta['last_process'] and ctx.meta['output_mode'] != 'native':
            with open_file(output, 'w', encoding='utf-8') as fp:
                fp = cast(IO[Any], fp)
                logger.info('Serializing as {} into {}'.format(ctx.meta['output_mode'], output))
                res.save(f'{output}.png')
                from kraken import serialization
                fp.write(serialization.serialize([],
                                                 image_name=f'{output}.png',
                                                 image_size=res.size,
                                                 template=ctx.meta['output_mode']))
        else:
            form = None
            ext = os.path.splitext(output)[1]
            if ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '']:
                form = 'png'
                if ext:
                    logger.warning('jpeg does not support 1bpp images. Forcing to png.')
            res.save(output, format=form)
        ctx.meta['base_image'] = output
    except Exception:
        message('\u2717', fg='red')
        raise
    message('\u2713', fg='green')


def segmenter(legacy, model, text_direction, scale, maxcolseps, black_colseps,
              remove_hlines, pad, mask, device, input, output) -> None:
    import json

    from kraken import pageseg
    from kraken import blla

    ctx = click.get_current_context()

    if ctx.meta['first_process']:
        if ctx.meta['input_format_type'] != 'image':
            input = get_input_parser(ctx.meta['input_format_type'])(input)['image']
        ctx.meta['first_process'] = False

    if 'base_image' not in ctx.meta:
        ctx.meta['base_image'] = input

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
        if legacy:
            res = pageseg.segment(im, text_direction, scale, maxcolseps, black_colseps, no_hlines=remove_hlines, pad=pad, mask=mask)
        else:
            res = blla.segment(im, text_direction, mask=mask, model=model, device=device)
    except Exception:
        message('\u2717', fg='red')
        raise
    if ctx.meta['last_process'] and ctx.meta['output_mode'] != 'native':
        with open_file(output, 'w', encoding='utf-8') as fp:
            fp = cast(IO[Any], fp)
            logger.info('Serializing as {} into {}'.format(ctx.meta['output_mode'], output))
            from kraken import serialization
            from kraken.rpred import ocr_record
            if 'type' in res and res['type'] == 'baselines':
                records = [ocr_record('', '', '', bl) for bl in res['lines']]
            else:
                records = []
                for line in res['boxes']:
                    xmin, xmax = min(line[::2]), max(line[::2])
                    ymin, ymax = min(line[1::2]), max(line[1::2])
                    records.append(ocr_record('', [], [], [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]))
            fp.write(serialization.serialize(records,
                                             image_name=ctx.meta['base_image'],
                                             image_size=im.size,
                                             regions=res['regions'] if 'regions' in res else None,
                                             template=ctx.meta['output_mode']))
    else:
        with open_file(output, 'w') as fp:
            fp = cast(IO[Any], fp)
            json.dump(res, fp)
    message('\u2713', fg='green')


def recognizer(model, pad, no_segmentation, bidi_reordering, script_ignore, input, output) -> None:

    import json

    from kraken import rpred

    ctx = click.get_current_context()

    bounds = None
    if 'base_image' not in ctx.meta:
        ctx.meta['base_image'] = input

    if ctx.meta['first_process']:
        if ctx.meta['input_format_type'] != 'image':
            doc = get_input_parser(ctx.meta['input_format_type'])(input)
            ctx.meta['base_image'] = doc['image']
            doc['text_direction'] = 'horizontal-lr'
            bounds = doc
    try:
        im = Image.open(ctx.meta['base_image'])
    except IOError as e:
        raise click.BadParameter(str(e))

    if not bounds and ctx.meta['base_image'] != input:
        with open_file(input, 'r') as fp:
            try:
                fp = cast(IO[Any], fp)
                bounds = json.load(fp)
            except ValueError as e:
                raise click.UsageError(f'{input} invalid segmentation: {str(e)}')
    elif not bounds:
        if no_segmentation:
            bounds = {'script_detection': False,
                      'text_direction': 'horizontal-lr',
                      'boxes': [(0, 0) + im.size]}
        else:
            raise click.UsageError('No line segmentation given. Add one with the input or run `segment` first.')
    elif no_segmentation:
        logger.warning('no_segmentation mode enabled but segmentation defined. Ignoring --no-segmentation option.')

    scripts = set()
    # script detection
    if 'script_detection' in bounds and bounds['script_detection']:
        it = rpred.mm_rpred(model, im, bounds, pad,
                            bidi_reordering=bidi_reordering,
                            script_ignore=script_ignore)
    else:
        it = rpred.rpred(model['default'], im, bounds, pad,
                         bidi_reordering=bidi_reordering)

    preds = []

    with log.progressbar(it, label='Processing') as bar:
        for pred in bar:
            preds.append(pred)

    ctx = click.get_current_context()
    with open_file(output, 'w', encoding='utf-8') as fp:
        fp = cast(IO[Any], fp)
        message(f'Writing recognition results for {ctx.meta["orig_file"]}\t', nl=False)
        logger.info('Serializing as {} into {}'.format(ctx.meta['output_mode'], output))
        if ctx.meta['output_mode'] != 'native':
            from kraken import serialization
            fp.write(serialization.serialize(preds, ctx.meta['base_image'],
                                             Image.open(ctx.meta['base_image']).size,
                                             ctx.meta['text_direction'],
                                             scripts,
                                             bounds['regions'] if 'regions' in bounds else None,
                                             ctx.meta['output_mode']))
        else:
            fp.write('\n'.join(s.prediction for s in preds))
        message('\u2713', fg='green')


@click.group(chain=True)
@click.version_option()
@click.option('-i', '--input',
              type=(click.Path(exists=True),  # type: ignore
                    click.Path(writable=True)),
              multiple=True,
              help='Input-output file pairs. Each input file (first argument) is mapped to one '
                   'output file (second argument), e.g. `-i input.png output.txt`')
@click.option('-I', '--batch-input', multiple=True, help='Glob expression to add multiple files at once.')
@click.option('-o', '--suffix', default='', show_default=True,
              help='Suffix for output files from batch and PDF inputs.')
@click.option('-v', '--verbose', default=0, count=True, show_default=True)
@click.option('-f', '--format-type', type=click.Choice(['image', 'alto', 'page', 'pdf', 'xml']), default='image',
              help='Sets the default input type. In image mode inputs are image '
                   'files, alto/page expects XML files in the respective format, pdf '
                   'expects PDF files with numbered suffixes added to output file '
                   'names as needed.')
@click.option('-p', '--pdf-format', default='{src}_{idx:06d}',
              show_default=True,
              help='Format for output of PDF files. valid fields '
                   'are `src` (source file), `idx` (page number), and `uuid` (v4 uuid). '
                   '`-o` suffixes are appended to this format string.')
@click.option('-h', '--hocr', 'serializer',
              help='Switch between hOCR, ALTO, abbyyXML, PageXML or "native"'
              'output. Native are plain image files for image, JSON for'
              'segmentation, and text for transcription output.',
              flag_value='hocr')
@click.option('-a', '--alto', 'serializer', flag_value='alto')
@click.option('-y', '--abbyy', 'serializer', flag_value='abbyyxml')
@click.option('-x', '--pagexml', 'serializer', flag_value='pagexml')
@click.option('-n', '--native', 'serializer', flag_value='native', default=True,
              show_default=True)
@click.option('-d', '--device', default='cpu', show_default=True,
              help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('-r', '--raise-on-error/--no-raise-on-error', default=False, show_default=True,
              help='Raises the exception that caused processing to fail in the case of an error')
def cli(input, batch_input, suffix, verbose, format_type, pdf_format, serializer, device, raise_on_error):
    """
    Base command for recognition functionality.

    Inputs are defined as one or more pairs `-i input_file output_file`
    followed by one or more chainable processing commands. Likewise, verbosity
    is set on all subcommands with the `-v` switch.
    """
    ctx = click.get_current_context()
    ctx.meta['device'] = device
    ctx.meta['input_format_type'] = format_type if format_type != 'pdf' else 'image'
    ctx.meta['raise_failed'] = raise_on_error
    ctx.meta['output_mode'] = serializer
    log.set_logger(logger, level=30-min(10*verbose, 20))


@cli.resultcallback()
def process_pipeline(subcommands, input, batch_input, suffix, verbose, format_type, pdf_format, **args):
    """
    Helper function calling the partials returned by each subcommand and
    placing their respective outputs in temporary files.
    """
    import glob
    import uuid
    import tempfile

    input = list(input)
    # expand batch inputs
    if batch_input and suffix:
        for batch_expr in batch_input:
            for in_file in glob.glob(batch_expr, recursive=True):
                input.append((in_file, '{}{}'.format(os.path.splitext(in_file)[0], suffix)))

    # parse pdfs
    if format_type == 'pdf':
        import pyvips

        if not batch_input:
            logger.warning('PDF inputs not added with batch option. Manual output filename will be ignored and `-o` utilized.')
        new_input = []
        num_pages = 0
        for (fpath, _) in input:
            doc = pyvips.Image.new_from_file(fpath, dpi=300, n=-1, access="sequential")
            if 'n-pages' in doc.get_fields():
                num_pages += doc.get('n-pages')

        with log.progressbar(length=num_pages, label='Extracting PDF pages') as bar:
            for (fpath, _) in input:
                try:
                    doc = pyvips.Image.new_from_file(fpath, dpi=300, n=-1, access="sequential")
                    if 'n-pages' not in doc.get_fields():
                        logger.warning('{fpath} does not contain pages. Skipping.')
                        continue
                    n_pages = doc.get('n-pages')

                    dest_dict = {'idx': -1, 'src': fpath, 'uuid': None}
                    for i in range(0, n_pages):
                        dest_dict['idx'] += 1
                        dest_dict['uuid'] = str(uuid.uuid4())
                        fd, filename = tempfile.mkstemp(suffix='.png')
                        os.close(fd)
                        doc = pyvips.Image.new_from_file(fpath, dpi=300, page=i, access="sequential")
                        logger.info(f'Saving temporary image {fpath}:{dest_dict["idx"]} to {filename}')
                        doc.write_to_file(filename)
                        new_input.append((filename, pdf_format.format(**dest_dict) + suffix))
                        bar.update(1)
                except pyvips.error.Error:
                    logger.warning(f'{fpath} is not a PDF file. Skipping.')
        input = new_input

    ctx = click.get_current_context()

    for io_pair in input:
        ctx.meta['first_process'] = True
        ctx.meta['last_process'] = False
        ctx.meta['orig_file'] = io_pair[0]
        if 'base_image' in ctx.meta:
            del ctx.meta['base_image']
        try:
            tmps = [tempfile.mkstemp() for cmd in subcommands[1:]]
            for tmp in tmps:
                os.close(tmp[0])
            fc = [io_pair[0]] + [tmp[1] for tmp in tmps] + [io_pair[1]]
            for idx, (task, input, output) in enumerate(zip(subcommands, fc, fc[1:])):
                if len(fc)-2 == idx:
                    ctx.meta['last_process'] = True
                task(input=input, output=output)
        except Exception as e:
            logger.error(f'Failed processing {io_pair[0]}: {str(e)}')
            if ctx.meta['raise_failed'] is True:
                raise
        finally:
            for f in fc[1:-1]:
                os.unlink(f)
            # clean up temporary PDF image files
            if format_type == 'pdf':
                logger.debug(f'unlinking {fc[0]}')
                os.unlink(fc[0])


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
@click.pass_context
@click.option('-i', '--model',
              default=None,
              show_default=True, type=click.Path(exists=True),
              help='Baseline detection model to use')
@click.option('-x/-bl', '--boxes/--baseline', default=True, show_default=True,
              help='Switch between legacy box segmenter and neural baseline segmenter')
@click.option('-d', '--text-direction', default='horizontal-lr',
              show_default=True,
              type=click.Choice(['horizontal-lr', 'horizontal-rl',
                                 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction')
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
def segment(ctx, model, boxes, text_direction, scale, maxcolseps,
            black_colseps, remove_hlines, pad, mask):
    """
    Segments page images into text lines.
    """
    if model and boxes:
        logger.warning(f'Baseline model ({model}) given but legacy segmenter selected. Forcing to -bl.')
        boxes = False

    if boxes == False:
        if not model:
            model = SEGMENTATION_DEFAULT_MODEL
        from kraken.lib.vgsl import TorchVGSLModel
        message(f'Loading ANN {model}\t', nl=False)
        try:
            model = TorchVGSLModel.load_model(model)
            model.to(ctx.meta['device'])
        except Exception:
            message('\u2717', fg='red')
            ctx.exit(1)
        message('\u2713', fg='green')

    return partial(segmenter, boxes, model, text_direction, scale, maxcolseps,
                   black_colseps, remove_hlines, pad, mask,
                   ctx.meta['device'])


def _validate_mm(ctx, param, value):
    """
    Maps model mappings to a dictionary.
    """
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
    except Exception:
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
@click.option('-d', '--text-direction', default='horizontal-tb',
              show_default=True,
              type=click.Choice(['horizontal-tb', 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction in serialization output')
@click.option('--threads', default=1, show_default=True, type=click.IntRange(1),
              help='Number of threads to use for OpenMP parallelization.')
def ocr(ctx, model, pad, reorder, no_segmentation, text_direction, threads):
    """
    Recognizes text in line images.
    """
    from kraken.lib import models

    if ctx.meta['input_format_type'] != 'image' and no_segmentation:
        raise click.BadParameter('no_segmentation mode is incompatible with page/alto inputs')

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
            raise click.BadParameter(f'No model for {k} found')
        message(f'Loading ANN {k}\t', nl=False)
        try:
            rnn = models.load_any(location, device=ctx.meta['device'])
            nm[k] = rnn
        except Exception:
            message('\u2717', fg='red')
            ctx.exit(1)
        message('\u2713', fg='green')

    if 'default' in nm:
        from collections import defaultdict

        nn = defaultdict(lambda: nm['default'])  # type: Dict[str, models.TorchSeqRecognizer]
        nn.update(nm)
        nm = nn
    # thread count is global so setting it once is sufficient
    nm[k].nn.set_num_threads(threads)

    # set output mode
    ctx.meta['text_direction'] = text_direction
    return partial(recognizer,
                   model=nm,
                   pad=pad,
                   no_segmentation=no_segmentation,
                   bidi_reordering=reorder,
                   script_ignore=ign_scripts)


@cli.command('show')
@click.pass_context
@click.argument('model_id')
def show(ctx, model_id):
    """
    Retrieves model metadata from the repository.
    """
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
    message(f'Model name: {filename}')
    ctx.exit(0)


if __name__ == '__main__':
    cli()
