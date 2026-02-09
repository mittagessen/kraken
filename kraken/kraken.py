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
import uuid
import click
import logging
import warnings
import importlib
import dataclasses

from PIL import Image
from pathlib import Path
from itertools import chain
from functools import partial
from importlib import resources
from platformdirs import user_data_dir
from typing import IO, Any, Callable, cast

from rich import print
from rich.tree import Tree
from rich.table import Table
from rich.console import Group
from rich.traceback import install
from rich.markdown import Markdown

from kraken.lib import log
from kraken.registry import PRECISIONS
from kraken.configs import Config, RecognitionInferenceConfig, SegmentationInferenceConfig
from kraken.ketos.util import to_ptl_device

warnings.simplefilter('ignore', UserWarning)

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# install rich traceback handler
install(suppress=[click])

APP_NAME = 'kraken'
SEGMENTATION_DEFAULT_MODEL = resources.files(APP_NAME).joinpath('blla.mlmodel')
DEFAULT_MODEL = ['en_best.mlmodel']

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


def message(msg: str, **styles) -> None:
    if logger.getEffectiveLevel() >= 30:
        click.secho(msg, **styles)


def get_input_parser(type_str: str) -> Callable[[str], dict[str, Any]]:
    if type_str in ['alto', 'page', 'xml']:
        from kraken.lib.xml import XMLPage
        return XMLPage
    elif type_str == 'image':
        return Image.open


# chainable functions of functional components (binarization/segmentation/recognition)

def binarizer(threshold, zoom, escale, border, perc, range, low, high, input, output) -> None:
    from kraken import binarization

    ctx = click.get_current_context()
    if ctx.meta['first_process']:
        if ctx.meta['input_format_type'] != 'image':
            input = get_input_parser(ctx.meta['input_format_type'])(input).imagename
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
            with click.open_file(output, 'w', encoding='utf-8') as fp:
                fp = cast('IO[Any]', fp)
                logger.info('Serializing as {} into {}'.format(ctx.meta['output_mode'], output))
                res.save(f'{output}.png')
                from kraken import serialization
                fp.write(serialization.serialize([],
                                                 image_name=f'{output}.png',
                                                 image_size=res.size,
                                                 template=ctx.meta['output_template'],
                                                 template_source='custom' if ctx.meta['output_mode'] == 'template' else 'native',
                                                 processing_steps=ctx.meta['steps'],
                                                 sub_line_segmentation=ctx.meta["subline_segmentation"]))
        else:
            form = None
            ext = Path(output).suffix
            if ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '']:
                form = 'png'
                if ext:
                    logger.warning('jpeg does not support 1bpp images. Forcing to png.')
            res.save(output, format=form)
        ctx.meta['base_image'] = output
    except Exception:
        if ctx.meta['raise_failed']:
            raise
        message('\u2717', fg='red')
        ctx.exit(1)
    message('\u2713', fg='green')


def segmenter(legacy, model, config, input, output) -> None:
    import json

    ctx = click.get_current_context()

    if ctx.meta['first_process']:
        if ctx.meta['input_format_type'] != 'image':
            input = get_input_parser(ctx.meta['input_format_type'])(input).imagename
        ctx.meta['first_process'] = False

    if 'base_image' not in ctx.meta:
        ctx.meta['base_image'] = input

    try:
        im = Image.open(input)
    except IOError as e:
        raise click.BadParameter(str(e))
    message(f'Segmenting {ctx.meta["orig_file"]}\t', nl=False)
    try:
        if legacy:
            from kraken import pageseg
            res = pageseg.segment(im,
                                  config.text_direction,
                                  config.legacy_scale,
                                  config.legacy_maxcolseps,
                                  config.legacy_black_colseps,
                                  no_hlines=config.legacy_no_hlines,
                                  pad=config.bbox_line_padding)
        else:
            from kraken.tasks import SegmentationTaskModel
            task = SegmentationTaskModel.load_model(model)
            res = task.predict(im=im, config=config)
    except Exception:
        if ctx.meta['raise_failed']:
            raise
        message('\u2717', fg='red')
        ctx.exit(1)
    if ctx.meta['last_process'] and ctx.meta['output_mode'] != 'native':
        with click.open_file(output, 'w', encoding='utf-8') as fp:
            fp = cast('IO[Any]', fp)
            logger.info('Serializing as {} into {}'.format(ctx.meta['output_mode'], output))
            from kraken import serialization
            fp.write(serialization.serialize(res,
                                             image_size=im.size,
                                             template=ctx.meta['output_template'],
                                             template_source='custom' if ctx.meta['output_mode'] == 'template' else 'native',
                                             processing_steps=ctx.meta['steps'],
                                             sub_line_segmentation=ctx.meta["subline_segmentation"]))
    else:
        with click.open_file(output, 'w') as fp:
            fp = cast('IO[Any]', fp)
            json.dump(dataclasses.asdict(res), fp)
    message('\u2713', fg='green')


def recognizer(model, no_segmentation, config, input, output) -> None:

    import dataclasses
    import json

    from kraken.lib import vgsl  # NOQA
    from kraken.tasks import RecognitionTaskModel
    from kraken.containers import BBoxLine, Segmentation
    from kraken.lib.progress import KrakenProgressBar

    ctx = click.get_current_context()

    bounds = None
    if 'base_image' not in ctx.meta:
        ctx.meta['base_image'] = input

    if ctx.meta['first_process']:
        if ctx.meta['input_format_type'] != 'image':
            doc = get_input_parser(ctx.meta['input_format_type'])(input)
            ctx.meta['base_image'] = doc.imagename
            if doc.base_dir and config.bidi_reordering is True:
                message(f'Setting base text direction for BiDi reordering to {doc.base_dir} (from XML input file)')
                config.bidi_reordering = doc.base_dir
            bounds = doc.to_container()
    try:
        im = Image.open(ctx.meta['base_image'])
    except IOError as e:
        raise click.BadParameter(str(e))

    if not bounds and ctx.meta['base_image'] != input:
        with click.open_file(input, 'r') as fp:
            try:
                fp = cast('IO[Any]', fp)
                bounds = Segmentation(**json.load(fp))
            except ValueError as e:
                raise click.UsageError(f'{input} invalid segmentation: {str(e)}')
    elif not bounds:
        if no_segmentation:
            bounds = Segmentation(type='bbox',
                                  text_direction=config.text_direction,
                                  imagename=ctx.meta['base_image'],
                                  script_detection=False,
                                  lines=[BBoxLine(id=f'_{uuid.uuid4()}',
                                                  bbox=(0, 0, im.width, im.height))])
        else:
            raise click.UsageError('No line segmentation given. Add one with the input or run `segment` first.')
    elif no_segmentation:
        logger.warning('no_segmentation mode enabled but segmentation defined. Ignoring --no-segmentation option.')

    predictor = RecognitionTaskModel.load_model(model)
    it = predictor.predict(im=im, segmentation=bounds, config=config)

    preds = []

    with KrakenProgressBar() as progress:
        pred_task = progress.add_task('Processing', total=len(bounds.lines), visible=True if not ctx.meta['verbose'] else False)
        for pred in it:
            preds.append(pred)
            progress.update(pred_task, advance=1)
    results = dataclasses.replace(bounds, lines=preds, imagename=ctx.meta['base_image'])

    ctx = click.get_current_context()
    with click.open_file(output, 'w', encoding='utf-8') as fp:
        fp = cast('IO[Any]', fp)
        message(f'Writing recognition results for {ctx.meta["orig_file"]}\t', nl=False)
        logger.info('Serializing as {} into {}'.format(ctx.meta['output_mode'], output))
        if ctx.meta['output_mode'] != 'native':
            from kraken import serialization
            fp.write(serialization.serialize(results=results,
                                             image_size=Image.open(ctx.meta['base_image']).size,
                                             writing_mode=ctx.meta['text_direction'],
                                             scripts=None,
                                             template=ctx.meta['output_template'],
                                             template_source='custom' if ctx.meta['output_mode'] == 'template' else 'native',
                                             processing_steps=ctx.meta['steps'],
                                             sub_line_segmentation=ctx.meta["subline_segmentation"]))
        else:
            fp.write('\n'.join(s.prediction for s in preds))
        message('\u2713', fg='green')


@click.group(context_settings=dict(show_default=True,
                                   default_map={**Config().__dict__,
                                                'segment': SegmentationInferenceConfig().__dict__,
                                                'ocr': RecognitionInferenceConfig().__dict__}),
             chain=True)
@click.version_option()
@click.option('-i', '--input',
              type=(click.Path(exists=True, dir_okay=False, path_type=Path),  # type: ignore
                    click.Path(writable=True, dir_okay=False, path_type=Path)),
              multiple=True,
              help='Input-output file pairs. Each input file (first argument) is mapped to one '
                   'output file (second argument), e.g. `-i input.png output.txt`')
@click.option('-I', '--batch-input', multiple=True, help='Glob expression to add multiple files at once.')
@click.option('-o', '--suffix', default='', help='Suffix for output files from batch and PDF inputs.')
@click.option('-v', '--verbose', default=0, count=True)
@click.option('-f', '--format-type', type=click.Choice(['image', 'alto', 'page', 'pdf', 'xml']), default='image',
              help='Sets the default input type. In image mode inputs are image '
                   'files, alto/page expects XML files in the respective format, pdf '
                   'expects PDF files with numbered suffixes added to output file '
                   'names as needed.')
@click.option('-p', '--pdf-format', default='{src}_{idx:06d}',
              help='Format for output of PDF files. valid fields '
                   'are `src` (source file), `idx` (page number), and `uuid` (v4 uuid). '
                   '`-o` suffixes are appended to this format string.')
@click.option('-h', '--hocr', 'serializer',
              help='Switch between hOCR, ALTO, abbyyXML, PageXML or "native" '
              'output. Native are plain image files for image, JSON for '
              'segmentation, and text for transcription output.',
              flag_value='hocr')
@click.option('-a', '--alto', 'serializer', flag_value='alto')
@click.option('-y', '--abbyy', 'serializer', flag_value='abbyyxml')
@click.option('-x', '--pagexml', 'serializer', flag_value='pagexml')
@click.option('-n', '--native', 'serializer', flag_value='native', default=True)
@click.option('-t', '--template', type=click.Path(exists=True, dir_okay=False))
@click.option('-d', '--device', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--precision',
              type=click.Choice(PRECISIONS),
              help='Numerical precision to use for inference. Default is 32-bit single-point precision.')
@click.option('-r', '--raise-on-error/--no-raise-on-error', default=False, help='Raises the exception that caused processing to fail in the case of an error')
@click.option('--threads', 'num_threads', type=click.IntRange(1), help='Maximum size of OpenMP/BLAS thread pool.')
@click.option('--subline-segmentation/--no-subline-segmentation', 'subline_segmentation',
              is_flag=True,
              default=True,
              help="Enable/disable subline segmentation in the serialization format (Default: enabled)")
def cli(**kwargs):
    """
    Base command for recognition functionality.

    Inputs are defined as one or more pairs `-i input_file output_file`
    followed by one or more chainable processing commands. Likewise, verbosity
    is set on all subcommands with the `-v` switch.
    """
    ctx = click.get_current_context()
    params = ctx.params
    ctx.meta['accelerator'], ctx.meta['device'] = to_ptl_device(params['device'])
    ctx.meta['precision'] = params['precision']
    ctx.meta['input_format_type'] = params['format_type'] if params['format_type'] != 'pdf' else 'image'
    ctx.meta['raise_failed'] = params['raise_on_error']
    if not params['template']:
        ctx.meta['output_mode'] = params['serializer']
        ctx.meta['output_template'] = params['serializer']
    else:
        ctx.meta['output_mode'] = 'template'
        ctx.meta['output_template'] = params['template']
    ctx.meta['verbose'] = params['verbose']
    ctx.meta['steps'] = []
    ctx.meta['subline_segmentation'] = params['subline_segmentation']
    ctx.meta['num_threads'] = params['num_threads']

    log.set_logger(logger, level=30 - min(10 * params['verbose'], 20))


@cli.result_callback()
def process_pipeline(subcommands, input, batch_input, suffix, verbose, format_type, pdf_format, **args):
    """
    Helper function calling the partials returned by each subcommand and
    placing their respective outputs in temporary files.
    """
    import glob
    import tempfile

    from threadpoolctl import threadpool_limits

    from kraken.containers import ProcessingStep
    from kraken.lib.progress import KrakenProgressBar

    ctx = click.get_current_context()

    input = list(input)
    # expand batch inputs
    if batch_input and suffix:
        for batch_expr in batch_input:
            for in_file in glob.glob(str(Path(batch_expr).expanduser()), recursive=True):
                input.append((Path(in_file), Path(in_file).with_suffix(suffix)))

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

        with KrakenProgressBar() as progress:
            pdf_parse_task = progress.add_task('Extracting PDF pages', total=num_pages, visible=True if not ctx.meta['verbose'] else False)
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
                        dest_dict['uuid'] = f'_{uuid.uuid4()}'
                        fd, filename = tempfile.mkstemp(suffix='.png')
                        os.close(fd)
                        doc = pyvips.Image.new_from_file(fpath, dpi=300, page=i, access="sequential")
                        logger.info(f'Saving temporary image {fpath}:{dest_dict["idx"]} to {filename}')
                        doc.write_to_file(filename)
                        new_input.append((filename, pdf_format.format(**dest_dict) + suffix))
                        progress.update(pdf_parse_task, advance=1)
                except pyvips.error.Error:
                    num_pages -= n_pages
                    progress.update(pdf_parse_task, total=num_pages)
                    logger.warning(f'{fpath} is not a PDF file. Skipping.')
        input = new_input
        ctx.meta['steps'].insert(0, ProcessingStep(id=f'_{uuid.uuid4()}',
                                                   category='preprocessing',
                                                   description='PDF image extraction',
                                                   settings={}))

    for io_pair in input:
        ctx.meta['first_process'] = True
        ctx.meta['last_process'] = False
        ctx.meta['orig_file'] = io_pair[0]
        if 'base_image' in ctx.meta:
            del ctx.meta['base_image']
        try:
            tmps = [tempfile.mkstemp() for _ in subcommands[1:]]
            for tmp in tmps:
                os.close(tmp[0])
            fc = [io_pair[0]] + [tmp[1] for tmp in tmps] + [io_pair[1]]
            for idx, (task, input, output) in enumerate(zip(subcommands, fc, fc[1:])):
                if len(fc) - 2 == idx:
                    ctx.meta['last_process'] = True
                with threadpool_limits(limits=ctx.meta['num_threads']):
                    task(input=input, output=output)
        except Exception as e:
            logger.error(f'Failed processing {io_pair[0]}: {str(e)}')
            if ctx.meta['raise_failed']:
                raise
        finally:
            for f in fc[1:-1]:
                os.unlink(f)
            # clean up temporary PDF image files
            if format_type == 'pdf':
                logger.debug(f'unlinking {fc[0]}')
                os.unlink(fc[0])


@click.command('binarize')
@click.pass_context
@click.option('--threshold', default=0.5, type=click.FLOAT)
@click.option('--zoom', default=0.5, type=click.FLOAT)
@click.option('--escale', default=1.0, type=click.FLOAT)
@click.option('--border', default=0.1, type=click.FLOAT)
@click.option('--perc', default=80, type=click.IntRange(1, 100))
@click.option('--range', default=20, type=click.INT)
@click.option('--low', default=5, type=click.IntRange(1, 100))
@click.option('--high', default=90, type=click.IntRange(1, 100))
def binarize(ctx, threshold, zoom, escale, border, perc, range, low, high):
    """
    Binarizes page images.
    """
    from kraken.containers import ProcessingStep

    ctx.meta['steps'].append(ProcessingStep(id=f'_{uuid.uuid4()}',
                                            category='preprocessing',
                                            description='Image binarization',
                                            settings={'threshold': threshold,
                                                      'zoom': zoom,
                                                      'escale': escale,
                                                      'border': border,
                                                      'perc': perc,
                                                      'range': range,
                                                      'low': low,
                                                      'high': high}))

    return partial(binarizer, threshold, zoom, escale, border, perc, range,
                   low, high)


@click.command('segment')
@click.pass_context
@click.option('-i', '--model', type=str, help='Baseline/region detection model(s) to use')
@click.option('-x/-bl', '--boxes/--baseline', default=True,
              help='Switch between legacy box segmenter and neural baseline segmenter')
@click.option('-d', '--text-direction',
              type=click.Choice(['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction')
@click.option('--scale', 'legacy_scale', type=float)
@click.option('-m', '--maxcolseps', 'legacy_maxcolseps', type=int)
@click.option('-b/-w', '--black-colseps/--white_colseps', 'legacy_black_colseps')
@click.option('-r/-l', '--remove-hlines/--hlines', 'legacy_no_hlines')
@click.option('-p', '--pad', 'bbox_line_padding', type=int, help='Left and right padding around lines. Only for BBox segmenter.')
@click.option('--input-pad', 'input_padding', type=int, help='Padding to add around input image.')
def segment(ctx, **kwargs):
    """
    Segments page images into text lines.
    """
    from kraken.containers import ProcessingStep
    params = ctx.params
    config = SegmentationInferenceConfig(**params, **ctx.meta)

    if params['model'] and params['boxes']:
        logger.warning(f'Baseline model ({params["model"]}) given but legacy segmenter selected. Forcing to -bl.')
        params['boxes'] = False

    if params['boxes'] is False:
        if not params['model']:
            params['model'] = SEGMENTATION_DEFAULT_MODEL
        model = Path(params['model'])
        ctx.meta['steps'].append(ProcessingStep(id=f'_{uuid.uuid4()}',
                                                category='processing',
                                                description='Baseline and region segmentation',
                                                settings={'model': model.name,
                                                          'text_direction': config.text_direction}))

        # first try to find the segmentation models by their given names, then
        # look in the kraken config folder
        locations = []
        location = None
        search = chain([model],
                       Path(user_data_dir('htrmopo')).rglob(str(model)),
                       Path(click.get_app_dir('kraken')).rglob(str(model)))
        for loc in search:
            if loc.is_file():
                location = loc
                locations.append(loc)
                break
        if not location:
            raise click.BadParameter(f'No model for {model} found')
    else:
        model = None
        ctx.meta['steps'].append(ProcessingStep(id=f'_{uuid.uuid4()}',
                                                category='processing',
                                                description='bounding box segmentation',
                                                settings={'text_direction': config.text_direction,
                                                          'scale': config.legacy_scale,
                                                          'maxcolseps': config.legacy_maxcolseps,
                                                          'black_colseps': config.legacy_black_colseps,
                                                          'no_hlines': config.legacy_no_hlines,
                                                          'pad': config.bbox_line_padding}))

    return partial(segmenter, params['boxes'], model, config)


@click.command('ocr')
@click.pass_context
@click.option('-m',
              '--model',
              help='Path to recognition model weights.')
@click.option('-B',
              '--batch-size',
              type=int,
              help='Number of lines per forward pass batch.')
@click.option('-p',
              '--pad',
              'padding',
              type=int,
              help='Left and right padding around lines')
@click.option('-t',
              '--temperature',
              type=float,
              help='Softmax temperature')
@click.option('--num-line-workers',
              type=click.IntRange(0),
              help='Number of line extraction worker processes. 0 for in-process extraction.')
@click.option('-n',
              '--reorder/--no-reorder',
              'bidi_reordering',
              help='Reorder code points to logical order in output.')
@click.option('--base-dir',
              type=click.Choice(['L', 'R', 'auto']),
              help='Set base text direction. This should be set to the '
              'direction used during the creation of the training data. If '
              'set to `auto` it will be overridden by any explicit value '
              'given in the input files.')
@click.option('-s',
              '--no-segmentation',
              default=False,
              is_flag=True,
              help='Enables non-segmentation mode treating each input image as a whole line.')
@click.option('-d',
              '--text-direction',
              type=click.Choice(['horizontal-tb', 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction in serialization output')
@click.option('--no-legacy-polygons',
              'no_legacy_polygons',
              is_flag=True,
              default=False,
              help="Force disable the legacy polygon extractor")
def ocr(ctx, **kwargs):
    """
    Recognizes text in line images.
    """
    from kraken.containers import ProcessingStep

    params = ctx.params
    config = RecognitionInferenceConfig(**params, **ctx.meta)

    if ctx.meta['input_format_type'] != 'image' and params['no_segmentation']:
        raise click.BadParameter('no_segmentation mode is incompatible with page/alto inputs')

    if config.bidi_reordering and params['base_dir'] != 'auto':
        config.bidi_reordering = params['base_dir']

    # first we try to find the model in the absolute path, then ~/.kraken
    search = chain([Path(params['model'])],
                   Path(user_data_dir('htrmopo')).rglob(params['model']),
                   Path(click.get_app_dir('kraken')).rglob(params['model']))

    location = None
    for loc in search:
        if loc.is_file():
            location = loc
            break
    if not location:
        raise click.BadParameter(f'No model path for {params["model"]} found')

    ctx.meta['steps'].append(ProcessingStep(id=f'_{uuid.uuid4()}',
                                            category='processing',
                                            description='Text line recognition',
                                            settings={'text_direction': config.text_direction,
                                                      'models': params['model'],
                                                      'pad': config.padding,
                                                      'bidi_reordering': config.bidi_reordering}))

    # set output mode
    ctx.meta['text_direction'] = params['text_direction']
    return partial(recognizer,
                   model=location,
                   no_segmentation=params['no_segmentation'],
                   config=config)


@click.command('show')
@click.pass_context
@click.option('-V', '--metadata-version',
              default='highest',
              type=click.Choice(['v0', 'v1', 'highest']),
              help='Version of metadata to fetch if multiple exist in repository.')
@click.argument('model_id')
def show(ctx, metadata_version, model_id):
    """
    Retrieves model metadata from the repository.
    """
    from htrmopo.util import iso15924_to_name, iso639_3_to_name
    from kraken.repo import get_description
    from kraken.lib.util import is_printable, make_printable

    def _render_creators(creators):
        o = []
        for creator in creators:
            c_text = creator['name']
            if (orcid := creator.get('orcid', None)) is not None:
                c_text += f' ({orcid})'
            if (affiliation := creator.get('affiliation', None)) is not None:
                c_text += f' ({affiliation})'
            o.append(c_text)
        return o

    def _render_metrics(metrics):
        if metrics:
            return [f'{k}: {v:.2f}' for k, v in metrics.items()]
        return ''

    if metadata_version == 'highest':
        metadata_version = None

    try:
        desc = get_description(model_id,
                               version=metadata_version,
                               filter_fn=lambda record: getattr(record, 'software_name', None) == 'kraken' or 'kraken_pytorch' in record.keywords)
    except ValueError as e:
        logger.error(e)
        ctx.exit(1)

    if desc.version == 'v0':
        chars = []
        combining = []
        for char in sorted(desc.graphemes):
            if not is_printable(char):
                combining.append(make_printable(char))
            else:
                chars.append(char)

        table = Table(title=desc.summary, show_header=False)
        table.add_column('key', justify="left", no_wrap=True)
        table.add_column('value', justify="left", no_wrap=False)
        table.add_row('DOI', desc.doi)
        table.add_row('concept DOI', desc.concept_doi)
        table.add_row('publication date', desc.publication_date.isoformat())
        table.add_row('model type', Group(*desc.model_type))
        table.add_row('script', Group(*[iso15924_to_name(x) for x in desc.script]))
        table.add_row('alphabet', Group(' '.join(chars), ', '.join(combining)))
        table.add_row('keywords', Group(*desc.keywords))
        table.add_row('metrics', Group(*_render_metrics(desc.metrics)))
        table.add_row('license', desc.license)
        table.add_row('creators', Group(*_render_creators(desc.creators)))
        table.add_row('description', desc.description)
    elif desc.version == 'v1':
        table = Table(title=desc.summary, show_header=False)
        table.add_column('key', justify="left", no_wrap=True)
        table.add_column('value', justify="left", no_wrap=False)
        table.add_row('DOI', desc.doi)
        table.add_row('concept DOI', desc.concept_doi)
        table.add_row('publication date', desc.publication_date.isoformat())
        table.add_row('model type', Group(*desc.model_type))
        table.add_row('language', Group(*[iso639_3_to_name(x) for x in desc.language]))
        table.add_row('script', Group(*[iso15924_to_name(x) for x in desc.script]))
        table.add_row('keywords', Group(*desc.keywords) if desc.keywords else '')
        table.add_row('datasets', Group(*desc.datasets) if desc.datasets else '')
        table.add_row('metrics', Group(*_render_metrics(desc.metrics)) if desc.metrics else '')
        table.add_row('base model', Group(*desc.base_model) if desc.base_model else '')
        table.add_row('software', desc.software_name)
        table.add_row('software_hints', Group(*desc.software_hints) if desc.software_hints else '')
        table.add_row('license', desc.license)
        table.add_row('creators', Group(*_render_creators(desc.creators)))
        table.add_row('description', Markdown(desc.description))

    print(table)


@click.command('list')
@click.option('--all', 'model_type', flag_value='all', default=True, help='List both segmentation and recognition models.')
@click.option('--recognition', 'model_type', flag_value='recognition', help='Only list recognition models.')
@click.option('--segmentation', 'model_type', flag_value='segmentation', help='Only list segmentation models.')
@click.option('-l', '--language', default=None, multiple=True, help='Filter for language by ISO 639-3 codes')
@click.option('-s', '--script', default=None, multiple=True, help='Filter for script by ISO 15924 codes')
@click.option('-k', '--keyword', default=None, multiple=True, help='Filter by keyword.')
@click.pass_context
def list_models(ctx, model_type, language, script, keyword):
    """
    Lists models in the repository.

    Multiple filters of different type are ANDed, specifying a filter of a
    single type multiple times will OR those values:

    --script Arab --script Syrc -> Arab OR Syrc script models

    --script Arab --language urd -> Arab script AND Urdu language models
    """
    from kraken.repo import get_listing
    from kraken.lib.progress import KrakenProgressBar

    if language:
        logger.warning('Filtering by language is only supported for v1 records. '
                       'You might not find the model(s) you are looking for if '
                       'they do not have a v1 metadata record.')

    def _filter_fn(record):
        if getattr(record, 'software_name', None) != 'kraken' and 'kraken_pytorch' not in record.keywords:
            return False
        if model_type != 'all' and model_type not in record.model_type:
            return False
        if script and not any(filter(lambda s: s in script, record.script)):
            return False
        if language and not any(filter(lambda s: s in language, getattr(record, 'language', tuple()))):
            return False
        if keyword and not any(filter(lambda s: s in keyword, record.keywords)):
            return False
        return True

    with KrakenProgressBar() as progress:
        download_task = progress.add_task('Retrieving model list', total=0, visible=True if not ctx.meta['verbose'] else False)
        repository = get_listing(callback=lambda total, advance: progress.update(download_task, total=total, advance=advance),
                                 filter_fn=_filter_fn)

    if model_type is not None:
        repository
    table = Table(show_header=True)
    table.add_column('DOI', justify="left", no_wrap=True)
    table.add_column('summary', justify="left", no_wrap=False)
    table.add_column('model type', justify="left", no_wrap=False)
    table.add_column('keywords', justify="left", no_wrap=False)

    for k, records in repository.items():
        t = Tree(k)
        [t.add(x.doi) for x in records]
        table.add_row(t,
                      Group(*[''] + [x.summary for x in records]),
                      Group(*[''] + ['; '.join(x.model_type) for x in records]),
                      Group(*[''] + ['; '.join(x.keywords) for x in records]))

    print(table)


@click.command('get')
@click.pass_context
@click.argument('model_id')
def get(ctx, model_id):
    """
    Retrieves a model from the repository.
    """
    from htrmopo import get_model

    from kraken.repo import get_description
    from kraken.lib.progress import KrakenDownloadProgressBar

    try:
        get_description(model_id,
                        filter_fn=lambda record: getattr(record, 'software_name', None) == 'kraken' or 'kraken_pytorch' in record.keywords)
    except ValueError as e:
        logger.error(e)
        ctx.exit(1)

    with KrakenDownloadProgressBar() as progress:
        download_task = progress.add_task('Processing', total=0, visible=True if not ctx.meta['verbose'] else False)
        model_dir = get_model(model_id,
                              callback=lambda total, advance: progress.update(download_task, total=total, advance=advance))
    model_candidates = list(filter(lambda x: x.suffix == '.mlmodel', model_dir.iterdir()))
    message(f'Model dir: {model_dir} (model files: {", ".join(x.name for x in model_candidates)})')


for subcommand in sorted(importlib.metadata.entry_points(group='kraken.cli')):
    cli.add_command(subcommand.load(), name=subcommand.name)


if __name__ == '__main__':
    cli()
