# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from future import standard_library
standard_library.install_aliases()

import click
import csv
import os
import urllib.request
import tempfile

from PIL import Image
from click import open_file
from urllib.parse import urljoin
from itertools import cycle
from collections import namedtuple
from functools import partial
from multiprocessing import Queue, Pool, cpu_count
from kraken import binarization
from kraken import pageseg
from kraken import rpred
from kraken import html
from kraken.lib import models

APP_NAME = 'kraken'
MODEL_URL = 'http://l.unchti.me/'
DEFAULT_MODEL = 'en-default.pyrnn.hdf5'
LEGACY_MODEL_DIR = '/usr/local/share/ocropus'

spinner = cycle([u'⣾', u'⣽', u'⣻', u'⢿', u'⡿', u'⣟', u'⣯', u'⣷'])

message = namedtuple('message', 'func args remainder')
result_message = namedtuple('result_message', 'func args state data')


def binarizer(threshold, zoom, escale, border, perc, range, low, high, input,
              output, queue, result_queue, remainder):
    try:
        im = Image.open(input)
    except IOError as e:
        raise click.BadParameter(e.message)
    res = binarization.nlbin(im, threshold, zoom, escale, border, perc, range,
                             low, high)
    res.save(output, format='png')
    if remainder:
        args = {'input': remainder[0][1], 'output': remainder[0][2]}
        queue.put(message(remainder[0][0], args, remainder[1:]))


def segmenter(scale, black_colseps, input, output, queue, result_queue,
              remainder):
    try:
        im = Image.open(input)
    except IOError as e:
        raise click.BadParameter(e.message)
    res = pageseg.segment(im, scale, black_colseps)
    if remainder:
        result_queue.put(result_message(partial(segmenter),
                                        {},
                                        'new',
                                        len(res) - 1))
    with open_file(output, 'w') as fp:
        for box in enumerate(res):
            if remainder:
                args = {'input': input,
                        'box_id': box[0],
                        'box': box[1],
                        'output': remainder[0][2]}
                queue.put(message(remainder[0][0], args, remainder[1:]))
            else:
                fp.write(u'{},{},{},{}\n'.format(*box[1]).encode('utf-8'))


def recognizer(model, pad, input, output, queue, result_queue, remainder,
               lines=None, box=None, box_id=0):
    try:
        im = Image.open(input)
    except IOError as e:
        raise click.BadParameter(e.message)

    # if a line file is given (and no specific boxes to recognize) we split it
    # into a bounding box array and reschedule recognition for each box
    if lines and not box:
        with open_file(lines, 'r') as fp:
            bounds = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2
                      in csv.reader(fp)]
            result_queue.put(result_message(partial(recognizer), {}, 'new',
                                            len(bounds)))
            for box in enumerate(bounds):
                args = {'input': input,
                        'box_id': box[0],
                        'box': box[1],
                        'output': remainder[0][2]}
                queue.put(message(remainder[0][0], args, remainder[1:]))
    else:
        it = rpred.rpred(model, im, [box], pad)
        result_queue.put(result_message(partial(recognizer),
                         {'box_id': box_id, 'input': input, 'output': output},
                         'result', next(it)))


@click.group(chain=True, invoke_without_command=True)
@click.option('-i', '--input', type=(click.Path(exists=True),
                                     click.Path(writable=True)), multiple=True)
@click.option('-c', '--concurrency', default=cpu_count(), type=click.INT)
@click.option('-v', '--verbose', count=True)
def cli(input, concurrency, verbose):
    pass


@cli.resultcallback()
def process_pipeline(subcommands, input, concurrency, verbose):
    if not len(subcommands):
        subcommands = [binarize.callback(),
                       segment.callback(),
                       ocr.callback()]
    q = Queue()
    rq = Queue()

    def pipeline_worker(queue, result_queue):
        while True:
            msg = queue.get(block=True, timeout=None)
            rq.put(result_message(*msg[:2], state='running', data=None))
            try:
                msg.func(queue=queue, result_queue=rq, remainder=msg.remainder,
                         **msg.args)
            except Exception as e:
                rq.put(result_message(*msg[:2], state='error', data=e))
                continue
            rq.put(result_message(*msg[:2], state='finished', data=None))

    Pool(processes=concurrency, initializer=pipeline_worker, initargs=(q, rq))

    expected = 0
    temps = []
    for io_pair in input:
        # create temporary files for intermediate results
        fc = [io_pair[0]] + [tempfile.mkstemp()[1] for cmd in subcommands[1:]] + [io_pair[1]]
        temps.extend(fc[1:-1])
        chain = zip(subcommands, fc, fc[1:])
        # we expect len(chain) finished/failed tasks
        expected += len(chain)
        q.put(message(chain[0][0], {'input': chain[0][1], 'output': chain[0][2]}, chain[1:]))

    finished = 0
    results = {}
    if verbose == 1:
        click.echo('Waiting for {} finished processes.'.format(expected))
    while expected != finished:
        st = rq.get(block=True, timeout=None)
        if st.state == 'finished':
            finished += 1
        elif st.state == 'new':
            expected += st.data
            if verbose == 1:
                click.echo('Spawned {} new tasks'.format(st.data))
            continue
        elif st.state == 'error':
            if not verbose:
                click.secho(u'\b\u2717', fg='red', nl=False)
                click.echo('\033[?25h\n', nl=False)
                for f in temps:
                    os.unlink(f)
            raise st.data
        elif st.state == 'result':
            if st.args['output'] not in results:
                results[st.args['output']] = []
            results[st.args['output']].append((st.args['box_id'], st.data))

        if not verbose:
            click.echo(u'\r\033[?25lProcessing\t{}'.format(next(spinner)), nl=False)
        elif verbose == 1:
            click.echo('{} {} on {}'.format(st.state.title(),
                                            st.func.func.__name__,
                                            st.args['input']), nl=False)
            if 'box_id' in st.args:
                click.echo(':line {}'.format(st.args['box_id']),
                           nl=False)
            click.echo()

    if not verbose:
        click.secho(u'\b\u2713', fg='green', nl=False)
        click.echo('\033[?25h\n', nl=False)

    # sort results for each output and create final document
    ctx = click.get_current_context()
    for dest, preds in results.items():
        preds = sorted(preds, key=lambda pred: pred[0])
        with open_file(dest, 'w', encoding='utf-8')as fp:
            iopair = next(iopair for iopair in input if iopair[1] == dest)
            click.echo('Writing recognition results for {}\t'.format(iopair[0]), nl=False)
            if ctx.meta['mode'] == 'hocr':
                fp.write(unicode(html.hocr((x[1] for x in preds), iopair[0])))
            else:
                fp.write(u'\n'.join(s[1].prediction for s in preds))
            click.secho(u'\u2713', fg='green')

    for f in temps:
        os.unlink(f)

@click.command('binarize')
@click.option('--threshold', default=0.5, type=click.FLOAT)
@click.option('--zoom', default=0.5, type=click.FLOAT)
@click.option('--escale', default=1.0, type=click.FLOAT)
@click.option('--border', default=0.1, type=click.FLOAT)
@click.option('--perc', default=80, type=click.IntRange(1, 100))
@click.option('--range', default=20, type=click.INT)
@click.option('--low', default=5, type=click.IntRange(1, 100))
@click.option('--high', default=90, type=click.IntRange(1, 100))
def binarize(threshold=0.5, zoom=0.5, escale=1.0, border=0.1, perc=80,
             range=20, low=5, high=90):
    return partial(binarizer, threshold, zoom, escale, border, perc, range,
                   low, high)


@click.command('segment')
@click.option('--scale', default=None, type=click.FLOAT)
@click.option('-b/-w', '--black_colseps/--white_colseps', default=False)
def segment(scale=None, black_colseps=False):
    return partial(segmenter, scale, black_colseps)

@click.command('ocr')
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
def ocr(ctx, model=DEFAULT_MODEL, pad=16, hocr=False, lines=None, conv=True):
    # we do the locating and loading of the model here to spare us the overhead
    # in each worker.

    # first we try to find the model in the absolue path, then ~/.kraken, then
    # LEGACY_MODEL_DIR
    search = [model,
              os.path.join(click.get_app_dir(APP_NAME, force_posix=True), model),
              os.path.join(LEGACY_MODEL_DIR, model)]
    # if automatic conversion is enabled we look for an converted model in
    # ~/.kraken
    if conv is True:
        search.insert(0, os.path.join(click.get_app_dir(APP_NAME,
                                      force_posix=True),
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
        ctx.exit(1)
    click.secho(u'\u2713', fg='green')

    # convert input model to HDF5
    if conv and rnn.kind == 'pyrnn':
        name, _ = os.path.splitext(os.path.basename(model))
        op = os.path.join(click.get_app_dir(APP_NAME, force_posix=True), name +
                          '.hdf5')
        try:
            os.makedirs(click.get_app_dir(APP_NAME, force_posix=True))
        except OSError:
            pass
        models.pyrnn_to_hdf5(rnn, op)

    # set output mode
    if hocr:
        ctx.meta['mode'] = 'hocr'
    else:
        ctx.meta['mode'] = 'text'
    return partial(recognizer, model=rnn, pad=pad, lines=lines)


@click.command('download')
@click.pass_context
def download(ctx):
    default_model = urllib.request.urlopen(urljoin(MODEL_URL, DEFAULT_MODEL))
    try:
        os.makedirs(click.get_app_dir(APP_NAME, force_posix=True))
    except OSError:
        pass
    # overwrite next function for iterator to return 8192 octets instead of
    # line
    default_model.next = lambda: default_model.read(8192)
    fs = int(default_model.info()["Content-Length"])
    with open_file(os.path.join(click.get_app_dir(APP_NAME, force_posix=True),
                                DEFAULT_MODEL), 'wb') as fp:
        with click.progressbar(length=fs,
                               label='Downloading default model',
                               fill_char=click.style('#', fg='green')) as dl:
            for buf in default_model:
                if not buf:
                    raise StopIteration()
                dl.update(len(buf))
                fp.write(buf)
    ctx.exit(0)

cli.add_command(binarize)
cli.add_command(segment)
cli.add_command(ocr)
cli.add_command(download)

if __name__ == '__main__':
    cli()
