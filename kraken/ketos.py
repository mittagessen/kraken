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

import os
import time
import click
import errno
import base64
import logging
import unicodedata
import numpy as np

from PIL import Image
from lxml import html
from io import BytesIO
from itertools import cycle
from bidi.algorithm import get_display

from kraken import rpred
from kraken import linegen
from kraken import pageseg
from kraken import transcribe
from kraken import binarization
from kraken.lib import log
from kraken.lib import models
from kraken.train import GroundTruthContainer, compute_error
from kraken.lib.exceptions import KrakenCairoSurfaceException
from kraken.lib.exceptions import KrakenInputException

standard_library.install_aliases()

APP_NAME = 'kraken'

logger = logging.getLogger('kraken')

spinner = cycle([u'⣾', u'⣽', u'⣻', u'⢿', u'⡿', u'⣟', u'⣯', u'⣷'])

def spin(msg):
    if logger.getEffectiveLevel() >= 30:
        click.echo(u'\r\033[?25l{}\t{}'.format(msg, next(spinner)), nl=False)

def message(msg, **styles):
    if logger.getEffectiveLevel() >= 30:
        click.secho(msg, **styles)


@click.group()
@click.version_option()
@click.option('-v', '--verbose', default=0, count=True)
def cli(verbose):
    ctx = click.get_current_context()
    log.set_logger(logger, level=30-10*verbose)
    ctx.meta['verbose'] = verbose


@cli.command('train')
@click.pass_context
@click.option('-l', '--lineheight', default=48, help='Line image height after normalization')
@click.option('-p', '--pad', type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('-S', '--hiddensize', default=100, help='LSTM units in hidden layer')
@click.option('-o', '--output', type=click.Path(), default='model.clstm', help='Output model file')
@click.option('-i', '--load', type=click.Path(exists=True, readable=True), help='Load existing file to continue training')
@click.option('-F', '--savefreq', default=1000, help='Model save frequency during training')
@click.option('-R', '--report', default=1000, help='Report creation frequency')
@click.option('-N', '--ntrain', default=1000000, help='Iterations to train.')
@click.option('-r', '--lrate', default=1e-4, help='LSTM learning rate')
@click.option('-m', '--momentum', default=0.9, help='LSTM momentum')
@click.option('-p', '--partition', default=0.9, help='Ground truth data partition ratio between train/test set')
@click.option('-u', '--normalization', type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']), default=None, help='Normalize ground truth')
@click.option('-n', '--reorder/--no-reorder', default=True, help='Reorder code points to display order')
@click.argument('ground_truth', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def train(ctx, lineheight, pad, hiddensize, output, load, savefreq, report,
          ntrain, lrate, momentum, partition, normalization, reorder,
          ground_truth):
    """
    Trains a model from image-text pairs.
    """
    if load is None:
        message(u'Training from scratch net yet supported.')
        ctx.exit(1)
    logger.info(u'Building ground truth set from {} line images'.format(len(ground_truth)))
    spin(u'Building ground truth set')

    gt_set = GroundTruthContainer()

    for line in ground_truth:
        gt_set.add(line, normalization=normalization, reorder=reorder)
        logger.debug(u'Adding {}'.format(line))
        spin(u'Building ground truth set')
    gt_set.repartition(partition)

    logger.info(u'Training set {} lines, test set {} lines, alphabet {} symbols'.format(len(gt_set.training_set), len(gt_set.test_set), len(gt_set.training_alphabet)))
    logger.debug(u'grapheme\tcount')
    for k, v in sorted(gt_set.training_alphabet.items(), key=lambda x : x[1], reverse=True):
        if unicodedata.combining(k) or k.isspace():
            k = unicodedata.name(k)
        else:
            k = '\t' + k
        logger.debug(u'{}\t{}'.format(k, v))

    message(u'\b\u2713', fg='green', nl=False)
    message('\033[?25h\n', nl=False)

    if load:
        logger.info(u'Loading existing model from {} '.format(load))
        spin('Loading model')

        rnn = models.ClstmSeqRecognizer(load)

        message(u'\b\u2713', fg='green', nl=False)
        message('\033[?25h\n', nl=False)

    else:
        ctx.exit(1)

    logger.info(u'Setting learning rate ({}) and momentum ({}) '.format(lrate, momentum))
    rnn.setLearningRate(lrate, momentum)

    for trial in xrange(ntrain):
        line, s = gt_set.sample()
        res = rnn.trainString(line, s)
        logger.debug(u'TRU: {1}\n[{0:2.4f}] OUT: {2}'.format(s, res))
        spin('Training')

        if trial and not trial % savefreq:
            rnn.save_model('{}_{}'.format(output, trial))
            logger.info(u'Saving to {}_{}'.format(output, trial))

        if trial and not trial % report:
            c, e = compute_error(rnn, gt_set.test_set)
            logger.info(u'Accuracy report ({}) {:0.4f} {} {}'.format(trial, (c-e)/c, c, e))


@cli.command('extract')
@click.pass_context
@click.option('-u', '--normalization',
              type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']), default=None,
              help='Normalize ground truth')
@click.option('-n', '--reorder/--no-reorder', default=True,
              help='Reorder transcribed lines to display order')
@click.option('-r', '--rotate/--no-rotate', default=True,
              help='Skip rotation of vertical lines')
@click.option('-o', '--output', type=click.Path(), default='training',
              help='Output directory')
@click.argument('transcriptions', nargs=-1, type=click.File(lazy=True))
def extract(ctx, normalization, reorder, rotate, output, transcriptions):
    """
    Extracts image-text pairs from a transcription environment created using
    ``ketos transcribe``.
    """
    try:
        os.mkdir(output)
    except:
        pass
    idx = 0
    manifest = []
    for fp in transcriptions:
        logger.info(u'Reading {}'.format(fp.name))
        spin('Reading transcription')
        doc = html.parse(fp)
        td = doc.find(".//meta[@itemprop='text_direction']")
        if td is None:
            td = 'horizontal-tb'
        else:
            td = td.attrib['content']

        im = None
        for section in doc.xpath('//section'):
            img = section.xpath('.//img')[0].get('src')
            fd = BytesIO(base64.b64decode(img.split(',')[1]))
            im = Image.open(fd)
            if not im:
                logger.info(u'Skipping {} because image not found'.format(fp.name))
                break
            for line in section.iter('li'):
                if line.get('contenteditable') and not u''.join(line.itertext().isspace()):
                    l = im.crop([int(x) for x in line.get('data-bbox').split(',')])
                    if rotate and td.startswith('vertical'):
                        im.rotate(90, expand=True)
                    l.save('{}/{:06d}.png'.format(output, idx))
                    manifest.append('{:06d}.png'.format(idx))
                    text = u''.join(line.itertext())
                    if normalization:
                        text = unicodedata.normalize(normalization, text)
                    with open('{}/{:06d}.gt.txt'.format(output, idx), 'wb') as t:
                        if reorder:
                            t.write(get_display(text).encode('utf-8'))
                        else:
                            t.write(text.encode('utf-8'))
                    idx += 1
    logger.info(u'Extracted {} lines'.format(idx))
    with open('{}/manifest.txt'.format(output), 'w') as fp:
        fp.write('\n'.join(manifest))
    message(u'\b\u2713', fg='green', nl=False)
    message('\033[?25h\n', nl=False)


@cli.command('transcribe')
@click.pass_context
@click.option('-d', '--text-direction', default='horizontal-tb',
              type=click.Choice(['horizontal-tb', 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction')
@click.option('--scale', default=None, type=click.FLOAT)
@click.option('-m', '--maxcolseps', default=2, type=click.INT)
@click.option('-b/-w', '--black_colseps/--white_colseps', default=False)
@click.option('-f', '--font', default='',
              help='Font family to use')
@click.option('-fs', '--font-style', default=None,
              help='Font style to use')
@click.option('-p', '--prefill', default=None,
              help='Use given model for prefill mode.')
@click.option('-o', '--output', type=click.File(mode='wb'), default='transcription.html',
              help='Output file')
@click.argument('images', nargs=-1, type=click.File(mode='rb', lazy=True))
def transcription(ctx, text_direction, scale, maxcolseps, black_colseps, font,
                  font_style, prefill, output, images):
    ti = transcribe.TranscriptionInterface(font, font_style)

    if prefill:
        logger.info('Loading model {}'.format(prefill))
        spin('Loading RNN')
        prefill = models.load_any(prefill.encode('utf-8'))
        message(u'\b\u2713', fg='green', nl=False)
        message('\033[?25h\n', nl=False)

    for fp in images:
        logger.info('Reading {}'.format(fp.name))
        spin('Reading images')
        im = Image.open(fp)
        if not binarization.is_bitonal(im):
            logger.info(u'Binarizing page')
            im = binarization.nlbin(im)
        logger.info(u'Segmenting page')
        res = pageseg.segment(im, text_direction, scale, maxcolseps, black_colseps)
        if prefill:
            it = rpred.rpred(prefill, im, res)
            preds = []
            for pred in it:
                logger.info('{}'.format(pred.prediction))
                spin('Recognizing')
                preds.append(pred)
            message(u'\b\u2713', fg='green', nl=False)
            message('\033[?25h\n', nl=False)
            ti.add_page(im, res, records=preds)
        else:
            ti.add_page(im, res)
        fp.close()
    message(u'\b\u2713', fg='green', nl=False)
    message('\033[?25h\n', nl=False)
    logger.info(u'Writing transcription to {}'.format(output.name))
    spin('Writing output')
    ti.write(output)
    message(u'\b\u2713', fg='green', nl=False)
    message('\033[?25h\n', nl=False)


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
@click.option('-d', '--disable-degradation', is_flag=True, help='Dont degrade '
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
    lines = set()
    if not text:
        return
    st_time = time.time()
    for t in text:
        with click.open_file(t, encoding=encoding) as fp:
            logger.info('Reading {}'.format(t))
            spin('Reading texts')
            for l in fp:
                lines.add(l.rstrip('\r\n'))
    if normalization:
        lines = set([unicodedata.normalize(normalization, line) for line in lines])
    if strip:
        lines = set([line.strip() for line in lines])
    if max_length:
        lines = set([line for line in lines if len(line) < max_length])
    logger.info('Read {} lines'.format(len(lines)))
    message(u'\b\u2713', fg='green', nl=False)
    message('\033[?25h\n', nl=False)
    message('Read {} unique lines'.format(len(lines)))
    if maxlines and maxlines < len(lines):
        message('Sampling {} lines\t'.format(maxlines), nl=False)
        lines = list(lines)
        lines = [lines[idx] for idx in np.random.randint(0, len(lines), maxlines)]
        message(u'\u2713', fg='green')
    try:
        os.makedirs(output)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

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
    message(u'Σ (len: {})'.format(len(alphabet)))
    message(u'Symbols: {}'.format(''.join(chars)))
    if combining:
        message(u'Combining Characters: {}'.format(', '.join(combining)))
    lg = linegen.LineGenerator(font, font_size, font_weight, language)
    for idx, line in enumerate(lines):
        logger.info(line)
        spin('Writing images')
        try:
            if renormalize:
                im = lg.render_line(unicodedata.normalize(renormalize, line))
            else:
                im = lg.render_line(line)
        except KrakenCairoSurfaceException as e:
            logger.info('{}: {} {}'.format(e.message, e.width, e.height))
            message(u'\b\u2717', fg='red')
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
    message(u'\b\u2713', fg='green', nl=False)
    message('\033[?25h\n', nl=False)


if __name__ == '__main__':
    cli()
