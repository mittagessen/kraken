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
import os
import glob
import uuid
import click
import logging
import unicodedata

from bidi.algorithm import get_display

from typing import cast, Set

from kraken.lib import log
from kraken.lib.exceptions import KrakenCairoSurfaceException
from kraken.lib.exceptions import KrakenEncodeException
from kraken.lib.exceptions import KrakenInputException

APP_NAME = 'kraken'

logger = logging.getLogger('kraken')


def message(msg, **styles):
    if logger.getEffectiveLevel() >= 30:
        click.secho(msg, **styles)


@click.group()
@click.version_option()
@click.option('-v', '--verbose', default=0, count=True)
def cli(verbose):
    log.set_logger(logger, level=30-min(10*verbose, 20))


def _validate_manifests(ctx, param, value):
    images = []
    for manifest in value:
        images.extend([x.rstrip('\r\n') for x in manifest.readlines() if os.path.isfile(x.rstrip('\r\n'))])
    return images


def _expand_gt(ctx, param, value):
    images = []
    for expression in value:
        images.extend([x for x in glob.iglob(expression, recursive=True) if os.path.isfile(x)])
    return images


@cli.command('train')
@click.pass_context
@click.option('-p', '--pad', show_default=True, type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model', help='Output model file')
@click.option('-s', '--spec', show_default=True,
              default='[1,48,0,1 Cr3,3,32 Do0.1,2 Mp2,2 Cr3,3,64 Do0.1,2 Mp2,2 S1(1x12)1,3 Lbx100 Do]',
              help='VGSL spec of the network to train. CTC layer will be added automatically.')
@click.option('-a', '--append', show_default=True, default=None, type=click.INT,
              help='Removes layers before argument and then appends spec. Only works when loading an existing model')
@click.option('-i', '--load', show_default=True, type=click.Path(exists=True, readable=True), help='Load existing file to continue training')
@click.option('-F', '--savefreq', show_default=True, default=1, type=click.INT, help='Model save frequency in epochs during training')
@click.option('-R', '--report', show_default=True, default=1, type=click.INT, help='Report creation frequency in epochs')
@click.option('-q', '--quit', show_default=True, default='early', type=click.Choice(['early', 'dumb']),
              help='Stop condition for training. Set to `early` for early stooping or `dumb` for fixed number of epochs')
@click.option('-N', '--epochs', show_default=True, default=-1, help='Number of epochs to train for')
@click.option('--lag', show_default=True, default=5, help='Number of epochs to wait before stopping training without improvement')
@click.option('--min-delta', show_default=True, default=0.005, help='Minimum improvement between epochs to reset early stopping')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--optimizer', show_default=True, default='Adam', type=click.Choice(['Adam', 'SGD', 'RMSprop']), help='Select optimizer')
@click.option('-r', '--lrate', show_default=True, default=2e-3, help='Learning rate')
@click.option('-m', '--momentum', show_default=True, default=0.9, help='Momentum')
@click.option('-w', '--weight-decay', show_default=True, default=0.0, help='Weight decay')
@click.option('--schedule', show_default=True, type=click.Choice(['constant', '1cycle']), default='constant',
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--epoch` option.')
@click.option('-p', '--partition', show_default=True, default=0.9, help='Ground truth data partition ratio between train/validation set')
@click.option('-u', '--normalization', show_default=True, type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              default=None, help='Ground truth normalization')
@click.option('-c', '--codec', show_default=True, default=None, type=click.File(mode='r', lazy=True),
              help='Load a codec JSON definition (invalid if loading existing model)')
@click.option('--resize', show_default=True, default='fail', type=click.Choice(['add', 'both', 'fail']),
              help='Codec/output layer resizing option. If set to `add` code '
                   'points will be added, `both` will set the layer to match exactly '
                   'the training data, `fail` will abort if training data and model '
                   'codec do not match.')
@click.option('-n', '--reorder/--no-reorder', show_default=True, default=True, help='Reordering of code points to display order')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--preload/--no-preload', show_default=True, default=None, help='Hard enable/disable for training data preloading')
@click.option('--threads', show_default=True, default=1, help='Number of OpenMP threads when running on CPU.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def train(ctx, pad, output, spec, append, load, savefreq, report, quit, epochs,
          lag, min_delta, device, optimizer, lrate, momentum, weight_decay,
          schedule, partition, normalization, codec, resize, reorder,
          training_files, evaluation_files, preload, threads, ground_truth):
    """
    Trains a model from image-text pairs.
    """
    import re
    import torch
    import shutil
    import numpy as np

    from torch.utils.data import DataLoader

    from kraken.lib import models, vgsl, train
    from kraken.lib.train import EarlyStopping, EpochStopping, TrainStopper, TrainScheduler, add_1cycle
    from kraken.lib.codec import PytorchCodec
    from kraken.lib.dataset import GroundTruthDataset, compute_error, generate_input_transforms

    logger.info('Building ground truth set from {} line images'.format(len(ground_truth) + len(training_files)))

    if not load and append:
        raise click.BadOptionUsage('append', 'append option requires loading an existing model')

    if resize != 'fail' and not load:
        raise click.BadOptionUsage('resize', 'resize option requires loading an existing model')

    # load model if given. if a new model has to be created we need to do that
    # after data set initialization, otherwise to output size is still unknown.
    nn = None
    if load:
        logger.info('Loading existing model from {} '.format(load))
        message('Loading model {}'.format(load), nl=False)
        nn = vgsl.TorchVGSLModel.load_model(load)
        message('\u2713', fg='green', nl=False)

    # preparse input sizes from vgsl string to seed ground truth data set
    # sizes and dimension ordering.
    if not nn:
        spec = spec.strip()
        if spec[0] != '[' or spec[-1] != ']':
            raise click.BadOptionUsage('spec', 'VGSL spec {} not bracketed'.format(spec))
        blocks = spec[1:-1].split(' ')
        m = re.match(r'(\d+),(\d+),(\d+),(\d+)', blocks[0])
        if not m:
            raise click.BadOptionUsage('spec', 'Invalid input spec {}'.format(blocks[0]))
        batch, height, width, channels = [int(x) for x in m.groups()]
    else:
        batch, channels, height, width = nn.input
    try:
        transforms = generate_input_transforms(batch, height, width, channels, pad)
    except KrakenInputException as e:
        raise click.BadOptionUsage('spec', str(e))

    # disable automatic partition when given evaluation set explicitly
    if evaluation_files:
        partition = 1
    ground_truth = list(ground_truth)

    # merge training_files into ground_truth list
    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    np.random.shuffle(ground_truth)

    if len(ground_truth) > 2500 and not preload:
        logger.info('Disabling preloading for large (>2500) training data set. Enable by setting --preload parameter')
        preload = False
    # implicit preloading enabled for small data sets
    if preload is None:
        preload = True

    tr_im = ground_truth[:int(len(ground_truth) * partition)]
    if evaluation_files:
        logger.debug('Using {} lines from explicit eval set'.format(len(evaluation_files)))
        te_im = evaluation_files
    else:
        te_im = ground_truth[int(len(ground_truth) * partition):]
        logger.debug('Taking {} lines from training for evaluation'.format(len(te_im)))

    gt_set = GroundTruthDataset(normalization=normalization, reorder=reorder, im_transforms=transforms, preload=preload)
    with log.progressbar(tr_im, label='Building training set') as bar:
        for im in bar:
            logger.debug('Adding line {} to training set'.format(im))
            try:
                gt_set.add(im)
            except FileNotFoundError as e:
                logger.warning('{}: {}. Skipping.'.format(e.strerror, e.filename))
            except KrakenInputException as e:
                logger.warning(str(e))

    val_set = GroundTruthDataset(normalization=normalization, reorder=reorder, im_transforms=transforms, preload=preload)
    with log.progressbar(te_im, label='Building validation set') as bar:
        for im in bar:
            logger.debug('Adding line {} to validation set'.format(im))
            try:
                val_set.add(im)
            except FileNotFoundError as e:
                logger.warning('{}: {}. Skipping.'.format(e.strerror, e.filename))
            except KrakenInputException as e:
                logger.warning(str(e))

    logger.info('Training set {} lines, validation set {} lines, alphabet {} symbols'.format(len(gt_set._images), len(val_set._images), len(gt_set.alphabet)))
    alpha_diff = set(gt_set.alphabet).symmetric_difference(set(val_set.alphabet))
    if alpha_diff:
        logger.warn('alphabet mismatch {}'.format(alpha_diff))
    logger.info('grapheme\tcount')
    for k, v in sorted(gt_set.alphabet.items(), key=lambda x: x[1], reverse=True):
        if unicodedata.combining(k) or k.isspace():
            try:
                k = unicodedata.name(k)
            except ValueError as e:
                k = '0x{:x}'.format(ord(k))
        else:
            k = '\t' + k
        logger.info(u'{}\t{}'.format(k, v))

    logger.debug('Encoding training set')

    # use model codec when given
    if append:
        # is already loaded
        nn = cast(vgsl.TorchVGSLModel, nn)
        gt_set.encode(codec)
        message('Slicing and dicing model ', nl=False)
        # now we can create a new model
        spec = '[{} O1c{}]'.format(spec[1:-1], gt_set.codec.max_label()+1)
        logger.info('Appending {} to existing model {} after {}'.format(spec, nn.spec, append))
        nn.append(append, spec)
        nn.add_codec(gt_set.codec)
        message('\u2713', fg='green')
        logger.info('Assembled model spec: {}'.format(nn.spec))
    elif load:
        # is already loaded
        nn = cast(vgsl.TorchVGSLModel, nn)

        # prefer explicitly given codec over network codec if mode is 'both'
        codec = codec if (codec and resize == 'both') else nn.codec

        try:
            gt_set.encode(codec)
        except KrakenEncodeException as e:
            message('Network codec not compatible with training set')
            alpha_diff = set(gt_set.alphabet).difference(set(codec.c2l.keys()))
            if resize == 'fail':
                logger.error('Training data and model codec alphabets mismatch: {}'.format(alpha_diff))
                ctx.exit(code=1)
            elif resize == 'add':
                message('Adding missing labels to network ', nl=False)
                logger.info('Resizing codec to include {} new code points'.format(len(alpha_diff)))
                codec.c2l.update({k: [v] for v, k in enumerate(alpha_diff, start=codec.max_label()+1)})
                nn.add_codec(PytorchCodec(codec.c2l))
                logger.info('Resizing last layer in network to {} outputs'.format(codec.max_label()+1))
                nn.resize_output(codec.max_label()+1)
                message('\u2713', fg='green')
            elif resize == 'both':
                message('Fitting network exactly to training set ', nl=False)
                logger.info('Resizing network or given codec to {} code sequences'.format(len(gt_set.alphabet)))
                gt_set.encode(None)
                ncodec, del_labels = codec.merge(gt_set.codec)
                logger.info('Deleting {} output classes from network ({} retained)'.format(len(del_labels), len(codec)-len(del_labels)))
                gt_set.encode(ncodec)
                nn.resize_output(ncodec.max_label()+1, del_labels)
                message('\u2713', fg='green')
            else:
                raise click.BadOptionUsage('resize', 'Invalid resize value {}'.format(resize))
    else:
        gt_set.encode(codec)
        logger.info('Creating new model {} with {} outputs'.format(spec, gt_set.codec.max_label()+1))
        spec = '[{} O1c{}]'.format(spec[1:-1], gt_set.codec.max_label()+1)
        nn = vgsl.TorchVGSLModel(spec)
        # initialize weights
        message('Initializing model ', nl=False)
        nn.init_weights()
        nn.add_codec(gt_set.codec)
        # initialize codec
        message('\u2713', fg='green')

    train_loader = DataLoader(gt_set, batch_size=1, shuffle=True, pin_memory=True)

    # don't encode validation set as the alphabets may not match causing encoding failures
    val_set.training_set = list(zip(val_set._images, val_set._gt))

    logger.debug('Constructing {} optimizer (lr: {}, momentum: {})'.format(optimizer, lrate, momentum))

    # set mode to trainindg
    nn.train()

    # set number of OpenMP threads
    logger.debug('Set OpenMP threads to {}'.format(threads))
    nn.set_num_threads(threads)

    logger.debug('Moving model to device {}'.format(device))
    rec = models.TorchSeqRecognizer(nn, train=True, device=device)
    optim = getattr(torch.optim, optimizer)(nn.nn.parameters(), lr=0)

    tr_it = TrainScheduler(optim)
    if schedule == '1cycle':
        add_1cycle(tr_it, epochs * len(gt_set), lrate, momentum, momentum - 0.10, weight_decay)
    else:
        # constant learning rate scheduler
        tr_it.add_phase(1, [lrate, lrate], [momentum, momentum], weight_decay, train.annealing_const)

    st_it = cast(TrainStopper, None)  # type: TrainStopper
    if quit == 'early':
        st_it = EarlyStopping(train_loader, min_delta, lag)
    elif quit == 'dumb':
        st_it = EpochStopping(train_loader, epochs)
    else:
        raise click.BadOptionUsage('quit', 'Invalid training interruption scheme {}'.format(quit))

    for epoch, loader in enumerate(st_it):
        if epoch and not epoch % savefreq:
            logger.info('Saving to {}_{}'.format(output, epoch))
            try:
                nn.save_model('{}_{}.mlmodel'.format(output, epoch))
            except Exception as e:
                logger.error('Saving model failed: {}'.format(str(e)))
        if not epoch % report:
            logger.debug('Starting evaluation run')
            nn.eval()
            chars, error = compute_error(rec, list(val_set))
            nn.train()
            accuracy = (chars-error)/chars
            logger.info('Accuracy report ({}) {:0.4f} {} {}'.format(epoch, accuracy, chars, error))
            message('Accuracy report ({}) {:0.4f} {} {}'.format(epoch, accuracy, chars, error))
            st_it.update(accuracy)
        with log.progressbar(label='epoch {}/{}'.format(epoch, epochs), length=len(loader), show_pos=True) as bar:
            for trial, (input, target) in enumerate(loader):
                tr_it.step()
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                input = input.requires_grad_()
                o = nn.nn(input)
                # height should be 1 by now
                if o.size(2) != 1:
                    raise KrakenInputException('Expected dimension 3 to be 1, actual {}'.format(output.size()))
                o = o.squeeze(2)
                optim.zero_grad()
                # NCW -> WNC
                loss = nn.criterion(o.permute(2, 0, 1),  # type: ignore
                                    target,
                                    (o.size(2),),
                                    (target.size(1),))
                logger.info('trial {} - loss {}'.format(trial, float(loss)))
                loss.backward()
                optim.step()
                bar.update(1)
    if quit == 'early':
        message('Moving best model {0}_{1}.mlmdel ({2}) to {0}_best.mlmodel'.format(output, st_it.best_epoch, st_it.best_loss))
        logger.info('Moving best model {0}_{1}.mlmdel ({2}) to {0}_best.mlmodel'.format(output, st_it.best_epoch, st_it.best_loss))
        shutil.copy('{}_{}.mlmodel'.format(output, st_it.best_epoch), '{}_best.mlmodel'.format(output))


@cli.command('test')
@click.pass_context
@click.option('-m', '--model', show_default=True, type=click.Path(exists=True, readable=True),
              multiple=True, help='Model(s) to evaluate')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('-p', '--pad', show_default=True, type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('--threads', show_default=True, default=1, help='Number of OpenMP threads when running on CPU.')
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def test(ctx, model, evaluation_files, device, pad, threads, test_set):
    """
    Evaluate on a test set.
    """
    import numpy as np
    from PIL import Image

    from kraken.lib import models
    from kraken.lib.dataset import _fast_levenshtein, generate_input_transforms

    logger.info('Building test set from {} line images'.format(len(test_set) + len(evaluation_files)))

    if not model:
        raise click.UsageError('No model to evaluate given.')

    nn = {}
    for p in model:
        message('Loading model {}\t'.format(p), nl=False)
        nn[p] = models.load_any(p)
        message('\u2713', fg='green')

    test_set = list(test_set)

    # set number of OpenMP threads
    logger.debug('Set OpenMP threads to {}'.format(threads))
    next(iter(nn.values())).nn.set_num_threads(threads)

    # merge training_files into ground_truth list
    if evaluation_files:
        test_set.extend(evaluation_files)

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    def _get_text(im):
        with open(os.path.splitext(im)[0] + '.gt.txt', 'r') as fp:
            return fp.read()

    acc_list = []
    for p, net in nn.items():
        chars = 0
        error = 0
        message('Evaluating {}'.format(p))
        logger.info('Evaluating {}'.format(p))
        batch, channels, height, width = net.nn.input
        ts = generate_input_transforms(batch, height, width, channels, pad)
        with log.progressbar(test_set, label='Evaluating') as bar:
            for im_path in bar:
                i = ts(Image.open(im_path))
                text = _get_text(im_path)
                pred = net.predict_string(i)
                chars += len(text)
                error += _fast_levenshtein(pred, text)
        acc_list.append((chars-error)/chars)
        logger.info('Accuracy report {:0.4f} {} {}'.format(acc_list[-1], chars, error))
        message('Accuracy report {:0.4f} {} {}'.format(acc_list[-1], chars, error))
    logger.info('Average accuracy: {:0.4f}, (stddev: {:0.4f})'.format(np.mean(acc_list), np.std(acc_list)))
    message('Average accuracy: {:0.4f}, (stddev: {:0.4f})'.format(np.mean(acc_list), np.std(acc_list)))


@cli.command('extract')
@click.pass_context
@click.option('-b', '--binarize/--no-binarize', show_default=True, default=True,
              help='Binarize color/grayscale images')
@click.option('-u', '--normalization', show_default=True,
              type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']), default=None,
              help='Normalize ground truth')
@click.option('-s', '--normalize-whitespace/--no-normalize-whitespace',
              show_default=True, default=True, help='Normalizes unicode whitespace')
@click.option('-n', '--reorder/--no-reorder', default=False, show_default=True,
              help='Reorder transcribed lines to display order')
@click.option('-r', '--rotate/--no-rotate', default=True, show_default=True,
              help='Skip rotation of vertical lines')
@click.option('-o', '--output', type=click.Path(), default='training', show_default=True,
              help='Output directory')
@click.option('--format', default='{idx:06d}', show_default=True, help='Format for extractor output. valid fields are `src` (source file), `idx` (line number), and `uuid` (v4 uuid)')
@click.argument('transcriptions', nargs=-1, type=click.File(lazy=True))
def extract(ctx, binarize, normalization, normalize_whitespace, reorder,
            rotate, output, format, transcriptions):
    """
    Extracts image-text pairs from a transcription environment created using
    ``ketos transcribe``.
    """
    import regex
    import base64

    from io import BytesIO
    from PIL import Image
    from lxml import html, etree

    from kraken import binarization

    try:
        os.mkdir(output)
    except Exception:
        pass

    text_transforms = []
    if normalization:
        text_transforms.append(lambda x: unicodedata.normalize(normalization, x))
    if normalize_whitespace:
        text_transforms.append(lambda x: regex.sub('\s', ' ', x))
    if reorder:
        text_transforms.append(get_display)

    idx = 0
    manifest = []
    with log.progressbar(transcriptions, label='Reading transcriptions') as bar:
        for fp in bar:
            logger.info('Reading {}'.format(fp.name))
            doc = html.parse(fp)
            etree.strip_tags(doc, etree.Comment)
            td = doc.find(".//meta[@itemprop='text_direction']")
            if td is None:
                td = 'horizontal-lr'
            else:
                td = td.attrib['content']

            im = None
            dest_dict = {'output': output, 'idx': 0, 'src': fp.name, 'uuid': str(uuid.uuid4())}
            for section in doc.xpath('//section'):
                img = section.xpath('.//img')[0].get('src')
                fd = BytesIO(base64.b64decode(img.split(',')[1]))
                im = Image.open(fd)
                if not im:
                    logger.info('Skipping {} because image not found'.format(fp.name))
                    break
                if binarize:
                    im = binarization.nlbin(im)
                for line in section.iter('li'):
                    if line.get('contenteditable') and (not u''.join(line.itertext()).isspace() and u''.join(line.itertext())):
                        dest_dict['idx'] = idx
                        dest_dict['uuid'] = str(uuid.uuid4())
                        logger.debug('Writing line {:06d}'.format(idx))
                        l_img = im.crop([int(x) for x in line.get('data-bbox').split(',')])
                        if rotate and td.startswith('vertical'):
                            im.rotate(90, expand=True)
                        l_img.save(('{output}/' + format + '.png').format(**dest_dict))
                        manifest.append((format + '.png').format(**dest_dict))
                        text = u''.join(line.itertext()).strip()
                        for func in text_transforms:
                            text = func(text)
                        with open(('{output}/' + format + '.gt.txt').format(**dest_dict), 'wb') as t:
                            t.write(text.encode('utf-8'))
                        idx += 1
    logger.info('Extracted {} lines'.format(idx))
    with open('{}/manifest.txt'.format(output), 'w') as fp:
        fp.write('\n'.join(manifest))


@cli.command('transcribe')
@click.pass_context
@click.option('-d', '--text-direction', default='horizontal-lr',
              type=click.Choice(['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction')
@click.option('--scale', default=None, type=click.FLOAT)
@click.option('--bw/--orig', default=True,
              help="Put nonbinarized images in output")
@click.option('-m', '--maxcolseps', default=2, type=click.INT)
@click.option('-b/-w', '--black_colseps/--white_colseps', default=False)
@click.option('-f', '--font', default='',
              help='Font family to use')
@click.option('-fs', '--font-style', default=None,
              help='Font style to use')
@click.option('-p', '--prefill', default=None,
              help='Use given model for prefill mode.')
@click.option('-p', '--pad', show_default=True, type=(int, int), default=(0, 0),
              help='Left and right padding around lines')
@click.option('-o', '--output', type=click.File(mode='wb'), default='transcription.html',
              help='Output file')
@click.argument('images', nargs=-1, type=click.File(mode='rb', lazy=True))
def transcription(ctx, text_direction, scale, bw, maxcolseps,
                  black_colseps, font, font_style, prefill, pad, output,
                  images):
    from PIL import Image

    from kraken import rpred
    from kraken import pageseg
    from kraken import transcribe
    from kraken import binarization

    from kraken.lib import models
    from kraken.lib.util import is_bitonal

    ti = transcribe.TranscriptionInterface(font, font_style)

    if prefill:
        logger.info('Loading model {}'.format(prefill))
        message('Loading RNN', nl=False)
        prefill = models.load_any(prefill)
        message('\u2713', fg='green')

    with log.progressbar(images, label='Reading images') as bar:
        for fp in bar:
            logger.info('Reading {}'.format(fp.name))
            im = Image.open(fp)
            im_bin = im
            if not is_bitonal(im):
                logger.info('Binarizing page')
                im_bin = binarization.nlbin(im)
            logger.info('Segmenting page')
            if bw:
                im = im_bin
            res = pageseg.segment(im_bin, text_direction, scale, maxcolseps, black_colseps, pad=pad)
            if prefill:
                it = rpred.rpred(prefill, im_bin, res)
                preds = []
                logger.info('Recognizing')
                for pred in it:
                    logger.debug('{}'.format(pred.prediction))
                    preds.append(pred)
                ti.add_page(im, res, records=preds)
            else:
                ti.add_page(im, res)
            fp.close()
    logger.info('Writing transcription to {}'.format(output.name))
    message('Writing output', nl=False)
    ti.write(output)
    message('\u2713', fg='green')


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
    import errno
    import numpy as np

    from kraken import linegen

    lines: Set[str] = set()
    if not text:
        return
    with log.progressbar(text, label='Reading texts') as bar:
        for t in text:
            with click.open_file(t, encoding=encoding) as fp:
                logger.info('Reading {}'.format(t))
                for l in fp:
                    lines.add(l.rstrip('\r\n'))
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
        if unicodedata.combining(char):
            combining.append(unicodedata.name(char))
        else:
            chars.append(char)
    message('Σ (len: {})'.format(len(alphabet)))
    message('Symbols: {}'.format(''.join(chars)))
    if combining:
        message('Combining Characters: {}'.format(', '.join(combining)))
    lg = linegen.LineGenerator(font, font_size, font_weight, language)
    with log.progressbar(lines, label='Writing images') as bar:
        for idx, line in enumerate(bar):
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


if __name__ == '__main__':
    cli()
