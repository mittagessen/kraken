#
# Copyright 2022 Benjamin Kiessling
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
kraken.ketos.train
~~~~~~~~~~~~~~~~~~

Command line driver for recognition training and evaluation.
"""
import click
import logging
import pathlib

from typing import List

from kraken.lib.progress import KrakenProgressBar
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.default_specs import RECOGNITION_HYPER_PARAMS, RECOGNITION_SPEC
from .util import _validate_manifests, _expand_gt, message, to_ptl_device

logging.captureWarnings(True)
logger = logging.getLogger('kraken')


@click.command('train')
@click.pass_context
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=RECOGNITION_HYPER_PARAMS['batch_size'], help='batch sample size')
@click.option('--pad', show_default=True, type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model', help='Output model file')
@click.option('-s', '--spec', show_default=True, default=RECOGNITION_SPEC,
              help='VGSL spec of the network to train. CTC layer will be added automatically.')
@click.option('-a', '--append', show_default=True, default=None, type=click.INT,
              help='Removes layers before argument and then appends spec. Only works when loading an existing model')
@click.option('-i', '--load', show_default=True, type=click.Path(exists=True,
              readable=True), help='Load existing file to continue training')
@click.option('-F', '--freq', show_default=True, default=RECOGNITION_HYPER_PARAMS['freq'], type=click.FLOAT,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q',
              '--quit',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['quit'],
              type=click.Choice(['early',
                                 'dumb']),
              help='Stop condition for training. Set to `early` for early stooping or `dumb` for fixed number of epochs')
@click.option('-N',
              '--epochs',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['epochs'],
              help='Number of epochs to train for')
@click.option('--min-epochs',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['min_epochs'],
              help='Minimal number of epochs to train for when using early stopping.')
@click.option('--lag',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['lag'],
              help='Number of evaluations (--report frequence) to wait before stopping training without improvement')
@click.option('--min-delta',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['min_delta'],
              type=click.FLOAT,
              help='Minimum improvement between epochs to reset early stopping. Default is scales the delta by the best loss')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--optimizer',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['optimizer'],
              type=click.Choice(['Adam',
                                 'SGD',
                                 'RMSprop']),
              help='Select optimizer')
@click.option('-r', '--lrate', show_default=True, default=RECOGNITION_HYPER_PARAMS['lrate'], help='Learning rate')
@click.option('-m', '--momentum', show_default=True, default=RECOGNITION_HYPER_PARAMS['momentum'], help='Momentum')
@click.option('-w', '--weight-decay', show_default=True, type=float,
              default=RECOGNITION_HYPER_PARAMS['weight_decay'], help='Weight decay')
@click.option('--warmup', show_default=True, type=int,
              default=RECOGNITION_HYPER_PARAMS['warmup'], help='Number of samples to ramp up to `lrate` initial learning rate.')
@click.option('--freeze-backbone', show_default=True, type=int,
              default=RECOGNITION_HYPER_PARAMS['freeze_backbone'], help='Number of samples to keep the backbone (everything but last layer) frozen.')
@click.option('--schedule',
              show_default=True,
              type=click.Choice(['constant',
                                 '1cycle',
                                 'exponential',
                                 'cosine',
                                 'step',
                                 'reduceonplateau']),
              default=RECOGNITION_HYPER_PARAMS['schedule'],
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--epoch` option.')
@click.option('-g',
              '--gamma',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['gamma'],
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['step_size'],
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['rop_patience'],
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['cos_t_max'],
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('-p', '--partition', show_default=True, default=0.9,
              help='Ground truth data partition ratio between train/validation set')
@click.option('--fixed-splits/--ignore-fixed-split', show_default=True, default=False,
              help='Whether to honor fixed splits in binary datasets.')
@click.option('-u', '--normalization', show_default=True, type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              default=RECOGNITION_HYPER_PARAMS['normalization'], help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace', show_default=True,
              default=RECOGNITION_HYPER_PARAMS['normalize_whitespace'], help='Normalizes unicode whitespace')
@click.option('-c', '--codec', show_default=True, default=None, type=click.File(mode='r', lazy=True),
              help='Load a codec JSON definition (invalid if loading existing model)')
@click.option('--resize', show_default=True, default='fail', type=click.Choice(['add', 'both', 'fail']),
              help='Codec/output layer resizing option. If set to `add` code '
                   'points will be added, `both` will set the layer to match exactly '
                   'the training data, `fail` will abort if training data and model '
                   'codec do not match.')
@click.option('--reorder/--no-reorder', show_default=True, default=True, help='Reordering of code points to display order')
@click.option('--base-dir', show_default=True, default='auto',
              type=click.Choice(['L', 'R', 'auto']), help='Set base text '
              'direction.  This should be set to the direction used during the '
              'creation of the training data. If set to `auto` it will be '
              'overridden by any explicit value given in the input files.')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--workers', show_default=True, default=1, help='Number of OpenMP threads and workers when running on CPU.')
@click.option('--load-hyper-parameters/--no-load-hyper-parameters', show_default=True, default=False,
              help='When loading an existing model, retrieve hyperparameters from the model')
@click.option('--repolygonize/--no-repolygonize', show_default=True,
              default=False, help='Repolygonizes line data in ALTO/PageXML '
              'files. This ensures that the trained model is compatible with the '
              'segmenter in kraken even if the original image files either do '
              'not contain anything but transcriptions and baseline information '
              'or the polygon data was created using a different method. Will '
              'be ignored in `path` mode. Note that this option will be slow '
              'and will not scale input images to the same size as the segmenter '
              'does.')
@click.option('--force-binarization/--no-binarization', show_default=True,
              default=False, help='Forces input images to be binary, otherwise '
              'the appropriate color format will be auto-determined through the '
              'network specification. Will be ignored in `path` mode.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page', 'binary']), default='path',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both line definitions and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with `.gt.txt` text files '
              'containing the transcription. In binary mode files are datasets '
              'files containing pre-extracted text lines.')
@click.option('--augment/--no-augment',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['augment'],
              help='Enable image augmentation')
@click.option('--failed-sample-threshold', show_default=True, default=10,
              help='Abort if more than `n` samples fail to load.')
@click.option('--logger', 'pl_logger', show_default=True, type=click.Choice([None, 'tensorboard']), default=None,
              help='Logger used by PyTorch Lightning to track metrics such as loss and accuracy.')
@click.option('--log-dir', show_default=True, type=click.Path(exists=True, dir_okay=True, writable=True),
              help='Path to directory where the logger will store the logs. If not set, a directory will be created in the current working directory.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def train(ctx, batch_size, pad, output, spec, append, load, freq, quit, epochs,
          min_epochs, lag, min_delta, device, optimizer, lrate, momentum,
          weight_decay, warmup, freeze_backbone, schedule, gamma, step_size,
          sched_patience, cos_max, partition, fixed_splits, normalization,
          normalize_whitespace, codec, resize, reorder, base_dir,
          training_files, evaluation_files, workers, load_hyper_parameters,
          repolygonize, force_binarization, format_type, augment,
          failed_sample_threshold, pl_logger, log_dir, ground_truth):
    """
    Trains a model from image-text pairs.
    """
    if not load and append:
        raise click.BadOptionUsage('append', 'append option requires loading an existing model')

    if resize != 'fail' and not load:
        raise click.BadOptionUsage('resize', 'resize option requires loading an existing model')

    if not (0 <= freq <= 1) and freq % 1.0 != 0:
        raise click.BadOptionUsage('freq', 'freq needs to be either in the interval [0,1.0] or a positive integer.')

    if augment:
        try:
            import albumentations # NOQA
        except ImportError:
            raise click.BadOptionUsage('augment', 'augmentation needs the `albumentations` package installed.')

    if pl_logger == 'tensorboard':
        try:
            import tensorboard
        except ImportError:
            raise click.BadOptionUsage('logger', 'tensorboard logger needs the `tensorboard` package installed.')

    if log_dir is None:
        log_dir = pathlib.Path.cwd()

    import json
    import shutil
    from kraken.lib.train import RecognitionModel, KrakenTrainer

    hyper_params = RECOGNITION_HYPER_PARAMS.copy()
    hyper_params.update({'freq': freq,
                         'pad': pad,
                         'batch_size': batch_size,
                         'quit': quit,
                         'epochs': epochs,
                         'min_epochs': min_epochs,
                         'lag': lag,
                         'min_delta': min_delta,
                         'optimizer': optimizer,
                         'lrate': lrate,
                         'momentum': momentum,
                         'weight_decay': weight_decay,
                         'warmup': warmup,
                         'freeze_backbone': freeze_backbone,
                         'schedule': schedule,
                         'gamma': gamma,
                         'step_size': step_size,
                         'rop_patience': sched_patience,
                         'cos_t_max': cos_max,
                         'normalization': normalization,
                         'normalize_whitespace': normalize_whitespace,
                         'augment': augment,
                         'pl_logger': pl_logger,})

    # disable automatic partition when given evaluation set explicitly
    if evaluation_files:
        partition = 1
    ground_truth = list(ground_truth)

    # merge training_files into ground_truth list
    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    if reorder and base_dir != 'auto':
        reorder = base_dir

    if codec:
        logger.debug(f'Loading codec file from {codec}')
        codec = json.load(codec)

    try:
        accelerator, device = to_ptl_device(device)
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    if hyper_params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(hyper_params['freq'])}
    else:
        val_check_interval = {'val_check_interval': hyper_params['freq']}

    model = RecognitionModel(hyper_params=hyper_params,
                             output=output,
                             spec=spec,
                             append=append,
                             model=load,
                             reorder=reorder,
                             training_data=ground_truth,
                             evaluation_data=evaluation_files,
                             partition=partition,
                             binary_dataset_split=fixed_splits,
                             num_workers=workers,
                             load_hyper_parameters=load_hyper_parameters,
                             repolygonize=repolygonize,
                             force_binarization=force_binarization,
                             format_type=format_type,
                             codec=codec,
                             resize=resize)

    trainer = KrakenTrainer(accelerator=accelerator,
                            devices=device,
                            max_epochs=hyper_params['epochs'] if hyper_params['quit'] == 'dumb' else -1,
                            min_epochs=hyper_params['min_epochs'],
                            freeze_backbone=hyper_params['freeze_backbone'],
                            failed_sample_threshold=failed_sample_threshold,
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            deterministic=ctx.meta['deterministic'],
                            pl_logger=pl_logger,
                            log_dir=log_dir,
                            **val_check_interval)
    try:
        trainer.fit(model)
    except KrakenInputException as e:
        if e.args[0].startswith('Training data and model codec alphabets mismatch') and resize == 'fail':
            raise click.BadOptionUsage('resize', 'Mismatched training data for loaded model. Set option `--resize` to `add` or `both`')
        else:
            raise e

    if quit == 'early':
        message('Moving best model {0}_{1}.mlmodel ({2}) to {0}_best.mlmodel'.format(
            output, model.best_epoch, model.best_metric))
        logger.info('Moving best model {0}_{1}.mlmodel ({2}) to {0}_best.mlmodel'.format(
            output, model.best_epoch, model.best_metric))
        shutil.copy(f'{output}_{model.best_epoch}.mlmodel', f'{output}_best.mlmodel')


@click.command('test')
@click.pass_context
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=RECOGNITION_HYPER_PARAMS['batch_size'], help='Batch sample size')
@click.option('-m', '--model', show_default=True, type=click.Path(exists=True, readable=True),
              multiple=True, help='Model(s) to evaluate')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--pad', show_default=True, type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('--workers', show_default=True, default=1, help='Number of OpenMP threads when running on CPU.')
@click.option('--reorder/--no-reorder', show_default=True, default=True, help='Reordering of code points to display order')
@click.option('--base-dir', show_default=True, default='auto',
              type=click.Choice(['L', 'R', 'auto']), help='Set base text '
              'direction.  This should be set to the direction used during the '
              'creation of the training data. If set to `auto` it will be '
              'overridden by any explicit value given in the input files.')
@click.option('-u', '--normalization', show_default=True, type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              default=None, help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace',
              show_default=True, default=True, help='Normalizes unicode whitespace')
@click.option('--repolygonize/--no-repolygonize', show_default=True,
              default=False, help='Repolygonizes line data in ALTO/PageXML '
              'files. This ensures that the trained model is compatible with the '
              'segmenter in kraken even if the original image files either do '
              'not contain anything but transcriptions and baseline information '
              'or the polygon data was created using a different method. Will '
              'be ignored in `path` mode. Note, that this option will be slow '
              'and will not scale input images to the same size as the segmenter '
              'does.')
@click.option('--force-binarization/--no-binarization', show_default=True,
              default=False, help='Forces input images to be binary, otherwise '
              'the appropriate color format will be auto-determined through the '
              'network specification. Will be ignored in `path` mode.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page', 'binary']), default='path',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information. In `binary` mode files are '
              'collections of pre-extracted text line images.')
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def test(ctx, batch_size, model, evaluation_files, device, pad, workers,
         reorder, base_dir, normalization, normalize_whitespace, repolygonize,
         force_binarization, format_type, test_set):
    """
    Evaluate on a test set.
    """
    if not model:
        raise click.UsageError('No model to evaluate given.')

    import numpy as np
    from torch.utils.data import DataLoader

    from kraken.serialization import render_report
    from kraken.lib import models
    from kraken.lib.xml import preparse_xml_data
    from kraken.lib.dataset import (global_align, compute_confusions,
                                    PolygonGTDataset, GroundTruthDataset,
                                    ImageInputTransforms,
                                    ArrowIPCRecognitionDataset,
                                    collate_sequences)

    logger.info('Building test set from {} line images'.format(len(test_set) + len(evaluation_files)))

    nn = {}
    for p in model:
        message('Loading model {}\t'.format(p), nl=False)
        nn[p] = models.load_any(p)
        message('\u2713', fg='green')

    test_set = list(test_set)

    # set number of OpenMP threads
    next(iter(nn.values())).nn.set_num_threads(1)

    if evaluation_files:
        test_set.extend(evaluation_files)

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    if format_type in ['xml', 'page', 'alto']:
        if repolygonize:
            message('Repolygonizing data')
        test_set = preparse_xml_data(test_set, format_type, repolygonize)
        valid_norm = False
        DatasetClass = PolygonGTDataset
    elif format_type == 'binary':
        DatasetClass = ArrowIPCRecognitionDataset
        if repolygonize:
            logger.warning('Repolygonization enabled in `binary` mode. Will be ignored.')
        test_set = [{'file': file} for file in test_set]
        valid_norm = False
    else:
        DatasetClass = GroundTruthDataset
        if force_binarization:
            logger.warning('Forced binarization enabled in `path` mode. Will be ignored.')
            force_binarization = False
        if repolygonize:
            logger.warning('Repolygonization enabled in `path` mode. Will be ignored.')
        test_set = [{'image': img} for img in test_set]
        valid_norm = True

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    if reorder and base_dir != 'auto':
        reorder = base_dir

    acc_list = []
    for p, net in nn.items():
        algn_gt: List[str] = []
        algn_pred: List[str] = []
        chars = 0
        error = 0
        message('Evaluating {}'.format(p))
        logger.info('Evaluating {}'.format(p))
        batch, channels, height, width = net.nn.input
        ts = ImageInputTransforms(batch, height, width, channels, (pad, 0), valid_norm, force_binarization)
        ds = DatasetClass(normalization=normalization,
                          whitespace_normalization=normalize_whitespace,
                          reorder=reorder,
                          im_transforms=ts)
        for line in test_set:
            try:
                ds.add(**line)
            except KrakenInputException as e:
                logger.info(e)
        # don't encode validation set as the alphabets may not match causing encoding failures
        ds.no_encode()
        ds_loader = DataLoader(ds,
                               batch_size=batch_size,
                               num_workers=workers,
                               pin_memory=True,
                               collate_fn=collate_sequences)

        with KrakenProgressBar() as progress:
            batches = len(ds_loader)
            pred_task = progress.add_task('Evaluating', total=batches, visible=True if not ctx.meta['verbose'] else False)

            for batch in ds_loader:
                im = batch['image']
                text = batch['target']
                lens = batch['seq_lens']
                try:
                    pred = net.predict_string(im, lens)
                    for x, y in zip(pred, text):
                        chars += len(y)
                        c, algn1, algn2 = global_align(y, x)
                        algn_gt.extend(algn1)
                        algn_pred.extend(algn2)
                        error += c
                except FileNotFoundError as e:
                    batches -= 1
                    progress.update(pred_task, total=batches)
                    logger.warning('{} {}. Skipping.'.format(e.strerror, e.filename))
                except KrakenInputException as e:
                    batches -= 1
                    progress.update(pred_task, total=batches)
                    logger.warning(str(e))
                progress.update(pred_task, advance=1)

        acc_list.append((chars - error) / chars)
        confusions, scripts, ins, dels, subs = compute_confusions(algn_gt, algn_pred)
        rep = render_report(p, chars, error, confusions, scripts, ins, dels, subs)
        logger.info(rep)
        message(rep)
    logger.info('Average accuracy: {:0.2f}%, (stddev: {:0.2f})'.format(np.mean(acc_list) * 100, np.std(acc_list) * 100))
    message('Average accuracy: {:0.2f}%, (stddev: {:0.2f})'.format(np.mean(acc_list) * 100, np.std(acc_list) * 100))
