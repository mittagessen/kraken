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
kraken.ketos.pretrain
~~~~~~~~~~~~~~~~~~~~~

Command line driver for unsupervised recognition pretraining
"""
import logging

import click
from PIL import Image

from kraken.lib.register import OPTIMIZERS, SCHEDULERS, STOPPERS
from kraken.lib.default_specs import (RECOGNITION_PRETRAIN_HYPER_PARAMS,
                                      RECOGNITION_SPEC)

from .util import _expand_gt, _validate_manifests, message, to_ptl_device

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


@click.command('pretrain')
@click.pass_context
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['batch_size'], help='batch sample size')
@click.option('--pad', show_default=True, type=click.INT,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['pad'],
              help='Left and right padding around lines')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model', help='Output model file')
@click.option('-s', '--spec', show_default=True, default=RECOGNITION_SPEC,
              help='VGSL spec of the network to train.')
@click.option('-i', '--load', show_default=True, type=click.Path(exists=True,
              readable=True), help='Load existing file to continue training')
@click.option('-F', '--freq', show_default=True, default=RECOGNITION_PRETRAIN_HYPER_PARAMS['freq'], type=click.FLOAT,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q',
              '--quit',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['quit'],
              type=click.Choice(STOPPERS),
              help='Stop condition for training. Set to `early` for early stooping or `fixed` for fixed number of epochs')
@click.option('-N',
              '--epochs',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['epochs'],
              help='Number of epochs to train for')
@click.option('--min-epochs',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['min_epochs'],
              help='Minimal number of epochs to train for when using early stopping.')
@click.option('--lag',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['lag'],
              help='Number of evaluations (--report frequency) to wait before stopping training without improvement')
@click.option('--min-delta',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['min_delta'],
              type=click.FLOAT,
              help='Minimum improvement between epochs to reset early stopping. Default is scales the delta by the best loss')
@click.option('--optimizer',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['optimizer'],
              type=click.Choice(OPTIMIZERS),
              help='Select optimizer')
@click.option('-r', '--lrate', show_default=True, default=RECOGNITION_PRETRAIN_HYPER_PARAMS['lrate'], help='Learning rate')
@click.option('-m', '--momentum', show_default=True, default=RECOGNITION_PRETRAIN_HYPER_PARAMS['momentum'], help='Momentum')
@click.option('-w', '--weight-decay', show_default=True, type=float,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['weight_decay'], help='Weight decay')
@click.option('--warmup', show_default=True, type=float,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['warmup'], help='Number of samples to ramp up to `lrate` initial learning rate.')
@click.option('--schedule',
              show_default=True,
              type=click.Choice(SCHEDULERS),
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['schedule'],
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--epoch` option.')
@click.option('-g',
              '--gamma',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['gamma'],
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['step_size'],
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['rop_patience'],
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['cos_t_max'],
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('--cos-min-lr',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['cos_min_lr'],
              help='Minimal final learning rate for cosine LR scheduler.')
@click.option('-p', '--partition', show_default=True, default=0.9,
              help='Ground truth data partition ratio between train/validation set')
@click.option('--fixed-splits/--ignore-fixed-splits', show_default=True, default=False,
              help='Whether to honor fixed splits in binary datasets.')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--load-hyper-parameters/--no-load-hyper-parameters', show_default=True, default=False,
              help='When loading an existing model, retrieve hyperparameters from the model')
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
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['augment'],
              help='Enable image augmentation')
@click.option('-mw', '--mask-width', show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['mask_width'],
              help='Width of sampled masks at scale of the sampled tensor, e.g. '
                   '4X subsampling in convolutional layers with mask width 3 results '
                   'in an effective mask width of 12.')
@click.option('-mp', '--mask-probability',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['mask_prob'],
              help='Probability of a particular position being the start position of a mask.')
@click.option('-nn', '--num-negatives',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['num_negatives'],
              help='Number of negative samples for the contrastive loss.')
@click.option('-lt', '--logit-temp',
              show_default=True,
              default=RECOGNITION_PRETRAIN_HYPER_PARAMS['logit_temp'],
              help='Multiplicative factor for the logits used in contrastive loss.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
@click.option('--legacy-polygons', show_default=True, default=False, is_flag=True, help='Use the legacy polygon extractor.')
def pretrain(ctx, batch_size, pad, output, spec, load, freq, quit, epochs,
             min_epochs, lag, min_delta, optimizer, lrate, momentum,
             weight_decay, warmup, schedule, gamma, step_size, sched_patience,
             cos_max, cos_min_lr, partition, fixed_splits, training_files,
             evaluation_files, load_hyper_parameters,
             force_binarization, format_type, augment, mask_probability,
             mask_width, num_negatives, logit_temp, ground_truth,
             legacy_polygons):
    """
    Trains a model from image-text pairs.
    """
    if not (0 <= freq <= 1) and freq % 1.0 != 0:
        raise click.BadOptionUsage('freq', 'freq needs to be either in the interval [0,1.0] or a positive integer.')

    if augment:
        try:
            import albumentations  # NOQA
        except ImportError:
            raise click.BadOptionUsage('augment', 'augmentation needs the `albumentations` package installed.')

    import shutil

    from threadpoolctl import threadpool_limits

    from kraken.lib.pretrain import (PretrainDataModule,
                                     RecognitionPretrainModel)
    from kraken.lib.train import KrakenTrainer

    hyper_params = RECOGNITION_PRETRAIN_HYPER_PARAMS.copy()
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
                         'schedule': schedule,
                         'gamma': gamma,
                         'step_size': step_size,
                         'rop_patience': sched_patience,
                         'cos_t_max': cos_max,
                         'cos_min_lr': cos_min_lr,
                         'augment': augment,
                         'mask_prob': mask_probability,
                         'mask_width': mask_width,
                         'num_negatives': num_negatives,
                         'logit_temp': logit_temp})

    # disable automatic partition when given evaluation set explicitly
    if evaluation_files:
        partition = 1
    ground_truth = list(ground_truth)

    # merge training_files into ground_truth list
    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    try:
        accelerator, device = to_ptl_device(ctx.meta['device'])
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    if hyper_params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(hyper_params['freq'])}
    else:
        val_check_interval = {'val_check_interval': hyper_params['freq']}

    model = RecognitionPretrainModel(hyper_params=hyper_params,
                                     output=output,
                                     spec=spec,
                                     model=load,
                                     load_hyper_parameters=load_hyper_parameters,
                                     legacy_polygons=legacy_polygons)

    data_module = PretrainDataModule(batch_size=hyper_params.pop('batch_size'),
                                     pad=hyper_params.pop('pad'),
                                     augment=hyper_params.pop('augment'),
                                     training_data=ground_truth,
                                     evaluation_data=evaluation_files,
                                     partition=partition,
                                     binary_dataset_split=fixed_splits,
                                     num_workers=ctx.meta['workers'],
                                     height=model.height,
                                     width=model.width,
                                     channels=model.channels,
                                     force_binarization=force_binarization,
                                     format_type=format_type,
                                     legacy_polygons=legacy_polygons,)

    model.len_train_set = len(data_module.train_dataloader())

    trainer = KrakenTrainer(accelerator=accelerator,
                            devices=device,
                            precision=ctx.meta['precision'],
                            max_epochs=hyper_params['epochs'] if hyper_params['quit'] == 'fixed' else -1,
                            min_epochs=hyper_params['min_epochs'],
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            deterministic=ctx.meta['deterministic'],
                            **val_check_interval)

    with threadpool_limits(limits=ctx.meta['threads']):
        trainer.fit(model, datamodule=data_module)

    if model.best_epoch == -1:
        logger.warning('Model did not improve during training.')
        ctx.exit(1)

    if not model.current_epoch:
        logger.warning('Training aborted before end of first epoch.')
        ctx.exit(1)

    if quit == 'early':
        message(f'Moving best model {model.best_model} ({model.best_metric}) to {output}_best.mlmodel')
        logger.info(f'Moving best model {model.best_model} ({model.best_metric}) to {output}_best.mlmodel')
        shutil.copy(f'{model.best_model}', f'{output}_best.mlmodel')
