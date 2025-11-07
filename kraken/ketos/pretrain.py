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

from kraken.registry import OPTIMIZERS, SCHEDULERS, STOPPERS

from .util import _expand_gt, _validate_manifests, message, to_ptl_device

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


@click.command('pretrain')
@click.pass_context
@click.option('-B', '--batch-size', type=int, help='batch sample size')
@click.option('--pad', 'padding', type=int, help='Left and right padding around lines')
@click.option('-o', '--output', 'checkpoint_path', type=click.Path(), help='Output checkpoint path')
@click.option('-s', '--spec', help='VGSL spec of the network to train.')
@click.option('-i', '--load', type=click.Path(exists=True, readable=True),
              help='Load existing file to continue training')
@click.option('-F',
              '--freq',
              type=float,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q',
              '--quit',
              type=click.Choice(STOPPERS),
              help='Stop condition for training. Set to `early` for early stooping or `fixed` for fixed number of epochs')
@click.option('-N',
              '--epochs',
              type=int,
              help='Number of epochs to train for')
@click.option('--min-epochs',
              type=int,
              help='Minimal number of epochs to train for when using early stopping.')
@click.option('--lag',
              type=int,
              help='Number of evaluations (--report frequency) to wait before stopping training without improvement')
@click.option('--min-delta',
              type=float,
              help='Minimum improvement between epochs to reset early stopping. Default is scales the delta by the best loss')
@click.option('--optimizer',
              type=click.Choice(OPTIMIZERS),
              help='Select optimizer')
@click.option('-r',
              '--lrate',
              type=float,
              help='Learning rate')
@click.option('-m',
              '--momentum',
              type=float,
              help='Momentum')
@click.option('-w',
              '--weight-decay',
              type=float, help='Weight decay')
@click.option('--warmup',
              type=float,
              help='Number of samples to ramp up to `lrate` initial learning rate.')
@click.option('--schedule',
              type=click.Choice(SCHEDULERS),
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--epoch` option.')
@click.option('-g',
              '--gamma',
              type=float,
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              type=int,
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              'rop_patience',
              type=int,
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              'cos_max_t',
              type=int,
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('--cos-min-lr',
              type=float,
              help='Minimal final learning rate for cosine LR scheduler.')
@click.option('-p',
              '--partition',
              type=float,
              help='Ground truth data partition ratio between train/validation set')
@click.option('--fixed-splits/--ignore-fixed-splits', default=False,
              help='Whether to honor fixed splits in binary datasets.')
@click.option('-t', '--training-files', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--load-hyper-parameters/--no-load-hyper-parameters', default=False,
              help='When loading an existing model, retrieve hyperparameters from the model')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page', 'binary']),
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both line definitions and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with `.gt.txt` text files '
              'containing the transcription. In binary mode files are datasets '
              'files containing pre-extracted text lines.')
@click.option('--augment/--no-augment',
              help='Enable image augmentation')
@click.option('-mw',
              '--mask-width',
              type=int,
              help='Width of sampled masks at scale of the sampled tensor, e.g. '
                   '4X subsampling in convolutional layers with mask width 3 results '
                   'in an effective mask width of 12.')
@click.option('-mp',
              '--mask-probability',
              type=float,
              help='Probability of a particular position being the start position of a mask.')
@click.option('-nn',
              '--num-negatives',
              type=int,
              help='Number of negative samples for the contrastive loss.')
@click.option('-lt',
              '--logit-temp',
              type=float,
              help='Multiplicative factor for the logits used in contrastive loss.')
@click.option('--legacy-polygons', default=False, is_flag=True, help='Use the legacy polygon extractor.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def pretrain(ctx, **kwargs):
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
                                     num_workers=ctx.meta['num_workers'],
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
