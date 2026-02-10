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
import importlib
from pathlib import Path

import click
from PIL import Image

from kraken.registry import OPTIMIZERS, SCHEDULERS, STOPPERS

from .util import _expand_gt, _validate_manifests, message

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


@click.command('pretrain')
@click.pass_context
@click.option('-B',
              '--batch-size',
              type=int,
              help='batch sample size')
@click.option('--pad',
              'padding',
              type=int,
              help='Left and right padding around lines')
@click.option('-o',
              '--output',
              'checkpoint_path',
              type=click.Path(), help='Output checkpoint path')
@click.option('-s',
              '--spec',
              help='VGSL spec of the network to train.')
@click.option('-i',
              '--load',
              type=click.Path(exists=True, readable=True),
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
@click.option('--gradient-clip-val',
              type=float,
              help='Gradient clip value')
@click.option('--accumulate-grad-batches',
              type=int,
              help='Number of batches to accumulate gradient across.')
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
@click.option('--legacy-polygons', is_flag=True, help='Use the legacy polygon extractor.')
@click.option('-ss',
              '--step-size',
              type=int,
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              'rop_patience',
              type=int,
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              'cos_t_max',
              type=int,
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('--cos-min-lr',
              type=float,
              help='Minimal final learning rate for cosine LR scheduler.')
@click.option('-p',
              '--partition',
              type=float,
              help='Ground truth data partition ratio between train/validation set')
@click.option('-t', '--training-data', 'training_data', multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-data', 'evaluation_data', multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
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
              'mask_prob',
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
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def pretrain(ctx, **kwargs):
    """
    Trains a model from image-text pairs.
    """
    params = ctx.params.copy()
    params.update(ctx.meta)
    resume = params.pop('resume', None)
    load = params.pop('load', None)
    training_data = params.pop('training_data', [])
    ground_truth = list(params.pop('ground_truth', []))

    if sum(map(bool, [resume, load])) > 1:
        raise click.BadOptionsUsage('load', 'load/resume options are mutually exclusive.')

    if params.get('augment'):
        try:
            import albumentations  # NOQA
        except ImportError:
            raise click.BadOptionUsage('augment', 'augmentation needs the `albumentations` package installed.')

    if params.get('pl_logger') == 'tensorboard':
        try:
            import tensorboard  # NOQA
        except ImportError:
            raise click.BadOptionUsage('logger', 'tensorboard logger needs the `tensorboard` package installed.')

    from threadpoolctl import threadpool_limits
    from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint

    from kraken.lib.pretrain import (PretrainDataModule,
                                     RecognitionPretrainModel)
    from kraken.train import KrakenTrainer

    from kraken.configs import VGSLPreTrainingConfig, VGSLPreTrainingDataConfig

    # disable automatic partition when given evaluation set explicitly
    if params['evaluation_data']:
        params['partition'] = 1

    # merge training_data into ground_truth list
    if training_data:
        ground_truth.extend(training_data)

    params['training_data'] = ground_truth

    if len(params['training_data']) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    if params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(params['freq'])}
    else:
        val_check_interval = {'val_check_interval': params['freq']}

    cbs = [OnExceptionCheckpoint(dirpath=params.get('checkpoint_path'),
                                 filename='checkpoint_abort')]
    checkpoint_callback = ModelCheckpoint(dirpath=Path(params.pop('checkpoint_path')),
                                          save_top_k=10,
                                          monitor='CE',
                                          mode='min',
                                          auto_insert_metric_name=False,
                                          filename='checkpoint_{epoch:02d}-{val_metric:.4f}')
    cbs.append(checkpoint_callback)

    dm_config = VGSLPreTrainingDataConfig(**params)
    m_config = VGSLPreTrainingConfig(**params)

    if resume:
        data_module = PretrainDataModule.load_from_checkpoint(resume)
    else:
        data_module = PretrainDataModule(dm_config)

    trainer = KrakenTrainer(accelerator=ctx.meta['accelerator'],
                            devices=ctx.meta['device'],
                            precision=ctx.meta['precision'],
                            max_epochs=params['epochs'] if params['quit'] == 'fixed' else -1,
                            min_epochs=params['min_epochs'],
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            deterministic=ctx.meta['deterministic'],
                            enable_model_summary=False,
                            accumulate_grad_batches=params['accumulate_grad_batches'],
                            callbacks=cbs,
                            gradient_clip_val=params['gradient_clip_val'],
                            num_sanity_val_steps=0,
                            use_distributed_sampler=False,
                            **val_check_interval)

    with trainer.init_module(empty_init=False if (load or resume) else True):
        if load:
            message(f'Loading from checkpoint {load}.')
            if load.endswith('ckpt'):
                model = RecognitionPretrainModel.load_from_checkpoint(load, config=m_config, weights_only=True)
            else:
                model = RecognitionPretrainModel.load_from_weights(load, m_config)
        elif resume:
            message(f'Resuming from checkpoint {resume}.')
            model = RecognitionPretrainModel.load_from_checkpoint(resume, weights_only=True)
        else:
            message('Initializing new model.')
            model = RecognitionPretrainModel(m_config)

    with threadpool_limits(limits=ctx.meta['num_threads']):
        if resume:
            trainer.fit(model, data_module, ckpt_path=resume)
        else:
            trainer.fit(model, data_module)

    score = checkpoint_callback.best_model_score.item()
    message(f'Best model checkpoint: {checkpoint_callback.best_model_path}')

    try:
        (entry_point,) = importlib.metadata.entry_points(group='kraken.writers', name='coreml')
        writer = entry_point.load()
    except ValueError:
        raise click.UsageError('weights_format', 'Unknown format `coreml` for weights.')

    weight_path = Path(checkpoint_callback.best_model_path).with_name(f'best_{score:.4f}.coreml')
    model = RecognitionPretrainModel.load_from_checkpoint(checkpoint_callback.best_model_path,
                                                          config=m_config,
                                                          weights_only=True)
    opath = writer([model.net], weight_path)
    message(f'Converting best model {checkpoint_callback.best_model_path} (score: {score:.4f}) to weights file {opath}')
