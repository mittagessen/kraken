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
kraken.ketos.ro
~~~~~~~~~~~~~~~

Command line driver for reading order training, evaluation, and handling.
"""
import logging
import pathlib

import click
from PIL import Image

from kraken.ketos.util import (_expand_gt, _validate_manifests, message,
                               to_ptl_device)
from kraken.lib.default_specs import READING_ORDER_HYPER_PARAMS

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


@click.command('rotrain')
@click.pass_context
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=READING_ORDER_HYPER_PARAMS['batch_size'], help='batch sample size')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model', help='Output model file')
@click.option('-i', '--load', show_default=True, type=click.Path(exists=True,
              readable=True), help='Load existing file to continue training')
@click.option('-F', '--freq', show_default=True, default=READING_ORDER_HYPER_PARAMS['freq'], type=click.FLOAT,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q',
              '--quit',
              show_default=True,
              default=READING_ORDER_HYPER_PARAMS['quit'],
              type=click.Choice(['early',
                                 'fixed']),
              help='Stop condition for training. Set to `early` for early stopping or `fixed` for fixed number of epochs')
@click.option('-N',
              '--epochs',
              show_default=True,
              default=READING_ORDER_HYPER_PARAMS['epochs'],
              help='Number of epochs to train for')
@click.option('--min-epochs',
              show_default=True,
              default=['min_epochs'],
              help='Minimal number of epochs to train for when using early stopping.')
@click.option('--lag',
              show_default=True,
              default=READING_ORDER_HYPER_PARAMS['lag'],
              help='Number of evaluations (--report frequency) to wait before stopping training without improvement')
@click.option('--min-delta',
              show_default=True,
              default=READING_ORDER_HYPER_PARAMS['min_delta'],
              type=click.FLOAT,
              help='Minimum improvement between epochs to reset early stopping. By default it scales the delta by the best loss')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--precision', default='32', type=click.Choice(['32', '16']), help='set tensor precision')
@click.option('--optimizer',
              show_default=True,
              default=READING_ORDER_HYPER_PARAMS['optimizer'],
              type=click.Choice(['Adam',
                                 'SGD',
                                 'RMSprop',
                                 'Lamb']),
              help='Select optimizer')
@click.option('-r', '--lrate', show_default=True, default=READING_ORDER_HYPER_PARAMS['lrate'], help='Learning rate')
@click.option('-m', '--momentum', show_default=True, default=READING_ORDER_HYPER_PARAMS['momentum'], help='Momentum')
@click.option('-w', '--weight-decay', show_default=True,
              default=READING_ORDER_HYPER_PARAMS['weight_decay'], help='Weight decay')
@click.option('--warmup', show_default=True, type=float,
              default=READING_ORDER_HYPER_PARAMS['warmup'], help='Number of samples to ramp up to `lrate` initial learning rate.')
@click.option('--schedule',
              show_default=True,
              type=click.Choice(['constant',
                                 '1cycle',
                                 'exponential',
                                 'cosine',
                                 'step',
                                 'reduceonplateau']),
              default=READING_ORDER_HYPER_PARAMS['schedule'],
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--step-size` option.')
@click.option('-g',
              '--gamma',
              show_default=True,
              default=READING_ORDER_HYPER_PARAMS['gamma'],
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              show_default=True,
              default=READING_ORDER_HYPER_PARAMS['step_size'],
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              show_default=True,
              default=READING_ORDER_HYPER_PARAMS['rop_patience'],
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              show_default=True,
              default=READING_ORDER_HYPER_PARAMS['cos_t_max'],
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('--cos-min-lr',
              show_default=True,
              default=READING_ORDER_HYPER_PARAMS['cos_min_lr'],
              help='Minimal final learning rate for cosine LR scheduler.')
@click.option('-p', '--partition', show_default=True, default=0.9,
              help='Ground truth data partition ratio between train/validation set')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--workers', show_default=True, default=1, type=click.IntRange(0), help='Number of worker proesses.')
@click.option('--threads', show_default=True, default=1, type=click.IntRange(1), help='Maximum size of OpenMP/BLAS thread pool.')
@click.option('--load-hyper-parameters/--no-load-hyper-parameters', show_default=True, default=False,
              help='When loading an existing model, retrieve hyper-parameters from the model')
@click.option('-f', '--format-type', type=click.Choice(['xml', 'alto', 'page']), default='xml',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images.')
@click.option('--logger', 'pl_logger', show_default=True, type=click.Choice(['tensorboard']), default=None,
              help='Logger used by PyTorch Lightning to track metrics such as loss and accuracy.')
@click.option('--log-dir', show_default=True, type=click.Path(exists=True, dir_okay=True, writable=True),
              help='Path to directory where the logger will store the logs. If not set, a directory will be created in the current working directory.')
@click.option('--level', show_default=True, type=click.Choice(['baselines', 'regions']), default='baselines',
              help='Selects level to train reading order model on.')
@click.option('--reading-order', show_default=True, default=None,
              help='Select reading order to train. Defaults to `line_implicit`/`region_implicit`')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def rotrain(ctx, batch_size, output, load, freq, quit, epochs, min_epochs, lag,
            min_delta, device, precision, optimizer, lrate, momentum,
            weight_decay, warmup, schedule, gamma, step_size, sched_patience,
            cos_max, cos_min_lr, partition, training_files, evaluation_files,
            workers, threads, load_hyper_parameters, format_type, pl_logger,
            log_dir, level, reading_order, ground_truth):
    """
    Trains a baseline labeling model for layout analysis
    """
    import shutil

    from threadpoolctl import threadpool_limits

    from kraken.lib.ro import ROModel
    from kraken.lib.train import KrakenTrainer

    if not (0 <= freq <= 1) and freq % 1.0 != 0:
        raise click.BadOptionUsage('freq', 'freq needs to be either in the interval [0,1.0] or a positive integer.')

    if pl_logger == 'tensorboard':
        try:
            import tensorboard  # NOQA
        except ImportError:
            raise click.BadOptionUsage('logger', 'tensorboard logger needs the `tensorboard` package installed.')

    if log_dir is None:
        log_dir = pathlib.Path.cwd()

    logger.info('Building ground truth set from {} document images'.format(len(ground_truth) + len(training_files)))

    # populate hyperparameters from command line args
    hyper_params = READING_ORDER_HYPER_PARAMS.copy()
    hyper_params.update({'batch_size': batch_size,
                         'freq': freq,
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
                         'pl_logger': pl_logger,
                         }
                        )

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
        accelerator, device = to_ptl_device(device)
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    if hyper_params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(hyper_params['freq'])}
    else:
        val_check_interval = {'val_check_interval': hyper_params['freq']}

    if load:
        model = ROModel.load_from_checkpoint(load,
                                             training_data=ground_truth,
                                             evaluation_data=evaluation_files,
                                             partition=partition,
                                             num_workers=workers,
                                             load_hyper_parameters=load_hyper_parameters,
                                             format_type=format_type)
    else:
        model = ROModel(hyper_params,
                        output=output,
                        training_data=ground_truth,
                        evaluation_data=evaluation_files,
                        partition=partition,
                        num_workers=workers,
                        load_hyper_parameters=load_hyper_parameters,
                        format_type=format_type,
                        level=level,
                        reading_order=reading_order)

    message(f'Training RO on following {level} types:')
    for k, v in model.train_set.dataset.class_mapping.items():
        message(f'  {k}\t{v}')

    if len(model.train_set) == 0:
        raise click.UsageError('No valid training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    trainer = KrakenTrainer(accelerator=accelerator,
                            devices=device,
                            max_epochs=hyper_params['epochs'] if hyper_params['quit'] == 'fixed' else -1,
                            min_epochs=hyper_params['min_epochs'],
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            deterministic=ctx.meta['deterministic'],
                            precision=int(precision),
                            pl_logger=pl_logger,
                            log_dir=log_dir,
                            **val_check_interval)

    with threadpool_limits(limits=threads):
        trainer.fit(model)

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


@click.command('roadd')
@click.pass_context
@click.option('-o', '--output', show_default=True, type=click.Path(), default='combined_seg.mlmodel', help='Combined output model file')
@click.option('-r', '--ro-model', show_default=True, type=click.Path(exists=True, readable=True), help='Reading order model to load into segmentation model')
@click.option('-i', '--seg-model', show_default=True, type=click.Path(exists=True, readable=True), help='Segmentation model to load')
def roadd(ctx, output, ro_model, seg_model):
    """
    Combines a reading order model with a segmentation model.
    """
    from kraken.lib import vgsl
    from kraken.lib.ro import ROModel

    message(f'Adding {ro_model} reading order model to {seg_model}.')
    ro_net = ROModel.load_from_checkpoint(ro_model)
    message('Line classes known to RO model:')
    for k, v in ro_net.class_mapping.items():
        message(f'  {k}\t{v}')
    seg_net = vgsl.TorchVGSLModel.load_model(seg_model)
    if seg_net.model_type != 'segmentation':
        raise click.UsageError(f'Model {seg_model} is invalid {seg_net.model_type} model (expected `segmentation`).')
    message('Line classes known to segmentation model:')
    for k, v in seg_net.user_metadata['class_mapping']['baselines'].items():
        message(f'  {k}\t{v}')
    if ro_net.class_mapping.keys() != seg_net.user_metadata['class_mapping']['baselines'].keys():
        raise click.UsageError(f'Model {seg_model} and {ro_model} class mappings mismatch.')

    seg_net.aux_layers = {'ro_model': ro_net.ro_net}
    seg_net.user_metadata['ro_class_mapping'] = ro_net.class_mapping
    message(f'Saving combined model to {output}')
    seg_net.save_model(output)
