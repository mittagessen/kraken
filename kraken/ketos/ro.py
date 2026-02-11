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
import click
import logging

from PIL import Image

from pathlib import Path
from kraken.ketos.util import _expand_gt, _validate_manifests, message, _create_class_map

from kraken.registry import OPTIMIZERS, SCHEDULERS, STOPPERS

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


@click.command('rotrain')
@click.pass_context
@click.option('-B', '--batch-size', type=int, help='batch sample size')
@click.option('--weights-format', default='safetensors', help='Output weights format.')
@click.option('-o', '--output', 'checkpoint_path', type=click.Path(), help='Output model file')
@click.option('-i', '--load', type=click.Path(exists=True, readable=True), help='Load existing checkpoint or weights file to train from.')
@click.option('--resume', type=click.Path(exists=True, readable=True), help='Load a checkpoint to continue training')
@click.option('-F',
              '--freq',
              type=float,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q',
              '--quit',
              type=click.Choice(STOPPERS),
              help='Stop condition for training. Set to `early` for early stopping or `fixed` for fixed number of epochs')
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
              help='Minimum improvement between epochs to reset early stopping. By default it scales the delta by the best loss')
@click.option('--optimizer',
              type=click.Choice(OPTIMIZERS),
              help='Select optimizer')
@click.option('-r', '--lrate', type=float, help='Learning rate')
@click.option('-m', '--momentum', type=float, help='Momentum')
@click.option('-w', '--weight-decay', type=float, help='Weight decay')
@click.option('--gradient-clip-val',
              type=float,
              help='Gradient clip value')
@click.option('--accumulate-grad-batches',
              type=int,
              help='Number of batches to accumulate gradient across.')
@click.option('--warmup', type=int, help='Number of samples to ramp up to `lrate` initial learning rate.')
@click.option('--schedule',
              type=click.Choice(SCHEDULERS),
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--step-size` option.')
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
              'cos_t_max',
              type=int,
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('--cos-min-lr',
              type=float,
              help='Minimal final learning rate for cosine LR scheduler.')
@click.option('-p', '--partition', type=float, help='Ground truth data partition ratio between train/validation set')
@click.option('-t', '--training-data', 'training_data', multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-data', 'evaluation_data', multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('-f', '--format-type', type=click.Choice(['xml', 'alto', 'page']),
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images.')
@click.option('--logger', 'pl_logger', type=click.Choice(['tensorboard']),
              help='Logger used by PyTorch Lightning to track metrics such as loss and accuracy.')
@click.option('--log-dir', type=click.Path(exists=True, dir_okay=True, writable=True),
              help='Path to directory where the logger will store the logs. If not set, a directory will be created in the current working directory.')
@click.option('--level', type=click.Choice(['baselines', 'regions']),
              help='Selects level to train reading order model on.')
@click.option('--reading-order', help='Select reading order to train. Defaults to `line_implicit`/`region_implicit`')
@click.option('--class-mapping', type=click.UNPROCESSED, hidden=True)
@click.option('--class-mapping-from-ckpt',
              type=click.Path(exists=True, readable=True),
              help='Extract class mapping from a segmentation checkpoint (.ckpt). '
                   'Uses --level to select baseline or region mapping.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def rotrain(ctx, **kwargs):
    """
    Trains a baseline labeling model for layout analysis
    """
    params = ctx.params
    resume = params.pop('resume', None)
    load = params.pop('load', None)
    training_data = params.pop('training_data', [])
    ground_truth = list(params.pop('ground_truth', []))

    class_mapping_from_ckpt = params.pop('class_mapping_from_ckpt', None)

    # parse class_mapping from YAML list-of-tuples
    if isinstance(params.get('class_mapping'), list):
        params['class_mapping'] = _create_class_map(params['class_mapping'])
    elif params.get('class_mapping') is None:
        params.pop('class_mapping', None)

    if class_mapping_from_ckpt:
        if 'class_mapping' in params:
            raise click.UsageError('--class-mapping and --class-mapping-from-ckpt are mutually exclusive.')
        from kraken.models.convert import load_from_checkpoint
        try:
            module = load_from_checkpoint(class_mapping_from_ckpt)
            data_config = module.hparams['data_config']
            if params.get('level', 'baselines') == 'baselines':
                cm = data_config.line_class_mapping
            else:
                cm = data_config.region_class_mapping
            params['class_mapping'] = dict(cm)
        except Exception as e:
            raise click.UsageError(
                f'Failed to extract class mapping from {class_mapping_from_ckpt}: {e}. '
                'This usually happens when the checkpoint\'s serialized config references '
                'training data paths that have moved or been deleted.'
            )

    if sum(map(bool, [resume, load])) > 1:
        raise click.BadOptionsUsage('load', 'load/resume options are mutually exclusive.')

    if params.get('pl_logger') == 'tensorboard':
        try:
            import tensorboard  # NOQA
        except ImportError:
            raise click.BadOptionUsage('logger', 'tensorboard logger needs the `tensorboard` package installed.')

    from threadpoolctl import threadpool_limits

    from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint

    from kraken.lib import vgsl  # NOQA
    from kraken.lib.ro import ROModel, RODataModule
    from kraken.train import KrakenTrainer
    from kraken.configs import ROTrainingConfig, ROTrainingDataConfig
    from kraken.models.convert import convert_models

    # disable automatic partition when given evaluation set explicitly
    if params['evaluation_data']:
        params['partition'] = 1

    # merge training_data into ground_truth list
    if training_data:
        ground_truth.extend(training_data)

    params['training_data'] = ground_truth

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    if params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(params['freq'])}
    else:
        val_check_interval = {'val_check_interval': params['freq']}

    cbs = [OnExceptionCheckpoint(dirpath=params.get('checkpoint_path'),
                                 filename='checkpoint_abort')]
    checkpoint_callback = ModelCheckpoint(dirpath=params.pop('checkpoint_path'),
                                          save_top_k=10,
                                          monitor='val_metric',
                                          mode='max',
                                          auto_insert_metric_name=False,
                                          filename='checkpoint_{epoch:02d}-{val_metric:.4f}')
    cbs.append(checkpoint_callback)

    dm_config = ROTrainingDataConfig(**params, **ctx.meta)
    m_config = ROTrainingConfig(**params)

    if resume:
        data_module = RODataModule.load_from_checkpoint(resume)
    else:
        data_module = RODataModule(dm_config)

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
                            **val_check_interval)

    with trainer.init_module(empty_init=False if (load or resume) else True):
        if load:
            message(f'Loading from checkpoint {load}.')
            if load.endswith('ckpt'):
                model = ROModel.load_from_checkpoint(load, config=m_config)
            else:
                model = ROModel.load_from_weights(load, m_config)
        elif resume:
            message(f'Resuming from checkpoint {resume}.')
            model = ROModel.load_from_checkpoint(resume)
        else:
            message('Initializing new model.')
            model = ROModel(m_config)

    with threadpool_limits(limits=ctx.meta['num_threads']):
        if resume:
            trainer.fit(model, data_module, ckpt_path=resume)
        else:
            trainer.fit(model, data_module)

    score = checkpoint_callback.best_model_score.item()
    weight_path = Path(checkpoint_callback.best_model_path).with_name(f'best_{score:.4f}.{params.get("weights_format")}')
    opath = convert_models([checkpoint_callback.best_model_path], weight_path, weights_format=params['weights_format'])
    message(f'Converting best model {checkpoint_callback.best_model_path} (score: {score:.4f}) to weights {opath}')


@click.command('roadd')
@click.pass_context
@click.option('-o', '--output', type=click.Path(), default='combined_seg.mlmodel', help='Combined output model file')
@click.option('-r', '--ro-model', type=click.Path(exists=True, readable=True), help='Reading order model to load into segmentation model')
@click.option('-i', '--seg-model', type=click.Path(exists=True, readable=True), help='Segmentation model to load')
def roadd(ctx, output, ro_model, seg_model):
    """
    Combines a reading order model with a segmentation model.
    """
    from kraken.lib import vgsl
    from kraken.lib.ro import ROModel
    from kraken.models import load_models

    message(f'Adding {ro_model} reading order model to {seg_model}.')

    # Load ROMLP — try as weights file first, fall back to checkpoint
    try:
        models = load_models(ro_model, tasks=['reading_order'])
        if len(models) != 1:
            raise ValueError(f'Found {len(models)} reading order models in {ro_model}.')
        ro_mlp = models[0]
    except ValueError:
        ro_module = ROModel.load_from_checkpoint(ro_model)
        ro_mlp = ro_module.net

    ro_class_mapping = ro_mlp.user_metadata.get('class_mapping', {})
    level = ro_mlp.user_metadata.get('level', 'baselines')

    message(f'RO model level: {level}')
    message('Classes known to RO model:')
    for k, v in ro_class_mapping.items():
        message(f'  {k}\t{v}')

    seg_net = vgsl.TorchVGSLModel.load_model(seg_model)
    if 'segmentation' not in seg_net.model_type:
        raise click.UsageError(f'Model {seg_model} is invalid {seg_net.model_type} model (expected `segmentation`).')

    seg_section = 'baselines' if level == 'baselines' else 'regions'
    aux_key = 'ro_model' if level == 'baselines' else 'ro_model_regions'

    message(f'{seg_section.capitalize()} classes known to segmentation model:')
    for k, v in seg_net.user_metadata['class_mapping'][seg_section].items():
        message(f'  {k}\t{v}')
    diff = set(ro_class_mapping.keys()).symmetric_difference(set(seg_net.user_metadata['class_mapping'][seg_section].keys()))
    diff.discard('default')
    if len(diff):
        raise click.UsageError(f'Model {seg_model} and {ro_model} class mappings mismatch.')

    if not hasattr(seg_net, 'aux_layers') or seg_net.aux_layers is None:
        seg_net.aux_layers = {}
    seg_net.aux_layers[aux_key] = ro_mlp
    message(f'Saving combined model to {output}')
    seg_net.save_model(output)
