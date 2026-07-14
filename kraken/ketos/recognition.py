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

from pathlib import Path
from threadpoolctl import threadpool_limits

from kraken.registry import OPTIMIZERS, SCHEDULERS, STOPPERS

from .util import (_arch_names, _expand_gt, _resolve_module_class,
                   _user_supplied_params, _validate_manifests, message)

logging.captureWarnings(True)
logger = logging.getLogger('kraken')


# option names on train/test that do not map to config-class fields
_NON_CONFIG_PARAMS = frozenset({'arch', 'load', 'resume', 'ground_truth',
                                'training_data', 'evaluation_data', 'test_data',
                                'test_set', 'model', 'base_dir', 'pl_logger',
                                'log_dir', 'no_legacy_polygons'})


def _check_arch_options(ctx: click.Context,
                        arch: str,
                        explicit: dict,
                        config_cls: type,
                        data_config_cls: type) -> None:
    """
    Rejects explicitly set options that are not fields of the selected
    architecture's training or data config.
    """
    from click.core import ParameterSource

    known = set(vars(config_cls())) | set(vars(data_config_cls())) | _NON_CONFIG_PARAMS
    for name in sorted(explicit.keys() - known):
        opt = next((param for param in ctx.command.params if param.name == name), None)
        flag = opt.opts[-1] if opt else name
        hint = ' (set in the --config file)' if ctx.get_parameter_source(name) is ParameterSource.DEFAULT_MAP else ''
        raise click.BadOptionUsage(name, f'Option {flag}{hint} is not supported by architecture {arch!r}.')


def _config_kwargs(ctx: click.Context, explicit: dict) -> dict:
    """
    Builds the config class kwargs from the run-level context values plus the
    explicitly set options, leaving everything else at the config defaults.
    """
    cfg_kwargs = {k: v for k, v in ctx.meta.items() if k != 'ketos_user_config'}
    cfg_kwargs.update({k: v for k, v in explicit.items() if k not in _NON_CONFIG_PARAMS})
    return cfg_kwargs


@click.command('train')
@click.pass_context
@click.option('--arch', type=click.Choice(_arch_names('recognition')), default='vgsl',
              help='Recognition architecture family to train. Auto-detected from '
                   '--load/--resume artifacts; only needed when training from scratch. '
                   'Defaults shown by --help for shared options are those of the vgsl '
                   'family; other families apply their own defaults to any option not '
                   'explicitly set.')
@click.option('-B', '--batch-size', type=int, help='batch sample size')
@click.option('--pad', 'padding', type=int, help='Left and right padding around lines')
@click.option('-o', '--output', 'checkpoint_path', default='model', help='Directory to save checkpoints into.')
@click.option('--weights-format', default='safetensors', help='Output weights format.')
@click.option('-s', '--spec', help='[vgsl] VGSL spec of the network to train. CTC layer will be added automatically.')
@click.option('--variant', type=click.Choice(['tiny', 'small', 'medium']),
              help='[ppocrv6] Model size. (ppocrv6 default: small)')
@click.option('--height', type=int,
              help='[ppocrv6] Input line height. (ppocrv6 default: 96)')
@click.option('--max-width', 'max_width', type=click.IntRange(min=1),
              help='[ppocrv6] Maximum line width in pixels after height-normalization. (ppocrv6 default: 2560)')
@click.option('-i', '--load', type=click.Path(exists=True, readable=True), help='Load existing file to continue training')
@click.option('--resume', type=click.Path(exists=True, readable=True), help='Load a checkpoint to continue training')
@click.option('-F', '--freq',
              type=click.FLOAT,
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
              help='Minimum improvement between epochs to reset early stopping.')
@click.option('--optimizer',
              type=click.Choice(OPTIMIZERS),
              help='Select optimizer')
@click.option('-r', '--lrate', type=float, help='Learning rate')
@click.option('-m', '--momentum', type=float, help='Momentum')
@click.option('-w', '--weight-decay', type=float, help='Weight decay')
@click.option('--gradient-clip-val', type=float, help='Gradient clip value')
@click.option('--accumulate-grad-batches', type=int, help='Number of batches to accumulate gradient across.')
@click.option('--warmup', type=int, help='Number of steps to ramp up to `lrate` initial learning rate.')
@click.option('--freeze-backbone', type=int, help='[vgsl] Number of samples to keep the backbone (everything but last layer) frozen.')
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
@click.option('--cos-max', 'cos_t_max', type=int, help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('--cos-min-lr',
              type=float,
              help='Minimal final learning rate for cosine LR scheduler.')
@click.option('-p', '--partition', type=float, help='Ground truth data partition ratio between train/validation set')
@click.option('-u', '--normalization', type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']), help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace', help='Normalizes unicode whitespace')
@click.option('-c', '--codec', type=click.UNPROCESSED,
              help='Load a codec JSON definition (invalid if loading existing model)')
@click.option('--resize',
              type=click.Choice([
                  'add', 'union',  # Deprecation: `add` is deprecated, `union` is the new value
                  'both', 'new',  # Deprecation: `both` is deprecated, `new` is the new value
                  'fail'
              ]),
              help='Codec/output layer resizing option. If set to `union` code '
                   'points will be added, `new` will set the layer to match exactly '
                   'the training data, `fail` will abort if training data and model '
                   'codec do not match.')
@click.option('--reorder/--no-reorder', 'bidi_reordering', help='Reordering of code points to display order')
@click.option('--base-dir',
              type=click.Choice(['L', 'R', 'auto']), default='auto', help='Set base text '
              'direction.  This should be set to the direction used during the '
              'creation of the training data. If set to `auto` it will be '
              'overridden by any explicit value given in the input files.')
@click.option('-t', '--training-data', 'training_data', multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-data', 'evaluation_data', multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page', 'binary']), default='path',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both line definitions and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with `.gt.txt` text files '
              'containing the transcription. In binary mode files are datasets '
              'files containing pre-extracted text lines.')
@click.option('--linetype', type=click.Choice(['baselines', 'bbox']),
              help='Forces the line type of the training data. If not set the '
              'type is determined automatically: baselines for XML data, bbox '
              'for path data, and the recorded type for binary datasets.')
@click.option('--augment/--no-augment',
              help='Enable image augmentation')
@click.option('--logger', 'pl_logger', type=click.Choice(['tensorboard', 'wandb']),
              help='Logger used by PyTorch Lightning to track metrics such as loss and accuracy.')
@click.option('--log-dir', type=click.Path(exists=True, dir_okay=True, writable=True),
              help='Path to directory where the logger will store the logs. If not set, a directory will be created in the current working directory.')
@click.option('--legacy-polygons', is_flag=True, help='Use the legacy polygon extractor.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def train(ctx, **kwargs):
    """
    Trains a model from image-text pairs.

    The architecture is selected with `--arch`. Options left unset take the
    selected architecture's defaults.
    """
    p = ctx.params
    explicit = _user_supplied_params(ctx)

    resume = p['resume']
    load = p['load']

    if sum(map(bool, [resume, load])) > 1:
        raise click.BadOptionUsage('load', 'load/resume options are mutually exclusive.')

    if p['pl_logger'] == 'tensorboard':
        try:
            import tensorboard  # NOQA
        except ImportError:
            raise click.BadOptionUsage('logger', 'tensorboard logger needs the `tensorboard` package installed.')

    import json

    from lightning.pytorch.callbacks import ModelCheckpoint

    from kraken.models.convert import convert_models
    from kraken.train import KrakenTrainer
    from kraken.train.utils import KrakenOnExceptionCheckpoint

    module_cls = _resolve_module_class(ctx, explicit, 'recognition', artifact=resume or load)
    arch = module_cls._arch
    config_cls = module_cls._config_class
    data_config_cls = module_cls._data_config_class
    dm_cls = module_cls._data_module_class

    _check_arch_options(ctx, arch, explicit, config_cls, data_config_cls)

    cfg_kwargs = _config_kwargs(ctx, explicit)

    if (codec := cfg_kwargs.get('codec')) is not None and not isinstance(codec, dict):
        with open(codec, 'rb') as fp:
            cfg_kwargs['codec'] = json.load(fp)

    # merge training_data manifests into ground_truth list
    ground_truth = list(p['ground_truth'])
    if p['training_data']:
        ground_truth.extend(p['training_data'])

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    cfg_kwargs['training_data'] = ground_truth

    # disable automatic partition when given evaluation set explicitly
    if p['evaluation_data']:
        cfg_kwargs['evaluation_data'] = p['evaluation_data']
        cfg_kwargs['partition'] = 1

    dm_config = data_config_cls(**cfg_kwargs)
    m_config = config_cls(**cfg_kwargs)

    if dm_config.bidi_reordering and p['base_dir'] != 'auto':
        dm_config.bidi_reordering = p['base_dir']

    if resume and (ignored := sorted(explicit.keys() - _NON_CONFIG_PARAMS)):
        logger.warning('Resuming from a checkpoint restores its full training state; '
                       f'explicitly set hyperparameters {ignored} are ignored.')

    if m_config.freq > 1:
        val_check_interval = {'check_val_every_n_epoch': int(m_config.freq)}
    else:
        val_check_interval = {'val_check_interval': m_config.freq}

    cbs = [KrakenOnExceptionCheckpoint(dirpath=p['checkpoint_path'],
                                       filename='checkpoint_abort')]
    checkpoint_callback = ModelCheckpoint(dirpath=Path(p['checkpoint_path']),
                                          save_top_k=10,
                                          monitor='val_metric',
                                          mode='max',
                                          auto_insert_metric_name=False,
                                          filename='checkpoint_{epoch:02d}-{val_metric:.4f}')
    cbs.append(checkpoint_callback)

    if resume:
        data_module = dm_cls.load_from_checkpoint(resume, weights_only=False)
    else:
        data_module = dm_cls(dm_config)

    trainer = KrakenTrainer(accelerator=ctx.meta['accelerator'],
                            devices=ctx.meta['device'],
                            precision=ctx.meta['precision'],
                            max_epochs=m_config.epochs if m_config.quit == 'fixed' else -1,
                            min_epochs=m_config.min_epochs,
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            deterministic=ctx.meta['deterministic'],
                            enable_model_summary=False,
                            accumulate_grad_batches=m_config.accumulate_grad_batches,
                            callbacks=cbs,
                            gradient_clip_val=m_config.gradient_clip_val,
                            num_sanity_val_steps=0,
                            **val_check_interval)

    with trainer.init_module(empty_init=False if (load or resume) else True):
        if load:
            message(f'Loading from checkpoint {load}.')
            if load.endswith('ckpt'):
                model = module_cls.load_from_checkpoint(load, config=m_config, weights_only=False)
            else:
                model = module_cls.load_from_weights(load, config=m_config)
        elif resume:
            message(f'Resuming from checkpoint {resume}.')
            model = module_cls.load_from_checkpoint(resume, weights_only=False)
        else:
            message('Initializing new model.')
            model = module_cls(m_config)

    try:
        with threadpool_limits(limits=ctx.meta['num_threads']):
            if resume:
                trainer.fit(model, data_module, ckpt_path=resume)
            else:
                trainer.fit(model, data_module)
    except ValueError as e:
        if e.args[0].startswith('Training data and model codec alphabets mismatch') and getattr(m_config, 'resize', 'fail') == 'fail':
            raise click.BadOptionUsage('resize', 'Mismatched training data for loaded model. Set option `--resize` to `new` or `add`')
        else:
            raise e

    score = checkpoint_callback.best_model_score.item()
    weight_path = Path(checkpoint_callback.best_model_path).with_name(f'best_{score:.4f}.{m_config.weights_format}')
    opath = convert_models([checkpoint_callback.best_model_path], weight_path, weights_format=m_config.weights_format)
    message(f'Converting best model {checkpoint_callback.best_model_path} (score: {score:.4f}) to weights file {opath}')


@click.command('test')
@click.pass_context
@click.option('--arch', type=click.Choice(_arch_names('recognition')), default='vgsl',
              help='Recognition architecture family of the tested model. '
                   'Auto-detected from the model file; only needed for artifacts '
                   'without architecture metadata.')
@click.option('-B', '--batch-size', type=int, help='Batch sample size')
@click.option('-m', '--model', type=click.Path(exists=True, readable=True), help='Model to evaluate')
@click.option('-e', '--test-data', 'test_data', multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page', 'binary']), default='path',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both line definitions and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with `.gt.txt` text files '
              'containing the transcription. In binary mode files are datasets '
              'files containing pre-extracted text lines.')
@click.option('--linetype', type=click.Choice(['baselines', 'bbox']),
              help='Forces the line type of the test data. If not set the type '
              'the model has been trained on is used for XML data, bbox for '
              'path data, and the recorded type for binary datasets.')
@click.option('--pad', 'padding', type=int, help='Left and right padding around lines')
@click.option('--reorder/--no-reorder', 'bidi_reordering', help='Reordering of code points to display order')
@click.option('--base-dir', type=click.Choice(['L', 'R', 'auto']), default='auto', help='Set base text '
              'direction.  This should be set to the direction used during the '
              'creation of the training data. If set to `auto` it will be '
              'overridden by any explicit value given in the input files.')
@click.option('-u', '--normalization', type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace',
              help='Normalizes unicode whitespace')
@click.option('--no-legacy-polygons', show_default=True, default=False, is_flag=True, help='Force disable the legacy polygon extractor.')
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def test(ctx, **kwargs):
    """
    Evaluate on a test set.
    """
    p = ctx.params
    explicit = _user_supplied_params(ctx)

    model = p['model']
    if not model:
        raise click.UsageError('No model to evaluate given.')

    from kraken.train import KrakenTrainer
    from kraken.serialization import render_report

    module_cls = _resolve_module_class(ctx, explicit, 'recognition', artifact=model)
    arch = module_cls._arch
    config_cls = module_cls._config_class
    data_config_cls = module_cls._data_config_class
    dm_cls = module_cls._data_module_class

    _check_arch_options(ctx, arch, explicit, config_cls, data_config_cls)

    cfg_kwargs = _config_kwargs(ctx, explicit)

    # merge test_data manifests into test_set list
    test_set = list(p['test_set'])
    if p['test_data']:
        test_set.extend(p['test_data'])
    cfg_kwargs['test_data'] = test_set

    trainer = KrakenTrainer(accelerator=ctx.meta['accelerator'],
                            devices=ctx.meta['device'],
                            precision=ctx.meta['precision'],
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            deterministic=ctx.meta['deterministic'],
                            enable_model_summary=False,
                            num_sanity_val_steps=0)

    m_config = config_cls(**cfg_kwargs)
    with trainer.init_module(empty_init=False):
        message(f'Loading from {model}.')
        if model.endswith('ckpt'):
            model = module_cls.load_from_checkpoint(model, config=m_config)
        else:
            model = module_cls.load_from_weights(model, m_config)

    dm_config = data_config_cls(**cfg_kwargs)

    if dm_config.bidi_reordering and p['base_dir'] != 'auto':
        dm_config.bidi_reordering = p['base_dir']

    dm_config.legacy_polygons = (not p['no_legacy_polygons']) and getattr(model.net, 'use_legacy_polygons', False)

    # evaluate XML data with the line type the model has been trained on
    # unless explicitly overridden
    if dm_config.linetype is None and dm_config.format_type in ('xml', 'alto', 'page'):
        dm_config.linetype = getattr(model.net, 'seg_type', None)

    data_module = dm_cls(dm_config)

    with threadpool_limits(limits=ctx.meta['num_threads']):
        test_metrics = trainer.test(model, data_module)

    rep = render_report(model=model,
                        chars=sum(test_metrics.character_counts.values()),
                        errors=test_metrics.num_errors,
                        char_accuracy=test_metrics.cer,
                        char_CI_accucary=test_metrics.case_insensitive_cer,  # Case insensitive
                        word_accuracy=test_metrics.wer,
                        char_confusions=test_metrics.confusions,
                        scripts=test_metrics.scripts,
                        insertions=test_metrics.insertions,
                        deletions=test_metrics.deletes,
                        substitutions=test_metrics.substitutions)

    logger.info(rep)
    message(rep)
