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

from threadpoolctl import threadpool_limits

from kraken.registry import OPTIMIZERS, SCHEDULERS, STOPPERS

from .util import _expand_gt, _validate_manifests, message

logging.captureWarnings(True)
logger = logging.getLogger('kraken')


@click.command('train')
@click.pass_context
@click.option('-B', '--batch-size', type=int, help='batch sample size')
@click.option('--pad', 'padding', type=int, help='Left and right padding around lines')
@click.option('-o', '--output', 'checkpoint_path', default='model', help='Directory to save checkpoints into.')
@click.option('--weights-format', default='safetensors', help='Output weights format.')
@click.option('-s', '--spec', help='VGSL spec of the network to train. CTC layer will be added automatically.')
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
@click.option('--freeze-backbone', type=int, help='Number of samples to keep the backbone (everything but last layer) frozen.')
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
@click.option('-c', '--codec', type=click.Path(exists=True, readable=True),
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
@click.option('-t', '--training-files', 'training_data', multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', 'evaluation_data',
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page', 'binary']), default='path',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both line definitions and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with `.gt.txt` text files '
              'containing the transcription. In binary mode files are datasets '
              'files containing pre-extracted text lines.')
@click.option('--augment/--no-augment',
              help='Enable image augmentation')
@click.option('--logger', 'pl_logger', type=click.Choice(['tensorboard']),
              help='Logger used by PyTorch Lightning to track metrics such as loss and accuracy.')
@click.option('--log-dir', type=click.Path(exists=True, dir_okay=True, writable=True),
              help='Path to directory where the logger will store the logs. If not set, a directory will be created in the current working directory.')
@click.option('--legacy-polygons', is_flag=True, help='Use the legacy polygon extractor.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def train(ctx, **kwargs):
    """
    Trains a model from image-text pairs.
    """
    params = ctx.meta.copy()
    params.update(ctx.params)

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

    import json

    from lightning.pytorch.callbacks import ModelCheckpoint

    from kraken.lib import vgsl  # NOQA
    from kraken.train import (KrakenTrainer, CRNNRecognitionModel,
                              CRNNRecognitionDataModule)
    from kraken.configs import VGSLRecognitionTrainingConfig, VGSLRecognitionTrainingDataConfig

    if (codec := params.get('codec')) is not None and not isinstance(codec, dict):
        with open(codec, 'rb') as fp:
            params['codec'] = json.load(fp)

    # disable automatic partition when given evaluation set explicitly
    if params['evaluation_data']:
        params['partition'] = 1

    # merge training_files into ground_truth list
    if training_data:
        ground_truth.extend(training_data)

    params['training_data'] = ground_truth

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    if params['bidi_reordering'] and params['base_dir'] != 'auto':
        params['bidi_reordering'] = params['base_dir']

    if params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(params['freq'])}
    else:
        val_check_interval = {'val_check_interval': params['freq']}

    cbs = []
    checkpoint_callback = ModelCheckpoint(dirpath=params.pop('checkpoint_path'),
                                          save_top_k=10,
                                          monitor='val_metric',
                                          mode='max',
                                          auto_insert_metric_name=False,
                                          filename='checkpoint_{epoch:02d}-{val_metric:.4f}')
    cbs.append(checkpoint_callback)

    dm_config = VGSLRecognitionTrainingDataConfig(**params)
    m_config = VGSLRecognitionTrainingConfig(**params)

    if resume:
        data_module = CRNNRecognitionDataModule.load_from_checkpoint(resume)
    else:
        data_module = CRNNRecognitionDataModule(dm_config)

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
                model = CRNNRecognitionModel.load_from_checkpoint(load, config=m_config)
            else:
                model = CRNNRecognitionModel.load_from_weights(load, config=m_config)
        elif resume:
            message(f'Resuming from checkpoint {resume}.')
            model = CRNNRecognitionModel.load_from_checkpoint(resume)
        else:
            message('Initializing new model.')
            model = CRNNRecognitionModel(m_config)

    try:
        with threadpool_limits(limits=ctx.meta['num_threads']):
            if resume:
                trainer.fit(model, data_module, ckpt_path=resume)
            else:
                trainer.fit(model, data_module)
    except ValueError as e:
        if e.args[0].startswith('Training data and model codec alphabets mismatch') and params['resize'] == 'fail':
            raise click.BadOptionUsage('resize', 'Mismatched training data for loaded model. Set option `--resize` to `new` or `add`')
        else:
            raise e

    score = checkpoint_callback.best_model_score.item()
    weight_path = Path(checkpoint_callback.best_model_path).with_name(f'best_{score}.{kwargs.pop("weights_format")}')
    message(f'Converting best model {checkpoint_callback.best_model_path} (score: {score}) to weights {weight_path}')

@click.command('test')
@click.pass_context
@click.option('-B', '--batch-size', type=int, help='Batch sample size')
@click.option('-m', '--model', type=click.Path(exists=True, readable=True), help='Model to evaluate')
@click.option('-e', '--test-files', 'test_data', multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page', 'binary']), default='path',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both line definitions and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with `.gt.txt` text files '
              'containing the transcription. In binary mode files are datasets '
              'files containing pre-extracted text lines.')
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
    params = ctx.meta.copy()
    params.update(ctx.params)

    model = params.pop('model')
    if not model:
        raise click.UsageError('No model to evaluate given.')

    test_data = params.pop('test_data', [])
    test_set = list(params.pop('test_set', []))

    # merge training_files into ground_truth list
    if test_data:
        test_set.extend(test_data)

    params['test_data'] = test_set

    if params['bidi_reordering'] and params['base_dir'] != 'auto':
        params['bidi_reordering'] = params['base_dir']

    from kraken.lib import vgsl  # NOQA
    from kraken.train import (KrakenTrainer, CRNNRecognitionModel,
                              CRNNRecognitionDataModule)
    from kraken.configs import VGSLRecognitionTrainingDataConfig, VGSLRecognitionTrainingConfig
    from kraken.serialization import render_report

    trainer = KrakenTrainer(accelerator=ctx.meta['accelerator'],
                            devices=ctx.meta['device'],
                            precision=ctx.meta['precision'],
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            deterministic=ctx.meta['deterministic'],
                            enable_model_summary=False,
                            num_sanity_val_steps=0)

    m_config = VGSLRecognitionTrainingConfig(**params)
    dm_config = VGSLRecognitionTrainingDataConfig(**params)
    data_module = CRNNRecognitionDataModule(dm_config)

    with trainer.init_module(empty_init=False):
        message(f'Loading from {model}.')
        if model.endswith('ckpt'):
            model = CRNNRecognitionModel.load_from_checkpoint(model, config=m_config)
        else:
            model = CRNNRecognitionModel.load_from_weights(model, m_config)

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
