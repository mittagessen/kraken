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
kraken.ketos.segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~

Command line driver for segmentation training and evaluation.
"""
import click
import logging

from PIL import Image
from pathlib import Path
from collections import Counter

from kraken.ketos.util import _expand_gt, _validate_manifests, message, _create_class_map

from kraken.registry import OPTIMIZERS, SCHEDULERS, STOPPERS

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


def _collect_observed_test_classes(test_data):
    """
    Collects raw class names seen in test data before class-map filtering.
    """
    from kraken.lib.dataset.utils import _get_type

    observed = {'baselines': Counter(), 'regions': Counter()}
    for sample in test_data or []:
        doc = sample.get('doc') if isinstance(sample, dict) else sample
        if doc is None:
            continue
        for line in getattr(doc, 'lines', []):
            observed['baselines'][_get_type(line.tags)] += 1
        for reg_type, regs in getattr(doc, 'regions', {}).items():
            observed['regions'][reg_type] += len(regs)
    return observed


def _build_segtest_class_diagnostics(model_mapping, dataset_mapping, dataset_stats, observed):
    """
    Builds rows for a unified class diagnostic table.
    """
    rows = []
    mismatches = []
    for section in ('baselines', 'regions'):
        model_cls = model_mapping.get(section, {})
        dataset_cls = dataset_mapping.get(section, {})
        section_stats = dataset_stats.get(section, {})
        observed_stats = observed.get(section, {})
        names = sorted(set(model_cls.keys()) | set(dataset_cls.keys()) | set(observed_stats.keys()))
        for name in names:
            model_idx = model_cls.get(name)
            dataset_idx = dataset_cls.get(name)
            observed_count = int(observed_stats.get(name, 0))
            effective_count = int(section_stats.get(name, 0))

            if model_idx is not None and dataset_idx is not None:
                status = 'ok' if model_idx == dataset_idx else 'index mismatch'
            elif dataset_idx is not None:
                status = 'missing in model mapping'
            elif model_idx is not None:
                status = 'missing in dataset mapping'
            elif observed_count > 0:
                status = 'unknown in test data'
            else:
                status = 'unknown'

            if observed_count > 0 and dataset_idx is None and model_idx is not None:
                status = 'ignored by dataset mapping'

            row = {'category': section,
                   'class_name': name,
                   'model_idx': '-' if model_idx is None else str(model_idx),
                   'dataset_idx': '-' if dataset_idx is None else str(dataset_idx),
                   'observed': str(observed_count),
                   'effective': str(effective_count),
                   'status': status}
            rows.append(row)
            if status != 'ok':
                mismatches.append(row)
    return rows, mismatches


@click.command('segtrain')
@click.pass_context
@click.option('-o', '--output', 'checkpoint_path', type=click.Path(), default='model', help='Output checkpoint path')
@click.option('--weights-format', default='safetensors', help='Output weights format.')
@click.option('-s', '--spec', help='VGSL spec of the baseline labeling network')
@click.option('--line-width', type=int, help='The height of each baseline in the target after scaling')
@click.option('--bl-tol', type=float, help='Tolerance in pixels for baseline detection metrics')
@click.option('--pad', 'padding', type=(int, int), help='Padding (left/right, top/bottom) around the page image')
@click.option('-i', '--load', type=click.Path(exists=True, readable=True), help='Load existing file to continue training')
@click.option('--resume', type=click.Path(exists=True, readable=True), help='Load a checkpoint to continue training')
@click.option('-F', '--freq', type=click.FLOAT,
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
              type=float,
              help='Weight decay')
@click.option('--gradient-clip-val',
              type=float,
              help='Gradient clip value')
@click.option('--accumulate-grad-batches',
              type=int,
              help='Number of batches to accumulate gradient across.')
@click.option('--warmup',
              type=int,
              help='Number of steps to ramp up to `lrate` initial learning rate.')
@click.option('--schedule',
              type=click.Choice(SCHEDULERS),
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--step-size` option.')
@click.option('-g',
              '--gamma',
              type=float,
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              type=float,
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
@click.option('-t', '--training-data', 'training_data', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-data', 'evaluation_data', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('-f',
              '--format-type',
              type=click.Choice(['xml', 'alto', 'page']),
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information.')
@click.option('--augment/--no-augment', help='Enable image augmentation')
@click.option('--resize',
              type=click.Choice([
                  'add', 'union',  # Deprecation: `add` is deprecated, `union` is the new value
                  'both', 'new',  # Deprecation: `both` is deprecated, `new` is the new value
                  'fail'
              ]),
              help='Output layer resizing option. If set to `add` new classes will be '
                   'added, `both` will set the layer to match exactly '
                   'the training data classes, `fail` will abort if training data and model '
                   'classes do not match.')
@click.option('-tl', '--topline', 'topline', flag_value=True,
              help='Switch for the baseline location in the scripts. '
                   'Set to topline if the data is annotated with a hanging baseline, as is '
                   'common with Hebrew, Bengali, Devanagari, etc. Set to '
                   ' centerline for scripts annotated with a central line.')
@click.option('-cl', '--centerline', 'topline', flag_value=None)
@click.option('-bl', '--baseline', 'topline', flag_value=False)
@click.option('--logger',
              'pl_logger',
              type=click.Choice(['tensorboard', 'wandb']),
              help='Logger used by PyTorch Lightning to track metrics such as loss and accuracy.')
@click.option('--log-dir',
              type=click.Path(exists=True, dir_okay=True, writable=True),
              help='Path to directory where the logger will store the logs. If not set, a directory will be created in the current working directory.')
@click.option('--line-class-mapping', type=click.UNPROCESSED, hidden=True)
@click.option('--region-class-mapping', type=click.UNPROCESSED, hidden=True)
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def segtrain(ctx, **kwargs):
    """
    Trains a baseline labeling model for layout analysis
    """
    params = ctx.meta.copy()
    params.update(ctx.params)
    resume = params.pop('resume', None)
    load = params.pop('load', None)
    training_data = params.pop('training_data', [])
    ground_truth = list(params.pop('ground_truth', []))

    if sum(map(bool, [resume, load])) > 1:
        raise click.BadOptionUsage('load', 'load/resume options are mutually exclusive.')

    if params.get('pl_logger') == 'tensorboard':
        try:
            import tensorboard  # NOQA
        except ImportError:
            raise click.BadOptionUsage('logger', 'tensorboard logger needs the `tensorboard` package installed.')

    # parse line_class_mapping from list into dictionary
    if isinstance(line_cls_map := params.get('line_class_mapping'), list):
        params['line_class_mapping'] = _create_class_map(line_cls_map)
    if isinstance(region_cls_map := params.get('region_class_mapping'), list):
        params['region_class_mapping'] = _create_class_map(region_cls_map)

    from threadpoolctl import threadpool_limits
    from lightning.pytorch.callbacks import ModelCheckpoint

    from kraken.lib import vgsl  # NOQA
    from kraken.train import (KrakenTrainer, BLLASegmentationDataModule,
                              BLLASegmentationModel)
    from kraken.train.utils import KrakenOnExceptionCheckpoint
    from kraken.configs import BLLASegmentationTrainingConfig, BLLASegmentationTrainingDataConfig
    from kraken.models.convert import convert_models

    # disable automatic partition when given evaluation set explicitly
    if params['evaluation_data']:
        params['partition'] = 1

    # merge training_data into ground_truth list
    if training_data:
        ground_truth.extend(training_data)

    params['training_data'] = ground_truth

    if len(ground_truth) == 0 and not resume:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    if params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(params['freq'])}
    else:
        val_check_interval = {'val_check_interval': params['freq']}

    cbs = [KrakenOnExceptionCheckpoint(dirpath=params.get('checkpoint_path'),
                                       filename='checkpoint_abort')]
    checkpoint_callback = ModelCheckpoint(dirpath=params.pop('checkpoint_path'),
                                          save_top_k=10,
                                          monitor='val_metric',
                                          mode='max',
                                          auto_insert_metric_name=False,
                                          filename='checkpoint_{epoch:02d}-{val_metric:.4f}')
    cbs.append(checkpoint_callback)

    dm_config = BLLASegmentationTrainingDataConfig(**params)
    m_config = BLLASegmentationTrainingConfig(**params)

    if resume:
        data_module = BLLASegmentationDataModule.load_from_checkpoint(resume, weights_only=False)
    else:
        data_module = BLLASegmentationDataModule(dm_config)

    from rich.console import Console
    from rich.table import Table

    ds = data_module.train_set.dataset
    canonical = ds.canonical_class_mapping
    merged = ds.merged_classes

    table = Table(title='Training Class Summary')
    table.add_column('Category')
    table.add_column('Class')
    table.add_column('Label Index', justify='right')
    table.add_column('Merged With')
    table.add_column('Count', justify='right')

    for section in ('baselines', 'regions'):
        for cls_name, idx in canonical[section].items():
            aliases = merged[section].get(cls_name, [])
            merged_str = ', '.join(aliases) if aliases else ''
            count = ds.class_stats[section].get(cls_name, 0)
            for alias in aliases:
                count += ds.class_stats[section].get(alias, 0)
            table.add_row(section, cls_name, str(idx), merged_str, str(count))

    Console(stderr=True).print(table)

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
                model = BLLASegmentationModel.load_from_checkpoint(load, config=m_config, weights_only=False)
            else:
                model = BLLASegmentationModel.load_from_weights(load, config=m_config)
        elif resume:
            message(f'Resuming from checkpoint {resume}.')
            model = BLLASegmentationModel.load_from_checkpoint(resume, weights_only=False)
        else:
            message('Initializing new model.')
            model = BLLASegmentationModel(m_config)

    with threadpool_limits(limits=ctx.meta['num_threads']):
        if resume:
            trainer.fit(model, data_module, ckpt_path=resume)
        else:
            trainer.fit(model, data_module)

    score = checkpoint_callback.best_model_score.item()
    weight_path = Path(checkpoint_callback.best_model_path).with_name(f'best_{score:.4f}.{params.get("weights_format")}')
    opath = convert_models([checkpoint_callback.best_model_path], weight_path, weights_format=params['weights_format'])
    message(f'Converting best model {checkpoint_callback.best_model_path} (score: {score:.4f}) to weights {opath}')


@click.command('segtest')
@click.pass_context
@click.option('-m', '--model', type=click.Path(exists=True, readable=True),
              multiple=False, help='Model(s) to evaluate')
@click.option('-e',
              '--test-data',
              'test_data',
              multiple=True,
              callback=_validate_manifests,
              type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-f',
              '--format-type',
              type=click.Choice(['xml', 'alto', 'page']),
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images.')
@click.option('--bl-tol', type=float, help='Tolerance in pixels for baseline detection metrics')
@click.option('--test-class-mapping-mode',
              type=click.Choice(['full', 'canonical', 'custom']),
              default='full',
              show_default=True,
              help='Class mapping mode for test dataset. `full` uses the '
                   'many-to-one mapping from the training checkpoint (falls '
                   'back to `canonical` for weights files), `canonical` uses '
                   'the one-to-one mapping, `custom` uses user-provided mappings.')
@click.option('--line-class-mapping', type=click.UNPROCESSED, hidden=True)
@click.option('--region-class-mapping', type=click.UNPROCESSED, hidden=True)
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def segtest(ctx, **kwargs):
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

    # merge test_data into test_set list
    if test_data:
        test_set.extend(test_data)

    params['test_data'] = test_set

    # parse custom class mappings
    if isinstance(line_cls_map := params.get('line_class_mapping'), list):
        params['line_class_mapping'] = _create_class_map(line_cls_map)
    if isinstance(region_cls_map := params.get('region_class_mapping'), list):
        params['region_class_mapping'] = _create_class_map(region_cls_map)

    from kraken.train import (KrakenTrainer, BLLASegmentationModel,
                              BLLASegmentationDataModule)
    from kraken.configs import BLLASegmentationTrainingConfig, BLLASegmentationTestDataConfig

    trainer = KrakenTrainer(accelerator=ctx.meta['accelerator'],
                            devices=ctx.meta['device'],
                            precision=ctx.meta['precision'],
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            deterministic=ctx.meta['deterministic'],
                            enable_model_summary=False,
                            num_sanity_val_steps=0)

    m_config = BLLASegmentationTrainingConfig(**params)
    dm_config = BLLASegmentationTestDataConfig(**params)
    data_module = BLLASegmentationDataModule(dm_config)

    with trainer.init_module(empty_init=False):
        message(f'Loading from {model}.')
        if model.endswith('ckpt'):
            model = BLLASegmentationModel.load_from_checkpoint(model, config=m_config, weights_only=False)
        else:
            model = BLLASegmentationModel.load_from_weights(model, m_config)

    from rich.console import Console
    from rich.table import Table

    console = Console()

    # initialize test data/model explicitly so diagnostics are based on effective
    # test mapping before running evaluation
    class _DMProxy:
        pass

    class _ModelProxy:
        pass

    dm_proxy = _DMProxy()
    dm_proxy.lightning_module = model
    data_module.trainer = dm_proxy
    data_module.setup('test')

    model_proxy = _ModelProxy()
    model_proxy.datamodule = data_module
    model.trainer = model_proxy
    model.setup('test')

    # class mapping diagnostics
    effective_mapping = data_module.test_set.dataset.class_mapping
    effective_stats = data_module.test_set.dataset.class_stats
    observed = _collect_observed_test_classes(data_module.test_data)
    mode = params.get('test_class_mapping_mode', 'full')
    if mode == 'full' and hasattr(model, '_full_class_mapping'):
        model_mapping = model._full_class_mapping
        model_mapping_src = 'full'
    else:
        model_mapping = model.net.user_metadata['class_mapping']
        model_mapping_src = 'canonical'

    diag_rows, mismatch_rows = _build_segtest_class_diagnostics(model_mapping,
                                                                effective_mapping,
                                                                effective_stats,
                                                                observed)
    diag = Table(title=f'Class Mapping Diagnostics (model={model_mapping_src}, dataset=effective)')
    diag.add_column('Category')
    diag.add_column('Class Name')
    diag.add_column('Model Idx')
    diag.add_column('Dataset Idx')
    diag.add_column('Observed')
    diag.add_column('Effective')
    diag.add_column('Status')
    for row in diag_rows:
        diag.add_row(row['category'],
                     row['class_name'],
                     row['model_idx'],
                     row['dataset_idx'],
                     row['observed'],
                     row['effective'],
                     row['status'])
    console.print(diag)

    if mismatch_rows:
        mismatch = Table('Category', 'Class Name', 'Model Idx', 'Dataset Idx', 'Observed', 'Effective', 'Mismatch')
        for row in mismatch_rows:
            mismatch.add_row(row['category'],
                             row['class_name'],
                             row['model_idx'],
                             row['dataset_idx'],
                             row['observed'],
                             row['effective'],
                             row['status'])
        console.print(mismatch)

    test_metrics = trainer.test(model, data_module)

    # pixel metrics table (aux + regions only)
    class_mapping = effective_mapping
    idx_to_name = {}
    for cat in ('aux', 'regions'):
        for name, idx in class_mapping[cat].items():
            idx_to_name[idx] = (cat, name)
    pixel_idxs = sorted(idx_to_name.keys())

    table = Table('Category', 'Class Name', 'Pixel Accuracy', 'IOU', 'Object Count')
    class_iu = test_metrics.class_iu.tolist()
    class_pixel_accuracy = test_metrics.class_pixel_accuracy.tolist()
    for idx in pixel_idxs:
        cat, class_name = idx_to_name[idx]
        count = f'{data_module.test_set.dataset.class_stats[cat][class_name]}' if cat != 'aux' else 'N/A'
        table.add_row(cat, class_name, f'{class_pixel_accuracy[idx]:.3f}', f'{class_iu[idx]:.3f}', count)
    console.print(table)

    # baseline detection metrics table
    if test_metrics.bl_f1 is not None:
        bl_table = Table('Class', 'Precision', 'Recall', 'F1')
        bl_table.add_row('Overall',
                         f'{test_metrics.bl_precision:.3f}',
                         f'{test_metrics.bl_recall:.3f}',
                         f'{test_metrics.bl_f1:.3f}')
        if test_metrics.bl_detection_per_class:
            for cls_name, m in test_metrics.bl_detection_per_class.items():
                bl_table.add_row(cls_name,
                                 f'{m["precision"]:.3f}',
                                 f'{m["recall"]:.3f}',
                                 f'{m["f1"]:.3f}')
        console.print(bl_table)
