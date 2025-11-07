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
import logging
import pathlib

import click
from PIL import Image

from kraken.ketos.util import (_expand_gt, _validate_manifests, message,
                               to_ptl_device, _validate_merging)

from kraken.registry import OPTIMIZERS, SCHEDULERS, STOPPERS

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


@click.command('segtrain')
@click.pass_context
@click.option('-o', '--output', 'checkpoint_path', type=click.Path(), default='model', help='Output model file')
@click.option('--weights-format', default='safetensors', help='Output weights format.')
@click.option('-s', '--spec', help='VGSL spec of the baseline labeling network')
@click.option('--line-width', type=int, help='The height of each baseline in the target after scaling')
@click.option('--pad', 'padding', type=(int, int), help='Padding (left/right, top/bottom) around the page image')
@click.option('-i', '--load', type=click.Path(exists=True, readable=True), help='Load existing file to continue training')
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
@click.option('-r', '--lrate', type=float, help='Learning rate')
@click.option('-m', '--momentum', type=float, help='Momentum')
@click.option('-w', '--weight-decay', type=float, help='Weight decay')
@click.option('--warmup', type=int, help='Number of steps to ramp up to `lrate` initial learning rate.')
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
              type=int,
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              type=int,
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('--cos-min-lr',
              type=float,
              help='Minimal final learning rate for cosine LR scheduler.')
@click.option('-p', '--partition', type=float, help='Ground truth data partition ratio between train/validation set')
@click.option('-t', '--training-files', 'training_data', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', 'evaluation_data', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('-f', '--format-type', type=click.Choice(['xml', 'alto', 'page']),
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information.')
@click.option('--augment/--no-augment', help='Enable image augmentation')
@click.option('--resize', default='fail',
              type=click.Choice([
                  'add', 'union',  # Deprecation: `add` is deprecated, `union` is the new value
                  'both', 'new',  # Deprecation: `both` is deprecated, `new` is the new value
                  'fail'
              ]),
              help='Output layer resizing option. If set to `add` new classes will be '
                   'added, `both` will set the layer to match exactly '
                   'the training data classes, `fail` will abort if training data and model '
                   'classes do not match.')
@click.option('-tl', '--topline', 'topline', flag_value='topline',
              help='Switch for the baseline location in the scripts. '
                   'Set to topline if the data is annotated with a hanging baseline, as is '
                   'common with Hebrew, Bengali, Devanagari, etc. Set to '
                   ' centerline for scripts annotated with a central line.')
@click.option('-cl', '--centerline', 'topline', flag_value='centerline')
@click.option('-bl', '--baseline', 'topline', flag_value='baseline')
@click.option('--logger', 'pl_logger', type=click.Choice(['tensorboard']), help='Logger used by PyTorch Lightning to track metrics such as loss and accuracy.')
@click.option('--log-dir', type=click.Path(exists=True, dir_okay=True, writable=True),
              help='Path to directory where the logger will store the logs. If not set, a directory will be created in the current working directory.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def segtrain(ctx, **kwargs):
    """
    Trains a baseline labeling model for layout analysis
    """
    import shutil

    from threadpoolctl import threadpool_limits

    from kraken.lib.train import KrakenTrainer, SegmentationModel

    if resize != 'fail' and not load:
        raise click.BadOptionUsage('resize', 'resize option requires loading an existing model')

    if not (0 <= freq <= 1) and freq % 1.0 != 0:
        raise click.BadOptionUsage('freq', 'freq needs to be either in the interval [0,1.0] or a positive integer.')

    if augment:
        try:
            import albumentations  # NOQA
        except ImportError:
            raise click.BadOptionUsage('augment', 'augmentation needs the `albumentations` package installed.')

    if pl_logger == 'tensorboard':
        try:
            import tensorboard  # NOQA
        except ImportError:
            raise click.BadOptionUsage('logger', 'tensorboard logger needs the `tensorboard` package installed.')

    if log_dir is None:
        log_dir = pathlib.Path.cwd()

    logger.info('Building ground truth set from {} document images'.format(len(ground_truth) + len(training_files)))

    # populate hyperparameters from command line args
    hyper_params = SEGMENTATION_HYPER_PARAMS.copy()
    hyper_params.update({'line_width': line_width,
                         'padding': pad,
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
                         'augment': augment,
                         'gamma': gamma,
                         'step_size': step_size,
                         'rop_patience': sched_patience,
                         'cos_t_max': cos_max,
                         'cos_min_lr': cos_min_lr,
                         })

    # disable automatic partition when given evaluation set explicitly
    if evaluation_files:
        partition = 1
    ground_truth = list(ground_truth)

    # merge training_files into ground_truth list
    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    loc = {'topline': True,
           'baseline': False,
           'centerline': None}

    topline = loc[topline]

    try:
        accelerator, device = to_ptl_device(device)
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    if hyper_params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(hyper_params['freq'])}
    else:
        val_check_interval = {'val_check_interval': hyper_params['freq']}

    model = SegmentationModel()

    message('Training line types:')
    for k, v in model.train_set.dataset.class_mapping['baselines'].items():
        message(f'  {k}\t{v}\t{model.train_set.dataset.class_stats["baselines"][k]}')
    message('Training region types:')
    for k, v in model.train_set.dataset.class_mapping['regions'].items():
        message(f'  {k}\t{v}\t{model.train_set.dataset.class_stats["regions"][k]}')

    if len(model.train_set) == 0:
        raise click.UsageError('No valid training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    trainer = KrakenTrainer(accelerator=accelerator,
                            devices=device,
                            precision=ctx.meta['precision'],
                            max_epochs=hyper_params['epochs'] if hyper_params['quit'] == 'fixed' else -1,
                            min_epochs=hyper_params['min_epochs'],
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            deterministic=ctx.meta['deterministic'],
                            pl_logger=pl_logger,
                            log_dir=log_dir,
                            **val_check_interval)

    with threadpool_limits(limits=ctx.meta['threads']):
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


@click.command('segtest')
@click.pass_context
@click.option('-m', '--model', type=click.Path(exists=True, readable=True),
              multiple=False, help='Model(s) to evaluate')
@click.option('-e', '--test-data', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-f', '--format-type', type=click.Choice(['xml', 'alto', 'page']), default='xml',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images.')
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def segtest(ctx, **kwargs):
    """
    Evaluate on a test set.
    """
    if not model:
        raise click.UsageError('No model to evaluate given.')

    import torch
    import torch.nn.functional as F
    from threadpoolctl import threadpool_limits
    from torch.utils.data import DataLoader
    from lightning.fabric import Fabric

    from kraken.lib.xml import XMLPage
    from kraken.lib.vgsl import TorchVGSLModel
    from kraken.lib.progress import KrakenProgressBar
    from kraken.lib.train import BaselineSet, ImageInputTransforms

    logger.info('Building test set from {} documents'.format(len(test_set) + len(evaluation_files)))

    message('Loading model {}\t'.format(model), nl=False)

    try:
        accelerator, device = to_ptl_device(ctx.meta['device'])
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    fabric = Fabric(accelerator=accelerator,
                    devices=device,
                    precision=ctx.meta['precision'])

    with fabric.init_tensor(), fabric.init_module():
        nn = TorchVGSLModel.load_model(model)
        nn.eval()

    message('\u2713', fg='green')

    test_set = list(test_set)
    if evaluation_files:
        test_set.extend(evaluation_files)

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    _batch, _channels, _height, _width = nn.input
    transforms = ImageInputTransforms(
        _batch,
        _height, _width, _channels, 0,
        valid_norm=False, force_binarization=force_binarization
    )
    if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
        logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
        torch.multiprocessing.set_sharing_strategy('file_system')

    if not valid_regions:
        valid_regions = None
    if not valid_baselines:
        valid_baselines = None

    if suppress_regions:
        valid_regions = []
        merge_regions = None
    if suppress_baselines:
        valid_baselines = []
        merge_baselines = None

    dataset = BaselineSet(line_width=nn.user_metadata["hyper_params"]["line_width"],
                          im_transforms=transforms,
                          augmentation=False,
                          valid_baselines=valid_baselines,
                          merge_baselines=merge_baselines,
                          valid_regions=valid_regions,
                          merge_regions=merge_regions,
                          merge_all_baselines=merge_all_baselines,
                          merge_all_regions=merge_all_regions,
                          class_mapping=nn.user_metadata["class_mapping"])

    for file in test_set:
        try:
            dataset.add(XMLPage(file).to_container())
        except Exception as e:
            logger.warning(str(e))

    baselines_diff = set(dataset.class_stats["baselines"].keys()).difference(dataset.class_mapping["baselines"].keys())
    regions_diff = set(dataset.class_stats["regions"].keys()).difference(dataset.class_mapping["regions"].keys())

    if baselines_diff:
        message(f'Model baseline types missing in test set: {", ".join(sorted(list(baselines_diff)))}')

    if regions_diff:
        message(f'Model region types missing in the test set: {", ".join(sorted(list(regions_diff)))}')

    if len(dataset) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    ds_loader = DataLoader(dataset, batch_size=1, num_workers=ctx.meta['workers'], pin_memory=True)

    pages = []

    lines_idx = list(dataset.class_mapping["baselines"].values())
    regions_idx = list(dataset.class_mapping["regions"].values())

    with KrakenProgressBar() as progress:
        batches = len(ds_loader)
        pred_task = progress.add_task('Evaluating', total=batches, visible=True if not ctx.meta['verbose'] else False)
        with torch.inference_mode(), threadpool_limits(limits=ctx.meta['threads']), fabric.init_tensor():
            for batch in ds_loader:
                x, y = batch['image'], batch['target']
                try:
                    pred, _ = nn.nn(x)
                    # scale target to output size
                    y = F.interpolate(y, size=(pred.size(2), pred.size(3))).squeeze(0).bool()
                    pred = pred.squeeze() > threshold
                    pred = pred.view(pred.size(0), -1)
                    y = y.view(y.size(0), -1)
                    pages.append({
                        'intersections': (y & pred).sum(dim=1, dtype=torch.double),
                        'unions': (y | pred).sum(dim=1, dtype=torch.double),
                        'corrects': torch.eq(y, pred).sum(dim=1, dtype=torch.double),
                        'cls_cnt': y.sum(dim=1, dtype=torch.double),
                        'all_n': torch.tensor(y.size(1), dtype=torch.double)
                    })
                    if lines_idx:
                        y_baselines = y[lines_idx].sum(dim=0, dtype=torch.bool)
                        pred_baselines = pred[lines_idx].sum(dim=0, dtype=torch.bool)
                        pages[-1]["baselines"] = {
                            'intersections': (y_baselines & pred_baselines).sum(dim=0, dtype=torch.double),
                            'unions': (y_baselines | pred_baselines).sum(dim=0, dtype=torch.double),
                        }
                    if regions_idx:
                        y_regions_idx = y[regions_idx].sum(dim=0, dtype=torch.bool)
                        pred_regions_idx = pred[regions_idx].sum(dim=0, dtype=torch.bool)
                        pages[-1]["regions"] = {
                            'intersections': (y_regions_idx & pred_regions_idx).sum(dim=0, dtype=torch.double),
                            'unions': (y_regions_idx | pred_regions_idx).sum(dim=0, dtype=torch.double),
                        }
                except Exception as e:
                    batches -= 1
                    progress.update(pred_task, total=batches)
                    logger.warning(str(e))
                progress.update(pred_task, advance=1)

    # Accuracy / pixel
    message(f"Mean Accuracy: {mean_accuracy.item():.3f}")
    message(f"Mean IOU: {mean_iu.item():.3f}")
    message(f"Frequency-weighted IOU: {freq_iu.item():.3f}")

    # Region accuracies
    if lines_idx:
        line_intersections = torch.stack([x["baselines"]['intersections'] for x in pages]).sum()
        line_unions = torch.stack([x["baselines"]['unions'] for x in pages]).sum()
        smooth = torch.finfo(torch.float).eps
        line_iu = (line_intersections + smooth) / (line_unions + smooth)
        message(f"Class-independent Baseline IOU: {line_iu.item():.3f}")

    # Region accuracies
    if regions_idx:
        region_intersections = torch.stack([x["regions"]['intersections'] for x in pages]).sum()
        region_unions = torch.stack([x["regions"]['unions'] for x in pages]).sum()
        smooth = torch.finfo(torch.float).eps
        region_iu = (region_intersections + smooth) / (region_unions + smooth)
        message(f"Class-independent Region IOU: {region_iu.item():.3f}")

    from rich.console import Console
    from rich.table import Table

    table = Table('Category', 'Class Name', 'Pixel Accuracy', 'IOU', 'Object Count')

    class_iu = class_iu.tolist()
    class_pixel_accuracy = class_pixel_accuracy.tolist()
    for (cat, class_name), iu, pix_acc in zip(
        [(cat, key) for (cat, subcategory) in dataset.class_mapping.items() for key in subcategory],
        class_iu,
        class_pixel_accuracy
    ):
        table.add_row(cat, class_name, f'{pix_acc:.3f}', f'{iu:.3f}', f'{dataset.class_stats[cat][class_name]}' if cat != "aux" else 'N/A')

    console = Console()
    console.print(table)
