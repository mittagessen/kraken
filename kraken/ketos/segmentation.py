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
import pathlib
import logging

from PIL import Image
from typing import Dict

from kraken.lib.progress import KrakenProgressBar
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.default_specs import SEGMENTATION_HYPER_PARAMS, SEGMENTATION_SPEC

from kraken.ketos.util import _validate_manifests, _expand_gt, message, to_ptl_device

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


def _validate_merging(ctx, param, value):
    """
    Maps baseline/region merging to a dict of merge structures.
    """
    if not value:
        return None
    merge_dict = {}  # type: Dict[str, str]
    try:
        for m in value:
            k, v = m.split(':')
            merge_dict[v] = k  # type: ignore
    except Exception:
        raise click.BadParameter('Mappings must be in format target:src')
    return merge_dict


@click.command('segtrain')
@click.pass_context
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model', help='Output model file')
@click.option('-s', '--spec', show_default=True,
              default=SEGMENTATION_SPEC,
              help='VGSL spec of the baseline labeling network')
@click.option('--line-width',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['line_width'],
              help='The height of each baseline in the target after scaling')
@click.option('--pad', show_default=True, type=(int, int), default=(0, 0),
              help='Padding (left/right, top/bottom) around the page image')
@click.option('-i', '--load', show_default=True, type=click.Path(exists=True,
              readable=True), help='Load existing file to continue training')
@click.option('-F', '--freq', show_default=True, default=SEGMENTATION_HYPER_PARAMS['freq'], type=click.FLOAT,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q',
              '--quit',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['quit'],
              type=click.Choice(['early',
                                 'dumb']),
              help='Stop condition for training. Set to `early` for early stopping or `dumb` for fixed number of epochs')
@click.option('-N',
              '--epochs',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['epochs'],
              help='Number of epochs to train for')
@click.option('--min-epochs',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['min_epochs'],
              help='Minimal number of epochs to train for when using early stopping.')
@click.option('--lag',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['lag'],
              help='Number of evaluations (--report frequence) to wait before stopping training without improvement')
@click.option('--min-delta',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['min_delta'],
              type=click.FLOAT,
              help='Minimum improvement between epochs to reset early stopping. By default it scales the delta by the best loss')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--precision', default='32', type=click.Choice(['32', '16']), help='set tensor precision')
@click.option('--optimizer',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['optimizer'],
              type=click.Choice(['Adam',
                                 'SGD',
                                 'RMSprop',
                                 'Lamb']),
              help='Select optimizer')
@click.option('-r', '--lrate', show_default=True, default=SEGMENTATION_HYPER_PARAMS['lrate'], help='Learning rate')
@click.option('-m', '--momentum', show_default=True, default=SEGMENTATION_HYPER_PARAMS['momentum'], help='Momentum')
@click.option('-w', '--weight-decay', show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['weight_decay'], help='Weight decay')
@click.option('--warmup', show_default=True, type=float,
              default=SEGMENTATION_HYPER_PARAMS['warmup'], help='Number of samples to ramp up to `lrate` initial learning rate.')
@click.option('--schedule',
              show_default=True,
              type=click.Choice(['constant',
                                 '1cycle',
                                 'exponential',
                                 'cosine',
                                 'step',
                                 'reduceonplateau']),
              default=SEGMENTATION_HYPER_PARAMS['schedule'],
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--step-size` option.')
@click.option('-g',
              '--gamma',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['gamma'],
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['step_size'],
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['rop_patience'],
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['cos_t_max'],
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('-p', '--partition', show_default=True, default=0.9,
              help='Ground truth data partition ratio between train/validation set')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--workers', show_default=True, default=1, help='Number of OpenMP threads and workers when running on CPU.')
@click.option('--load-hyper-parameters/--no-load-hyper-parameters', show_default=True, default=False,
              help='When loading an existing model, retrieve hyper-parameters from the model')
@click.option('--force-binarization/--no-binarization', show_default=True,
              default=False, help='Forces input images to be binary, otherwise '
              'the appropriate color format will be auto-determined through the '
              'network specification. Will be ignored in `path` mode.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page']), default='xml',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information.')
@click.option('--suppress-regions/--no-suppress-regions', show_default=True,
              default=False, help='Disables region segmentation training.')
@click.option('--suppress-baselines/--no-suppress-baselines', show_default=True,
              default=False, help='Disables baseline segmentation training.')
@click.option('-vr', '--valid-regions', show_default=True, default=None, multiple=True,
              help='Valid region types in training data. May be used multiple times.')
@click.option('-vb', '--valid-baselines', show_default=True, default=None, multiple=True,
              help='Valid baseline types in training data. May be used multiple times.')
@click.option('-mr',
              '--merge-regions',
              show_default=True,
              default=None,
              help='Region merge mapping. One or more mappings of the form `$target:$src` where $src is merged into $target.',
              multiple=True,
              callback=_validate_merging)
@click.option('-mb',
              '--merge-baselines',
              show_default=True,
              default=None,
              help='Baseline type merge mapping. Same syntax as `--merge-regions`',
              multiple=True,
              callback=_validate_merging)
@click.option('-br', '--bounding-regions', show_default=True, default=None, multiple=True,
              help='Regions treated as boundaries for polygonization purposes. May be used multiple times.')
@click.option('--failed-sample-threshold', show_default=True, default=10,
              help='Abort if more than `n` samples fail to load.')
@click.option('--augment/--no-augment',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['augment'],
              help='Enable image augmentation')
@click.option('--resize', show_default=True, default='fail', type=click.Choice(['add', 'both', 'fail']),
              help='Output layer resizing option. If set to `add` new classes will be '
                   'added, `both` will set the layer to match exactly '
                   'the training data classes, `fail` will abort if training data and model '
                   'classes do not match.')
@click.option('-tl', '--topline', 'topline', show_default=True, flag_value='topline',
              help='Switch for the baseline location in the scripts. '
                   'Set to topline if the data is annotated with a hanging baseline, as is '
                   'common with Hebrew, Bengali, Devanagari, etc. Set to '
                   ' centerline for scripts annotated with a central line.')
@click.option('-cl', '--centerline', 'topline', flag_value='centerline')
@click.option('-bl', '--baseline', 'topline', flag_value='baseline', default='baseline')
@click.option('--logger', 'pl_logger', show_default=True, type=click.Choice([None, 'tensorboard']), default=None,
              help='Logger used by PyTorch Lightning to track metrics such as loss and accuracy.')
@click.option('--log-dir', show_default=True, type=click.Path(exists=True, dir_okay=True, writable=True),
              help='Path to directory where the logger will store the logs. If not set, a directory will be created in the current working directory.')          
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def segtrain(ctx, output, spec, line_width, pad, load, freq, quit, epochs,
             min_epochs, lag, min_delta, device, precision, optimizer, lrate,
             momentum, weight_decay, warmup, schedule, gamma, step_size,
             sched_patience, cos_max, partition, training_files,
             evaluation_files, workers, load_hyper_parameters,
             force_binarization, format_type, suppress_regions,
             suppress_baselines, valid_regions, valid_baselines, merge_regions,
             merge_baselines, bounding_regions, failed_sample_threshold,
             augment, resize, topline, pl_logger, log_dir, ground_truth):
    """
    Trains a baseline labeling model for layout analysis
    """
    import shutil

    from kraken.lib.train import SegmentationModel, KrakenTrainer

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

    model = SegmentationModel(hyper_params,
                              output=output,
                              spec=spec,
                              model=load,
                              training_data=ground_truth,
                              evaluation_data=evaluation_files,
                              partition=partition,
                              num_workers=workers,
                              load_hyper_parameters=load_hyper_parameters,
                              force_binarization=force_binarization,
                              format_type=format_type,
                              suppress_regions=suppress_regions,
                              suppress_baselines=suppress_baselines,
                              valid_regions=valid_regions,
                              valid_baselines=valid_baselines,
                              merge_regions=merge_regions,
                              merge_baselines=merge_baselines,
                              bounding_regions=bounding_regions,
                              resize=resize,
                              topline=topline)

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
                            max_epochs=hyper_params['epochs'] if hyper_params['quit'] == 'dumb' else -1,
                            min_epochs=hyper_params['min_epochs'],
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            deterministic=ctx.meta['deterministic'],
                            failed_sample_threshold=failed_sample_threshold,
                            precision=int(precision),
                            pl_logger=pl_logger,
                            log_dir=log_dir,
                            **val_check_interval)

    trainer.fit(model)

    if quit == 'early':
        message('Moving best model {0}_{1}.mlmodel ({2}) to {0}_best.mlmodel'.format(
            output, model.best_epoch, model.best_metric))
        logger.info('Moving best model {0}_{1}.mlmodel ({2}) to {0}_best.mlmodel'.format(
            output, model.best_epoch, model.best_metric))
        shutil.copy(f'{output}_{model.best_epoch}.mlmodel', f'{output}_best.mlmodel')


@click.command('segtest')
@click.pass_context
@click.option('-m', '--model', show_default=True, type=click.Path(exists=True, readable=True),
              multiple=False, help='Model(s) to evaluate')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--workers', show_default=True, default=1, help='Number of OpenMP threads when running on CPU.')
@click.option('--force-binarization/--no-binarization', show_default=True,
              default=False, help='Forces input images to be binary, otherwise '
              'the appropriate color format will be auto-determined through the '
              'network specification. Will be ignored in `path` mode.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page']), default='xml',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information.')
@click.option('--suppress-regions/--no-suppress-regions', show_default=True,
              default=False, help='Disables region segmentation training.')
@click.option('--suppress-baselines/--no-suppress-baselines', show_default=True,
              default=False, help='Disables baseline segmentation training.')
@click.option('-vr', '--valid-regions', show_default=True, default=None, multiple=True,
              help='Valid region types in training data. May be used multiple times.')
@click.option('-vb', '--valid-baselines', show_default=True, default=None, multiple=True,
              help='Valid baseline types in training data. May be used multiple times.')
@click.option('-mr',
              '--merge-regions',
              show_default=True,
              default=None,
              help='Region merge mapping. One or more mappings of the form `$target:$src` where $src is merged into $target.',
              multiple=True,
              callback=_validate_merging)
@click.option('-mb',
              '--merge-baselines',
              show_default=True,
              default=None,
              help='Baseline type merge mapping. Same syntax as `--merge-regions`',
              multiple=True,
              callback=_validate_merging)
@click.option('-br', '--bounding-regions', show_default=True, default=None, multiple=True,
              help='Regions treated as boundaries for polygonization purposes. May be used multiple times.')
@click.option("--threshold", type=click.FloatRange(.01, .99), default=.3, show_default=True,
              help="Threshold for heatmap binarization. Training threshold is .3, prediction is .5")
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def segtest(ctx, model, evaluation_files, device, workers, threshold,
            force_binarization, format_type, test_set, suppress_regions,
            suppress_baselines, valid_regions, valid_baselines, merge_regions,
            merge_baselines, bounding_regions):
    """
    Evaluate on a test set.
    """
    if not model:
        raise click.UsageError('No model to evaluate given.')

    from torch.utils.data import DataLoader
    import torch
    import torch.nn.functional as F

    from kraken.lib.train import BaselineSet, ImageInputTransforms
    from kraken.lib.vgsl import TorchVGSLModel

    logger.info('Building test set from {} documents'.format(len(test_set) + len(evaluation_files)))

    message('Loading model {}\t'.format(model), nl=False)
    nn = TorchVGSLModel.load_model(model)
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

    test_set = BaselineSet(test_set,
                           line_width=nn.user_metadata["hyper_params"]["line_width"],
                           im_transforms=transforms,
                           mode=format_type,
                           augmentation=False,
                           valid_baselines=valid_baselines,
                           merge_baselines=merge_baselines,
                           valid_regions=valid_regions,
                           merge_regions=merge_regions)

    test_set.class_mapping = nn.user_metadata["class_mapping"]
    test_set.num_classes = sum([len(classDict) for classDict in test_set.class_mapping.values()])

    baselines_diff = set(test_set.class_stats["baselines"].keys()).difference(test_set.class_mapping["baselines"].keys())
    regions_diff = set(test_set.class_stats["regions"].keys()).difference(test_set.class_mapping["regions"].keys())

    if baselines_diff:
        message(f'Model baseline types missing in test set: {", ".join(sorted(list(baselines_diff)))}')

    if regions_diff:
        message(f'Model region types missing in the test set: {", ".join(sorted(list(regions_diff)))}')

    try:
        accelerator, device = to_ptl_device(device)
        if device:
            device = f'{accelerator}:{device}'
        else:
            device = accelerator
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    ds_loader = DataLoader(test_set, batch_size=1, num_workers=workers, pin_memory=True)

    nn.to(device)
    nn.eval()
    nn.set_num_threads(1)
    pages = []

    lines_idx = list(test_set.class_mapping["baselines"].values())
    regions_idx = list(test_set.class_mapping["regions"].values())

    with KrakenProgressBar() as progress:
        batches = len(ds_loader)
        pred_task = progress.add_task('Evaluating', total=batches, visible=True if not ctx.meta['verbose'] else False)
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
                    'all_n': torch.tensor(y.size(1), dtype=torch.double, device=device)
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

            except FileNotFoundError as e:
                batches -= 1
                progress.update(pred_task, total=batches)
                logger.warning('{} {}. Skipping.'.format(e.strerror, e.filename))
            except KrakenInputException as e:
                batches -= 1
                progress.update(pred_task, total=batches)
                logger.warning(str(e))
            progress.update(pred_task, advance=1)

    # Accuracy / pixel
    corrects = torch.stack([x['corrects'] for x in pages], -1).sum(dim=-1)
    all_n = torch.stack([x['all_n'] for x in pages]).sum()  # Number of pixel for all pages

    class_pixel_accuracy = corrects / all_n
    mean_accuracy = torch.mean(class_pixel_accuracy)

    intersections = torch.stack([x['intersections'] for x in pages], -1).sum(dim=-1)
    unions = torch.stack([x['unions'] for x in pages], -1).sum(dim=-1)
    smooth = torch.finfo(torch.float).eps
    class_iu = (intersections + smooth) / (unions + smooth)
    mean_iu = torch.mean(class_iu)

    cls_cnt = torch.stack([x['cls_cnt'] for x in pages]).sum()
    freq_iu = torch.sum(cls_cnt / cls_cnt.sum() * class_iu.sum())

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
        [(cat, key) for (cat, subcategory) in test_set.class_mapping.items() for key in subcategory],
        class_iu,
        class_pixel_accuracy
    ):
        table.add_row(cat, class_name, f'{pix_acc:.3f}', f'{iu:.3f}', f'{test_set.class_stats[cat][class_name]}' if cat != "aux" else 'N/A')

    console = Console()
    console.print(table)
