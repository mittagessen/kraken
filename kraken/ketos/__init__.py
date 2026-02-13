#
# Copyright 2015 Benjamin Kiessling
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
kraken.ketos
~~~~~~~~~~~~~

Command line drivers for training functionality.
"""
import click
import importlib
import logging

from PIL import Image
from rich.traceback import install

from kraken.lib import log
from kraken.registry import PRECISIONS

from .util import _load_config, to_ptl_device

from kraken.configs import (Config,
                            TrainingDataConfig,
                            VGSLPreTrainingConfig,
                            VGSLRecognitionTrainingConfig,
                            VGSLRecognitionTrainingDataConfig,
                            BLLASegmentationTrainingConfig,
                            BLLASegmentationTrainingDataConfig,
                            ROTrainingDataConfig,
                            ROTrainingConfig)

logging.captureWarnings(True)
logger = logging.getLogger('kraken')
# disable annoying lightning worker seeding log messages
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)
# install rich traceback handler
install(suppress=[click])

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


@click.group(context_settings=dict(show_default=True,
                                   default_map={**Config().__dict__,
                                                **TrainingDataConfig().__dict__,
                                                'train': {**VGSLRecognitionTrainingConfig().__dict__, **VGSLRecognitionTrainingDataConfig().__dict__},
                                                'test': VGSLRecognitionTrainingDataConfig().__dict__,
                                                'segtrain': {**BLLASegmentationTrainingConfig().__dict__, **BLLASegmentationTrainingDataConfig().__dict__},
                                                'segtest': {**BLLASegmentationTrainingConfig().__dict__, **BLLASegmentationTrainingDataConfig().__dict__},
                                                'pretrain': {**VGSLRecognitionTrainingDataConfig().__dict__, **VGSLPreTrainingConfig().__dict__},
                                                'rotrain': {**ROTrainingConfig().__dict__, **ROTrainingDataConfig().__dict__}}))
@click.version_option()
@click.pass_context
@click.option('-v', '--verbose', default=0, count=True)
@click.option('-d', '--device', show_default=True,
              help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--precision',
              type=click.Choice(PRECISIONS),
              help='Numerical precision to use for training. Default is 32-bit single-point precision.')
@click.option('--workers', 'num_workers', type=click.IntRange(0), help='Number of data loading worker processes.')
@click.option('--threads', 'num_threads', type=click.IntRange(1), help='Maximum size of OpenMP/BLAS thread pool.')
@click.option('-s', '--seed', default=None, type=click.INT,
              help='Seed for numpy\'s and torch\'s RNG. Set to a fixed value to '
                   'ensure reproducible random splits of data')
@click.option('-r', '--deterministic/--no-deterministic',
              help="Enables deterministic training. If no seed is given and enabled the seed will be set to 42.")
@click.option('--config',
              type=click.File(mode='r', lazy=True),
              help="Path to configuration file.",
              callback=_load_config,
              is_eager=True,
              expose_value=False,
              required=False)
def cli(ctx, **kwargs):
    params = ctx.params

    ctx.meta['deterministic'] = False if not params['deterministic'] else 'warn'
    if params['seed']:
        from lightning.pytorch import seed_everything
        seed_everything(params['seed'], workers=True)
    elif params['deterministic']:
        from lightning.pytorch import seed_everything
        seed_everything(42, workers=True)

    try:
        ctx.meta['accelerator'], ctx.meta['device'] = to_ptl_device(params['device'])
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    ctx.meta['verbose'] = params.get('verbose')
    ctx.meta['precision'] = params.get('precision')
    ctx.meta['num_workers'] = params.get('num_workers')
    ctx.meta['num_threads'] = params.get('num_threads')

    log.set_logger(logger, level=30 - min(10 * params['verbose'], 20))


for entry_point in sorted(importlib.metadata.entry_points(group='ketos.cli')):
    cli.add_command(entry_point.load(), name=entry_point.name)


if __name__ == '__main__':
    cli()
