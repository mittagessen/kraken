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
import logging

from PIL import Image
from rich.traceback import install

from kraken.lib import log

from .dataset import compile
from .linegen import line_generator
from .pretrain import pretrain
from .recognition import train, test
from .repo import publish
from .segmentation import segtrain, segtest
from .transcription import extract, transcription

APP_NAME = 'kraken'

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# install rich traceback handler
install(suppress=[click])

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


@click.group()
@click.version_option()
@click.pass_context
@click.option('-v', '--verbose', default=0, count=True)
@click.option('-s', '--seed', default=None, type=click.INT,
              help='Seed for numpy\'s and torch\'s RNG. Set to a fixed value to '
                   'ensure reproducible random splits of data')
@click.option('-r', '--deterministic/--no-deterministic', default=False,
              help="Enables deterministic training. If no seed is given and enabled the seed will be set to 42.")
def cli(ctx, verbose, seed, deterministic):
    ctx.meta['deterministic'] = deterministic
    if seed:
        from pytorch_lightning import seed_everything
        seed_everything(seed, workers=True)
    elif deterministic:
        from pytorch_lightning import seed_everything
        seed_everything(42, workers=True)

    ctx.meta['verbose'] = verbose
    log.set_logger(logger, level=30 - min(10 * verbose, 20))


cli.add_command(compile)
cli.add_command(pretrain)
cli.add_command(train)
cli.add_command(test)
cli.add_command(segtrain)
cli.add_command(segtest)
cli.add_command(publish)

# deprecated commands
cli.add_command(line_generator)
cli.add_command(extract)
cli.add_command(transcription)

if __name__ == '__main__':
    cli()
