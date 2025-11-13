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
kraken.ketos.weights
~~~~~~~~~~~~~~~~~~~~

Command line driver for checkpoint and weights handling.
"""
import click


@click.command('convert',
               epilog="""
This method can be used to convert checkpoints into weights:

ketos convert -o model.safetensors model.ckpt

or combine multiple related models into a single weights file:

ketos convert -o model.safetensors blla_line.ckpt blla_region.ckpt

It accepts weights and checkpoint interchangeably:

ketos convert -o model.safetensors model_1.ckpt model_2.safetensors

It can also convert certain models between coreml and safetensors serialization format:

ketos convert -o model.mlmodel --weights-format coreml model_1.safetensors
""")
@click.pass_context
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model.safetensors', help='Output weights file')
@click.option('--weights-format', default='safetensors', help='Output weights format.')
@click.argument('checkpoints', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def convert(ctx, output, weights_format, checkpoints):
    """
    Converts and combines one or more checkpoints/weights.
    """
    import importlib

    from kraken.models import load_models
    from kraken.lib.progress import KrakenProgressBar

    from .util import message

    try:
        (entry_point,) = importlib.metadata.entry_points(group='kraken.writers', name=weights_format)
        writer = entry_point.load()
    except ValueError:
        raise click.UsageError('weights_format', 'Unknown format `{weights_format}` for weights.')

    if not checkpoints:
        raise click.UsageError('No checkpoints to convert were provided.')

    def _find_module(path):
        for entry_point in importlib.metadata.entry_points(group='kraken.lightning_modules'):
            module = entry_point.load()
            try:
                return module.load_from_checkpoint(path)
            except ValueError:
                continue
        raise ValueError(f'No lightning module found for checkpoint {path}')

    models = []
    with KrakenProgressBar() as progress:
        load_task = progress.add_task('Loading checkpoints', total=len(checkpoints), start=True, visible=True if not ctx.meta['verbose'] else False)
        for ckpt in checkpoints:
            if ckpt.endswith('ckpt'):
                models.append(_find_module(ckpt).net)
            else:
                models.extend(load_models(ckpt))
        load_task.advance(load_task)

    opath = writer(models, output)
    message(f'Output file written to {opath}')
