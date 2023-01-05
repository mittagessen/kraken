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
kraken.ketos.repo
~~~~~~~~~~~~~~~~~

Command line driver for publishing models to the model repository.
"""
import os
import click
import logging

from kraken.lib.progress import KrakenDownloadProgressBar

from .util import message

logging.captureWarnings(True)
logger = logging.getLogger('kraken')


@click.command('publish')
@click.pass_context
@click.option('-i', '--metadata', show_default=True,
              type=click.File(mode='r', lazy=True), help='Metadata for the '
              'model. Will be prompted from the user if not given')
@click.option('-a', '--access-token', prompt=True, help='Zenodo access token')
@click.option('-p', '--private/--public', default=False, help='Disables Zenodo '
              'community inclusion request. Allows upload of models that will not show '
              'up on `kraken list` output')
@click.argument('model', nargs=1, type=click.Path(exists=False, readable=True, dir_okay=False))
def publish(ctx, metadata, access_token, private, model):
    """
    Publishes a model on the zenodo model repository.
    """
    import json
    import pkg_resources

    from jsonschema import validate
    from jsonschema.exceptions import ValidationError

    from kraken import repo
    from kraken.lib import models

    with pkg_resources.resource_stream('kraken', 'metadata.schema.json') as fp:
        schema = json.load(fp)

    nn = models.load_any(model)

    if not metadata:
        author = click.prompt('author')
        affiliation = click.prompt('affiliation')
        summary = click.prompt('summary')
        description = click.edit('Write long form description (training data, transcription standards) of the model here')
        accuracy_default = None
        # take last accuracy measurement in model metadata
        if 'accuracy' in nn.nn.user_metadata and nn.nn.user_metadata['accuracy']:
            accuracy_default = nn.nn.user_metadata['accuracy'][-1][1] * 100
        accuracy = click.prompt('accuracy on test set', type=float, default=accuracy_default)
        script = [
            click.prompt(
                'script',
                type=click.Choice(
                    sorted(
                        schema['properties']['script']['items']['enum'])),
                show_choices=True)]
        license = click.prompt(
            'license',
            type=click.Choice(
                sorted(
                    schema['properties']['license']['enum'])),
            show_choices=True)
        metadata = {
            'authors': [{'name': author, 'affiliation': affiliation}],
            'summary': summary,
            'description': description,
            'accuracy': accuracy,
            'license': license,
            'script': script,
            'name': os.path.basename(model),
            'graphemes': ['a']
        }
        while True:
            try:
                validate(metadata, schema)
            except ValidationError as e:
                message(e.message)
                metadata[e.path[-1]] = click.prompt(e.path[-1], type=float if e.schema['type'] == 'number' else str)
                continue
            break

    else:
        metadata = json.load(metadata)
        validate(metadata, schema)
    metadata['graphemes'] = [char for char in ''.join(nn.codec.c2l.keys())]
    with KrakenDownloadProgressBar() as progress:
        upload_task = progress.add_task('Uploading', total=0, visible=True if not ctx.meta['verbose'] else False)
        oid = repo.publish_model(model, metadata, access_token, lambda total, advance: progress.update(upload_task, total=total, advance=advance), private)
    message('model PID: {}'.format(oid))
