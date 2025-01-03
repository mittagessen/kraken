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
import re
import logging

import click

from pathlib import Path
from .util import message

logging.captureWarnings(True)
logger = logging.getLogger('kraken')


def _get_field_list(name):
    values = []
    while True:
        value = click.prompt(name, default=None)
        if value is not None:
            values.append(value)
        else:
            break
    return values


@click.command('publish')
@click.pass_context
@click.option('-i', '--metadata', show_default=True,
              type=click.File(mode='r', lazy=True), help='Model card file for the model.')
@click.option('-a', '--access-token', prompt=True, help='Zenodo access token')
@click.option('-d', '--doi', prompt=True, help='DOI of an existing record to update')
@click.option('-p', '--private/--public', default=False, help='Disables Zenodo '
              'community inclusion request. Allows upload of models that will not show '
              'up on `kraken list` output')
@click.argument('model', nargs=1, type=click.Path(exists=False, readable=True, dir_okay=False))
def publish(ctx, metadata, access_token, doi, private, model):
    """
    Publishes a model on the zenodo model repository.
    """
    import json
    import tempfile

    from htrmopo import publish_model, update_model

    pub_fn = publish_model

    from kraken.lib.vgsl import TorchVGSLModel
    from kraken.lib.progress import KrakenDownloadProgressBar

    _yaml_delim = r'(?:---|\+\+\+)'
    _yaml = r'(.*?)'
    _content = r'\s*(.+)$'
    _re_pattern = r'^\s*' + _yaml_delim + _yaml + _yaml_delim + _content
    _yaml_regex = re.compile(_re_pattern, re.S | re.M)

    nn = TorchVGSLModel.load_model(model)

    frontmatter = {}
    # construct metadata if none is given
    if metadata:
        frontmatter, content = _yaml_regex.match(metadata.read()).groups()
    else:
        frontmatter['summary'] = click.prompt('summary')
        content = click.edit('Write long form description (training data, transcription standards) of the model in markdown format here')

        creators = []
        while True:
            author = click.prompt('author', default=None)
            affiliation = click.prompt('affiliation', default=None)
            orcid = click.prompt('orcid', default=None)
            if author is not None:
                creators.append({'author': author})
            else:
                break
            if affiliation is not None:
                creators[-1]['affiliation'] = affiliation
            if orcid is not None:
                creators[-1]['orcid'] = orcid
        frontmatter['authors'] = creators
        frontmatter['license'] = click.prompt('license')
        frontmatter['language'] = _get_field_list('language')
        frontmatter['script'] = _get_field_list('script')

        if len(tags := _get_field_list('tag')):
            frontmatter['tags'] = tags + ['kraken_pytorch']
        if len(datasets := _get_field_list('dataset URL')):
            frontmatter['datasets'] = datasets
        if len(base_model := _get_field_list('base model URL')):
            frontmatter['base_model'] = base_model

    # take last metrics field, falling back to accuracy field in model metadata
    metrics = {}
    if 'metrics' in nn.user_metadata and nn.user_metadata['metrics']:
        metrics['cer'] = 100 - nn.user_metadata['metrics'][-1][1]['val_accuracy']
        metrics['wer'] = 100 - nn.user_metadata['metrics'][-1][1]['val_word_accuracy']
    elif 'accuracy' in nn.user_metadata and nn.user_metadata['accuracy']:
        metrics['cer'] = 100 - nn.user_metadata['accuracy']
    frontmatter['metrics'] = metrics
    software_hints = ['kind=vgsl']

    # some recognition-specific software hints
    if nn.model_type == 'recognition':
        software_hints.append([f'seg_type={nn.seg_type}', f'one_channel_mode={nn.one_channel_mode}', 'legacy_polygons={nn.user_metadata["legacy_polygons"]}'])
    frontmatter['software_hints'] = software_hints

    frontmatter['software_name'] = 'kraken'

    # build temporary directory
    with tempfile.TemporaryDirectory() as tmpdir, KrakenDownloadProgressBar() as progress:
        upload_task = progress.add_task('Uploading', total=0, visible=True if not ctx.meta['verbose'] else False)

        model = Path(model)
        tmpdir = Path(tmpdir)
        (tmpdir / model.name).symlink_to(model)
        # v0 metadata only supports recognition models
        if nn.model_type == 'recognition':
            v0_metadata = {
                'summary': frontmatter['summary'],
                'description': content,
                'license': frontmatter['license'],
                'script': frontmatter['script'],
                'name': model.name,
                'graphemes': [char for char in ''.join(nn.codec.c2l.keys())]
            }
            if frontmatter['metrics']:
                v0_metadata['accuracy'] = 100 - metrics['cer']
            with open(tmpdir / 'metadata.json', 'w') as fo:
                json.dump(v0_metadata, fo)
        kwargs = {'model': tmpdir,
                  'model_card': f'---\n{frontmatter}---\n{content}',
                  'access_token': access_token,
                  'callback': lambda total, advance: progress.update(upload_task, total=total, advance=advance),
                  'private': private}
        if doi:
            pub_fn = update_model
            kwargs['model_id'] = doi
        oid = pub_fn(**kwargs)
    message(f'model PID: {oid}')
