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
from difflib import get_close_matches

from .util import message

logging.captureWarnings(True)
logger = logging.getLogger('kraken')


def _validate_script(script: str) -> str:
    from htrmopo.util import _iso15924
    if script not in _iso15924:
        return get_close_matches(script, _iso15924.keys())
    return script


def _validate_language(language: str) -> str:
    from htrmopo.util import _iso639_3
    if language not in _iso639_3:
        return get_close_matches(language, _iso639_3.keys())
    return language


def _validate_license(license: str) -> str:
    from htrmopo.util import _licenses
    if license not in _licenses:
        return get_close_matches(license, _licenses.keys())
    return license


def _get_field_list(name,
                    validation_fn=lambda x: x,
                    required: bool = False):
    values = []
    while True:
        value = click.prompt(name, default='')
        if value:
            if (cand := validation_fn(value)) == value:
                values.append(value)
            else:
                message(f'Not a valid {name} value. Did you mean {cand}?')
        else:
            if click.confirm(f'All `{name}` values added?'):
                if required and not values:
                    message(f'`{name}` is a required field.')
                    continue
                else:
                    break
            else:
                continue
    return values


def _get_serialization_kind(path: Path) -> str:
    """
    Returns a software hint value for the model serialization format.
    """
    suffix = path.suffix.lower()
    if suffix == '.mlmodel':
        return 'coreml'
    if suffix == '.safetensors':
        return 'safetensors'
    if suffix:
        return suffix[1:]
    return 'unknown'


def _get_segmentation_model_hint(class_mapping: dict) -> str | None:
    """
    Returns a segmentation subtype hint from class mapping metadata.
    """
    has_lines = bool(class_mapping.get('baselines', {}))
    has_regions = bool(class_mapping.get('regions', {}))
    if has_lines and has_regions:
        return 'combined'
    if has_lines:
        return 'line'
    if has_regions:
        return 'region'
    return None


@click.command('publish')
@click.pass_context
@click.option('-i', '--metadata', show_default=True,
              type=click.File(mode='r', lazy=True), help='Model card file for the model.')
@click.option('-a', '--access-token', prompt=True, help='Zenodo access token')
@click.option('-d', '--doi', help='DOI of an existing record to update')
@click.option('-p', '--private/--public', default=False, help='Disables Zenodo '
              'community inclusion request. Allows upload of models that will not show '
              'up on `kraken list` output')
@click.argument('model', nargs=1, type=click.Path(exists=False, readable=True, dir_okay=False))
def publish(ctx, metadata, access_token, doi, private, model):
    """
    Publishes a model on the zenodo model repository.
    """
    import json
    import yaml
    import tempfile

    from htrmopo import publish_model, update_model

    from kraken.lib.progress import KrakenDownloadProgressBar
    from kraken.models import load_models

    pub_fn = publish_model
    model_path = Path(model)

    _yaml_delim = r'(?:---|\+\+\+)'
    _yaml = r'(.*?)'
    _content = r'\s*(.+)$'
    _re_pattern = r'^\s*' + _yaml_delim + _yaml + _yaml_delim + _content
    _yaml_regex = re.compile(_re_pattern, re.S | re.M)

    models = load_models(model_path)
    if not models:
        raise click.UsageError(f'No models found in {model_path}.')

    rec_models = [m for m in models if 'recognition' in m.model_type]
    seg_models = [m for m in models if 'segmentation' in m.model_type]
    ro_models = [m for m in models if 'reading_order' in m.model_type]

    # build deduplicated combined model type
    combined_model_type = []
    for m in models:
        for t in (m.model_type if isinstance(m.model_type, list) else [m.model_type]):
            if t not in combined_model_type:
                combined_model_type.append(t)

    frontmatter = {}
    # construct metadata if none is given
    if metadata:
        frontmatter, content = _yaml_regex.match(metadata.read()).groups()
        frontmatter = yaml.safe_load(frontmatter)
    else:
        frontmatter['summary'] = click.prompt('summary')
        content = click.edit('Write long form description (training data, transcription standards) of the model in markdown format here')

        creators = []
        message('To stop adding authors, leave the author name field empty.')
        while True:
            author = click.prompt('author name', default='')
            if author:
                creators.append({'name': author})
            else:
                if click.confirm('All authors added?'):
                    break
                else:
                    continue
            affiliation = click.prompt('affiliation', default='')
            orcid = click.prompt('orcid', default='')
            if affiliation is not None:
                creators[-1]['affiliation'] = affiliation
            if orcid is not None:
                creators[-1]['orcid'] = orcid
        if not creators:
            raise click.UsageError('The `authors` field is obligatory. Aborting')

        frontmatter['authors'] = creators
        while True:
            license = click.prompt('license')
            if (lic := _validate_license(license)) == license:
                frontmatter['license'] = license
                break
            else:
                message(f'Not a valid license identifier. Did you mean {lic}?')

        message('To stop adding values to the following fields, enter an empty field.')

        frontmatter['language'] = _get_field_list('language', _validate_language, required=True)
        frontmatter['script'] = _get_field_list('script', _validate_script, required=True)

        if len(tags := _get_field_list('tag')):
            frontmatter['tags'] = tags + ['kraken_pytorch']
        if len(datasets := _get_field_list('dataset URL')):
            frontmatter['datasets'] = datasets
        if len(base_model := _get_field_list('base model URL')):
            frontmatter['base_model'] = base_model

    software_hints = [f'kind={_get_serialization_kind(model_path)}']

    # version hint from the model requiring the highest version
    from packaging.version import Version
    versions = [getattr(m, '_kraken_min_version', None) for m in models]
    versions = [v for v in versions if v is not None]
    if versions:
        software_hints.append(f'version>={max(versions, key=Version)}')

    # recognition-specific hints and metrics
    if rec_models:
        nn_rec = rec_models[0]
        metrics = {}
        if len(nn_rec.user_metadata.get('metrics', '')):
            if (val_accuracy := nn_rec.user_metadata['metrics'][-1][1].get('val_accuracy', None)) is not None:
                metrics['cer'] = 100 - (val_accuracy * 100)
            if (val_word_accuracy := nn_rec.user_metadata['metrics'][-1][1].get('val_word_accuracy', None)) is not None:
                metrics['wer'] = 100 - (val_word_accuracy * 100)
        elif (accuracy := nn_rec.user_metadata.get('accuracy', None)) is not None:
            metrics['cer'] = 100 - accuracy
        frontmatter['metrics'] = metrics
        software_hints.extend([f'seg_type={nn_rec.seg_type}', f'one_channel_mode={nn_rec.one_channel_mode}', f'legacy_polygons={nn_rec.use_legacy_polygons}'])

    # segmentation-specific hints
    if seg_models:
        if seg_hint := _get_segmentation_model_hint(seg_models[0].user_metadata.get('class_mapping', {})):
            software_hints.append(f'segmentation={seg_hint}')

    # reading order hints
    for nn_ro in ro_models:
        ro_level = nn_ro.user_metadata.get('level', None)
        if ro_level:
            software_hints.append(f'ro_level={ro_level}')

    frontmatter['software_hints'] = software_hints

    frontmatter['software_name'] = 'kraken'
    frontmatter['model_type'] = combined_model_type

    # build temporary directory
    with tempfile.TemporaryDirectory() as tmpdir, KrakenDownloadProgressBar() as progress:
        upload_task = progress.add_task('Uploading', total=0, visible=True if not ctx.meta['verbose'] else False)

        model_path = model_path.resolve()
        tmpdir = Path(tmpdir)
        (tmpdir / model_path.name).resolve().symlink_to(model_path)
        if rec_models:
            # v0 metadata only supports recognition models
            nn_rec = rec_models[0]
            v0_metadata = {
                'summary': frontmatter['summary'],
                'description': content,
                'license': frontmatter['license'],
                'script': frontmatter['script'],
                'name': model_path.name,
                'graphemes': [char for char in ''.join(nn_rec.codec.c2l.keys())]
            }
            if frontmatter.get('metrics'):
                v0_metadata['accuracy'] = 100 - frontmatter['metrics']['cer']
            with open(tmpdir / 'metadata.json', 'w') as fo:
                json.dump(v0_metadata, fo)
        kwargs = {'model': tmpdir,
                  'model_card': f'---\n{yaml.dump(frontmatter)}---\n{content}',
                  'access_token': access_token,
                  'callback': lambda total, advance: progress.update(upload_task, total=total, advance=advance),
                  'private': private}
        if doi:
            pub_fn = update_model
            kwargs['model_id'] = doi
        oid = pub_fn(**kwargs)
    message(f'model PID: {oid}')
