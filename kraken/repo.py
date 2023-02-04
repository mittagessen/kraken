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
Accessors to the model repository on zenodo.
"""
import os
import json
import urllib
import logging
import requests

from os import PathLike
from pathlib import Path
from contextlib import closing
from typing import Callable, Any

from kraken.lib.exceptions import KrakenRepoException

__all__ = ['get_model', 'get_description', 'get_listing', 'publish_model']

logger = logging.getLogger(__name__)

MODEL_REPO = 'https://zenodo.org/api/'
SUPPORTED_MODELS = set(['kraken_pytorch'])


def publish_model(model_file: [str, PathLike] = None,
                  metadata: dict = None,
                  access_token: str = None,
                  callback: Callable[[int, int], Any] = lambda: None,
                  private: bool = False) -> str:
    """
    Publishes a model to the repository.

    Args:
        model_file: Path to read model from.
        metadata: Metadata dictionary
        access_token: Zenodo API access token
        callback: Function called with octet-wise progress.
        private: Whether to generate a community inclusion request that makes
                 the model recoverable by the public.
    """
    model_file = Path(model_file)
    fp = open(model_file, 'rb')
    _metadata = json.dumps(metadata)
    total = model_file.stat().st_size + len(_metadata) + 3
    headers = {"Content-Type": "application/json"}
    r = requests.post(f'{MODEL_REPO}deposit/depositions',
                      params={'access_token': access_token}, json={},
                      headers=headers)
    r.raise_for_status()
    callback(total, 1)
    deposition_id = r.json()['id']
    data = {'filename': 'metadata.json'}
    files = {'file': ('metadata.json', _metadata)}
    r = requests.post(f'{MODEL_REPO}deposit/depositions/{deposition_id}/files',
                      params={'access_token': access_token}, data=data,
                      files=files)
    r.raise_for_status()
    callback(total, len(_metadata))
    data = {'filename': metadata['name']}
    files = {'file': fp}
    r = requests.post(f'{MODEL_REPO}deposit/depositions/{deposition_id}/files',
                      params={'access_token': access_token}, data=data,
                      files=files)
    r.raise_for_status()
    callback(total, model_file.stat().st_size)
    # fill zenodo metadata
    data = {'metadata': {
                        'title': metadata['summary'],
                        'upload_type': 'publication',
                        'publication_type': 'other',
                        'description': metadata['description'],
                        'creators': metadata['authors'],
                        'access_right': 'open',
                        'keywords': ['kraken_pytorch'],
                        'license': metadata['license']
                        }
            }

    if not private:
        data['metadata']['communities'] = [{'identifier': 'ocr_models'}]

    # add link to training data to metadata
    if 'source' in metadata:
        data['metadata']['related_identifiers'] = [{'relation': 'isSupplementTo', 'identifier': metadata['source']}]
    r = requests.put(f'{MODEL_REPO}deposit/depositions/{deposition_id}',
                     params={'access_token': access_token},
                     data=json.dumps(data),
                     headers=headers)
    r.raise_for_status()
    callback(total, 1)
    r = requests.post(f'{MODEL_REPO}deposit/depositions/{deposition_id}/actions/publish',
                      params={'access_token': access_token})
    r.raise_for_status()
    callback(total, 1)
    return r.json()['doi']


def get_model(model_id: str, path: str, callback: Callable[[int, int], Any] = lambda total, advance: None) -> str:
    """
    Retrieves a model and saves it to a path.

    Args:
        model_id (str): DOI of the model
        path (str): Destination to write model to.
        callback (func): Function called for every 1024 octet chunk received.

    Returns:
        The identifier the model can be called through on the command line.
        Will usually be the file name of the model.
    """
    logger.info(f'Saving model {model_id} to {path}')
    r = requests.get(f'{MODEL_REPO}records', params={'q': f'doi:"{model_id}"'})
    r.raise_for_status()
    callback(0, 0)
    resp = r.json()
    if resp['hits']['total'] != 1:
        logger.error(f'Found {resp["hits"]["total"]} models when querying for id \'{model_id}\'')
        raise KrakenRepoException(f'Found {resp["hits"]["total"]} models when querying for id \'{model_id}\'')

    metadata = resp['hits']['hits'][0]
    model_url = [x['links']['self'] for x in metadata['files'] if x['type'] == 'mlmodel'][0]
    # callable model identifier
    nat_id = os.path.basename(urllib.parse.urlparse(model_url).path)
    spath = os.path.join(path, nat_id)
    logger.debug(f'downloading model file {model_url} to {spath}')
    with closing(requests.get(model_url, stream=True)) as r:
        file_size = int(r.headers['Content-length'])
        with open(spath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                callback(file_size, len(chunk))
                f.write(chunk)
    return nat_id


def get_description(model_id: str, callback: Callable[..., Any] = lambda: None) -> dict:
    """
    Fetches the metadata for a single model from the zenodo repository.

    Args:
        model_id (str): DOI of the model.
        callback (callable): Optional function called once per HTTP request.

    Returns:
        Dict
    """
    logger.info(f'Retrieving metadata for {model_id}')
    r = requests.get(f'{MODEL_REPO}records', params={'q': f'doi:"{model_id}"'})
    r.raise_for_status()
    callback()
    resp = r.json()
    if resp['hits']['total'] != 1:
        logger.error(f'Found {resp["hits"]["total"]} models when querying for id \'{model_id}\'')
        raise KrakenRepoException(f'Found {resp["hits"]["total"]} models when querying for id \'{model_id}\'')
    record = resp['hits']['hits'][0]
    metadata = record['metadata']
    if 'keywords' not in metadata:
        logger.error('No keywords included on deposit')
        raise KrakenRepoException('No keywords included on deposit.')
    model_type = SUPPORTED_MODELS.intersection(metadata['keywords'])
    if not model_type:
        msg = 'Unsupported model type(s): {}'.format(', '.join(metadata['keywords']))
        logger.error(msg)
        raise KrakenRepoException(msg)
    meta_json = None
    for file in record['files']:
        if file['key'] == 'metadata.json':
            callback()
            r = requests.get(file['links']['self'])
            r.raise_for_status()
            callback()
            try:
                meta_json = r.json()
            except Exception:
                msg = f'Metadata for \'{record["metadata"]["title"]}\' ({record["metadata"]["doi"]}) not in JSON format'
                logger.error(msg)
                raise KrakenRepoException(msg)
    if not meta_json:
        msg = 'Mo metadata.jsn found for \'{}\' ({})'.format(record['metadata']['title'], record['metadata']['doi'])
        logger.error(msg)
        raise KrakenRepoException(msg)
    # merge metadata.json into DataCite
    metadata.update({'graphemes': meta_json['graphemes'],
                     'summary': meta_json['summary'],
                     'script': meta_json['script'],
                     'link': record['links']['latest'],
                     'type': [x.split('_')[1] for x in model_type],
                     'accuracy': meta_json['accuracy']})
    return metadata


def get_listing(callback: Callable[[int, int], Any] = lambda total, advance: None) -> dict:
    """
    Fetches a listing of all kraken models from the zenodo repository.

    Args:
        callback (Callable): Function called after each HTTP request.

    Returns:
        Dict of models with each model.
    """
    logger.info('Retrieving model list')
    records = []
    r = requests.get('{}{}'.format(MODEL_REPO, 'records'), params={'communities': 'ocr_models'})
    r.raise_for_status()
    callback(1, 1)
    resp = r.json()
    if not resp['hits']['total']:
        logger.error('No models found in community \'ocr_models\'')
        raise KrakenRepoException('No models found in repository \'ocr_models\'')
    logger.debug('Total of {} records in repository'.format(resp['hits']['total']))
    total = resp['hits']['total']
    callback(total, 0)
    records.extend(resp['hits']['hits'])
    while 'next' in resp['links']:
        logger.debug('Fetching next page')
        r = requests.get(resp['links']['next'])
        r.raise_for_status()
        resp = r.json()
        logger.debug('Found {} new records'.format(len(resp['hits']['hits'])))
        records.extend(resp['hits']['hits'])
    logger.debug('Retrieving model metadata')
    models = {}
    # fetch metadata.jsn for each model
    for record in records:
        if 'keywords' not in record['metadata']:
            continue
        model_type = SUPPORTED_MODELS.intersection(record['metadata']['keywords'])
        if not model_type:
            continue
        for file in record['files']:
            if file['key'] == 'metadata.json':
                callback(total, 1)
                r = requests.get(file['links']['self'])
                r.raise_for_status()
                try:
                    metadata = r.json()
                except Exception:
                    msg = f'Metadata for \'{record["metadata"]["title"]}\' ({record["metadata"]["doi"]}) not in JSON format'
                    logger.error(msg)
                    raise KrakenRepoException(msg)
        # merge metadata.jsn into DataCite
        key = record['metadata']['doi']
        models[key] = record['metadata']
        models[key].update({'graphemes': metadata['graphemes'],
                            'summary': metadata['summary'],
                            'script': metadata['script'],
                            'link': record['links']['latest'],
                            'type': [x.split('_')[1] for x in model_type]})
    return models
