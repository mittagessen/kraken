# -*- coding: utf-8 -*-
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
from collections import defaultdict
from typing import Callable, Any, BinaryIO
from contextlib import closing

from kraken.lib.exceptions import KrakenRepoException

import urllib
import requests
import json
import os
import logging

__all__ = ['get_model', 'get_description', 'get_listing']

logger = logging.getLogger(__name__)

MODEL_REPO = 'https://zenodo.org/api/'
SUPPORTED_MODELS = set(['kraken_pytorch'])

def publish_model(model_file: BinaryIO = None, metadata: dict = None, access_token: str = None, callback: Callable[..., Any] = lambda: None) -> str:
    """
    Publishes a model to the repository.

    Args:
        model_file (file): I/O stream to read model from.
        metadata (dict):
        access_token (str):
        callback (func): Function called for every 1024 octet chunk uploaded.
    """
    headers = {"Content-Type": "application/json"}
    r = requests.post('{}deposit/depositions'.format(MODEL_REPO),
                      params={'access_token': access_token}, json={},
                      headers=headers)
    r.raise_for_status()
    callback()
    deposition_id = r.json()['id']
    data = {'filename': 'metadata.json'}
    files = {'file': ('metadata.json', json.dumps(metadata))}
    r = requests.post('{}deposit/depositions/{}/files'.format(MODEL_REPO, deposition_id),
                      params={'access_token': access_token}, data=data,
                      files=files)
    r.raise_for_status()
    callback()
    data = {'filename': metadata['name']}
    files = {'file': open(model_file, 'rb')}
    r = requests.post('{}deposit/depositions/{}/files'.format(MODEL_REPO, deposition_id),
                      params={'access_token': access_token}, data=data,
                      files=files)
    r.raise_for_status()
    callback()
    # fill zenodo metadata
    data = {'metadata': {
                        'title': metadata['summary'],
                        'upload_type': 'publication',
                        'publication_type': 'other',
                        'description': metadata['description'],
                        'creators': metadata['authors'],
                        'access_right': 'open',
                        'communities': [{'identifier': 'ocr_models'}],
                        'keywords': ['kraken_pytorch'],
                        'license': metadata['license'],
                        }
           }
    # add link to training data to metadata
    if 'source' in metadata:
        data['metadata']['related_identifiers'] = [{'relation': 'isSupplementTo', 'identifier': metadata['source']}]
    r = requests.put('{}deposit/depositions/{}'.format(MODEL_REPO, deposition_id),
                     params={'access_token': access_token},
                     data=json.dumps(data),
                     headers=headers)
    r.raise_for_status()
    callback()
    r = requests.post('{}deposit/depositions/{}/actions/publish'.format(MODEL_REPO, deposition_id),
                      params={'access_token': access_token})
    r.raise_for_status()
    callback()
    return r.json()['doi']

def get_model(model_id: str, path: str, callback: Callable[..., Any] = lambda: None) -> str:
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
    logger.info('Saving model {} to {}'.format(model_id, path))
    r = requests.get('{}{}'.format(MODEL_REPO, 'records'), params={'q': 'doi:"{}"'.format(model_id)})
    r.raise_for_status()
    callback()
    resp = r.json()
    if  resp['hits']['total'] != 1:
        logger.error('Found {} models when querying for id \'{}\''.format(resp['hits']['total'], model_id))
        raise KrakenRepoException('Found {} models when querying for id \'{}\''.format(resp['hits']['total'], model_id))

    metadata = resp['hits']['hits'][0]
    model_url = [x['links']['self'] for x in metadata['files'] if x['type'] == 'mlmodel'][0]
    # callable model identifier 
    nat_id = os.path.basename(urllib.parse.urlparse(model_url).path)
    spath = os.path.join(path, nat_id)
    logger.debug('downloading model file {} to {}'.format(model_url, spath))
    with closing(requests.get(model_url, stream=True)) as r:
        with open(spath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                callback()
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
    logger.info('Retrieving metadata for {}'.format(model_id))
    r = requests.get('{}{}'.format(MODEL_REPO, 'records'), params={'q': 'doi:"{}"'.format(model_id)})
    r.raise_for_status()
    callback()
    resp = r.json()
    if  resp['hits']['total'] != 1:
        logger.error('Found {} models when querying for id \'{}\''.format(resp['hits']['total'], model_id))
        raise KrakenRepoException('Found {} models when querying for id \'{}\''.format(model_id))
    record = resp['hits']['hits'][0]
    metadata = record['metadata']
    if 'keywords' not in metadata:
        logger.error('No keywords included on deposit')
        raise KrakenRepoException('No keywords included on deposit.')
    model_type = SUPPORTED_MODELS.intersection(metadata['keywords'])
    if not model_type:
        msg = 'Unsupported model type(s): {}'.format(', '.format(metadata['keywords']))
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
            except:
                msg = 'Metadata for \'{}\' ({}) not in JSON format'.format(record['metadata']['title'], record['metadata']['doi'])
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


def get_listing(callback: Callable[..., Any] = lambda: None) -> dict:
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
    callback()
    resp = r.json()
    if not resp['hits']['total']:
        logger.error('No models found in community \'ocr_models\'')
        raise KrakenRepoException('No models found in repository \'ocr_models\'')
    logger.debug('Total of {} records in repository'.format(resp['hits']['total']))
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
                callback()
                r = requests.get(file['links']['self'])
                r.raise_for_status()
                try:
                    metadata = r.json()
                except:
                    msg = 'Metadata for \'{}\' ({}) not in JSON format'.format(record['metadata']['title'], record['metadata']['doi'])
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
