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

# -*- coding: utf-8 -*-
"""
Access functions to the model repository on github.
"""
from collections import defaultdict
from contextlib import closing

from kraken.lib.exceptions import KrakenRepoException

import base64
import requests
import json
import os
import logging

__all__ = ['get_model', 'get_description', 'get_listing']

logger = logging.getLogger(__name__)

MODEL_REPO = 'https://api.github.com/repos/mittagessen/kraken-models/'


def get_model(model_id, path, callback):
    """
    Retrieves a model and saves it to a path.

    Args:
        model_id (str): Identifier of the model
        path (str): Destination to write model to.
        callback (func): Function called for every 1024 octet chunk received.
    """
    logger.info(u'Saving model {} to {}'.format(model_id, path))
    logger.debug(u'Retrieving head of model repository')
    r = requests.get('{}{}'.format(MODEL_REPO, 'git/refs/heads/master'))
    callback()
    resp = r.json()
    if 'object' not in resp:
        logger.error(u'No \'object\' field in repo head API response.')
        raise KrakenRepoException('{}: {}'.format(r.status_code, resp['message']))
    head = resp['object']['sha']
    logger.debug(u'Retrieving tree of model repository')
    r = requests.get('{}{}{}'.format(MODEL_REPO, 'git/trees/', head), params={'recursive': 1})
    callback()
    resp = r.json()
    if 'tree' not in resp:
        logger.error(u'No \'tree\' field in repo API response.')
        raise KrakenRepoException('{}: {}'.format(r.status_code, resp['message']))
    url = None
    for el in resp['tree']:
        components = el['path'].split('/')
        if len(components) > 2 and components[1] == model_id and components[2] == 'DESCRIPTION':
            logger.debug(u'Retrieving description for {}'.format(components[1]))
            raw = base64.b64decode(requests.get(el['url']).json()['content']).decode('utf-8')
            desc = json.loads(raw)
            spath = os.path.join(path, desc['name'])
        elif len(components) > 2 and components[1] == model_id:
            url = el['url']
            break
    if not url:
        logger.error(u'Model {} not in repository.'.format(model_id))
        raise KrakenRepoException('Modle {} not in repository'.format(model_id))
    with closing(requests.get(url, headers={'Accept': 'application/vnd.github.v3.raw'},
                 stream=True)) as r:
        with open(spath, 'wb') as f:
            logger.debug(u'Downloading model')
            for chunk in r.iter_content(chunk_size=1024):
                callback()
                f.write(chunk)


def get_description(model_id):
    logger.info(u'Retrieving metadata for {}'.format(model_id))
    logger.debug(u'Retrieving head of model repository')
    r = requests.get('{}{}'.format(MODEL_REPO, 'git/refs/heads/master'))
    resp = r.json()
    if 'object' not in resp:
        logger.error(u'No \'object\' field in repo head API response.')
        raise KrakenRepoException('{}: {}'.format(r.status_code, resp['message']))
    head = resp['object']['sha']
    logger.debug(u'Retrieving tree of model repository')
    r = requests.get('{}{}{}'.format(MODEL_REPO, 'git/trees/', head), params={'recursive': 1})
    resp = r.json()
    if 'tree' not in resp:
        logger.error(u'No \'tree\' field in repo API response.')
        raise KrakenRepoException('{}: {}'.format(r.status_code, resp['message']))
    for el in resp['tree']:
        components = el['path'].split('/')
        if len(components) > 2 and components[1] == model_id and components[2] == 'DESCRIPTION':
            logger.debug(u'Retrieving description for {}'.format(components[1]))
            raw = base64.b64decode(requests.get(el['url']).json()['content']).decode('utf-8')
            return defaultdict(str, json.loads(raw))

def get_listing(callback):
    logger.info(u'Retrieving model list')
    r = requests.get('{}{}'.format(MODEL_REPO, 'git/refs/heads/master'))
    callback()
    resp = r.json()
    if 'object' not in resp:
        logger.error(u'No \'object\' field in repo head API response.')
        raise KrakenRepoException('{}: {}'.format(r.status_code, resp['message']))
    head = resp['object']['sha']
    logger.debug(u'Retrieving tree of model repository')
    r = requests.get('{}{}{}'.format(MODEL_REPO, 'git/trees/', head), params={'recursive': 1})
    callback()
    resp = r.json()
    if 'tree' not in resp:
        logger.error(u'No \'tree\' field in repo API response.')
        raise KrakenRepoException('{}: {}'.format(r.status_code, resp['message']))
    models = {}
    for el in resp['tree']:
        components = el['path'].split('/')
        # new model
        if len(components) == 2:
            models[components[1]] = {'type': components[0]}
        if len(components) > 2 and components[2] == 'DESCRIPTION':
            logger.debug(u'Retrieving description for {}'.format(components[1]))
            r = requests.get(el['url'])
            if not r.ok:
                logger.error(u'Requests to \'{}\' failed with status {}'.format(el['url'], r.status_code))
                raise KrakenRepoException('{}: {}'.format(r.status_code, r.json()['message']))
            raw = base64.b64decode(requests.get(el['url']).json()['content']).decode('utf-8')
            callback()
            try:
                models[components[1]].update(json.loads(raw))
            except Exception:
                del models[components[1]]
        elif len(components) > 2 and components[1] in models:
            models[components[1]]['model'] = el['url']
    return models
