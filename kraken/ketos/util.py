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
kraken.ketos.util
~~~~~~~~~~~~~~~~~~~~

Command line driver helpers
"""
import os
import glob
import click
import logging

from typing import List, Optional, Tuple

logging.captureWarnings(True)
logger = logging.getLogger('kraken')


def _validate_manifests(ctx, param, value):
    images = []
    for manifest in value:
        for entry in manifest.readlines():
            im_p = entry.rstrip('\r\n')
            if os.path.isfile(im_p):
                images.append(im_p)
            else:
                logger.warning('Invalid entry "{}" in {}'.format(im_p, manifest.name))
    return images


def _expand_gt(ctx, param, value):
    images = []
    for expression in value:
        images.extend([x for x in glob.iglob(expression, recursive=True) if os.path.isfile(x)])
    return images


def message(msg, **styles):
    if logger.getEffectiveLevel() >= 30:
        click.secho(msg, **styles)


def to_ptl_device(device: str) -> Tuple[str, Optional[List[int]]]:
    if any([device == x for x in ['cpu', 'mps']]):
        return device, None
    elif any([device.startswith(x) for x in ['tpu', 'cuda', 'hpu', 'ipu']]):
        dev, idx = device.split(':')
        if dev == 'cuda':
            dev = 'gpu'
        return dev, [int(idx)]
    raise Exception(f'Invalid device {device} specified')
