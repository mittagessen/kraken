import os
import glob
import click
import logging

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
