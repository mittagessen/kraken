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
import yaml
import shlex
import logging
import difflib
from collections import defaultdict
from typing import Optional, Any, TYPE_CHECKING

import click

if TYPE_CHECKING:
    from os import PathLike

logging.captureWarnings(True)
logger = logging.getLogger('kraken')


def _create_class_map(cls_map):
    """
    Converts the list as a parameter
    """
    default = None
    for idx, (cls, label) in enumerate(cls_map):
        if '*' in cls:
            def default():  # NOQA
                return label
            cls_map.pop(idx)
            break
    return defaultdict(default, cls_map)


def _recursive_update(a: dict[str, Any],
                      b: dict[str, Any],
                      cmd: Optional[click.BaseCommand] = None) -> dict[str, Any]:
    """Like standard ``dict.update()``, but recursive so sub-dict gets updated.

    Warns on keys present in ``b`` but not in ``a`` and not valid option names
    for the command ``cmd`` and suggests alternatives.
    """
    valid_keys = set(a.keys())
    if cmd is not None:
        for param in cmd.params:
            if param.name:
                valid_keys.add(param.name)
        if isinstance(cmd, click.MultiCommand):
            valid_keys.update(cmd.list_commands(None))
    for k, v in b.items():
        if k not in valid_keys:
            matches = difflib.get_close_matches(k, valid_keys)
            msg = f'Ignoring unknown configuration key "{k}" in experiment file.'
            if matches:
                msg += f' Did you mean "{matches[0]}"?'
            logger.warning(msg)
        subcmd = None
        if cmd is not None and isinstance(cmd, click.MultiCommand):
            subcmd = cmd.get_command(None, k)
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = _recursive_update(a[k], v, subcmd)
        elif isinstance(v, dict) and subcmd is not None:
            a[k] = _recursive_update({}, v, subcmd)
        else:
            a[k] = v
    return a


def _load_config(ctx: click.Context,
                 param: click.Parameter,
                 path: 'PathLike') -> None:
    """
    Fetch parameters values from configuration file and sets them as defaults.
    """
    logger.info(f"Load configuration matching {path}")
    if path:
        try:
            conf = yaml.safe_load(path)
            # keep the raw config so commands can tell experiment file values
            # from seeded defaults (both are ParameterSource.DEFAULT_MAP).
            ctx.meta['ketos_user_config'] = conf or {}
            # Update the default_map.
            if ctx.default_map is None:
                ctx.default_map = {}
            ctx.default_map = _recursive_update(ctx.default_map, conf, ctx.command)
        except FileNotFoundError:
            logger.critical(f"No configuration file {path} found.")


def _user_supplied_params(ctx: click.Context) -> dict[str, Any]:
    """
    Returns the subset of ``ctx.params`` the user set explicitly, on the
    command line/environment or in the subcommand's section of the `--config`
    file. Options left at their defaults are excluded so per-architecture
    config classes can apply their own defaults.
    """
    from click.core import ParameterSource

    explicit_sources = (ParameterSource.COMMANDLINE,
                        ParameterSource.ENVIRONMENT,
                        ParameterSource.PROMPT)
    yaml_keys = set((ctx.meta.get('ketos_user_config') or {}).get(ctx.info_name) or {})
    explicit = {}
    for name, value in ctx.params.items():
        source = ctx.get_parameter_source(name)
        if source in explicit_sources or (source is ParameterSource.DEFAULT_MAP and name in yaml_keys):
            explicit[name] = value
    return explicit


def _arch_names(task: str) -> list[str]:
    """
    Lists the architecture names registered for `task` in the
    ``kraken.<task>_archs`` entry point group without importing them, for use
    in a ``click.Choice`` at module load time.
    """
    import importlib.metadata
    return sorted(ep.name for ep in importlib.metadata.entry_points(group=f'kraken.{task}_archs'))


def _resolve_arch(task: str, arch: str) -> type:
    """
    Loads the trainer module class registered as `arch` for `task` in the
    ``kraken.<task>_archs`` entry point group.
    """
    import importlib.metadata
    eps = importlib.metadata.entry_points(group=f'kraken.{task}_archs', name=arch)
    if not eps:
        raise click.BadParameter(f'Unknown {task} architecture {arch!r}. Available: '
                                 f'{", ".join(_arch_names(task)) or "(none)"}',
                                 param_hint='arch')
    return tuple(eps)[0].load()


def _resolve_module_class(ctx: click.Context,
                          explicit: dict,
                          task: str,
                          artifact=None) -> type:
    """
    Resolves the trainer module class for `task`, using `--arch` or, when an
    `artifact` (checkpoint or weights file) is given, the architecture detected
    in it. `--arch` is thus only required when training from scratch. Raises if
    an explicit `--arch` conflicts with `artifact` or `artifact` is not a
    trainable model for `task`.
    """
    arch = ctx.params['arch']
    arch_explicit = 'arch' in explicit

    if artifact and str(artifact).endswith('.ckpt'):
        from kraken.models.convert import find_checkpoint_module
        try:
            module_cls = find_checkpoint_module(artifact)
        except Exception as e:
            if arch_explicit:
                logger.warning(f'Could not determine architecture of {artifact} ({e}); '
                               f'trusting --arch {arch}.')
                return _resolve_arch(task, arch)
            raise click.UsageError(f'Could not determine the model architecture of {artifact} '
                                   f'({e}). Pass --arch explicitly.')
        detected = getattr(module_cls, '_arch', None)
        if detected is None or getattr(module_cls, '_task', None) != task:
            raise click.UsageError(f'{artifact} is a {module_cls.__name__} checkpoint, not a '
                                   f'trainable {task} model.')
        if arch_explicit and arch != detected:
            raise click.BadOptionUsage('arch', f'--arch {arch} conflicts with the {detected!r} '
                                               f'model found in {artifact}.')
        return module_cls
    elif artifact:
        from kraken.models.convert import find_weights_archs
        detected = find_weights_archs(artifact, task=task)
        if detected is None:
            logger.info(f'Could not determine architecture from {artifact}; assuming --arch {arch}.')
        elif arch_explicit:
            if arch not in detected:
                raise click.BadOptionUsage('arch', f'--arch {arch} conflicts with the '
                                                   f'{sorted(detected)} model(s) in {artifact}.')
        elif len(detected) == 1:
            arch = next(iter(detected))
        else:
            raise click.UsageError(f'{artifact} contains models of multiple architectures '
                                   f'{sorted(detected)}. Pass --arch to select one.')
    return _resolve_arch(task, arch)


def _validate_merging(ctx, param, value):
    """
    Maps baseline/region merging to a dict of merge structures.
    """
    if not value:
        return None
    merge_dict: dict[str, str] = {}
    try:
        for m in value:
            lexer = shlex.shlex(m, posix=True)
            lexer.wordchars += r'\/.+-()=^&;,.$'
            tokens = list(lexer)
            if len(tokens) != 3:
                raise ValueError
            k, _, v = tokens
            merge_dict[v] = k  # type: ignore
    except Exception:
        raise click.BadParameter('Mappings must be in format target:src')
    return merge_dict


def _validate_manifests(ctx, param, value):
    images = []
    if value is None:
        return None
    for manifest in value:
        try:
            for entry in manifest.readlines():
                im_p = entry.rstrip('\r\n')
                if os.path.isfile(im_p):
                    images.append(im_p)
                else:
                    logger.warning('Invalid entry "{}" in {}'.format(im_p, manifest.name))
        except UnicodeDecodeError:
            raise click.BadOptionUsage(param,
                                       f'File {manifest.name} is not a text file. Please '
                                       'ensure that the argument to `-t`/`-e` is a manifest '
                                       'file containing paths to training data (one per '
                                       'line).',
                                       ctx=ctx)
    return images


def _expand_gt(ctx, param, value):
    images = []
    for expression in value:
        images.extend([x for x in glob.iglob(expression, recursive=True) if os.path.isfile(x)])
    return images


def message(msg, **styles):
    if logger.getEffectiveLevel() >= 30:
        click.secho(msg, **styles)


def to_ptl_device(device: str) -> tuple[str, Optional[list[int]]]:
    if device.strip() == 'auto':
        return 'auto', 'auto'
    devices = device.split(',')
    if devices[0] in ['cpu', 'mps']:
        return devices[0], 'auto'
    elif any([devices[0].startswith(x) for x in ['tpu', 'cuda', 'hpu', 'ipu']]):
        devices = [device.split(':') for device in devices]
        devices = [(x[0].strip(), x[1].strip()) for x in devices]
        if len(set(x[0] for x in devices)) > 1:
            raise Exception('Can only use a single type of device at a time.')
        dev, _ = devices[0]
        if dev == 'cuda':
            dev = 'gpu'
        return dev, [int(x[1]) for x in devices]
    raise Exception(f'Invalid device {device} specified')
