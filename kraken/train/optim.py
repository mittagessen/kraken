#
# Copyright 2026
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
kraken.train.optim
~~~~~~~~~~~~~~~~~~~~

A Muon + AdamW hybrid optimizer: Muon optimises the hidden 2-D linear weight
matrices, AdamW handles everything else.

The wrapper is adapted from Keller Jordan's reference Muon implementation
(https://github.com/KellerJordan/Muon).
"""
import logging
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)

__all__ = ['MuonWithAuxAdam', 'get_parameter_groups']


class MuonWithAuxAdam(Optimizer):
    """
    Lightning-compatible wrapper applying ``torch.optim.Muon`` to the parameter
    groups flagged ``use_muon=True`` and ``torch.optim.AdamW`` to the rest. The
    Muon groups must come first in ``param_groups``.
    """

    def __init__(self, param_groups, *, lr, weight_decay, momentum):
        normalized_groups = []
        muon_group_count = 0
        for group in param_groups:
            group = dict(group)
            if 'use_muon' not in group:
                raise ValueError('MuonWithAuxAdam parameter groups must define a `use_muon` flag.')
            group.setdefault('lr', lr)
            group.setdefault('weight_decay', weight_decay)
            if group['use_muon']:
                group.setdefault('momentum', momentum)
                muon_group_count += 1
            normalized_groups.append(group)

        super().__init__(normalized_groups, {})
        self._muon_group_count = muon_group_count
        self._muon_optimizer = None
        self._adamw_optimizer = None

        muon_groups = self.param_groups[:self._muon_group_count]
        adamw_groups = self.param_groups[self._muon_group_count:]
        if muon_groups:
            self._muon_optimizer = torch.optim.Muon(muon_groups, adjust_lr_fn='match_rms_adamw')
        if adamw_groups:
            self._adamw_optimizer = torch.optim.AdamW(adamw_groups)

        # Lightning moves optimizer state via optimizer.state, so both child
        # optimizers share the wrapper's state mapping.
        self.state = defaultdict(dict)
        self._sync_child_optimizers()

    def _sync_child_optimizers(self):
        if self._muon_optimizer is not None:
            self._muon_optimizer.param_groups = self.param_groups[:self._muon_group_count]
            self._muon_optimizer.state = self.state
        if self._adamw_optimizer is not None:
            self._adamw_optimizer.param_groups = self.param_groups[self._muon_group_count:]
            self._adamw_optimizer.state = self.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if self._muon_optimizer is not None:
            loss = self._muon_optimizer.step(closure=closure)
            closure = None
        if self._adamw_optimizer is not None:
            adamw_loss = self._adamw_optimizer.step(closure=closure)
            if loss is None:
                loss = adamw_loss
        return loss

    def load_state_dict(self, state_dict):
        saved_groups = state_dict.get('param_groups', [])
        if len(saved_groups) != len(self.param_groups) or any('use_muon' not in g for g in saved_groups):
            logger.warning('Ignoring incompatible optimizer state from checkpoint and '
                           'reinitializing MuonWithAuxAdam.')
            return
        super().load_state_dict(state_dict)
        self._sync_child_optimizers()


def _get_parameter_owners(model: nn.Module) -> dict:
    owners = {}
    for module_name, module in model.named_modules():
        prefix = f'{module_name}.' if module_name else ''
        for param_name, _ in module.named_parameters(recurse=False):
            owners[f'{prefix}{param_name}'] = module
    return owners


def _should_use_muon(name: str, parameter: nn.Parameter, owner: Optional[nn.Module]) -> bool:
    # Muon only for hidden linear weight matrices.
    if parameter.ndim != 2:
        return False
    if not isinstance(owner, nn.Linear):
        return False
    # XXX: Too specific to ppocrv6
    if name.endswith(('head.fc.weight', 'head.fc2.weight', 'tgt_word_prj.weight')):
        return False
    return True


def get_parameter_groups(model: nn.Module, base_lr: Optional[float] = None) -> list:
    """
    Split a model's parameters into a Muon group (hidden linear 2-D weights) and
    two AdamW groups: 2-D weights with weight decay, and 1-D params (biases,
    norm gains) with no weight decay. The Muon group is placed first.
    """
    owners = _get_parameter_owners(model)
    muon, muon_names = [], []
    decay, decay_names = [], []
    no_decay, no_decay_names = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        owner = owners.get(name)
        if _should_use_muon(name, p, owner):
            muon.append(p); muon_names.append(name)
        elif p.ndim >= 2:
            decay.append(p); decay_names.append(name)
        else:
            no_decay.append(p); no_decay_names.append(name)

    groups = []
    if muon:
        groups.append({'params': muon, 'param_names': muon_names,
                       'name': 'muon_hidden_linear', 'use_muon': True})
    if decay:
        groups.append({'params': decay, 'param_names': decay_names,
                       'name': 'adamw_decay', 'use_muon': False})
    if no_decay:
        groups.append({'params': no_decay, 'param_names': no_decay_names,
                       'name': 'adamw_no_decay', 'use_muon': False, 'weight_decay': 0.0})
    return groups
