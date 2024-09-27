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
Model configuration helpers
"""
import logging

import torch
from torch.optim import lr_scheduler

logger = logging.getLogger(__name__)


def _configure_optimizer_and_lr_scheduler(hparams, params, len_train_set=None, loss_tracking_mode='max'):
    optimizer = hparams.get("optimizer")
    lrate = hparams.get("lrate")
    momentum = hparams.get("momentum")
    weight_decay = hparams.get("weight_decay")
    schedule = hparams.get("schedule")
    gamma = hparams.get("gamma")
    cos_t_max = hparams.get("cos_t_max")
    cos_min_lr = hparams.get("cos_min_lr")
    step_size = hparams.get("step_size")
    rop_factor = hparams.get("rop_factor")
    rop_patience = hparams.get("rop_patience")
    epochs = hparams.get("epochs")
    completed_epochs = hparams.get("completed_epochs")

    # XXX: Warmup is not configured here because it needs to be manually done in optimizer_step()
    logger.debug(f'Constructing {optimizer} optimizer (lr: {lrate}, momentum: {momentum})')
    if optimizer == 'Adam':
        optim = torch.optim.Adam(params, lr=lrate, weight_decay=weight_decay)
    else:
        optim = getattr(torch.optim, optimizer)(params,
                                                lr=lrate,
                                                momentum=momentum,
                                                weight_decay=weight_decay)
    lr_sched = {}
    if schedule == 'exponential':
        lr_sched = {'scheduler': lr_scheduler.ExponentialLR(optim, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'cosine':
        lr_sched = {'scheduler': lr_scheduler.CosineAnnealingLR(optim,
                                                                cos_t_max,
                                                                cos_min_lr,
                                                                last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'step':
        lr_sched = {'scheduler': lr_scheduler.StepLR(optim, step_size, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'reduceonplateau':
        lr_sched = {'scheduler': lr_scheduler.ReduceLROnPlateau(optim,
                                                                mode=loss_tracking_mode,
                                                                factor=rop_factor,
                                                                patience=rop_patience),
                    'interval': 'step'}
    elif schedule == '1cycle':
        if epochs <= 0:
            raise ValueError('1cycle learning rate scheduler selected but '
                             'number of epochs is less than 0 '
                             f'({epochs}).')
        last_epoch = completed_epochs*len_train_set if completed_epochs else -1
        lr_sched = {'scheduler': lr_scheduler.OneCycleLR(optim,
                                                         max_lr=lrate,
                                                         epochs=epochs,
                                                         steps_per_epoch=len_train_set,
                                                         last_epoch=last_epoch),
                    'interval': 'step'}
    elif schedule != 'constant':
        raise ValueError(f'Unsupported learning rate scheduler {schedule}.')

    ret = {'optimizer': optim}
    if lr_sched:
        ret['lr_scheduler'] = lr_sched

    if schedule == 'reduceonplateau':
        lr_sched['monitor'] = 'val_metric'
        lr_sched['strict'] = False
        lr_sched['reduce_on_plateau'] = True

    return ret
