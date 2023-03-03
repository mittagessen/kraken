#
# Copyright 2023 Benjamin Kiessling
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
Pytorch-lightning modules for reading order training.

Adapted from:
"""
import re
import math
import torch
import logging
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from os import PathLike
from dataclasses import dataclass, field
from torch.nn import Module
from typing import Dict, Optional, Sequence, Union, Any, Literal, List

from kraken.lib import vgsl, default_specs, layers
from kraken.lib.dataset import ROSet
from kraken.lib.train import _configure_optimizer_and_lr_scheduler
from kraken.lib.ro.layers import MLP

from torch.utils.data import DataLoader, random_split, Subset


logger = logging.getLogger(__name__)

@dataclass
class DummyVGSLModel:
    hyper_params: Dict[str, int] = field(default_factory=dict)
    user_metadata: Dict[str, List] = field(default_factory=dict)
    one_channel_mode: Literal['1', 'L']  = '1'
    ptl_module: Module = None
    model_type: str = 'unknown'

    def __post_init__(self):
        self.hyper_params: Dict[str, int] = {'completed_epochs': 0}
        self.user_metadata: Dict[str, List] = {'accuracy': [], 'metrics': []}

    def save_model(self, filename):
        self.ptl_module.save_checkpoint(filename)


class ROModel(pl.LightningModule):
    def __init__(self,
                 hyper_params: Dict[str, Any] = None,
                 output: str = 'model',
                 training_data: Union[Sequence[Union[PathLike, str]], Sequence[Dict[str, Any]]] = None,
                 evaluation_data: Optional[Union[Sequence[Union[PathLike, str]], Sequence[Dict[str, Any]]]] = None,
                 partition: Optional[float] = 0.9,
                 num_workers: int = 1,
                 format_type: Literal['alto', 'page', 'xml'] = 'xml',
                 load_hyper_parameters: bool = False,
                 level: Literal['baselines', 'regions'] = 'baselines',
                 reading_order: Optional[str] = None):
        """
        A LightningModule encapsulating the unsupervised pretraining setup for
        a text recognition model.

        Setup parameters (load, training_data, evaluation_data, ....) are
        named, model hyperparameters (everything in
        `kraken.lib.default_specs.RECOGNITION_HYPER_PARAMS`) are in in the
        `hyper_params` argument.

        Args:
            hyper_params (dict): Hyperparameter dictionary containing all fields
                                 from
                                 kraken.lib.default_specs.RECOGNITION_PRETRAIN_HYPER_PARAMS
            **kwargs: Setup parameters, i.e. CLI parameters of the train() command.
        """
        super().__init__()
        self.hyper_params = default_specs.READING_ORDER_HYPER_PARAMS
        if hyper_params:
            self.hyper_params.update(hyper_params)

        if not evaluation_data:
            np.random.shuffle(training_data)
            training_data = training_data[:int(partition*len(training_data))]
            evaluation_data = training_data[int(partition*len(training_data)):]
        train_set = ROSet(training_data,
                          mode=format_type,
                          level=level,
                          ro_id=reading_order)
        self.train_set = Subset(train_set, range(len(train_set)))
        val_set = ROSet(evaluation_data,
                        mode=format_type,
                        class_mapping=train_set.class_mapping,
                        level=level,
                        ro_id=reading_order)
        self.val_set = Subset(val_set, range(len(val_set)))

        if len(self.train_set) == 0 or len(self.val_set) == 0:
            raise ValueError('No valid training data was provided to the train '
                             'command. Please add valid XML, line, or binary data.')

        logger.info(f'Training set {len(self.train_set)} lines, validation set '
                    f'{len(self.val_set)} lines')

        self.output = output
        self.criterion = torch.nn.BCELoss()

        self.num_workers = num_workers

        self.best_epoch = -1
        self.best_metric = torch.inf

        logger.info(f'Creating new RO model')
        self.ro_net = torch.jit.script(MLP(train_set.get_feature_dim(), 128))

        if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
            logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
            torch.multiprocessing.set_sharing_strategy('file_system')

        self.nn = DummyVGSLModel(ptl_module=self)

        self.save_hyperparameters()

    def forward(self, x):
        return self.ro_net(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch['sample'], batch['target']
        yhat = self.ro_net(x)
        loss = self.criterion(yhat.squeeze(), y)
        self.log('val_metric', loss)
        return loss

    def validation_epoch_end(self, outputs):
        val_metric = np.mean([x.cpu() for x in outputs])
        if val_metric < self.best_metric:
            logger.debug(f'Updating best metric from {self.best_metric} ({self.best_epoch}) to {val_metric} ({self.current_epoch})')
            self.best_epoch = self.current_epoch
            self.best_metric = val_metric
        logger.info(f'validation run: val_metric {val_metric}')
        self.log('val_metric', val_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        x, y = batch['sample'], batch['target']
        yhat = self.ro_net(x)
        loss = self.criterion(yhat.squeeze(), y)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return _configure_optimizer_and_lr_scheduler(self.hparams.hyper_params,
                                                     self.ro_net.parameters(),
                                                     len_train_set=len(self.train_set),
                                                     loss_tracking_mode='min')

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hyper_params['batch_size'],
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.hyper_params['batch_size'],
                          num_workers=self.num_workers,
                          pin_memory=True)

    def save_checkpoint(self, filename):
        self.trainer.save_checkpoint(filename)
