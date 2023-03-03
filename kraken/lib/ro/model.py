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
from typing import Dict, Optional, Sequence, Union, Any, Literal

from kraken.lib import vgsl, default_specs, layers
from kraken.lib.dataset import ROSet
from kraken.lib.train import _configure_optimizer_and_lr_scheduler
from kraken.lib.ro.layers import MLP

from torch.utils.data import DataLoader, random_split, Subset


logger = logging.getLogger(__name__)


class ROModel(pl.LightningModule):
    def __init__(self,
                 hyper_params: Dict[str, Any] = None,
                 output: str = 'model',
                 model: Optional[Union[PathLike, str]] = None,
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
        hyper_params_ = default_specs.READING_ORDER_HYPER_PARAMS
        if model:
            logger.info(f'Loading existing model from {model} ')
            self.nn = vgsl.TorchVGSLModel.load_model(model)

            if self.nn.model_type not in [None, 'segmentation']:
                raise ValueError(f'Model {model} is of type {self.nn.model_type} while `segmentation` is expected.')

            if load_hyper_parameters:
                hp = self.nn.hyper_params
            else:
                hp = {}
            hyper_params_.update(hp)
        else:
            self.ro_net = None

        if hyper_params:
            hyper_params_.update(hyper_params)
        self.save_hyperparameters(hyper_params_)

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

        self.model = model
        self.output = output
        self.criterion = torch.nn.BCELoss()

        self.num_workers = num_workers

        logger.info(f'Creating new RO model')
        self.ro_net = torch.jit.script(MLP(train_set.get_feature_dim(), 128))

        if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
            logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
            torch.multiprocessing.set_sharing_strategy('file_system')

        logger.info('Encoding training set')

    def forward(self, x):
        return self.ro_net(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch['sample'], batch['target']
        yhat = self.ro_net(x)
        loss = self.criterion(yhat.squeeze(), y)
        self.log('loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch['sample'], batch['target']
        yhat = self.ro_net(x)
        loss = self.criterion(yhat.squeeze(), y)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return _configure_optimizer_and_lr_scheduler(self.hparams,
                                                     self.ro_net.parameters(),
                                                     len_train_set=len(self.train_set),
                                                     loss_tracking_mode='min')

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)
