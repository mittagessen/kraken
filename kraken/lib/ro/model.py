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
import logging
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Optional,
                    Sequence, Union)

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchmetrics.aggregation import MeanMetric

from kraken.configs import ROTrainingConfig, ROTrainingDataConfig
from kraken.lib.dataset import PageWiseROSet, PairWiseROSet
from kraken.lib.ro.layers import ROMLP
from kraken.lib.segmentation import _greedy_order_decoder
from kraken.lib.train import _configure_optimizer_and_lr_scheduler

if TYPE_CHECKING:
    from os import PathLike

    from torch.nn import Module

logger = logging.getLogger(__name__)


@dataclass
class DummyVGSLModel:
    hyper_params: Dict[str, int] = field(default_factory=dict)
    user_metadata: Dict[str, List] = field(default_factory=dict)
    one_channel_mode: Literal['1', 'L'] = '1'
    ptl_module: 'Module' = None
    model_type: str = 'unknown'

    def __post_init__(self):
        self.hyper_params: Dict[str, int] = {'completed_epochs': 0}
        self.user_metadata: Dict[str, List] = {'accuracy': [], 'metrics': []}

    def save_model(self, filename):
        self.ptl_module.save_checkpoint(filename)


def spearman_footrule_distance(s, t):
    return (s - t).abs().sum() / (0.5 * (len(s) ** 2 - (len(s) % 2)))


class RODataModule(L.LightningDataModule):
    def __init__(self,
                 data_config: ROTrainingDataConfig):
        super().__init__()

        self.save_hyperparameters()

    def setup(self, stage: str):
        """
        Builds the datasets.
        """
        training_data = self.hparams.training_data
        evaluation_data = self.hparams.evaluation_data
        partition = self.hparams.partition

        valid_entities = self.hparams.valid_entities
        merge_entities = self.hparams.merge_entities
        merge_all_entities = self.hparams.merge_all_entities

        if not valid_entities:
            valid_entities = None

        if not evaluation_data:
            np.random.shuffle(training_data)
            training_data = training_data[:int(partition*len(training_data))]
            evaluation_data = training_data[int(partition*len(training_data)):]
        train_set = PairWiseROSet(training_data,
                                  mode=self.hparams.format_type,
                                  level=self.hparams.level,
                                  ro_id=self.hparams.reading_order,
                                  class_mapping=self.hparams.class_mapping,
                                  valid_entities=valid_entities,
                                  merge_entities=merge_entities,
                                  merge_all_entities=merge_all_entities)
        self.train_set = Subset(train_set, range(len(train_set)))
        self.class_mapping = train_set.class_mapping
        val_set = PageWiseROSet(evaluation_data,
                                mode=self.hparams.format_type,
                                class_mapping=self.class_mapping,
                                level=self.hparams.level,
                                ro_id=self.hparams.reading_order,
                                valid_entities=valid_entities,
                                merge_entities=merge_entities,
                                merge_all_entities=merge_all_entities)

        self.val_set = Subset(val_set, range(len(val_set)))

        if len(self.train_set) == 0 or len(self.val_set) == 0:
            raise ValueError('No valid training data was provided to the train '
                             'command. Please add valid XML, line, or binary data.')

        logger.info(f'Training set {len(self.train_set)} lines, validation set '
                    f'{len(self.val_set)} lines')

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=1,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)

    def get_feature_dim(self):
        return self.train_set.dataset.get_feature_dim()

    def get_class_mapping(self):
        return self.class_mapping


class ROModel(L.LightningModule):
    def __init__(self,
                 config: ROTrainingConfig,
                 model: Optional[BaseModel] = None) -> None:
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

        if model:
            self.net = model

            self.batch, self.channels, self.height, self.width = self.net.input
        else:
            logger.info('Creating new RO model')
            self.net = ROMLP(feature_dim, feature_dim * 2)
           
        self.output = output
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.nn = DummyVGSLModel(ptl_module=self)

        self.val_losses = MeanMetric()
        self.val_spearman = MeanMetric()

        self.save_hyperparameters()

    def forward(self, x):
        return F.sigmoid(self.net(x))

    def validation_step(self, batch, batch_idx):
        xs, ys, num_lines = batch['sample'], batch['target'], batch['num_lines']
        logits = self.net(xs).squeeze()
        yhat = F.sigmoid(logits)
        order = torch.zeros((num_lines, num_lines))
        idx = 0
        for i in range(num_lines):
            for j in range(num_lines):
                if i != j:
                    order[i, j] = yhat[idx]
                    idx += 1
        path = _greedy_order_decoder(order)
        spearman_dist = spearman_footrule_distance(torch.tensor(range(num_lines)), path)
        self.log('val_spearman', spearman_dist)
        loss = self.criterion(logits, ys.squeeze())
        self.val_losses.update(loss)
        self.val_spearman.update(spearman_dist)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            val_metric = self.val_spearman.compute()
            val_loss = self.val_losses.compute()

            logger.info(f'validation run: val_spearman {val_metric} val_loss {val_loss}')
            self.log('val_spearman', val_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_metric', val_metric, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.val_spearman.reset()
        self.val_losses.reset()

    def training_step(self, batch, batch_idx):
        x, y = batch['sample'], batch['target']
        logits = self.net(x)
        loss = self.criterion(logits.squeeze(), y)
        self.log('loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.hyper_params['quit'] == 'early':
            callbacks.append(EarlyStopping(monitor='val_metric',
                                           mode='min',
                                           patience=self.hparams.hyper_params['lag'],
                                           stopping_threshold=0.0))
        if self.hparams.hyper_params['pl_logger']:
            callbacks.append(LearningRateMonitor(logging_interval='step'))
        return callbacks

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return _configure_optimizer_and_lr_scheduler(self.hparams.hyper_params,
                                                     self.net.parameters(),
                                                     loss_tracking_mode='min')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lrate` in `warmup`
        # steps.
        if self.hparams.hyper_params['warmup'] and self.trainer.global_step < self.hparams.hyper_params['warmup']:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.hyper_params['warmup'])
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.hyper_params['lrate']

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.hyper_params['warmup'] or self.trainer.global_step >= self.hparams.hyper_params['warmup']:
            # step OneCycleLR each batch if not in warmup phase
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()
            # step every other scheduler epoch-wise
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)
