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
from typing import TYPE_CHECKING, Literal, Optional, Union

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
from kraken.train.utils import configure_optimizer_and_lr_scheduler

if TYPE_CHECKING:
    from os import PathLike
    from torch.nn import Module
    from kraken.models import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class DummyVGSLModel:
    hyper_params: dict[str, int] = field(default_factory=dict)
    user_metadata: dict[str, list] = field(default_factory=dict)
    one_channel_mode: Literal['1', 'L'] = '1'
    ptl_module: 'Module' = None
    model_type: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.hyper_params: dict[str, int] = {'completed_epochs': 0}
        self.user_metadata: dict[str, list] = {'accuracy': [], 'metrics': []}

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
        training_data = self.hparams.data_config.training_data
        evaluation_data = self.hparams.data_config.evaluation_data
        partition = self.hparams.data_config.partition

        if not evaluation_data:
            np.random.shuffle(training_data)
            training_data = training_data[:int(partition*len(training_data))]
            evaluation_data = training_data[int(partition*len(training_data)):]
        train_set = PairWiseROSet(training_data,
                                  mode=self.hparams.data_config.format_type,
                                  level=self.hparams.data_config.level,
                                  ro_id=self.hparams.data_config.reading_order,
                                  class_mapping=self.hparams.data_config.class_mapping)
        self.train_set = Subset(train_set, range(len(train_set)))
        self.class_mapping = train_set.class_mapping
        self.hparams.data_config.class_mapping = dict(self.class_mapping)
        val_set = PageWiseROSet(evaluation_data,
                                mode=self.hparams.data_config.format_type,
                                class_mapping=self.class_mapping,
                                level=self.hparams.data_config.level,
                                ro_id=self.hparams.data_config.reading_order)

        self.val_set = Subset(val_set, range(len(val_set)))

        if len(self.train_set) == 0 or len(self.val_set) == 0:
            raise ValueError('No valid training data was provided to the train '
                             'command. Please add valid XML, line, or binary data.')

        logger.info(f'Training set {len(self.train_set)} lines, validation set '
                    f'{len(self.val_set)} lines')

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.data_config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=1,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True)

    def get_feature_dim(self):
        return self.train_set.dataset.get_feature_dim()

    def get_class_mapping(self):
        return self.class_mapping


class ROModel(L.LightningModule):
    def __init__(self,
                 config: ROTrainingConfig,
                 model: Optional['BaseModel'] = None) -> None:
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
        self.save_hyperparameters(ignore=['model'])

        if not isinstance(config, ROTrainingConfig):
            raise ValueError(f'config attribute is {type(config)} not ROTrainingConfig.')

        if model:
            self.net = model

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.nn = DummyVGSLModel(ptl_module=self)

        self.val_losses = MeanMetric()
        self.val_spearman = MeanMetric()

    @classmethod
    def load_from_weights(cls,
                          path: Union[str, 'PathLike'],
                          config: ROTrainingConfig) -> 'ROModel':
        """
        Initializes the module from a model weights file.
        """
        from kraken.models import load_models
        models = load_models(path, tasks=['reading_order'])
        if len(models) != 1:
            raise ValueError(f'Found {len(models)} reading order models in model file.')
        return cls(config=config, model=models[0])

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

    def on_load_checkpoint(self, checkpoint):
        """
        Reconstruct the model here and not in setup() as otherwise the weight
        loading will fail.
        """
        from kraken.models import create_model
        if not isinstance(checkpoint['hyper_parameters']['config'], ROTrainingConfig):
            raise ValueError('Checkpoint is not a reading order model.')
        data_config = checkpoint['datamodule_hyper_parameters']['data_config']
        self.net = create_model('ROMLP',
                                class_mapping=data_config.class_mapping,
                                level=data_config.level)

    def on_save_checkpoint(self, checkpoint):
        """
        Save hyperparameters a second time so we can set parameters that
        shouldn't be overwritten in on_load_checkpoint.
        """
        checkpoint['_module_config'] = self.hparams.config

    def setup(self, stage: Optional[str] = None):
        if stage in [None, 'fit']:
            set_class_mapping = dict(self.trainer.datamodule.get_class_mapping())
            if not getattr(self, 'net', None):
                logger.info('Creating new RO model')
                self.net = ROMLP(class_mapping=set_class_mapping,
                                 level=self.trainer.datamodule.hparams.data_config.level)
            else:
                net_class_mapping = self.net.user_metadata['class_mapping']
                if set_class_mapping.keys() != net_class_mapping.keys():
                    diff = set(set_class_mapping.keys()).symmetric_difference(set(net_class_mapping.keys()))
                    raise ValueError(f'Training data and model class mapping differ: {diff}')
                # backfill train_set/val_set mapping if key-equal as the actual
                # numbering in the train_set might be different
                num_classes = max(0, *net_class_mapping.values()) + 1
                self.trainer.datamodule.train_set.dataset.class_mapping = net_class_mapping
                self.trainer.datamodule.train_set.dataset.num_classes = num_classes
                self.trainer.datamodule.val_set.dataset.class_mapping = net_class_mapping
                self.trainer.datamodule.val_set.dataset.num_classes = num_classes

            # store canonical (one-to-one) class mapping in model metadata for inference
            self.net.user_metadata['class_mapping'] = self.trainer.datamodule.train_set.dataset.canonical_class_mapping

            logger.info('Training types:')
            for k, v in set_class_mapping.items():
                logger.info(f'  {k}\t{v}\t{self.trainer.datamodule.train_set.dataset.class_stats[k]}')

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.config.quit == 'early':
            callbacks.append(EarlyStopping(monitor='val_metric',
                                           mode='min',
                                           patience=self.hparams.config.lag,
                                           stopping_threshold=0.0))
        return callbacks

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return configure_optimizer_and_lr_scheduler(self.hparams.config,
                                                    self.net.parameters(),
                                                    loss_tracking_mode='min')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lrate` in `warmup`
        # steps.
        if self.hparams.config.warmup and self.trainer.global_step < self.hparams.config.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.config.warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.config.lrate

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.config.warmup or self.trainer.global_step >= self.hparams.config.warmup:
            # step OneCycleLR each batch if not in warmup phase
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()
            # step every other scheduler epoch-wise
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)
