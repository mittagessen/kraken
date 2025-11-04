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
# or implied. See the License for the vgslific language governing
# permissions and limitations under the License.
"""
BLLA segmentation model trainer.
"""
import re
import torch
import logging
import lightning as L
import torch.nn.functional as F

from lightning.pytorch.callbacks import EarlyStopping

from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset, random_split
from torchmetrics.classification import MultilabelAccuracy, MultilabelJaccardIndex

from kraken.lib.xml import XMLPage
from kraken.registry import create_model
from kraken.lib.vgsl import BLLASegmentationTrainingConfig, BLLASegmentationTrainingDataConfig
from kraken.lib.dataset import BaselineSet, ImageInputTransforms
from kraken.train.utils import configure_optimizer_and_lr_scheduler

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from kraken.models import BaseModel

logger = logging.getLogger(__name__)


__all__ = ['BLLASegmentationDataModule', 'BLLASegmentationModel']


class BLLASegmentationDataModule(L.LightningDataModule):
    def __init__(self,
                 data_config: BLLASegmentationTrainingDataConfig,
                 parsing_callback: Callable[[int, int], None] = lambda pos, total: None):
        """
        A LightningDataModule encapsulating the training data for a page
        segmentation model.

        Args:
            data_config: Configuration object to set dataset parameters.
            parsing_callback: A callback that will be called after each input
                              file has been parsed with the current position
                              and total number of files to process. Will be
                              ignored if the training data aready contains
                              `Segmentation` objects.
        """
        super().__init__()
        self.save_hyperparameters()

        if data_config.format_type in ['xml', 'page', 'alto']:
            total = len(data_config.training_data) + len(data_config.evaluation_data if data_config.evaluation_data else 0)
            logger.info(f'Parsing {len(data_config.training_data)} XML files for training data')
            training_data = []
            for pos, file in enumerate(data_config.training_data):
                try:
                    training_data.append(XMLPage(file, filetype=data_config.format_type).to_container())
                except Exception as e:
                    logger.warning(f'Failed to parse {file}: {e}')
                parsing_callback(pos, total)
            if data_config.evaluation_data:
                evaluation_data = []
                logger.info(f'Parsing {len(data_config.evaluation_data)} XML files for validation data')
                for pos, file in enumerate(data_config.evaluation_data, len(data_config.training_data)):
                    try:
                        evaluation_data.append(XMLPage(file, filetype=data_config.format_type).to_container())
                    except Exception as e:
                        logger.warning(f'Failed to parse {file}: {e}')
                parsing_callback(pos, total)
        elif data_config.format_type is None:
            training_data = data_config.training_data
            evaluation_data = data_config.evaluation_data
        else:
            raise ValueError(f'format_type {data_config.format_type} not in [alto, page, xml, None].')

        if not training_data:
            raise ValueError('No training data provided. Please add some.')

        train_set = BaselineSet(line_width=data_config.line_width,
                                im_transforms=None,  # transforms get filled in once we have access to model hyperparams
                                augmentation=data_config.augment,
                                class_mapping={'aux': {'_start_separator': 0, '_end_separator': 1},
                                               'baselines': data_config.line_class_mapping,
                                               'regions': data_config.region_class_mapping})

        for page in training_data:
            train_set.add(page)

        if evaluation_data:
            val_set = BaselineSet(line_width=data_config.line_width,
                                  im_transforms=None,
                                  augmentation=False,
                                  class_mapping={k: dict(v) for k, v in train_set.class_mapping.items()})  # might be a defaultdict

            for page in evaluation_data:
                val_set.add(page)

            train_set = Subset(train_set, range(len(train_set)))
            val_set = Subset(val_set, range(len(val_set)))
        else:
            train_len = int(len(train_set)*data_config.partition)
            val_len = len(train_set) - train_len
            logger.info(f'No explicit validation data provided. Splitting off '
                        f'{val_len} (of {len(train_set)}) samples to validation '
                        'set.')
            train_set, val_set = random_split(train_set, (train_len, val_len))

        if len(train_set) == 0:
            raise ValueError('No valid training data provided. Please add some.')

        if len(val_set) == 0:
            raise ValueError('No valid validation data provided. Please add some.')

        self.train_set = train_set
        self.val_set = val_set

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            transforms = ImageInputTransforms(1,
                                              self.trainer.lightning_module.height,
                                              self.trainer.lightning_module.width,
                                              self.trainer.lightning_module.channels,
                                              self.trainer.lightning_module.hparams.config.padding,
                                              valid_norm=False)
            self.train_set.dataset.transforms = transforms
            self.val_set.dataset.transforms = transforms

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=1,
                          num_workers=self.hparams.data_config.num_workers,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=1,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True)


class BLLASegmentationModel(L.LightningModule):

    def __init__(self,
                 config: BLLASegmentationTrainingConfig,
                 model: Optional['BaseModel'] = None):
        """
        A LightningModule encapsulating the training setup for a page
        segmentation model.

        Args:
            config: Configuration for the training.
            model: Loaded model when instantiating from weights.
        """
        super().__init__()
        self.save_hyperparameters()

        if model:
            self.net = model

            if self.net.model_type not in [None, 'segmentation']:
                raise ValueError(f'Model {model} is of type {self.net.model_type} while `segmentation` is expected.')

            self.batch, self.channels, self.height, self.width = self.net.input
        else:
            self.net = None

            # this is ugly.
            vgsl = config.spec.strip()
            if vgsl[0] != '[' or vgsl[-1] != ']':
                raise ValueError(f'VGSL vgsl "{vgsl}" not bracketed')
            self.vgsl = vgsl
            blocks = vgsl[1:-1].split(' ')
            m = re.match(r'(\d+),(\d+),(\d+),(\d+)', blocks[0])
            if not m:
                raise ValueError(f'Invalid input vgsl {blocks[0]}')
            self.batch, self.height, self.width, self.channels = [int(x) for x in m.groups()]

        self.example_input_array = torch.Tensor(1,
                                                self.channels,
                                                self.height if self.height else 400,
                                                self.width if self.width else 300)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['target']
        output, _ = self.net(x)
        output = F.interpolate(output, size=(y.size(2), y.size(3)))
        loss = self.criterion(output, y)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['target']
        pred, _ = self.net(x)
        # scale target to output size
        y = F.interpolate(y, size=(pred.size(2), pred.size(3))).int()

        self.val_px_accuracy.update(pred, y)
        self.val_mean_accuracy.update(pred, y)
        self.val_mean_iu.update(pred, y)
        self.val_freq_iu.update(pred, y)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            pixel_accuracy = self.val_px_accuracy.compute()
            mean_accuracy = self.val_mean_accuracy.compute()
            mean_iu = self.val_mean_iu.compute()
            freq_iu = self.val_freq_iu.compute()

            if mean_iu > self.best_metric:
                logger.debug(f'Updating best metric from {self.best_metric} ({self.best_epoch}) to {mean_iu} ({self.current_epoch})')
                self.best_epoch = self.current_epoch
                self.best_metric = mean_iu

            logger.info(f'validation run: accuracy {pixel_accuracy} mean_acc {mean_accuracy} mean_iu {mean_iu} freq_iu {freq_iu}')

            self.log('val_accuracy', pixel_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_mean_acc', mean_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_mean_iu', mean_iu, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_freq_iu', freq_iu, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_metric', mean_iu, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # reset metrics even if sanity checking
        self.val_px_accuracy.reset()
        self.val_mean_accuracy.reset()
        self.val_mean_iu.reset()
        self.val_freq_iu.reset()

    def setup(self, stage: Optional[str] = None):
        # finalize models in case of appending/loading
        if stage in [None, 'fit']:
            set_class_mapping = self.trainer.datamodule.train_set.dataset.class_mapping
            if not self.net:
                vgsl = self.hparams.config.spec.strip()
                self.hparams.config.spec = f'[{vgsl[1:-1]} O2l{self.trainer.datamodule.train_set.dataset.num_classes}]'
                logger.info(f'Creating model {vgsl} with {self.trainer.datamodule.train_set.dataset.num_classes} outputs')
                self.net = create_model('TorchVGSLModel',
                                        vgsl=self.hparams.config.spec,
                                        topline=self.trainer.datamodule.hparams.data_config.topline,
                                        class_mapping=set_class_mapping)
            else:
                net_class_mapping = self.net.user_metadata['class_mapping']
                if set_class_mapping['baselines'].keys() != net_class_mapping['baselines'].keys() or \
                   set_class_mapping['regions'].keys() != net_class_mapping['regions'].keys():

                    bl_diff = set(set_class_mapping['baselines'].keys()).symmetric_difference(set(net_class_mapping['baselines'].keys()))
                    regions_diff = set(set_class_mapping['regions'].keys()).symmetric_difference(set(net_class_mapping['regions'].keys()))

                    if self.hparams.config.resize == 'fail':
                        raise ValueError(f'Training data and model class mapping differ (bl: {bl_diff}, regions: {regions_diff}')
                    elif self.hparams.config.resize == 'union':
                        new_bls = set_class_mapping['baselines'].keys() - net_class_mapping['baselines'].keys()
                        new_regions = set_class_mapping['regions'].keys() - net_class_mapping['regions'].keys()
                        cls_idx = max(max(net_class_mapping['baselines'].values()) if net_class_mapping['baselines'] else -1, # noqa
                                      max(net_class_mapping['regions'].values()) if net_class_mapping['regions'] else -1) # noqa
                        logger.info(f'Adding {len(new_bls) + len(new_regions)} missing types to network output layer.')
                        self.net.resize_output(cls_idx + len(new_bls) + len(new_regions) + 1)
                        for c in new_bls:
                            cls_idx += 1
                            net_class_mapping['baselines'][c] = cls_idx
                        for c in new_regions:
                            cls_idx += 1
                            net_class_mapping['regions'][c] = cls_idx
                    elif self.resize == 'new':
                        logger.info('Fitting network exactly to training set.')
                        new_bls = set_class_mapping['baselines'].keys() - net_class_mapping['baselines'].keys()
                        new_regions = set_class_mapping['regions'].keys() - net_class_mapping['regions'].keys()
                        del_bls = net_class_mapping['baselines'].keys() - set_class_mapping['baselines'].keys()
                        del_regions = net_class_mapping['regions'].keys() - set_class_mapping['regions'].keys()

                        logger.info(f'Adding {len(new_bls) + len(new_regions)} missing '
                                    f'types and removing {len(del_bls) + len(del_regions)} to network output layer ')
                        cls_idx = max(max(net_class_mapping['baselines'].values()) if net_class_mapping['baselines'] else -1, # noqa
                                      max(net_class_mapping['regions'].values()) if net_class_mapping['regions'] else -1) # noqa

                        del_indices = [net_class_mapping['baselines'][x] for x in del_bls]
                        del_indices.extend(net_class_mapping['regions'][x] for x in del_regions)
                        self.net.resize_output(cls_idx + len(new_bls) + len(new_regions) -
                                               len(del_bls) - len(del_regions) + 1, del_indices)

                        # delete old baseline/region types
                        cls_idx = min(min(net_class_mapping['baselines'].values()) if net_class_mapping['baselines'] else torch.inf, # noqa
                                      min(net_class_mapping['regions'].values()) if net_class_mapping['regions'] else torch.inf) # noqa

                        bls = {}
                        for k, v in sorted(net_class_mapping['baselines'].items(), key=lambda item: item[1]):
                            if k not in del_bls:
                                bls[k] = cls_idx
                                cls_idx += 1

                        regions = {}
                        for k, v in sorted(net_class_mapping['regions'].items(), key=lambda item: item[1]):
                            if k not in del_regions:
                                regions[k] = cls_idx
                                cls_idx += 1

                        net_class_mapping['baselines'] = bls
                        net_class_mapping['regions'] = regions

                        # add new baseline/region types
                        cls_idx -= 1
                        for c in new_bls:
                            cls_idx += 1
                            net_class_mapping['baselines'][c] = cls_idx
                        for c in new_regions:
                            cls_idx += 1
                            net_class_mapping['regions'][c] = cls_idx
                    else:
                        raise ValueError(f'invalid resize parameter value {self.resize}')
                # backfill train_set/val_set mapping if key-equal as the actual
                # numbering in the train_set might be different
                num_classes = sum(len(v) for v in net_class_mapping.values())
                self.trainer.datamodule.train_set.dataset.class_mapping = net_class_mapping
                self.trainer.datamodule.train_set.dataset.num_classes = num_classes
                self.trainer.datamodule.val_set.dataset.class_mapping = net_class_mapping
                self.trainer.datamodule.val_set.dataset.num_classes = num_classes

            # change topline/baseline switch
            loc = {None: 'centerline',
                   True: 'topline',
                   False: 'baseline'}

            topline = self.trainer.datamodule.hparams.data_config.topline
            if topline != (from_loc := self.net.user_metadata.get('topline', 'unset')):
                logger.warning(f'Changing baseline location from {from_loc} to {loc[topline]}.')
            self.net.user_metadata['topline'] = topline

            logger.info('Training line types:')
            for k, v in set_class_mapping['baselines'].items():
                logger.info(f'  {k}\t{v}\t{self.trainer.datamodule.train_set.dataset.class_stats["baselines"][k]}')
            logger.info('Training region types:')
            for k, v in set_class_mapping['regions'].items():
                logger.info(f'  {k}\t{v}\t{self.trainer.datamodule.train_set.dataset.class_stats["regions"][k]}')

            # set model type metadata field and dump class_mapping
            self.net.model_type = 'segmentation'

            # set up validation metrics after output classes have been determined
            num_classes = self.trainer.datamodule.train_set.dataset.num_classes
            self.val_px_accuracy = MultilabelAccuracy(average='micro', num_labels=num_classes)
            self.val_mean_accuracy = MultilabelAccuracy(average='macro', num_labels=num_classes)
            self.val_mean_iu = MultilabelJaccardIndex(average='macro', num_labels=num_classes)
            self.val_freq_iu = MultilabelJaccardIndex(average='weighted', num_labels=num_classes)

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.config.quit == 'early':
            callbacks.append(EarlyStopping(monitor='val_mean_iu',
                                           mode='max',
                                           patience=self.hparams.config.lag,
                                           stopping_threshold=1.0))

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
                                                    len_train_set=len(self.trainer.datamodule.train_set),
                                                    loss_tracking_mode='max')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lrate` in `warmup`
        # steps.
        if self.hparams.config.warmup and self.trainer.global_step < self.hparams.config.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams)
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
