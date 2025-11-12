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
from kraken.lib.dataset import BaselineSet, ImageInputTransforms
from kraken.configs import BLLASegmentationTrainingConfig, BLLASegmentationTrainingDataConfig
from kraken.train.utils import configure_optimizer_and_lr_scheduler, SegmentationTestMetrics

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from os import PathLike
    from kraken.models import BaseModel
    from kraken.containers import Segmentation

logger = logging.getLogger(__name__)


__all__ = ['BLLASegmentationDataModule', 'BLLASegmentationModel']


class BLLASegmentationDataModule(L.LightningDataModule):
    def __init__(self,
                 data_config: BLLASegmentationTrainingDataConfig):
        """
        A LightningDataModule encapsulating the training data for a page
        segmentation model.

        Args:
            data_config: Configuration object to set dataset parameters.
        """
        super().__init__()
        self.save_hyperparameters()

        all_files = [getattr(data_config, x) for x in ['training_data', 'evaluation_data', 'test_data']]

        if data_config.format_type in ['xml', 'page', 'alto']:

            def _parse_xml_set(ds_type, dataset) -> list[dict[str, 'Segmentation']]:
                if not dataset:
                    return None
                logger.info(f'Parsing {len(dataset) if dataset else 0} XML files for {ds_type} data')
                data = []
                for pos, file in enumerate(dataset):
                    try:
                        data.append(XMLPage(file, filetype=data_config.format_type).to_container())
                    except Exception as e:
                        logger.warning(f'Failed to parse {file}: {e}')
                return data

            training_data = _parse_xml_set('training', all_files[0])
            evaluation_data = _parse_xml_set('evaluation', all_files[1])
            self.test_data = _parse_xml_set('test', all_files[2])
        elif data_config.format_type is None:
            training_data = data_config.training_data
            logger.info(f'Using {len(training_data) if training_data else 0} Segmentation objects for training data')
            evaluation_data = data_config.evaluation_data
            logger.info(f'Using {len(evaluation_data) if evaluation_data else 0} Segmentation objects for evaluation data')
            self.test_data = data_config.test_data
            logger.info(f'Using {len(self.test_data) if data_config.test_data else 0} Segmentation objects for test data')
        else:
            raise ValueError(f'format_type {data_config.format_type} not in [alto, page, xml, None].')

        if training_data and evaluation_data:
            train_set = self._build_dataset(training_data,
                                            augmentation=data_config.augment,
                                            im_transforms=None,
                                            class_mapping={'aux': {'_start_separator': 0, '_end_separator': 1},
                                                           'baselines': self.hparams.data_config.line_class_mapping,
                                                           'regions': self.hparams.data_config.region_class_mapping})
            self.train_set = Subset(train_set, range(len(train_set)))
            val_set = self._build_dataset(evaluation_data,
                                          im_transforms=None,
                                          class_mapping={'aux': {'_start_separator': 0, '_end_separator': 1},
                                                         'baselines': self.hparams.data_config.line_class_mapping,
                                                         'regions': self.hparams.data_config.region_class_mapping})

            self.val_set = Subset(val_set, range(len(val_set)))
        elif training_data:
            train_set = self._build_dataset(training_data,
                                            augmentation=data_config.augment,
                                            im_transforms=None,
                                            class_mapping={'aux': {'_start_separator': 0, '_end_separator': 1},
                                                           'baselines': self.hparams.data_config.line_class_mapping,
                                                           'regions': self.hparams.data_config.region_class_mapping})

            train_len = int(len(train_set)*data_config.partition)
            val_len = len(train_set) - train_len
            logger.info(f'No explicit validation data provided. Splitting off '
                        f'{val_len} (of {len(train_set)}) samples to validation '
                        'set.')
            self.train_set, self.val_set = random_split(train_set, (train_len, val_len))
        elif self.test_data:
            pass
        else:
            raise ValueError('Invalid specification of training/evaluation/test data.')

    def _build_dataset(self,
                       data,
                       **kwargs):
        dataset = BaselineSet(line_width=self.hparams.data_config.line_width,
                              **kwargs)

        for page in data:
            dataset.add(page)

        return dataset

    def setup(self, stage: str = None):
        transforms = ImageInputTransforms(1,
                                          self.trainer.lightning_module.height,
                                          self.trainer.lightning_module.width,
                                          self.trainer.lightning_module.channels,
                                          self.trainer.lightning_module.hparams.config.padding,
                                          valid_norm=False)

        if stage == 'fit' or stage is None:
            if len(self.train_set) == 0:
                raise ValueError('No valid training data provided. Please add some.')
            if len(self.val_set) == 0:
                raise ValueError('No valid validation data provided. Please add some.')
            self.train_set.dataset.transforms = transforms
            self.val_set.dataset.transforms = transforms
            # make sure class mapping is serializable dict
            self.hparams.data_config.line_class_mapping = dict(self.train_set.dataset.class_mapping['baselines'])
            self.hparams.data_config.region_class_mapping = dict(self.train_set.dataset.class_mapping['regions'])
        elif stage == 'test':
            if len(self.test_data) == 0:
                raise ValueError('No valid test data provided. Please add some.')
            test_set = self._build_dataset(self.test_data,
                                           im_transforms=transforms,
                                           class_mapping=self.trainer.lightning_module.net.user_metadata['class_mapping'])
            self.test_set = Subset(test_set, range(len(test_set)))
            if len(self.test_set) == 0:
                raise ValueError('No valid test data provided. Please add some.')

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

    def test_dataloader(self):
        return DataLoader(self.test_set,
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
        self.save_hyperparameters(ignore=['model'])

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

    def on_test_epoch_start(self):
        self.pages = []

    def test_step(self, batch, batch_idx, test_dataloader=0):
        x, y = batch['image'], batch['target']
        pred, _ = self.net(x)
        # scale target to output size
        y = F.interpolate(y, size=(pred.size(2), pred.size(3))).squeeze(0).bool()
        pred = pred.squeeze() > 0.5
        pred = pred.view(pred.size(0), -1)
        y = y.view(y.size(0), -1)
        if self.line_cls_idxs:
            y_baselines = y[self.line_cls_idxs].sum(dim=0, dtype=torch.bool)
            pred_baselines = pred[self.line_cls_idxs].sum(dim=0, dtype=torch.bool)
            baseline_metrics = {'intersections': (y_baselines & pred_baselines).sum(dim=0),
                                'unions': (y_baselines | pred_baselines).sum(dim=0)}

        if self.region_cls_idxs:
            y_region_cls_idxs = y[self.region_cls_idxs].sum(dim=0, dtype=torch.bool)
            pred_region_cls_idxs = pred[self.region_cls_idxs].sum(dim=0, dtype=torch.bool)
            region_metrics = {'intersections': (y_region_cls_idxs & pred_region_cls_idxs).sum(dim=0),
                              'unions': (y_region_cls_idxs | pred_region_cls_idxs).sum(dim=0)}

        self.pages.append({'intersections': (y & pred).sum(dim=1),
                           'unions': (y | pred).sum(dim=1),
                           'corrects': torch.eq(y, pred).sum(dim=1),
                           'cls_cnt': y.sum(dim=1),
                           'all_n': torch.tensor(y.size(1)),
                           'baselines': baseline_metrics,
                           'regions': region_metrics})

    def on_test_epoch_end(self):
        corrects = torch.stack([x['corrects'] for x in self.pages], -1).sum(dim=-1)
        all_n = torch.stack([x['all_n'] for x in self.pages]).sum()  # Number of pixel for all pages

        class_pixel_accuracy = corrects / all_n
        mean_accuracy = torch.mean(class_pixel_accuracy)

        intersections = torch.stack([x['intersections'] for x in self.pages], -1).sum(dim=-1)
        unions = torch.stack([x['unions'] for x in self.pages], -1).sum(dim=-1)
        smooth = torch.finfo(torch.float).eps
        class_iu = (intersections + smooth) / (unions + smooth)
        mean_iu = torch.mean(class_iu)

        cls_cnt = torch.stack([x['cls_cnt'] for x in self.pages]).sum()
        freq_iu = torch.sum(cls_cnt / cls_cnt.sum() * class_iu.sum())

        self.test_metrics = SegmentationTestMetrics(class_pixel_accuracy=class_pixel_accuracy,
                                                    mean_accuracy=mean_accuracy,
                                                    class_iu=class_iu,
                                                    mean_iu=mean_iu,
                                                    freq_iu=freq_iu)

        if self.line_cls_idxs:
            line_intersections = torch.stack([x["baselines"]['intersections'] for x in self.pages]).sum()
            line_unions = torch.stack([x["baselines"]['unions'] for x in self.pages]).sum()
            smooth = torch.finfo(torch.float).eps
            line_iu = (line_intersections + smooth) / (line_unions + smooth)
            self.test_metrics.line_iu = line_iu

        if self.region_cls_idxs:
            region_intersections = torch.stack([x["regions"]['intersections'] for x in self.pages]).sum()
            region_unions = torch.stack([x["regions"]['unions'] for x in self.pages]).sum()
            smooth = torch.finfo(torch.float).eps
            region_iu = (region_intersections + smooth) / (region_unions + smooth)
            self.test_metrics.region_iu = region_iu

    def setup(self, stage: Optional[str] = None):
        # finalize models in case of appending/loading
        if stage in [None, 'fit']:
            set_class_mapping = self.trainer.datamodule.train_set.dataset.class_mapping
            if not self.net:
                vgsl = self.hparams.config.spec.strip()
                self.hparams.config.spec = f'[{vgsl[1:-1]} O2l{self.trainer.datamodule.train_set.dataset.num_classes}]'
                logger.info(f'Creating model {vgsl} with {self.trainer.datamodule.train_set.dataset.num_classes} outputs')
                from kraken.models import create_model
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
                    elif self.hparams.config.resize == 'new':
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
        elif stage == 'test':
            self.line_cls_idxs = list(self.trainer.datamodule.test_set.dataset.class_mapping['baselines'].values())
            self.region_cls_idxs = list(self.trainer.datamodule.test_set.dataset.class_mapping['regions'].values())

    def on_load_checkpoint(self, checkpoint):
        """
        Reconstruct the model from the spec here and not in setup() as
        otherwise the weight loading will fail.
        """
        from kraken.models import create_model
        if not isinstance(checkpoint['_module_config'], BLLASegmentationTrainingConfig):
            raise ValueError('Checkpoint is not a segmentation model.')

        data_config = checkpoint['datamodule_hyper_parameters']['data_config']
        self.net = create_model('TorchVGSLModel',
                                vgsl=checkpoint['_module_config'].spec,
                                model_type='segmentation',
                                topline=data_config.topline,
                                one_channel_mode=checkpoint['_one_channel_mode'],
                                class_mapping={'aux': {'_start_separator': 0, '_end_separator': 1},
                                               'baselines': data_config.line_class_mapping,
                                               'regions': data_config.region_class_mapping})

        self.batch, self.channels, self.height, self.width = self.net.input

    def on_save_checkpoint(self, checkpoint):
        """
        Save hyperparameters a second time so we can set parameters that
        shouldn't be overwritten in on_load_checkpoint.
        """
        checkpoint['_module_config'] = self.hparams.config
        checkpoint['_one_channel_mode'] = self.trainer.datamodule.train_set.dataset.im_mode

    @classmethod
    def load_from_weights(cls,
                          path: Union[str, 'PathLike'],
                          config: BLLASegmentationTrainingConfig) -> 'BLLASegmentationModel':
        """
        Initializes the module from a model weights file.
        """
        from kraken.models import load_models
        models = load_models(path, tasks=['segmentation'])
        if len(models) != 1:
            raise ValueError(f'Found {len(models)} segmentation models in model file.')
        return cls(config=config, model=models[0])

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
