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
BLLA segmentation model trainer.
"""
import logging
import re
import warnings
from typing import (TYPE_CHECKING, Callable, Dict, Literal, Optional,
                    Sequence, Union)

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset, random_split
from torchmetrics.classification import MultilabelAccuracy, MultilabelJaccardIndex

from kraken.lib.xml import XMLPage
from kraken.lib import default_specs, vgsl
from kraken.lib.dataset import BaselineSet, ImageInputTransforms
from kraken.train.utils import configure_optimizer_and_lr_scheduler

if TYPE_CHECKING:
    from os import PathLike
    from kraken.containers import Segmentation

logger = logging.getLogger(__name__)


class SegmentationModel(L.LightningModule):
    def __init__(self,
                 hyper_params: Dict = None,
                 load_hyper_parameters: bool = False,
                 progress_callback: Callable[[str, int], Callable[[None], None]] = lambda string, length: lambda: None,
                 message: Callable[[str], None] = lambda *args, **kwargs: None,
                 output: str = 'model',
                 spec: str = default_specs.SEGMENTATION_SPEC,
                 model: Optional[Union['PathLike', str]] = None,
                 training_data: Union[Sequence[Union['PathLike', str]], Sequence['Segmentation']] = None,
                 evaluation_data: Optional[Union[Sequence[Union['PathLike', str]], Sequence['Segmentation']]] = None,
                 partition: Optional[float] = 0.9,
                 num_workers: int = 1,
                 force_binarization: bool = False,
                 format_type: Literal['path', 'alto', 'page', 'xml', None] = 'path',
                 suppress_regions: bool = False,
                 suppress_baselines: bool = False,
                 valid_regions: Optional[Sequence[str]] = None,
                 valid_baselines: Optional[Sequence[str]] = None,
                 merge_regions: Optional[Dict[str, str]] = None,
                 merge_baselines: Optional[Dict[str, str]] = None,
                 merge_all_baselines: Optional[str] = None,
                 merge_all_regions: Optional[str] = None,
                 bounding_regions: Optional[Sequence[str]] = None,
                 resize: Literal['fail', 'both', 'new', 'add', 'union'] = 'fail',
                 topline: Union[bool, None] = False):
        """
        A LightningModule encapsulating the training setup for a page
        segmentation model.

        Setup parameters (load, training_data, evaluation_data, ....) are
        named, model hyperparameters (everything in
        `kraken.lib.default_specs.SEGMENTATION_HYPER_PARAMS`) are in in the
        `hyper_params` argument.

        Args:
            hyper_params (dict): Hyperparameter dictionary containing all fields
                                 from
                                 kraken.lib.default_specs.SEGMENTATION_HYPER_PARAMS
            **kwargs: Setup parameters, i.e. CLI parameters of the segtrain() command.
        """

        super().__init__()

        self.best_epoch = -1
        self.best_metric = 0.0
        self.best_model = None

        self.model = model
        self.num_workers = num_workers

        if resize == "add":
            resize = "union"
            warnings.warn("'add' value for resize has been deprecated. Use 'union' instead.", DeprecationWarning)
        elif resize == "both":
            resize = "new"
            warnings.warn("'both' value for resize has been deprecated. Use 'new' instead.", DeprecationWarning)
        self.resize = resize

        self.output = output
        self.bounding_regions = bounding_regions
        self.topline = topline

        hyper_params_ = default_specs.SEGMENTATION_HYPER_PARAMS.copy()

        if model:
            logger.info(f'Loading existing model from {model}')
            self.nn = vgsl.TorchVGSLModel.load_model(model)

            if self.nn.model_type not in [None, 'segmentation']:
                raise ValueError(f'Model {model} is of type {self.nn.model_type} while `segmentation` is expected.')

            if load_hyper_parameters:
                hp = self.nn.hyper_params
            else:
                hp = {}
            hyper_params_.update(hp)
            batch, channels, height, width = self.nn.input
        else:
            self.nn = None

            spec = spec.strip()
            if spec[0] != '[' or spec[-1] != ']':
                raise ValueError(f'VGSL spec "{spec}" not bracketed')
            self.spec = spec
            blocks = spec[1:-1].split(' ')
            m = re.match(r'(\d+),(\d+),(\d+),(\d+)', blocks[0])
            if not m:
                raise ValueError(f'Invalid input spec {blocks[0]}')
            batch, height, width, channels = [int(x) for x in m.groups()]

        if hyper_params:
            hyper_params_.update(hyper_params)

        self.hyper_params = hyper_params_
        self.save_hyperparameters()

        if format_type in ['xml', 'page', 'alto']:
            logger.info(f'Parsing {len(training_data)} XML files for training data')
            _training_data = []
            for file in training_data:
                try:
                    _training_data.append(XMLPage(file, format_type).to_container())
                except Exception as e:
                    logger.warning(f'Failed to parse {file}: {e}')
            training_data = _training_data
            if evaluation_data:
                _evaluation_data = []
                logger.info(f'Parsing {len(evaluation_data)} XML files for validation data')
                for file in evaluation_data:
                    try:
                        _evaluation_data.append(XMLPage(file, format_type).to_container())
                    except Exception as e:
                        logger.warning(f'Failed to parse {file}: {e}')
                evaluation_data = _evaluation_data
        elif not format_type:
            pass
        else:
            raise ValueError(f'format_type {format_type} not in [alto, page, xml, None].')

        if not training_data:
            raise ValueError('No training data provided. Please add some.')

        transforms = ImageInputTransforms(batch,
                                          height,
                                          width,
                                          channels,
                                          self.hyper_params['padding'],
                                          valid_norm=False,
                                          force_binarization=force_binarization)

        self.example_input_array = torch.Tensor(batch,
                                                channels,
                                                height if height else 400,
                                                width if width else 300)

        # set multiprocessing tensor sharing strategy
        if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
            logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
            torch.multiprocessing.set_sharing_strategy('file_system')

        if not valid_regions:
            valid_regions = None
        if not valid_baselines:
            valid_baselines = None

        if suppress_regions:
            valid_regions = []
            merge_regions = None
        if suppress_baselines:
            valid_baselines = []
            merge_baselines = None

        train_set = BaselineSet(line_width=self.hyper_params['line_width'],
                                im_transforms=transforms,
                                augmentation=self.hyper_params['augment'],
                                valid_baselines=valid_baselines,
                                merge_baselines=merge_baselines,
                                valid_regions=valid_regions,
                                merge_regions=merge_regions,
                                merge_all_baselines=merge_all_baselines,
                                merge_all_regions=merge_all_regions)

        for page in training_data:
            train_set.add(page)

        if evaluation_data:
            val_set = BaselineSet(line_width=self.hyper_params['line_width'],
                                  im_transforms=transforms,
                                  augmentation=False,
                                  valid_baselines=valid_baselines,
                                  merge_baselines=merge_baselines,
                                  valid_regions=valid_regions,
                                  merge_regions=merge_regions,
                                  merge_all_baselines=merge_all_baselines,
                                  merge_all_regions=merge_all_regions)

            for page in evaluation_data:
                val_set.add(page)

            train_set = Subset(train_set, range(len(train_set)))
            val_set = Subset(val_set, range(len(val_set)))
        else:
            train_len = int(len(train_set)*partition)
            val_len = len(train_set) - train_len
            logger.info(f'No explicit validation data provided. Splitting off '
                        f'{val_len} (of {len(train_set)}) samples to validation '
                        'set.')
            train_set, val_set = random_split(train_set, (train_len, val_len))

        if len(train_set) == 0:
            raise ValueError('No valid training data provided. Please add some.')

        if len(val_set) == 0:
            raise ValueError('No valid validation data provided. Please add some.')

        # overwrite class mapping in validation set
        val_set.dataset.num_classes = train_set.dataset.num_classes
        val_set.dataset.class_mapping = train_set.dataset.class_mapping

        self.train_set = train_set
        self.val_set = val_set

    def forward(self, x):
        return self.nn.nn(x)

    def training_step(self, batch, batch_idx):
        input, target = batch['image'], batch['target']
        output, _ = self.nn.nn(input)
        output = F.interpolate(output, size=(target.size(2), target.size(3)))
        loss = self.nn.criterion(output, target)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['target']
        pred, _ = self.nn.nn(x)
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
            if not self.model:
                self.spec = f'[{self.spec[1:-1]} O2l{self.train_set.dataset.num_classes}]'
                logger.info(f'Creating model {self.spec} with {self.train_set.dataset.num_classes} outputs')
                nn = vgsl.TorchVGSLModel(self.spec)
                if self.bounding_regions is not None:
                    nn.user_metadata['bounding_regions'] = self.bounding_regions
                nn.user_metadata['topline'] = self.topline
                self.nn = nn
            else:
                if self.train_set.dataset.class_mapping['baselines'].keys() != self.nn.user_metadata['class_mapping']['baselines'].keys() or \
                   self.train_set.dataset.class_mapping['regions'].keys() != self.nn.user_metadata['class_mapping']['regions'].keys():

                    bl_diff = set(self.train_set.dataset.class_mapping['baselines'].keys()).symmetric_difference(
                        set(self.nn.user_metadata['class_mapping']['baselines'].keys()))
                    regions_diff = set(self.train_set.dataset.class_mapping['regions'].keys()).symmetric_difference(
                        set(self.nn.user_metadata['class_mapping']['regions'].keys()))

                    if self.resize == 'fail':
                        raise ValueError(f'Training data and model class mapping differ (bl: {bl_diff}, regions: {regions_diff}')
                    elif self.resize == 'union':
                        new_bls = self.train_set.dataset.class_mapping['baselines'].keys() - self.nn.user_metadata['class_mapping']['baselines'].keys()
                        new_regions = self.train_set.dataset.class_mapping['regions'].keys() - self.nn.user_metadata['class_mapping']['regions'].keys()
                        cls_idx = max(max(self.nn.user_metadata['class_mapping']['baselines'].values()) if self.nn.user_metadata['class_mapping']['baselines'] else -1, # noqa
                                      max(self.nn.user_metadata['class_mapping']['regions'].values()) if self.nn.user_metadata['class_mapping']['regions'] else -1) # noqa
                        logger.info(f'Adding {len(new_bls) + len(new_regions)} missing types to network output layer.')
                        self.nn.resize_output(cls_idx + len(new_bls) + len(new_regions) + 1)
                        for c in new_bls:
                            cls_idx += 1
                            self.nn.user_metadata['class_mapping']['baselines'][c] = cls_idx
                        for c in new_regions:
                            cls_idx += 1
                            self.nn.user_metadata['class_mapping']['regions'][c] = cls_idx
                    elif self.resize == 'new':
                        logger.info('Fitting network exactly to training set.')
                        new_bls = self.train_set.dataset.class_mapping['baselines'].keys() - self.nn.user_metadata['class_mapping']['baselines'].keys()
                        new_regions = self.train_set.dataset.class_mapping['regions'].keys() - self.nn.user_metadata['class_mapping']['regions'].keys()
                        del_bls = self.nn.user_metadata['class_mapping']['baselines'].keys() - self.train_set.dataset.class_mapping['baselines'].keys()
                        del_regions = self.nn.user_metadata['class_mapping']['regions'].keys() - self.train_set.dataset.class_mapping['regions'].keys()

                        logger.info(f'Adding {len(new_bls) + len(new_regions)} missing '
                                    f'types and removing {len(del_bls) + len(del_regions)} to network output layer ')
                        cls_idx = max(max(self.nn.user_metadata['class_mapping']['baselines'].values()) if self.nn.user_metadata['class_mapping']['baselines'] else -1, # noqa
                                      max(self.nn.user_metadata['class_mapping']['regions'].values()) if self.nn.user_metadata['class_mapping']['regions'] else -1) # noqa

                        del_indices = [self.nn.user_metadata['class_mapping']['baselines'][x] for x in del_bls]
                        del_indices.extend(self.nn.user_metadata['class_mapping']['regions'][x] for x in del_regions)
                        self.nn.resize_output(cls_idx + len(new_bls) + len(new_regions) -
                                              len(del_bls) - len(del_regions) + 1, del_indices)

                        # delete old baseline/region types
                        cls_idx = min(min(self.nn.user_metadata['class_mapping']['baselines'].values()) if self.nn.user_metadata['class_mapping']['baselines'] else np.inf, # noqa
                                      min(self.nn.user_metadata['class_mapping']['regions'].values()) if self.nn.user_metadata['class_mapping']['regions'] else np.inf) # noqa

                        bls = {}
                        for k, v in sorted(self.nn.user_metadata['class_mapping']['baselines'].items(), key=lambda item: item[1]):
                            if k not in del_bls:
                                bls[k] = cls_idx
                                cls_idx += 1

                        regions = {}
                        for k, v in sorted(self.nn.user_metadata['class_mapping']['regions'].items(), key=lambda item: item[1]):
                            if k not in del_regions:
                                regions[k] = cls_idx
                                cls_idx += 1

                        self.nn.user_metadata['class_mapping']['baselines'] = bls
                        self.nn.user_metadata['class_mapping']['regions'] = regions

                        # add new baseline/region types
                        cls_idx -= 1
                        for c in new_bls:
                            cls_idx += 1
                            self.nn.user_metadata['class_mapping']['baselines'][c] = cls_idx
                        for c in new_regions:
                            cls_idx += 1
                            self.nn.user_metadata['class_mapping']['regions'][c] = cls_idx
                    else:
                        raise ValueError(f'invalid resize parameter value {self.resize}')
                # backfill train_set/val_set mapping if key-equal as the actual
                # numbering in the train_set might be different
                self.train_set.dataset.class_mapping = self.nn.user_metadata['class_mapping']
                self.val_set.dataset.class_mapping = self.nn.user_metadata['class_mapping']

            # updates model's hyper params with user-defined ones
            self.nn.hyper_params = self.hyper_params

            # change topline/baseline switch
            loc = {None: 'centerline',
                   True: 'topline',
                   False: 'baseline'}

            if 'topline' not in self.nn.user_metadata:
                logger.warning(f'Setting baseline location to {loc[self.topline]} from unset model.')
            elif self.nn.user_metadata['topline'] != self.topline:
                from_loc = loc[self.nn.user_metadata['topline']]
                logger.warning(f'Changing baseline location from {from_loc} to {loc[self.topline]}.')
            self.nn.user_metadata['topline'] = self.topline

            logger.info('Training line types:')
            for k, v in self.train_set.dataset.class_mapping['baselines'].items():
                logger.info(f'  {k}\t{v}\t{self.train_set.dataset.class_stats["baselines"][k]}')
            logger.info('Training region types:')
            for k, v in self.train_set.dataset.class_mapping['regions'].items():
                logger.info(f'  {k}\t{v}\t{self.train_set.dataset.class_stats["regions"][k]}')

            if len(self.train_set) == 0:
                raise ValueError('No valid training data was provided to the train command. Please add valid XML data.')

            # set model type metadata field and dump class_mapping
            self.nn.model_type = 'segmentation'
            self.nn.user_metadata['class_mapping'] = self.val_set.dataset.class_mapping

            # for model size/trainable parameter output
            self.net = self.nn.nn

            torch.set_num_threads(max(self.num_workers, 1))

            # set up validation metrics after output classes have been determined
            self.val_px_accuracy = MultilabelAccuracy(average='micro', num_labels=self.train_set.dataset.num_classes)
            self.val_mean_accuracy = MultilabelAccuracy(average='macro', num_labels=self.train_set.dataset.num_classes)
            self.val_mean_iu = MultilabelJaccardIndex(average='macro', num_labels=self.train_set.dataset.num_classes)
            self.val_freq_iu = MultilabelJaccardIndex(average='weighted', num_labels=self.train_set.dataset.num_classes)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=1,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=1,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def configure_callbacks(self):
        callbacks = []
        if self.hyper_params['quit'] == 'early':
            callbacks.append(EarlyStopping(monitor='val_mean_iu',
                                           mode='max',
                                           patience=self.hyper_params['lag'],
                                           stopping_threshold=1.0))

        return callbacks

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return configure_optimizer_and_lr_scheduler(self.hyper_params,
                                                    self.nn.nn.parameters(),
                                                    len_train_set=len(self.train_set),
                                                    loss_tracking_mode='max')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lrate` in `warmup`
        # steps.
        if self.hyper_params['warmup'] and self.trainer.global_step < self.hyper_params['warmup']:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hyper_params['warmup'])
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hyper_params['lrate']

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hyper_params['warmup'] or self.trainer.global_step >= self.hyper_params['warmup']:
            # step OneCycleLR each batch if not in warmup phase
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()
            # step every other scheduler epoch-wise
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)
