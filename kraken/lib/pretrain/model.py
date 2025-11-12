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
Pytorch-lightning modules for recognition model pretraining.

Pretraining is based on an image inpainting surrogate task that aims to
reconstruct randomly sampled masked patches from the initial convolutional
feature maps that have been replaced with a learnable embedding. The model is
trained with a contrastive loss where negative samples are randomly generated
from the unmasked parts of the sequence.

Apart from an improved sampling method the implementation is mostly a faithful
adaptation of:

Vogler, Nikolai, et al. "Lacuna Reconstruction: Self-supervised Pre-training
for Low-Resource Historical Document Transcription." arXiv preprint
arXiv:2112.08692 (2021).
"""
import logging
from itertools import chain
from typing import TYPE_CHECKING, Optional, Union

import lightning as L
import torch
import torch.nn.functional as F

from torchmetrics import MeanMetric

from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.utilities.memory import (garbage_collection_cuda,
                                                is_oom_error)
from torch.optim import lr_scheduler

from kraken.configs import VGSLPreTrainingConfig
from kraken.lib import vgsl
from kraken.lib.vgsl import layers
from kraken.models import BaseModel
from kraken.lib.codec import PytorchCodec
from kraken.lib.pretrain.layers import Wav2Vec2Mask
from kraken.train import CRNNRecognitionDataModule
from kraken.train.utils import configure_optimizer_and_lr_scheduler
from kraken.lib.dataset import ImageInputTransforms

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


class PretrainDataModule(CRNNRecognitionDataModule):

    def _build_dataset(self,
                       DatasetClass,
                       training_data,
                       **kwargs):
        dataset = DatasetClass(normalization=self.hparams.data_config.normalization,
                               whitespace_normalization=self.hparams.data_config.normalize_whitespace,
                               reorder=self.hparams.data_config.bidi_reordering,
                               im_transforms=None,
                               skip_empty_lines=False,
                               **kwargs)

        for sample in training_data:
            try:
                dataset.add(**sample)
            except Exception as e:
                logger.warning(str(e))
        if self.hparams.data_config.format_type == 'binary' and (self.hparams.data_config.normalization or
                                                                 self.hparams.data_config.normalize_whitespace or
                                                                 self.hparams.data_config.bidi_reordering):
            logger.debug('Text transformations modifying alphabet selected. Rebuilding alphabet')
            dataset.rebuild_alphabet()

        return dataset

    def setup(self, stage: Optional[str] = None):
        if stage in ['fit', None]:
            if getattr(self, 'train_set', None) is None or len(self.train_set) == 0:
                raise ValueError('No training data in dataset. Please supply some.')
            if getattr(self, 'val_set', None) is None or len(self.val_set) == 0:
                raise ValueError('No training data in dataset. Please supply some.')

            transforms = ImageInputTransforms(1,
                                              self.trainer.lightning_module.height,
                                              self.trainer.lightning_module.width,
                                              self.trainer.lightning_module.channels,
                                              (self.hparams.data_config.padding, 0),
                                              valid_norm=False)

            self.train_set.dataset.transforms = transforms
            self.val_set.dataset.transforms = transforms

            self.train_set.dataset.no_encode()
            self.val_set.dataset.no_encode()

            self.hparams.data_config.codec = PytorchCodec(' ')



class RecognitionPretrainModel(L.LightningModule):
    def __init__(self,
                 config: VGSLPreTrainingConfig,
                 model: Optional[BaseModel] = None):
        """
        A LightningModule encapsulating the unsupervised pretraining setup for
        a text recognition model.
        """
        super().__init__()
        self.save_hyperparameters()

        if model:
            self.net = model

            self.batch, self.channels, self.height, self.width = self.net.input
        else:
            from kraken.models import create_model
            self.net = create_model('TorchVGSLModel',
                                    vgsl=self.hparams.config.spec)
            self.net.init_weights()

        self.batch, self.channels, self.height, self.width = self.net.input

        for idx, layer in enumerate(self.net.nn.children()):
            if isinstance(layer, layers.TransposedSummarizingRNN):
                break

        self.features = self.net.nn[:idx]
        self.wav2vec2mask = Wav2Vec2Mask(self.net.nn[idx-1].output_shape[1],
                                         self.net.nn[-1].output_shape[1],
                                         self.hparams.config.mask_width,
                                         self.hparams.config.mask_prob,
                                         self.hparams.config.num_negatives)

        # add dummy codec and output layer
        if not isinstance(self.net.nn[-1], layers.LinSoftmax):
            self.net.append(len(self.net.nn), "[O1c2]")

        self.encoder = self.net.nn[idx:]

        self.val_ce = MeanMetric()

        self.example_input_array = torch.Tensor(self.batch,
                                                self.channels,
                                                self.height if self.height else 32,
                                                self.width if self.width else 400)

    def forward(self, x, seq_lens=None):
        return self.net(x, seq_lens)

    def _step(self, batch, batch_idx):
        try:
            # sequence batch
            if 'seq_lens' in batch:
                output = self.features(batch['image'], batch['seq_lens'])
            else:
                output = self.features(batch['image'])

            mask_output = self.wav2vec2mask(*output)

            # run contextual encoder, i.e. recurrent layers
            output, seq_lens = self.encoder(mask_output['output'], mask_output['seq_len'])

            # unmasked features in encoder output domain
            y = mask_output['unmasked_samples']
            # negative samples
            negatives = mask_output['negative_samples']
            N, C, H, W = output.shape
            output = output.transpose(1, 3).reshape(-1, W, C)
            # masked features after encoder
            x = output[mask_output['mask']].reshape_as(y)
            mask_n_neg = torch.cat([y.unsqueeze(0), negatives], dim=0)
            logits = torch.cosine_similarity(x.float(), mask_n_neg.float(), dim=-1).type_as(x)

            targets = logits.new_zeros(logits.size(1) * logits.size(2), dtype=torch.long)

            logits = logits.transpose(0, 2)
            logits = logits.reshape(-1, logits.size(-1))
            logits /= self.hparams.config.logit_temp

            loss = F.cross_entropy(logits, targets)
            return logits, targets, loss
        except RuntimeError as e:
            if is_oom_error(e):
                logger.warning('Out of memory error in trainer. Skipping batch and freeing caches.')
                garbage_collection_cuda()
            else:
                raise

    def validation_step(self, batch, batch_idx):
        o = self._step(batch, batch_idx)
        if o is not None:
            logits, targets, loss = o
            self.val_ce.update(loss)
            self.log('CE', loss, on_step=True, on_epoch=True)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            ce = self.val_ce.compute()
            logger.info(f'validation run: cross_enctropy: {ce}')
            self.log('val_ce', ce, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_ce.reset()

    def training_step(self, batch, batch_idx):
        o = self._step(batch, batch_idx)
        if o is not None:
            _, _, loss = o
            self.log('CE',
                     loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            return loss

    def on_load_checkpoint(self, checkpoint):
        """
        Reconstruct the model from the spec here and not in setup() as
        otherwise the weight loading will fail.
        """
        from kraken.lib import vgsl  # NOQA
        from kraken.models import create_model
        if not isinstance(checkpoint['hyper_parameters']['config'], VGSLPreTrainingConfig):
            raise ValueError('Checkpoint is not a recognition model.')

        data_config = checkpoint['datamodule_hyper_parameters']['data_config']
        self.net = create_model('TorchVGSLModel',
                                model_type='recognition',
                                use_legacy_polygons=data_config.use_legacy_polygons,
                                seg_type=checkpoint['_seg_type'],
                                one_channel_mode=checkpoint['_one_channel_mode'],
                                vgsl=checkpoint['_module_config'].spec,
                                codec=data_config['codec'])

        self.batch, self.channels, self.height, self.width = self.net.input

    def on_save_checkpoint(self, checkpoint):
        """
        Save hyperparameters a second time so we can set parameters that
        shouldn't be overwritten in on_load_checkpoint.
        """
        checkpoint['_module_config'] = self.hparams.config
        checkpoint['_one_channel_mode'] = self.trainer.datamodule.train_set.im_mode
        checkpoint['_seg_type'] = self.trainer.datamodule.train_set.seg_type

    @classmethod
    def load_from_weights(cls,
                          path: Union[str, 'PathLike'],
                          config: VGSLPreTrainingConfig) -> 'RecognitionPretrainModel':
        """
        Initializes the module from a model weights file.
        """
        from kraken.models import load_models
        models = load_models(path, tasks=['recognition'])
        if len(models) != 1:
            raise ValueError(f'Found {len(models)} recognition models in model file.')
        return cls(config=config, model=models[0])

    def configure_optimizers(self):
        return configure_optimizer_and_lr_scheduler(self.hparams.config,
                                                    chain(self.features.parameters(),
                                                          self.wav2vec2mask.parameters(),
                                                          self.encoder.parameters()),
                                                    len_train_set=len(self.trainer.datamodule.train_set),
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

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.config.quit == 'early':
            callbacks.append(EarlyStopping(monitor='CE',
                                           mode='min',
                                           patience=self.hparams.config.lag,
                                           stopping_threshold=0.0))

        return callbacks
