#
# Copyright 2026 Benjamin Kiessling
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
PP-OCRv6 text recognition network trainer.
"""
import math
import torch
import logging

from functools import partial
from typing import Optional, TYPE_CHECKING
from collections import Counter
from collections.abc import Callable

import torch.nn as nn
import torch.nn.functional as F

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics.text import CharErrorRate, WordErrorRate

from kraken.lib.codec import PytorchCodec
from kraken.lib.util import make_printable
from kraken.configs import (PPOCRv6RecognitionTrainingConfig,
                            PPOCRv6RecognitionTrainingDataConfig,
                            RecognitionInferenceConfig)
from kraken.lib import functional_im_transforms as F_t
from kraken.lib.dataset import compute_confusions, global_align, collate_sequences
from kraken.lib.exceptions import KrakenEncodeException
from kraken.lib.ppocr import MODEL_VARIANTS, PPOCRv6Model
from kraken.lib.ppocr.network import WIDTH_SUBSAMPLING
from kraken.lib.ppocr.nrtr import NRTRHead
from kraken.train.base import KrakenTrainerModule
from kraken.train.optim import MuonWithAuxAdam, get_parameter_groups
from kraken.train.utils import validation_worker_init_fn, configure_optimizer_and_lr_scheduler, RecognitionTestMetrics
from kraken.train.vgsl import VGSLRecognitionDataModule

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from kraken.models import BaseModel

__all__ = ['PPOCRv6RecognitionDataModule', 'PPOCRv6RecognitionModel']


def collate_static_width(batch, max_width: int):
    """
    Static-shape collation: pads the batch to exactly `max_width` so every
    batch has an identical shape. Samples not fitting `max_width` are already
    rejected at the dataset level.
    """
    out = collate_sequences(batch)
    width = out['image'].shape[-1]
    if width > max_width:
        raise ValueError(f'Batch of width {width} exceeds max_width ({max_width}).')
    if width < max_width:
        out['image'] = F.pad(out['image'], (0, max_width - width))
    return out


class PPOCRv6RecognitionDataModule(VGSLRecognitionDataModule):
    """
    Recognition datamodule for PP-OCRv6 models.

    Identical to the VGSL datamodule except for static shapes: every batch is
    padded to ``max_width`` and samples that are wider or whose targets do not
    fit into the subsampled output sequence are rejected at the dataset level.
    """

    def __init__(self, data_config: PPOCRv6RecognitionTrainingDataConfig):
        super().__init__(data_config)

    def _build_dataset(self, DatasetClass, training_data, **kwargs):
        return super()._build_dataset(DatasetClass,
                                      training_data,
                                      max_width=self.hparams.data_config.max_width,
                                      subsampling=WIDTH_SUBSAMPLING,
                                      **kwargs)

    @property
    def _collate_fn(self) -> Callable:
        return partial(collate_static_width,
                       max_width=self.hparams.data_config.max_width)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.trainer.lightning_module.hparams.config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=self.trainer.lightning_module.hparams.config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          collate_fn=self._collate_fn,
                          worker_init_fn=validation_worker_init_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          shuffle=False,
                          batch_size=self.trainer.lightning_module.hparams.config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          collate_fn=self._collate_fn,
                          worker_init_fn=validation_worker_init_fn)


class PPOCRv6RecognitionModel(KrakenTrainerModule):

    _task = 'recognition'
    _arch = 'ppocrv6'
    _model_class = PPOCRv6Model
    _config_class = PPOCRv6RecognitionTrainingConfig
    _data_config_class = PPOCRv6RecognitionTrainingDataConfig
    _data_module_class = PPOCRv6RecognitionDataModule

    def __init__(self,
                 config: PPOCRv6RecognitionTrainingConfig,
                 model: Optional['BaseModel'] = None):
        """
        A LightningModule encapsulating the training setup for a PP-OCRv6 text
        recognition model.

        Args:
            config: A training configuration object
            model: A loaded model to use with the module. Intended to be set by
                   `PPOCRv6RecognitionModel.load_from_weights()`.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        if not isinstance(config, PPOCRv6RecognitionTrainingConfig):
            raise ValueError(f'config attribute is {type(config)} not PPOCRv6RecognitionTrainingConfig.')

        self._loaded_model = model is not None

        if model:
            self.net = model

            if self.net.model_type and 'recognition' not in self.net.model_type:
                raise ValueError(f'Model {model} is of type {self.net.model_type} while `recognition` is expected.')

            self.batch, self.channels, self.height, self.width = self.net.input
            if self.net.variant != config.variant:
                logger.info(f'Loaded model is a {self.net.variant!r} variant; '
                            f'overriding configured {config.variant!r}.')
                config.variant = self.net.variant
            if self.height != config.height:
                logger.info(f'Loaded model expects line height {self.height}; '
                            f'overriding configured {config.height}.')
                config.height = self.height
        else:
            self.net = None
            # input geometry read by the datamodule's setup()
            self.batch = 1
            self.height = config.height
            self.width = 0
            self.channels = 3

        self._nrtr_pad_len = 0

        self.codec: Optional[PytorchCodec] = None
        self._val_codec: Optional[PytorchCodec] = None
        # CTC loss; class 0 is the blank
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')

        # guided training of CTC: an auxiliary NRTR decoder (training only)
        self.nrtr_head: Optional[nn.Module] = None
        self._bos = self._eos = None
        # label-smoothed CE over NRTR tokens, PAD (0) ignored
        self.nrtr_criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        # replaced by a compiled wrapper in setup()
        self._gtc_loss_fn = self._gtc_loss

    def on_fit_start(self):
        # silence the benign AccumulateGrad stream-mismatch warning from
        # torch.compile + DDP
        import torch.autograd.graph as autograd_graph
        fn = getattr(autograd_graph, 'set_warn_on_accumulate_grad_stream_mismatch', None)
        if fn is not None:
            fn(False)

    def forward(self, x, seq_lens=None):
        return self.net(x, seq_lens)

    # ------------------------------------------------------------------ setup
    def setup(self, stage: Optional[str] = None):
        # finalize models in case of appending/loading
        if stage in [None, 'fit']:
            self.val_cer = CharErrorRate()
            self.val_wer = WordErrorRate()

            if (codec := self.trainer.datamodule.hparams.data_config.codec):
                if not isinstance(codec, PytorchCodec):
                    logger.info('Instantiating codec')
                    self.trainer.datamodule.hparams.data_config.codec = PytorchCodec(codec)
                for k, v in self.trainer.datamodule.hparams.data_config.codec.c2l.items():
                    char = make_printable(k)
                    if char == k:
                        char = '\t' + char
                    logger.info(f'{char}\t{v}')

            logger.info('Encoding training set')
            train_set = self.trainer.datamodule.train_set.dataset
            val_set = self.trainer.datamodule.val_set.dataset

            if self.net:
                if self.hparams.config.resize == 'new' and self.trainer.datamodule.hparams.data_config.codec is not None:
                    codec = self.trainer.datamodule.hparams.data_config.codec
                elif self.net.codec is not None:
                    codec = self.net.codec
                else:
                    raise ValueError('No valid codec found in model.')

                codec.strict = True

                try:
                    train_set.encode(codec)
                except KrakenEncodeException:
                    alpha_diff = set(train_set.alphabet).difference(
                        set(codec.c2l.keys())
                    )
                    if self.hparams.config.resize == 'fail':
                        raise ValueError(f'Training data and model codec alphabets mismatch: {alpha_diff}')
                    elif self.hparams.config.resize == 'union':
                        logger.info(f'Resizing codec to include {len(alpha_diff)} new code points.')
                        codec = codec.add_labels(alpha_diff)
                        self.net.add_codec(codec)
                        logger.info(f'Resizing last layer in network to {codec.max_label + 1} outputs')
                        self.net.resize_output(codec.max_label + 1)
                        train_set.encode(codec)
                    elif self.hparams.config.resize == 'new':
                        logger.info('Resizing network or given codec to '
                                    f'{len(train_set.alphabet)} '
                                    'code sequences')
                        # same codec procedure as above, just with merging.
                        train_set.encode(None)
                        codec, del_labels = codec.merge(train_set.codec)
                        # Switch codec.
                        self.net.add_codec(codec)
                        logger.info(f'Deleting {len(del_labels)} output classes from network '
                                    f'({len(codec) - len(del_labels)} retained)')
                        self.net.resize_output(codec.max_label + 1, del_labels)
                        train_set.encode(codec)
                    else:
                        raise ValueError(f'invalid resize parameter value {self.hparams.config.resize}')
                codec.strict = False
                self.net.add_codec(codec)

                if train_set.seg_type != self.net.seg_type:
                    logger.warning(f'Neural network has been trained on {self.net.seg_type} image information but training set is {train_set.seg_type}.')
            else:
                codec = self.trainer.datamodule.hparams.data_config.codec
                train_set.encode(codec)
                num_classes = train_set.codec.max_label + 1
                logger.info(f'Building PP-OCRv6 {self.hparams.config.variant!r} recognizer '
                            f'with {num_classes} output classes ({self.channels}-channel, '
                            f'height {self.height}).')
                self.net = self._build_net(num_classes,
                                           train_set.codec.c2l,
                                           train_set.seg_type or 'baselines',
                                           None)

            self.codec = self.net.codec

            # build (or rebuild after an output resize) the auxiliary NRTR head
            num_classes = self.net.num_classes
            if self.nrtr_head is not None and self.nrtr_head.tgt_word_prj.out_features != num_classes + 2:
                logger.warning('Reinitializing auxiliary NRTR head after codec resize.')
                self.nrtr_head = None
            if self.nrtr_head is None:
                self.nrtr_head = self._build_nrtr_head(num_classes)

            # extend the codec with validation-only symbols so targets round-trip
            val_diff = set(val_set.alphabet).difference(
                set(self.codec.c2l.keys())
            )
            logger.info(f'Adding {len(val_diff)} dummy labels to validation set codec.')

            self._val_codec = self.codec.add_labels(val_diff)
            val_set.encode(self._val_codec)

            self.net.user_metadata['metrics'] = []

            if not self.net.seg_type:
                logger.info(f'Setting seg_type to {train_set.seg_type}.')
                self.net.seg_type = train_set.seg_type

            self.net.use_legacy_polygons = self.trainer.datamodule.use_legacy_polygons

            # Hand the finalised codec back to the datamodule.
            self.trainer.datamodule.hparams.data_config.codec = self.net.codec

            # fixed NRTR target length: longest repeat-free target fitting the
            # CTC output width plus BOS/EOS
            max_width = self.trainer.datamodule.hparams.data_config.max_width
            self._nrtr_pad_len = max_width // WIDTH_SUBSAMPLING + 2
            if self._nrtr_pad_len > (max_len := self.nrtr_head.positional_encoding.pe.shape[1]):
                raise ValueError(f'max_width {max_width} exceeds the NRTR decoder '
                                 f'capacity ({max_len} tokens).')
            self.net.nn.forward_train = torch.compile(self.net.nn.forward_train, dynamic=False)
            self._gtc_loss_fn = torch.compile(self._gtc_loss, dynamic=False)
        elif stage == 'test':
            if self.net is None:
                raise ValueError('No network to test; load a model before testing.')
            if self.codec is None:
                self.codec = self.net.codec
            if self.codec is None:
                raise ValueError('Model has no codec; cannot decode predictions.')
            self.test_cer = CharErrorRate()
            self.test_cer_case_insensitive = CharErrorRate()
            self.test_wer = WordErrorRate()

    def _build_nrtr_head(self, num_classes):
        # token vocab: 0=PAD/blank, 1..max_label=codepoints, then BOS, EOS
        max_label = num_classes - 1
        self._bos = max_label + 1
        self._eos = max_label + 2
        cfg = MODEL_VARIANTS[self.hparams.config.variant]['nrtr']
        return NRTRHead(in_channels=self.net.nn.backbone.out_channels,
                        vocab_size=max_label + 3,
                        d_model=cfg['dim'],
                        num_decoder_layers=cfg['num_decoder_layers'],
                        max_len=2048)

    def _build_net(self, num_classes, codec_c2l, seg_type, one_channel_mode):
        return PPOCRv6Model(variant=self.hparams.config.variant,
                            num_classes=num_classes,
                            height=self.height,
                            codec=codec_c2l,
                            seg_type=seg_type,
                            one_channel_mode=one_channel_mode,
                            legacy_polygons=False)

    def _gtc_loss(self, feat, tgt):
        # auxiliary NRTR decoder shares the backbone feature
        logits = self.nrtr_head(feat, tgt[:, :-1])          # (N, T-1, V)
        return self.nrtr_criterion(logits.reshape(-1, logits.shape[-1]),
                                   tgt[:, 1:].reshape(-1))

    def _nrtr_targets(self, target, target_lens, device):
        # build [BOS, c1, ..., cn, EOS] per sample, right-padded with 0 (PAD,
        # ignored by the CE) to a fixed token length for static compilation.
        # The dataset guarantees targets fit within the label budget.
        target = target.to(device).long()
        target_lens = target_lens.to(device)
        rows = torch.arange(target_lens.numel(), device=device)
        starts = target_lens.cumsum(0) - target_lens
        cols = torch.arange(target.numel(), device=device) - starts.repeat_interleave(target_lens) + 1
        padded = torch.zeros(target_lens.numel(), self._nrtr_pad_len, dtype=torch.long, device=device)
        padded[:, 0] = self._bos
        padded[rows.repeat_interleave(target_lens), cols] = target
        padded[rows, target_lens + 1] = self._eos
        return padded

    def training_step(self, batch, batch_idx):
        logits, out_lens, feat = self.net.nn.forward_train(batch['image'], batch['seq_lens'])
        # CTC: logits (N, C, 1, W) -> log-softmax over class dim, drop height,
        # permute to (W, N, C) as expected by nn.CTCLoss.
        log_probs = logits.log_softmax(1).squeeze(2).permute(2, 0, 1)
        ctc_loss = self.criterion(log_probs, batch['target'], out_lens, batch['target_lens'])
        bs = batch['image'].shape[0]
        self.log('train_ctc_loss', ctc_loss, on_step=True, on_epoch=True,
                 batch_size=bs, sync_dist=True)

        # GTC: auxiliary NRTR decoder shares the backbone feature.
        tgt = self._nrtr_targets(batch['target'], batch['target_lens'], feat.device)
        nrtr_loss = self._gtc_loss_fn(feat, tgt)
        loss = ctc_loss + nrtr_loss
        self.log('train_gtc_loss', nrtr_loss, on_step=True, on_epoch=True,
                 batch_size=bs, sync_dist=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=bs, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, out_lens = self.net(batch['image'], batch['seq_lens'])
        preds = logits.softmax(1).squeeze(2)  # (N, C, W)

        # Decode packed targets (encoded with the extended validation codec).
        targets = []
        idx = 0
        for offset in batch['target_lens'].tolist():
            chunk = [(int(x), 0, 0, 0.0) for x in batch['target'][idx:idx + offset]]
            targets.append(''.join(c[0] for c in self._val_codec.decode(chunk)))
            idx += offset

        decoded = [self.codec.decode(locs) for locs in RecognitionInferenceConfig().decoder(preds, out_lens)]
        for pred, target in zip(decoded, targets):
            pred_str = ''.join(c[0] for c in pred)
            self.val_cer.update(pred_str, target)
            self.val_wer.update(pred_str, target)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            accuracy = 1.0 - self.val_cer.compute()
            word_accuracy = 1.0 - self.val_wer.compute()

            logger.info(f'validation run: total chars {self.val_cer.total} errors {self.val_cer.errors} accuracy {accuracy}')
            self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_word_accuracy', word_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val_metric', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        # reset metrics even if not sanity checking
        self.val_cer.reset()
        self.val_wer.reset()

    def on_test_epoch_start(self):
        self.errors = 0
        self.characters = Counter()
        self.algn_gt: list[str] = []
        self.algn_pred: list[str] = []

        # mirror the ground truth normalization onto predictions (reordering
        # excluded as the network already emits in display order)
        data_config = self.trainer.datamodule.hparams.data_config
        self._pred_transforms: list[Callable[[str], str]] = []
        if data_config.normalization:
            self._pred_transforms.append(partial(F_t.text_normalize,
                                                 normalization=data_config.normalization))
        if data_config.normalize_whitespace:
            self._pred_transforms.append(F_t.text_whitespace_normalize)

    def test_step(self, batch, batch_idx, test_dataloader=0):
        preds, olens = self.net(batch['image'], batch['seq_lens'])
        preds = preds.squeeze(2)
        self.characters += Counter(''.join(batch['target']))
        for pred, target in zip([self.codec.decode(locs) for locs in RecognitionInferenceConfig().decoder(preds, olens)], batch['target']):
            pred_str = ''.join(x[0] for x in pred)
            for func in self._pred_transforms:
                pred_str = func(pred_str)
            c, algn1, algn2 = global_align(target, pred_str)
            self.errors += c
            self.algn_gt.extend(algn1)
            self.algn_pred.extend(algn2)

            self.test_cer.update(pred_str, target)
            self.test_cer_case_insensitive.update(pred_str.lower(), target.lower())
            self.test_wer.update(pred_str, target)

    def on_test_epoch_end(self):
        accuracy = (1.0 - self.test_cer.compute()).item()
        ci_accuracy = (1.0 - self.test_cer_case_insensitive.compute()).item()
        word_accuracy = (1.0 - self.test_wer.compute()).item()

        confusions, scripts, ins, dels, subs = compute_confusions(self.algn_gt, self.algn_pred)

        # reset metrics even if not sanity checking
        self.test_cer.reset()
        self.test_cer_case_insensitive.reset()
        self.test_wer.reset()

        self.test_metrics = RecognitionTestMetrics(character_counts=self.characters,
                                                   num_errors=self.errors,
                                                   cer=accuracy,
                                                   wer=word_accuracy,
                                                   case_insensitive_cer=ci_accuracy,
                                                   confusions=confusions,
                                                   scripts=scripts,
                                                   insertions=ins,
                                                   deletes=dels,
                                                   substitutions=subs)

    def _build_net_from_checkpoint(self, checkpoint):
        from kraken.models import create_model
        data_config = checkpoint['datamodule_hyper_parameters']['data_config']
        config = checkpoint['_module_config']
        return create_model('PPOCRv6Model',
                            model_type=['recognition'],
                            variant=config.variant,
                            num_classes=data_config.codec.max_label + 1,
                            height=config.height,
                            codec=data_config.codec.c2l,
                            seg_type=checkpoint['_seg_type'],
                            one_channel_mode=checkpoint['_one_channel_mode'],
                            legacy_polygons=data_config.legacy_polygons)

    def _post_load_checkpoint(self, checkpoint):
        super()._post_load_checkpoint(checkpoint)
        self.codec = self.net.codec
        # rebuild the auxiliary NRTR head if the checkpoint carries its weights
        if any(k.startswith('nrtr_head.') for k in checkpoint.get('state_dict', {})):
            self.nrtr_head = self._build_nrtr_head(self.net.num_classes)

    def _save_checkpoint_extras(self, checkpoint):
        checkpoint['_one_channel_mode'] = self.trainer.datamodule.train_set.dataset.im_mode
        checkpoint['_seg_type'] = self.trainer.datamodule.train_set.dataset.seg_type

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.config.quit == 'early':
            callbacks.append(EarlyStopping(monitor='val_accuracy',
                                           mode='max',
                                           patience=self.hparams.config.lag,
                                           stopping_threshold=1.0))

        return callbacks

    @property
    def _uses_muon(self) -> bool:
        return self.hparams.config.optimizer == 'AdamW+Muon'

    def configure_optimizers(self):
        cfg = self.hparams.config

        if not self._uses_muon:
            params = [p for m in (self.net, self.nrtr_head) for p in m.parameters()]
            return configure_optimizer_and_lr_scheduler(cfg,
                                                        params,
                                                        len_train_set=len(self.trainer.datamodule.train_set),
                                                        loss_tracking_mode='max')

        if cfg.schedule not in ('cosine', 'constant'):
            raise ValueError(f'Learning rate schedule {cfg.schedule!r} is not supported with the '
                             'AdamW+Muon optimizer. Use `cosine` or `constant`, or select a '
                             'standard optimizer.')
        opt_modules = nn.ModuleList([self.net, self.nrtr_head])
        groups = get_parameter_groups(opt_modules, base_lr=cfg.lrate)
        for g in groups:
            logger.info(f"optimizer group '{g['name']}': "
                        f"{sum(p.numel() for p in g['params']):,} params "
                        f"(muon={g['use_muon']})")
        opt = MuonWithAuxAdam(groups, lr=cfg.lrate, weight_decay=cfg.weight_decay,
                              momentum=cfg.momentum)

        if cfg.schedule != 'cosine':
            return opt

        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup = max(int(cfg.warmup), 0)
        min_ratio = cfg.cos_min_lr / cfg.lrate

        def lr_lambda(step):
            if warmup and step < warmup:
                return step / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total_steps - warmup))
            progress = min(1.0, max(0.0, progress))
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = lr_scheduler.LambdaLR(opt, lr_lambda)
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        if self._uses_muon:
            # warmup is baked into the LambdaLR schedule
            return

        # linear warmup between 0 and the initial learning rate `lrate` in `warmup`
        # steps.
        if self.hparams.config.warmup and self.trainer.global_step < self.hparams.config.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.config.warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.config.lrate

    def lr_scheduler_step(self, scheduler, metric):
        if self._uses_muon:
            # per-step warmup+cosine schedule
            scheduler.step()
            return
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
