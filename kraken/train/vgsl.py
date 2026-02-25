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
VGSL text recognition network trainer.
"""
import re
import torch
import logging
import lightning as L


from functools import partial
from typing import Optional, Union, TYPE_CHECKING
from collections import Counter
from torch.optim import lr_scheduler
from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics.text import CharErrorRate, WordErrorRate
from torch.utils.data import DataLoader, Subset, random_split

from kraken.lib.xml import XMLPage
from kraken.lib.codec import PytorchCodec
from kraken.lib.util import make_printable
from kraken.containers import Segmentation
from kraken.configs import (RecognitionInferenceConfig,
                            VGSLRecognitionTrainingConfig,
                            VGSLRecognitionTrainingDataConfig)
from kraken.lib.dataset import compute_confusions, global_align
from kraken.lib.dataset import (ArrowIPCRecognitionDataset, GroundTruthDataset,
                                ImageInputTransforms, PolygonGTDataset,
                                collate_sequences)
from kraken.lib.exceptions import KrakenEncodeException, KrakenInputException
from kraken.train.utils import validation_worker_init_fn, configure_optimizer_and_lr_scheduler, RecognitionTestMetrics

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from os import PathLike
    from kraken.models import BaseModel

__all__ = ['VGSLRecognitionDataModule', 'VGSLRecognitionModel']


class VGSLRecognitionDataModule(L.LightningDataModule):
    def __init__(self,
                 data_config: VGSLRecognitionTrainingDataConfig):
        """
        A LightningDataModule encapsulating the training data for a page
        segmentation model.

        Args:
            data_config: Configuration object to set dataset parameters.
        """
        super().__init__()
        self.save_hyperparameters()

        all_files = [getattr(data_config, x) for x in ['training_data', 'evaluation_data', 'test_data']]

        DatasetClass = GroundTruthDataset
        if data_config.format_type in ['xml', 'page', 'alto']:
            if data_config.binary_dataset_split:
                logger.warning('Internal binary dataset splits are enabled but using non-binary dataset files. Will be ignored.')
                data_config.binary_dataset_split = False

            def _parse_xml_set(ds_type, dataset) -> list[dict[str, Segmentation]]:
                if not dataset:
                    return None
                logger.info(f'Parsing {len(dataset) if dataset else 0} XML files for {ds_type} data')
                data = []
                for pos, file in enumerate(dataset):
                    try:
                        data.append({'page': XMLPage(file, filetype=data_config.format_type).to_container()})
                    except Exception as e:
                        logger.warning(f'Failed to parse {file}: {e}')
                return data

            training_data = _parse_xml_set('training', all_files[0])
            evaluation_data = _parse_xml_set('evaluation', all_files[1])
            test_data = _parse_xml_set('test', all_files[2])
            DatasetClass = partial(PolygonGTDataset, legacy_polygons=data_config.legacy_polygons)
        elif data_config.format_type == 'binary':
            DatasetClass = ArrowIPCRecognitionDataset
            training_data = [{'file': file} for file in all_files[0]] if all_files[0] else None
            evaluation_data = [{'file': file} for file in all_files[1]] if all_files[1] else None
            test_data = [{'file': file} for file in all_files[2]] if all_files[2] else None
        else:
            raise ValueError(f'format_type {data_config.format_type} not in [xml, page, alto, binary].')

        if training_data and evaluation_data:
            train_set = self._build_dataset(DatasetClass, training_data, augmentation=data_config.augment)
            self.train_set = Subset(train_set, range(len(train_set)))
            val_set = self._build_dataset(DatasetClass, evaluation_data)
            self.val_set = Subset(val_set, range(len(val_set)))
        elif training_data:
            train_set = self._build_dataset(DatasetClass, training_data)
            train_len = int(len(train_set) * data_config.partition)
            val_len = len(train_set) - train_len
            logger.info(f'No explicit validation data provided. Splitting off '
                        f'{val_len} (of {len(train_set)}) samples to validation '
                        'set. (Will disable alphabet mismatch detection.)')
            self.train_set, self.val_set = random_split(train_set, (train_len, val_len))
        elif test_data:
            test_set = self._build_dataset(DatasetClass, test_data)
            self.test_set = Subset(test_set, range(len(test_set)))
        else:
            raise ValueError('Invalid specification of training/evaluation/test data.')

    def _build_dataset(self,
                       DatasetClass,
                       training_data,
                       **kwargs):
        dataset = DatasetClass(normalization=self.hparams.data_config.normalization,
                               whitespace_normalization=self.hparams.data_config.normalize_whitespace,
                               reorder=self.hparams.data_config.bidi_reordering,
                               im_transforms=None,
                               **kwargs)

        for sample in training_data:
            try:
                dataset.add(**sample)
            except Exception as e:
                logger.warning(str(e))

        if self.hparams.data_config.format_type == 'binary' and (self.hparams.data_config.normalization or self.hparams.data_config.normalize_whitespace or self.hparams.data_config.bidi_reordering):
            logger.debug('Text transformations modifying alphabet selected. Rebuilding alphabet')
            dataset.rebuild_alphabet()

        return dataset

    def setup(self, stage: str = None):
        transforms = ImageInputTransforms(1,
                                          self.trainer.lightning_module.height,
                                          self.trainer.lightning_module.width,
                                          self.trainer.lightning_module.channels,
                                          (self.hparams.data_config.padding, 0),
                                          valid_norm=False)

        if stage in ['fit', None]:
            if getattr(self, 'train_set', None) is None or len(self.train_set) == 0:
                raise ValueError('No training data in dataset. Please supply some.')
            if getattr(self, 'val_set', None) is None or len(self.val_set) == 0:
                raise ValueError('No training data in dataset. Please supply some.')

            train_set = self.train_set.dataset
            val_set = self.val_set.dataset

            self.use_legacy_polygons = False

            # existing binary datasets might have been compiled with the legacy polygonizer
            if self.hparams.data_config.format_type == 'binary' and self.train_set:
                legacy_train_status = train_set.legacy_polygons_status
                if val_set.legacy_polygons_status != legacy_train_status:
                    logger.warning('Train and validation set have different legacy '
                                   f'polygon status: {legacy_train_status} and '
                                   f'{val_set.legacy_polygons_status}. Train set '
                                   'status prevails.')
                if legacy_train_status == "mixed":
                    logger.warning('Mixed legacy polygon status in training dataset. Consider recompilation.')
                    legacy_train_status = False
                if legacy_train_status != self.hparams.data_config.legacy_polygons:
                    logger.warning(f'Setting dataset legacy polygon status to {legacy_train_status} based on training set.')
                else:
                    logger.info(f'Setting dataset legacy polygon status to {legacy_train_status} based on training set.')
                self.use_legacy_polygons = legacy_train_status

            train_set.transforms = transforms
            val_set.transforms = transforms

            logger.info(f'Training set {len(train_set)} lines, validation set '
                        f'{len(val_set)} lines, alphabet {len(train_set.alphabet)} '
                        'symbols')
            alpha_diff_only_train = set(train_set.alphabet).difference(set(val_set.alphabet))
            alpha_diff_only_val = set(val_set.alphabet).difference(set(train_set.alphabet))
            if alpha_diff_only_train:
                logger.warning(f'alphabet mismatch: chars in training set only: '
                               f'{alpha_diff_only_train} (not included in accuracy test '
                               'during training)')
            if alpha_diff_only_val:
                logger.warning(f'alphabet mismatch: chars in validation set only: {alpha_diff_only_val} (not trained)')
            logger.info('grapheme\tcount')
            for k, v in sorted(train_set.alphabet.items(), key=lambda x: x[1], reverse=True):
                char = make_printable(k)
                if char == k:
                    char = '\t' + char
                logger.info(f'{char}\t{v}')
        elif stage == 'test':
            if getattr(self, 'test_set', None) is None or len(self.test_set) == 0:
                raise ValueError('No test data in dataset. Please supply some.')
            self.test_set.dataset.transforms = transforms
            self.test_set.dataset.no_encode()

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.trainer.lightning_module.hparams.config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_sequences)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=self.trainer.lightning_module.hparams.config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          collate_fn=collate_sequences,
                          worker_init_fn=validation_worker_init_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          shuffle=False,
                          batch_size=self.trainer.lightning_module.hparams.config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          collate_fn=collate_sequences,
                          worker_init_fn=validation_worker_init_fn)


class VGSLRecognitionModel(L.LightningModule):

    def __init__(self,
                 config: VGSLRecognitionTrainingConfig,
                 model: Optional['BaseModel'] = None):
        """
        A LightningModule encapsulating the training setup for a text
        recognition model.

        Args:
            config: A training configuration object
            model: A loaded model to use with the module. Intended to be set by
                   `VGSLRecognitionModel.load_from_weights()`.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        if not isinstance(config, VGSLRecognitionTrainingConfig):
            raise ValueError(f'config attribute is {type(config)} not VGSLRecognitionTrainingConfig.')

        self._loaded_model = model is not None

        if model:
            self.net = model

            if self.net.model_type and 'recognition' not in self.net.model_type:
                raise ValueError(f'Model {model} is of type {self.net.model_type} while `recognition` is expected.')

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

        self.example_input_array = torch.Tensor(self.batch,
                                                self.channels,
                                                self.height if self.height else 32,
                                                self.width if self.width else 400)

    def forward(self, x, seq_lens=None):
        return self.net(x, seq_lens)

    def training_step(self, batch, batch_idx):
        input, target = batch['image'], batch['target']
        # sequence batch
        if 'seq_lens' in batch:
            seq_lens, label_lens = batch['seq_lens'], batch['target_lens']
            target = (target, label_lens)
            o = self.net(input, seq_lens)
        else:
            o = self.net(input)

        seq_lens = o[1]
        output = o[0].log_softmax(1)
        target_lens = target[1]
        target = target[0]
        # height should be 1 by now
        if output.size(2) != 1:
            raise KrakenInputException('Expected dimension 3 to be 1, actual {}'.format(output.size(2)))
        output = output.squeeze(2)
        # NCW -> WNC
        loss = self.net.criterion(output.permute(2, 0, 1),  # type: ignore
                                  target,
                                  seq_lens,
                                  target_lens)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=batch['image'].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        preds, olens = self.net.forward(batch['image'], batch['seq_lens'])
        preds = preds.squeeze(2)
        idx = 0
        targets = []
        # decode packed target
        for offset in batch['target_lens']:
            targets.append(''.join([x[0] for x in self._val_codec.decode([(x, 0, 0, 0) for x in batch['target'][idx:idx + offset]])]))
            idx += offset
        for pred, target in zip([self.net.codec.decode(locs) for locs in RecognitionInferenceConfig().decoder(preds, olens)], targets):
            pred_str = ''.join(x[0] for x in pred)
            self.val_cer.update(pred_str, target)
            self.val_wer.update(pred_str, target)

        if self.logger and \
           self.trainer.state.stage != 'sanity_check' and \
           self.hparams.config.batch_size * batch_idx < 16 and \
           getattr(self.logger.experiment, 'add_image', None) is not None:
            for i in range(self.hparams.config.batch_size):
                count = self.hparams.config.batch_size * batch_idx + i
                if count < 16:
                    self.logger.experiment.add_image(f'Validation #{count}, target: {targets[i]}',
                                                     batch['image'][i],
                                                     self.global_step,
                                                     dataformats="CHW")
                    self.logger.experiment.add_text(f'Validation #{count}, target: {targets[i]}',
                                                    pred[i],
                                                    self.global_step)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            accuracy = 1.0 - self.val_cer.compute()
            word_accuracy = 1.0 - self.val_wer.compute()

            logger.info(f'validation run: total chars {self.val_cer.total} errors {self.val_cer.errors} accuracy {accuracy}')
            self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_word_accuracy', word_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_metric', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # reset metrics even if not sanity checking
        self.val_cer.reset()
        self.val_wer.reset()

    def on_test_epoch_start(self):
        self.errors = 0
        self.characters = Counter()
        self.algn_gt: list[str] = []
        self.algn_pred: list[str] = []

    def test_step(self, batch, batch_idx, test_dataloader=0):
        preds, olens = self.net.forward(batch['image'], batch['seq_lens'])
        preds = preds.squeeze(2)
        self.characters += Counter(''.join(batch['target']))
        for pred, target in zip([self.net.codec.decode(locs) for locs in RecognitionInferenceConfig().decoder(preds, olens)], batch['target']):
            pred_str = ''.join(x[0] for x in pred)
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
        self.test_cer_case_insensitive.compute()
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

            if self.logger and getattr(self.logger.experiment, 'add_image', None) is not None:
                train_set.no_encode()
                for i, idx in enumerate(torch.randint(len(train_set), (min(len(train_set), 16),))):
                    sample = train_set[idx.item()]
                    self.logger.experiment.add_image(f'train_set sample #{i}: {sample["target"]}', sample['image'])

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
                self.hparams.config.spec = self.net.spec

                if train_set.seg_type != self.net.seg_type:
                    logger.warning(f'Neural network has been trained on {self.net.seg_type} image information but training set is {train_set.seg_type}.')
            else:
                codec = self.trainer.datamodule.hparams.data_config.codec
                train_set.encode(codec)
                logger.info(f'Creating new model {self.hparams.config.spec} with {train_set.codec.max_label + 1} outputs')
                vgsl = self.hparams.config.spec.strip()
                self.hparams.config.spec = f'[{vgsl[1:-1]} O1c{train_set.codec.max_label + 1}]'
                from kraken.models import create_model
                self.net = create_model('TorchVGSLModel',
                                        model_type=['recognition'],
                                        vgsl=self.hparams.config.spec)
                # initialize weights
                self.net.init_weights()
                self.net.add_codec(train_set.codec)

            val_diff = set(val_set.alphabet).difference(
                set(train_set.codec.c2l.keys())
            )
            logger.info(f'Adding {len(val_diff)} dummy labels to validation set codec.')

            self._val_codec = self.net.codec.add_labels(val_diff)
            val_set.encode(self._val_codec)

            if self.net.one_channel_mode and train_set.im_mode != self.net.one_channel_mode:
                logger.warning(f'Neural network has been trained on mode {self.net.one_channel_mode} images, '
                               f'training set contains mode {train_set.im_mode} data. Consider binarizing your data.')

            self.net.user_metadata['metrics'] = []
            self.net.model_type = ['recognition']

            if not self.net.seg_type:
                logger.info(f'Setting seg_type to {train_set.seg_type}.')
                self.net.seg_type = train_set.seg_type

            data_cfg = self.trainer.datamodule.hparams.data_config
            use_legacy = self.trainer.datamodule.use_legacy_polygons
            force_legacy = data_cfg.legacy_polygons

            if data_cfg.format_type != 'binary' and force_legacy:
                desired_legacy = True
            elif use_legacy:
                desired_legacy = True
            else:
                desired_legacy = False

            if not self._loaded_model:
                if data_cfg.format_type != 'binary' and force_legacy:
                    logger.warning('the new model will be flagged to use legacy')
                elif data_cfg.format_type == 'binary':
                    if force_legacy and not use_legacy:
                        logger.warning('the new model will be flagged to use new')
                    elif (not force_legacy) and use_legacy:
                        logger.warning('the new model will be flagged to use legacy')

            if self._loaded_model and (not force_legacy) and self.net.use_legacy_polygons and not desired_legacy:
                logger.warning('model will be flagged to use new')

            self.net.use_legacy_polygons = desired_legacy

            self.trainer.datamodule.hparams.data_config.codec = self.net.codec
        elif stage == 'test':
            self.test_cer = CharErrorRate()
            self.test_cer_case_insensitive = CharErrorRate()
            self.test_wer = WordErrorRate()

    def on_load_checkpoint(self, checkpoint):
        """
        Reconstruct the model from the spec here and not in setup() as
        otherwise the weight loading will fail.
        """
        from kraken.models import create_model
        if not isinstance(checkpoint['hyper_parameters']['config'], VGSLRecognitionTrainingConfig):
            raise ValueError('Checkpoint is not a recognition model.')
        data_config = checkpoint['datamodule_hyper_parameters']['data_config']
        self.net = create_model('TorchVGSLModel',
                                model_type=['recognition'],
                                legacy_polygons=data_config.legacy_polygons,
                                seg_type=checkpoint['_seg_type'],
                                one_channel_mode=checkpoint['_one_channel_mode'],
                                vgsl=checkpoint['_module_config'].spec,
                                codec=data_config.codec.c2l)

        self.batch, self.channels, self.height, self.width = self.net.input

    def on_save_checkpoint(self, checkpoint):
        """
        Save hyperparameters a second time so we can set parameters that
        shouldn't be overwritten in on_load_checkpoint.
        """
        self.hparams.config.spec = self.net.spec
        checkpoint['_module_config'] = self.hparams.config
        checkpoint['_one_channel_mode'] = self.trainer.datamodule.train_set.dataset.im_mode
        checkpoint['_seg_type'] = self.trainer.datamodule.train_set.dataset.seg_type
        # populate validation metrics
        metrics = {k: v.item() if hasattr(v, 'item') else v
                   for k, v in self.trainer.callback_metrics.items()
                   if k.startswith('val_')}
        if metrics:
            self.net.user_metadata['metrics'].append((self.current_epoch, metrics))

    @classmethod
    def load_from_weights(cls,
                          path: Union[str, 'PathLike'],
                          config: VGSLRecognitionTrainingConfig) -> 'VGSLRecognitionModel':
        """
        Initializes the module from a model weights file.
        """
        from kraken.models import load_models
        models = load_models(path, tasks=['recognition'])
        if len(models) != 1:
            raise ValueError(f'Found {len(models)} recognition models in model file.')
        return cls(config=config, model=models[0])

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.config.quit == 'early':
            callbacks.append(EarlyStopping(monitor='val_accuracy',
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
