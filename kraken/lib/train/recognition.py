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
Text recognition model
"""
import re
import torch
import logging
import warnings
import numpy as np
import lightning as L
from typing import (TYPE_CHECKING, Any, Dict, Literal, Optional,
                    Sequence, Union)
from functools import partial

from lightning.pytorch.callbacks import EarlyStopping

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset, random_split
from torchmetrics.text import CharErrorRate, WordErrorRate

from kraken.containers import Segmentation
from kraken.lib import default_specs, models, vgsl
from kraken.lib.codec import PytorchCodec
from kraken.lib.dataset import (ArrowIPCRecognitionDataset, GroundTruthDataset,
                                ImageInputTransforms, PolygonGTDataset,
                                collate_sequences)
from kraken.lib.exceptions import KrakenEncodeException, KrakenInputException
from kraken.lib.util import make_printable, parse_gt_path
from kraken.lib.xml import XMLPage

from .utils import _configure_optimizer_and_lr_scheduler

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


class RecognitionModel(L.LightningModule):
    """
    A LightningModule encapsulating the training setup for a text
    recognition model.

    Setup parameters (load, training_data, evaluation_data, ....) are
    named, model hyperparameters (everything in
    `kraken.lib.default_specs.RECOGNITION_HYPER_PARAMS`) are in in the
    `hyper_params` argument.

    Args:
        hyper_params (dict): Hyperparameter dictionary containing all fields
                             from
                             kraken.lib.default_specs.RECOGNITION_HYPER_PARAMS
        **kwargs: Setup parameters, i.e. CLI parameters of the train() command.
    """
    def __init__(self,
                 hyper_params: Dict[str, Any] = None,
                 output: str = 'model',
                 spec: str = default_specs.RECOGNITION_SPEC,
                 append: Optional[int] = None,
                 model: Optional[Union['PathLike', str]] = None,
                 reorder: Union[bool, str] = True,
                 training_data: Union[Sequence[Union['PathLike', str]], Sequence[Dict[str, Any]]] = None,
                 evaluation_data: Optional[Union[Sequence[Union['PathLike', str]], Sequence[Dict[str, Any]]]] = None,
                 partition: Optional[float] = 0.9,
                 binary_dataset_split: bool = False,
                 num_workers: int = 1,
                 load_hyper_parameters: bool = False,
                 force_binarization: bool = False,
                 format_type: Literal['path', 'alto', 'page', 'xml', 'binary'] = 'path',
                 codec: Optional[Dict] = None,
                 resize: Literal['fail', 'both', 'new', 'add', 'union'] = 'fail',
                 legacy_polygons: bool = False):
        super().__init__()
        self.legacy_polygons = legacy_polygons
        hyper_params_ = default_specs.RECOGNITION_HYPER_PARAMS.copy()
        if model:
            logger.info(f'Loading existing model from {model} ')
            self.nn = vgsl.TorchVGSLModel.load_model(model)

            if self.nn.model_type not in [None, 'recognition']:
                raise ValueError(f'Model {model} is of type {self.nn.model_type} while `recognition` is expected.')

            if load_hyper_parameters:
                hp = self.nn.hyper_params
            else:
                hp = {}
            hyper_params_.update(hp)
        else:
            self.nn = None

        if hyper_params:
            hyper_params_.update(hyper_params)
        self.hyper_params = hyper_params_
        self.save_hyperparameters()

        self.reorder = reorder
        self.append = append
        self.model = model
        self.num_workers = num_workers
        if resize == "add":
            resize = "union"
            warnings.warn("'add' value for resize has been deprecated. Use 'union' instead.", DeprecationWarning)
        elif resize == "both":
            resize = "new"
            warnings.warn("'both' value for resize has been deprecated. Use 'new' instead.", DeprecationWarning)

        self.resize = resize
        self.format_type = format_type
        self.output = output

        self.best_epoch = -1
        self.best_metric = 0.0
        self.best_model = None

        DatasetClass = GroundTruthDataset
        valid_norm = True
        if format_type in ['xml', 'page', 'alto']:
            logger.info(f'Parsing {len(training_data)} XML files for training data')
            training_data = [{'page': XMLPage(file, format_type).to_container()} for file in training_data]
            if evaluation_data:
                logger.info(f'Parsing {len(evaluation_data)} XML files for validation data')
                evaluation_data = [{'page': XMLPage(file, format_type).to_container()} for file in evaluation_data]
            if binary_dataset_split:
                logger.warning('Internal binary dataset splits are enabled but using non-binary dataset files. Will be ignored.')
                binary_dataset_split = False
            DatasetClass = partial(PolygonGTDataset, legacy_polygons=legacy_polygons)
            valid_norm = False
        elif format_type == 'binary':
            DatasetClass = ArrowIPCRecognitionDataset
            valid_norm = False
            logger.info(f'Got {len(training_data)} binary dataset files for training data')
            training_data = [{'file': file} for file in training_data]
            if evaluation_data:
                logger.info(f'Got {len(evaluation_data)} binary dataset files for validation data')
                evaluation_data = [{'file': file} for file in evaluation_data]
        elif format_type == 'path':
            if force_binarization:
                logger.warning('Forced binarization enabled in `path` mode. Will be ignored.')
                force_binarization = False
            if binary_dataset_split:
                logger.warning('Internal binary dataset splits are enabled but using non-binary dataset files. Will be ignored.')
                binary_dataset_split = False
            logger.info(f'Got {len(training_data)} line strip images for training data')
            training_data = [{'line': parse_gt_path(im)} for im in training_data]
            if evaluation_data:
                logger.info(f'Got {len(evaluation_data)} line strip images for validation data')
                evaluation_data = [{'line': parse_gt_path(im)} for im in evaluation_data]
            valid_norm = True
        # format_type is None. Determine training type from container class types
        elif not format_type:
            if training_data[0].type == 'baselines':
                DatasetClass = partial(PolygonGTDataset, legacy_polygons=legacy_polygons)
                valid_norm = False
            else:
                if force_binarization:
                    logger.warning('Forced binarization enabled with box lines. Will be ignored.')
                    force_binarization = False
                if binary_dataset_split:
                    logger.warning('Internal binary dataset splits are enabled but using non-binary dataset files. Will be ignored.')
                    binary_dataset_split = False
            samples = []
            for sample in training_data:
                if isinstance(sample, Segmentation):
                    samples.append({'page': sample})
                else:
                    samples.append({'line': sample})
            training_data = samples
            if evaluation_data:
                samples = []
                for sample in evaluation_data:
                    if isinstance(sample, Segmentation):
                        samples.append({'page': sample})
                    else:
                        samples.append({'line': sample})
                evaluation_data = samples
        else:
            raise ValueError(f'format_type {format_type} not in [alto, page, xml, path, binary].')

        spec = spec.strip()
        if spec[0] != '[' or spec[-1] != ']':
            raise ValueError(f'VGSL spec {spec} not bracketed')
        self.spec = spec
        # preparse input sizes from vgsl string to seed ground truth data set
        # sizes and dimension ordering.
        if not self.nn:
            blocks = spec[1:-1].split(' ')
            m = re.match(r'(\d+),(\d+),(\d+),(\d+)', blocks[0])
            if not m:
                raise ValueError(f'Invalid input spec {blocks[0]}')
            batch, height, width, channels = [int(x) for x in m.groups()]
        else:
            batch, channels, height, width = self.nn.input

        self.transforms = ImageInputTransforms(batch,
                                               height,
                                               width,
                                               channels,
                                               (self.hparams.hyper_params['pad'], 0),
                                               valid_norm,
                                               force_binarization)

        self.example_input_array = torch.Tensor(batch,
                                                channels,
                                                height if height else 32,
                                                width if width else 400)

        if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
            logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
            torch.multiprocessing.set_sharing_strategy('file_system')

        val_set = None
        if evaluation_data:
            train_set = self._build_dataset(DatasetClass, training_data)
            self.train_set = Subset(train_set, range(len(train_set)))
            val_set = self._build_dataset(DatasetClass, evaluation_data)
            self.val_set = Subset(val_set, range(len(val_set)))
        elif binary_dataset_split:
            train_set = self._build_dataset(DatasetClass, training_data, split_filter='train')
            self.train_set = Subset(train_set, range(len(train_set)))
            val_set = self._build_dataset(DatasetClass, training_data, split_filter='validation')
            self.val_set = Subset(val_set, range(len(val_set)))
            logger.info(f'Found {len(self.train_set)} (train) / {len(self.val_set)} (val) samples in pre-encoded dataset')
        else:
            train_set = self._build_dataset(DatasetClass, training_data)
            train_len = int(len(train_set)*partition)
            val_len = len(train_set) - train_len
            logger.info(f'No explicit validation data provided. Splitting off '
                        f'{val_len} (of {len(train_set)}) samples to validation '
                        'set. (Will disable alphabet mismatch detection.)')
            self.train_set, self.val_set = random_split(train_set, (train_len, val_len))

        if len(self.train_set) == 0 or len(self.val_set) == 0:
            raise ValueError('No valid training data was provided to the train '
                             'command. Please add valid XML, line, or binary data.')

        if format_type == 'binary':
            legacy_train_status = self.train_set.dataset.legacy_polygons_status
            if self.val_set.dataset.legacy_polygons_status != legacy_train_status:
                logger.warning('Train and validation set have different legacy '
                               f'polygon status: {legacy_train_status} and '
                               f'{self.val_set.dataset.legacy_polygons_status}. Train set '
                               'status prevails.')
            if legacy_train_status == "mixed":
                logger.warning('Mixed legacy polygon status in training dataset. Consider recompilation.')
                legacy_train_status = False
            if legacy_polygons != legacy_train_status:
                logger.warning(f'Setting dataset legacy polygon status to {legacy_train_status} based on training set.')
                self.legacy_polygons = legacy_train_status

        logger.info(f'Training set {len(self.train_set)} lines, validation set '
                    f'{len(self.val_set)} lines, alphabet {len(train_set.alphabet)} '
                    'symbols')
        alpha_diff_only_train = set(self.train_set.dataset.alphabet).difference(set(self.val_set.dataset.alphabet))
        alpha_diff_only_val = set(self.val_set.dataset.alphabet).difference(set(self.train_set.dataset.alphabet))
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

        if codec:
            logger.info('Instantiating codec')
            self.codec = PytorchCodec(codec)
            for k, v in self.codec.c2l.items():
                char = make_printable(k)
                if char == k:
                    char = '\t' + char
                logger.info(f'{char}\t{v}')
        else:
            self.codec = None

        logger.info('Encoding training set')

        self.val_cer = CharErrorRate()
        self.val_wer = WordErrorRate()

    def _build_dataset(self,
                       DatasetClass,
                       training_data,
                       **kwargs):
        dataset = DatasetClass(normalization=self.hparams.hyper_params['normalization'],
                               whitespace_normalization=self.hparams.hyper_params['normalize_whitespace'],
                               reorder=self.reorder,
                               im_transforms=self.transforms,
                               augmentation=self.hparams.hyper_params['augment'],
                               **kwargs)

        for sample in training_data:
            try:
                dataset.add(**sample)
            except KrakenInputException as e:
                logger.warning(str(e))
        if self.format_type == 'binary' and (self.hparams.hyper_params['normalization'] or
                                             self.hparams.hyper_params['normalize_whitespace'] or
                                             self.reorder):
            logger.debug('Text transformations modifying alphabet selected. Rebuilding alphabet')
            dataset.rebuild_alphabet()

        return dataset

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
        output = o[0]
        target_lens = target[1]
        target = target[0]
        # height should be 1 by now
        if output.size(2) != 1:
            raise KrakenInputException('Expected dimension 3 to be 1, actual {}'.format(output.size(2)))
        output = output.squeeze(2)
        # NCW -> WNC
        loss = self.nn.criterion(output.permute(2, 0, 1),  # type: ignore
                                 target,
                                 seq_lens,
                                 target_lens)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.rec_nn.predict_string(batch['image'], batch['seq_lens'])
        idx = 0
        decoded_targets = []
        for offset in batch['target_lens']:
            decoded_targets.append(''.join([x[0] for x in self.val_codec.decode([(x, 0, 0, 0) for x in batch['target'][idx:idx+offset]])]))
            idx += offset
        self.val_cer.update(pred, decoded_targets)
        self.val_wer.update(pred, decoded_targets)

        if self.logger and self.trainer.state.stage != 'sanity_check' and self.hparams.hyper_params["batch_size"] * batch_idx < 16:
            for i in range(self.hparams.hyper_params["batch_size"]):
                count = self.hparams.hyper_params["batch_size"] * batch_idx + i
                if count < 16:
                    self.logger.experiment.add_image(f'Validation #{count}, target: {decoded_targets[i]}',
                                                     batch['image'][i],
                                                     self.global_step,
                                                     dataformats="CHW")
                    self.logger.experiment.add_text(f'Validation #{count}, target: {decoded_targets[i]}',
                                                    pred[i],
                                                    self.global_step)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            accuracy = 1.0 - self.val_cer.compute()
            word_accuracy = 1.0 - self.val_wer.compute()

            if accuracy > self.best_metric:
                logger.debug(f'Updating best metric from {self.best_metric} ({self.best_epoch}) to {accuracy} ({self.current_epoch})')
                self.best_epoch = self.current_epoch
                self.best_metric = accuracy
            logger.info(f'validation run: total chars {self.val_cer.total} errors {self.val_cer.errors} accuracy {accuracy}')
            self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_word_accuracy', word_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_metric', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # reset metrics even if not sanity checking
        self.val_cer.reset()
        self.val_wer.reset()

    def setup(self, stage: Optional[str] = None):
        # finalize models in case of appending/loading
        if stage in [None, 'fit']:

            # Log a few sample images before the datasets are encoded.
            # This is only possible for Arrow datasets, because the
            # other dataset types can only be accessed after encoding
            if self.logger and isinstance(self.train_set.dataset, ArrowIPCRecognitionDataset):
                for i in range(min(len(self.train_set), 16)):
                    idx = np.random.randint(len(self.train_set))
                    sample = self.train_set[idx]
                    self.logger.experiment.add_image(f'train_set sample #{i}: {sample["target"]}', sample['image'])

            if self.append:
                self.train_set.dataset.encode(self.codec)
                # now we can create a new model
                self.spec = '[{} O1c{}]'.format(self.spec[1:-1], self.train_set.dataset.codec.max_label + 1)
                logger.info(f'Appending {self.spec} to existing model {self.nn.spec} after {self.append}')
                self.nn.append(self.append, self.spec)
                self.nn.add_codec(self.train_set.dataset.codec)
                logger.info(f'Assembled model spec: {self.nn.spec}')
            elif self.model:
                self.spec = self.nn.spec

                # prefer explicitly given codec over network codec if mode is 'new'
                codec = self.codec if (self.codec and self.resize == 'new') else self.nn.codec

                codec.strict = True

                try:
                    self.train_set.dataset.encode(codec)
                except KrakenEncodeException:
                    alpha_diff = set(self.train_set.dataset.alphabet).difference(
                        set(codec.c2l.keys())
                    )
                    if self.resize == 'fail':
                        raise KrakenInputException(f'Training data and model codec alphabets mismatch: {alpha_diff}')
                    elif self.resize == 'union':
                        logger.info(f'Resizing codec to include '
                                    f'{len(alpha_diff)} new code points')
                        # Construct two codecs:
                        # 1. training codec containing only the vocabulary in the training dataset
                        # 2. validation codec = training codec + validation set vocabulary
                        # This keep the codec in the model from being 'polluted' by non-trained characters.
                        train_codec = codec.add_labels(alpha_diff)
                        self.nn.add_codec(train_codec)
                        logger.info(f'Resizing last layer in network to {train_codec.max_label+1} outputs')
                        self.nn.resize_output(train_codec.max_label + 1)
                        self.train_set.dataset.encode(train_codec)
                    elif self.resize == 'new':
                        logger.info(f'Resizing network or given codec to '
                                    f'{len(self.train_set.dataset.alphabet)} '
                                    f'code sequences')
                        # same codec procedure as above, just with merging.
                        self.train_set.dataset.encode(None)
                        train_codec, del_labels = codec.merge(self.train_set.dataset.codec)
                        # Switch codec.
                        self.nn.add_codec(train_codec)
                        logger.info(f'Deleting {len(del_labels)} output classes from network '
                                    f'({len(codec)-len(del_labels)} retained)')
                        self.nn.resize_output(train_codec.max_label + 1, del_labels)
                        self.train_set.dataset.encode(train_codec)
                    else:
                        raise ValueError(f'invalid resize parameter value {self.resize}')
                self.nn.codec.strict = False
                self.spec = self.nn.spec
            else:
                self.train_set.dataset.encode(self.codec)
                logger.info(f'Creating new model {self.spec} with {self.train_set.dataset.codec.max_label+1} outputs')
                self.spec = '[{} O1c{}]'.format(self.spec[1:-1], self.train_set.dataset.codec.max_label + 1)
                self.nn = vgsl.TorchVGSLModel(self.spec)
                self.nn.use_legacy_polygons = self.legacy_polygons
                # initialize weights
                self.nn.init_weights()
                self.nn.add_codec(self.train_set.dataset.codec)

            val_diff = set(self.val_set.dataset.alphabet).difference(
                set(self.train_set.dataset.codec.c2l.keys())
            )
            logger.info(f'Adding {len(val_diff)} dummy labels to validation set codec.')

            val_codec = self.nn.codec.add_labels(val_diff)
            self.val_set.dataset.encode(val_codec)
            self.val_codec = val_codec

            if self.nn.one_channel_mode and self.train_set.dataset.im_mode != self.nn.one_channel_mode:
                logger.warning(f'Neural network has been trained on mode {self.nn.one_channel_mode} images, '
                               f'training set contains mode {self.train_set.dataset.im_mode} data. Consider setting `force_binarization`')

            if self.format_type != 'path' and self.nn.seg_type == 'bbox':
                logger.warning('Neural network has been trained on bounding box image information but training set is polygonal.')

            self.nn.hyper_params = self.hparams.hyper_params
            self.nn.model_type = 'recognition'

            if not self.nn.seg_type:
                logger.info(f'Setting seg_type to {self.train_set.dataset.seg_type}.')
                self.nn.seg_type = self.train_set.dataset.seg_type

            self.rec_nn = models.TorchSeqRecognizer(self.nn, train=None, device=None)
            self.net = self.nn.nn

            torch.set_num_threads(max(self.num_workers, 1))

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.hyper_params['batch_size'],
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_sequences)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=self.hparams.hyper_params['batch_size'],
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=collate_sequences)

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.hyper_params['quit'] == 'early':
            callbacks.append(EarlyStopping(monitor='val_accuracy',
                                           mode='max',
                                           patience=self.hparams.hyper_params['lag'],
                                           stopping_threshold=1.0))

        return callbacks

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return _configure_optimizer_and_lr_scheduler(self.hparams.hyper_params,
                                                     self.nn.nn.parameters(),
                                                     len_train_set=len(self.train_set),
                                                     loss_tracking_mode='max')

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
