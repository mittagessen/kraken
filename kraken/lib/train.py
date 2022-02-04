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
Training loop interception helpers
"""
import re
import torch
import pathlib
import logging
import warnings
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from functools import partial
from torch.multiprocessing import Pool
from torch.optim import lr_scheduler
from typing import Callable, Dict, Optional, Sequence, Union, Any, List, Tuple
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelSummary

from kraken.lib import models, vgsl, default_specs, log
from kraken.lib.xml import preparse_xml_data
from kraken.lib.util import make_printable
from kraken.lib.codec import PytorchCodec
from kraken.lib.dataset import (ArrowIPCRecognitionDataset, BaselineSet,
                                GroundTruthDataset, PolygonGTDataset,
                                ImageInputTransforms, compute_error,
                                collate_sequences)
from kraken.lib.models import validate_hyper_parameters
from kraken.lib.exceptions import KrakenInputException, KrakenEncodeException

from torch.utils.data import DataLoader, random_split, Subset


logger = logging.getLogger(__name__)


def _star_fun(fun, kwargs):
    try:
        return fun(**kwargs)
    except FileNotFoundError as e:
        logger.warning(f'{e.strerror}: {e.filename}. Skipping.')
    except KrakenInputException as e:
        logger.warning(str(e))
    return None


class KrakenTrainer(pl.Trainer):
    def __init__(self,
                 callbacks: Optional[Union[List[Callback], Callback]] = None,
                 enable_progress_bar: bool = True,
                 enable_summary: bool = True,
                 min_epochs=5,
                 max_epochs=100,
                 *args,
                 **kwargs):
        kwargs['logger'] = False
        kwargs['enable_checkpointing'] = False
        kwargs['enable_progress_bar'] = enable_progress_bar
        kwargs['min_epochs'] = min_epochs
        kwargs['max_epochs'] = max_epochs
        kwargs['callbacks'] = ([] if 'callbacks' not in kwargs else kwargs['callbacks'])
        if not isinstance(kwargs['callbacks'], list):
            kwargs['callbacks'] = [kwargs['callbacks']]

        if enable_progress_bar:
            progress_bar_cb = KrakenProgressBar()
            kwargs['callbacks'].append(progress_bar_cb)

        if enable_summary:
            summary_cb = ModelSummary(max_depth=2)
            kwargs['callbacks'].append(summary_cb)
        else:
            kwargs['enable_model_summary'] = False

        super().__init__(*args, **kwargs)

    def on_validation_end(self):
        if not self.sanity_checking:
            # fill one_channel_mode after 1 iteration over training data set
            if self.current_epoch == 0 and self.model.nn.model_type == 'recognition':
                im_mode = self.model.train_set.dataset.im_mode
                if im_mode in ['1', 'L']:
                    logger.info(f'Setting model one_channel_mode to {im_mode}.')
                    self.model.nn.one_channel_mode = im_mode
            self.model.nn.hyper_params['completed_epochs'] += 1
            self.model.nn.user_metadata['accuracy'].append(((self.current_epoch+1)*len(self.model.train_set),
                                                            {k: float(v) for k, v in self.logged_metrics.items()}))

            logger.info('Saving to {}_{}'.format(self.model.output, self.current_epoch))
            self.model.nn.save_model(f'{self.model.output}_{self.current_epoch}.mlmodel')

    def fit(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=UserWarning,
                                    message='The dataloader,')
            super().fit(*args, **kwargs)


class KrakenProgressBar(pl.callbacks.progress.base.ProgressBarBase):

    def __init__(self, *args, **kwargs):
        """
        Creates a progress bar using click's progress indicators.

        Args:
            refresh_rate: Determines at which rate (in number of batches) the
                          progress bars get updated.  Set it to ``0`` to
                          disable the display.
            leave: Leaves the finished progress bar in the terminal at the end
                   of the epoch. Default: False
        """
        super().__init__()
        self.bar = None
        self.enable = True

    def on_train_epoch_start(self, trainer, pl_module):
        if self.bar:
            print()
        self.bar = log.progressbar(label=f"stage {trainer.current_epoch}/{trainer.max_epochs if pl_module.hparams.quit == 'dumb' else '∞'}",
                                   length=self.total_train_batches,
                                   show_pos=True)
        self.bar.__enter__()

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if self.bar:
            print()
        self.bar = log.progressbar(label='validating',
                                   length=self.total_val_batches,
                                   show_pos=True)
        self.bar.__enter__()

    def on_sanity_check_start(self, trainer, pl_module):
        if self.bar:
            print()
        self.bar = log.progressbar(label='val check',
                                   length=sum(trainer.num_sanity_val_batches),
                                   show_pos=True)
        self.bar.__enter__()

    def disable(self):
        self.enable = False

    def enable(self):
        self.enable = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.bar.update(1)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.bar.update(1)

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if not trainer.sanity_checking:
            metrics = self.get_metrics(trainer, pl_module)
            metrics.pop('loss')
            metrics.pop('val_metric')
            for k, v in metrics.items():
                print(f'{k}: {v:.5f} ', end='')
            if trainer.early_stopping_callback:
                print(f'early_stopping: {trainer.early_stopping_callback.wait_count}/{trainer.early_stopping_callback.patience} {trainer.early_stopping_callback.best_score:.5f}', end='')


class RecognitionModel(pl.LightningModule):
    def __init__(self,
                 hyper_params: Dict[str, Any] = None,
                 output: str = 'model',
                 spec: str = default_specs.RECOGNITION_SPEC,
                 append: Optional[int] = None,
                 model: Optional[Union[pathlib.Path, str]] = None,
                 reorder: Union[bool, str] = True,
                 training_data: Union[Sequence[Union[pathlib.Path, str]], Sequence[Dict[str, Any]]] = None,
                 evaluation_data: Optional[Union[Sequence[Union[pathlib.Path, str]], Sequence[Dict[str, Any]]]] = None,
                 partition: Optional[float] = 0.9,
                 binary_dataset_split: bool = False,
                 num_workers: int = 1,
                 load_hyper_parameters: bool = False,
                 repolygonize: bool = False,
                 force_binarization: bool = False,
                 format_type: str = 'path',
                 codec: Optional[Dict] = None,
                 resize: str = 'fail'):
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
        super().__init__()
        hyper_params_ = default_specs.RECOGNITION_HYPER_PARAMS
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
        self.save_hyperparameters(hyper_params_)

        self.reorder = reorder
        self.append = append
        self.model = model
        self.num_workers = num_workers
        self.resize = resize
        self.format_type = format_type
        self.output = output

        self.best_epoch = 0
        self.best_metric = 0.0

        DatasetClass = GroundTruthDataset
        valid_norm = True
        if format_type in ['xml', 'page', 'alto']:
            logger.info(f'Parsing {len(training_data)} XML files for training data')
            training_data = preparse_xml_data(training_data, format_type, repolygonize)
            if evaluation_data:
                logger.info(f'Parsing {len(evaluation_data)} XML files for validation data')
                evaluation_data = preparse_xml_data(evaluation_data, format_type, repolygonize)
            if binary_dataset_split:
                logger.warning('Internal binary dataset splits are enabled but using non-binary dataset files. Will be ignored.')
                binary_dataset_split = False
            DatasetClass = PolygonGTDataset
            valid_norm = False
        elif format_type == 'binary':
            DatasetClass = ArrowIPCRecognitionDataset
            if repolygonize:
                logger.warning('Repolygonization enabled in `binary` mode. Will be ignored.')
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
            if repolygonize:
                logger.warning('Repolygonization enabled in `path` mode. Will be ignored.')
            if binary_dataset_split:
                logger.warning('Internal binary dataset splits are enabled but using non-binary dataset files. Will be ignored.')
                binary_dataset_split = False
            logger.info(f'Got {len(training_data)} line strip images for training data')
            training_data = [{'image': im} for im in training_data]
            if evaluation_data:
                logger.info(f'Got {len(evaluation_data)} line strip images for validation data')
                evaluation_data = [{'image': im} for im in evaluation_data]
            valid_norm = True
        # format_type is None. Determine training type from length of training data entry
        elif not format_type:
            if len(training_data[0]) >= 4:
                DatasetClass = PolygonGTDataset
                valid_norm = False
            else:
                if force_binarization:
                    logger.warning('Forced binarization enabled with box lines. Will be ignored.')
                    force_binarization = False
                if repolygonize:
                    logger.warning('Repolygonization enabled with box lines. Will be ignored.')
                if binary_dataset_split:
                    logger.warning('Internal binary dataset splits are enabled but using non-binary dataset files. Will be ignored.')
                    binary_dataset_split = False
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
                                               self.hparams.pad,
                                               valid_norm,
                                               force_binarization)

        if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
            logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
            torch.multiprocessing.set_sharing_strategy('file_system')

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

    def _build_dataset(self,
                       DatasetClass,
                       training_data,
                       **kwargs):
        dataset = DatasetClass(normalization=self.hparams.normalization,
                               whitespace_normalization=self.hparams.normalize_whitespace,
                               reorder=self.reorder,
                               im_transforms=self.transforms,
                               augmentation=self.hparams.augment,
                               **kwargs)

        if (self.num_workers and self.num_workers > 1) and self.format_type != 'binary':
            with Pool(processes=self.num_workers) as pool:
                for im in pool.imap_unordered(partial(_star_fun, dataset.parse), training_data, 5):
                    logger.debug(f'Adding sample {im} to training set')
                    if im:
                        dataset.add(**im)
        else:
            for im in training_data:
                try:
                    dataset.add(**im)
                except KrakenInputException as e:
                    logger.warning(str(e))
        return dataset

    def forward(self, x, seq_lens):
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
        return loss

    def validation_step(self, batch, batch_idx):
        chars, error = compute_error(self.rec_nn, batch)
        chars = torch.tensor(chars)
        error = torch.tensor(error)
        return {'chars': chars, 'error': error}

    def validation_epoch_end(self, outputs):
        chars = torch.stack([x['chars'] for x in outputs]).sum()
        error = torch.stack([x['error'] for x in outputs]).sum()
        accuracy = (chars - error) / (chars + torch.finfo(torch.float).eps)
        if accuracy > self.best_metric:
            self.best_epoch = self.current_epoch
            self.best_metric = accuracy
        self.log_dict({'val_accuracy': accuracy, 'val_metric': accuracy}, prog_bar=True)

    def configure_optimizers(self):
        logger.debug(f'Constructing {self.hparams.optimizer} optimizer (lr: {self.hparams.lrate}, momentum: {self.hparams.momentum})')
        if self.hparams.optimizer == 'Adam':
            optim = torch.optim.Adam(self.nn.nn.parameters(), lr=self.hparams.lrate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Lamb':
            from pytorch_lamb import Lamb
            optim = Lamb(self.nn.nn.parameters(), lr=self.hparams.lrate, weight_decay=self.hparams.weight_decay)
        else:
            optim = getattr(torch.optim, self.hparams.optimizer)(self.nn.nn.parameters(),
                                                                 lr=self.hparams.lrate,
                                                                 momentum=self.hparams.momentum,
                                                                 weight_decay=self.hparams.weight_decay)
        lr_sched = {}
        if self.hparams.schedule == 'exponential':
            lr_sched['scheduler'] = lr_scheduler.ExponentialLR(optim, self.hparams.gamma)
        elif self.hparams.schedule == 'cosine':
            lr_sched['scheduler'] = lr_scheduler.CosineAnnealingLR(optim, self.hparams.cos_t_max)
        elif self.hparams.schedule == 'step':
            lr_sched['scheduler'] = lr_scheduler.StepLR(optim, self.hparams.step_size, self.hparams.gamma)
        elif self.hparams.schedule == 'reduceonplateau':
            lr_sched['scheduler'] = lr_scheduler.ReduceLROnPlateau(optim,
                                                                   mode='max',
                                                                   factor=self.hparams.rop_factor,
                                                                   patience=self.hparams.rop_patience)
        elif self.hparams.schedule == '1cycle':
            if self.hparams.epochs <= 0:
                raise ValueError('1cycle learning rate scheduler selected but '
                                 'number of epochs is less than 0 '
                                 f'({self.hparams.epochs}).')
            lr_sched['scheduler'] = lr_scheduler.OneCycleLR(optim, max_lr=1.0, epochs=self.hparams.epochs, steps_per_epoch=len(self.train_set))
            lr_sched['interval'] = 'step'
        elif self.hparams.schedule != 'constant':
            raise ValueError(f'Unsupported learning rate scheduler {self.hparams.schedule}.')

        if lr_sched:
            lr_sched['monitor'] = 'val_metric'

        return [optim], lr_sched if lr_sched else []

    def setup(self, stage: Optional[str] = None):
        # finalize models in case of appending/loading
        if stage in [None, 'fit']:
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

                # prefer explicitly given codec over network codec if mode is 'both'
                codec = self.codec if (self.codec and self.resize == 'both') else self.nn.codec

                codec.strict = True

                try:
                    self.train_set.dataset.encode(codec)
                except KrakenEncodeException:
                    alpha_diff = set(self.train_set.dataset.alphabet).difference(set(codec.c2l.keys()))
                    if self.resize == 'fail':
                        raise KrakenInputException(f'Training data and model codec alphabets mismatch: {alpha_diff}')
                    elif self.resize == 'add':
                        logger.info(f'Resizing codec to include {len(alpha_diff)} new code points')
                        codec = codec.add_labels(alpha_diff)
                        self.nn.add_codec(codec)
                        logger.info(f'Resizing last layer in network to {codec.max_label+1} outputs')
                        self.nn.resize_output(codec.max_label + 1)
                        self.train_set.dataset.encode(self.nn.codec)
                    elif self.resize == 'both':
                        logger.info(f'Resizing network or given codec to {self.train_set.dataset.alphabet} code sequences')
                        self.train_set.dataset.encode(None)
                        ncodec, del_labels = codec.merge(self.train_set.dataset.codec)
                        self.nn.add_codec(ncodec)
                        logger.info(f'Deleting {len(del_labels)} output classes from network '
                                    f'({len(codec)-len(del_labels)} retained)')
                        self.train_set.dataset.encode(ncodec)
                        self.nn.resize_output(ncodec.max_label + 1, del_labels)
                    else:
                        raise ValueError(f'invalid resize parameter value {self.resize}')
            else:
                self.train_set.dataset.encode(self.codec)
                logger.info(f'Creating new model {self.spec} with {self.train_set.dataset.codec.max_label+1} outputs')
                self.spec = '[{} O1c{}]'.format(self.spec[1:-1], self.train_set.dataset.codec.max_label + 1)
                self.nn = vgsl.TorchVGSLModel(self.spec)
                # initialize weights
                self.nn.init_weights()
                self.nn.add_codec(self.train_set.dataset.codec)

            self.val_set.dataset.encode(self.nn.codec)

            if self.nn.one_channel_mode and self.train_set.dataset.im_mode != self.nn.one_channel_mode:
                logger.warning(f'Neural network has been trained on mode {self.nn.one_channel_mode} images, '
                               f'training set contains mode {self.train_set.dataset.im_mode} data. Consider setting `force_binarization`')

            if self.format_type != 'path' and self.nn.seg_type == 'bbox':
                logger.warning('Neural network has been trained on bounding box image information but training set is polygonal.')

            self.nn.hyper_params = self.hparams
            self.nn.model_type = 'recognition'

            if not self.nn.seg_type:
                logger.info(f'Setting seg_type to {self.train_set.dataset.seg_type}.')
                self.nn.seg_type = self.train_set.dataset.seg_type

            self.rec_nn = models.TorchSeqRecognizer(self.nn, train=None, device=None)
            self.net = self.nn.nn

            torch.set_num_threads(max(self.num_workers, 1))

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_sequences)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=collate_sequences)

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.quit == 'early':
            callbacks.append(EarlyStopping(monitor='val_accuracy',
                                           mode='max',
                                           patience=self.hparams.lag,
                                           stopping_threshold=1.0,
                                           check_on_train_epoch_end=True))
        return callbacks


class SegmentationModel(pl.LightningModule):
    def __init__(self,
                 hyper_params: Dict = None,
                 load_hyper_parameters: bool = False,
                 progress_callback: Callable[[str, int], Callable[[None], None]] = lambda string, length: lambda: None,
                 message: Callable[[str], None] = lambda *args, **kwargs: None,
                 output: str = 'model',
                 spec: str = default_specs.SEGMENTATION_SPEC,
                 model: Optional[Union[pathlib.Path, str]] = None,
                 training_data: Union[Sequence[Union[pathlib.Path, str]], Sequence[Dict[str, Any]]] = None,
                 evaluation_data: Optional[Union[Sequence[Union[pathlib.Path, str]], Sequence[Dict[str, Any]]]] = None,
                 partition: Optional[float] = 0.9,
                 num_workers: int = 1,
                 force_binarization: bool = False,
                 format_type: str = 'path',
                 suppress_regions: bool = False,
                 suppress_baselines: bool = False,
                 valid_regions: Optional[Sequence[str]] = None,
                 valid_baselines: Optional[Sequence[str]] = None,
                 merge_regions: Optional[Dict[str, str]] = None,
                 merge_baselines: Optional[Dict[str, str]] = None,
                 bounding_regions: Optional[Sequence[str]] = None,
                 resize: str = 'fail',
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

        self.best_epoch = 0
        self.best_metric = 0.0

        self.model = model
        self.num_workers = num_workers
        self.resize = resize
        self.format_type = format_type
        self.output = output
        self.bounding_regions = bounding_regions
        self.topline = topline

        hyper_params_ = default_specs.SEGMENTATION_HYPER_PARAMS

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

        validate_hyper_parameters(hyper_params_)
        self.save_hyperparameters(hyper_params_)

        if not training_data:
            raise ValueError('No training data provided. Please add some.')

        transforms = ImageInputTransforms(batch, height, width, channels, 0, valid_norm=False, force_binarization=force_binarization)

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

        train_set = BaselineSet(training_data,
                                line_width=self.hparams.line_width,
                                im_transforms=transforms,
                                mode=format_type,
                                augmentation=self.hparams.augment,
                                valid_baselines=valid_baselines,
                                merge_baselines=merge_baselines,
                                valid_regions=valid_regions,
                                merge_regions=merge_regions)

        if format_type is None:
            for page in training_data:
                train_set.add(**page)

        if evaluation_data:
            val_set = BaselineSet(evaluation_data,
                                  line_width=self.hparams.line_width,
                                  im_transforms=transforms,
                                  mode=format_type,
                                  augmentation=False,
                                  valid_baselines=valid_baselines,
                                  merge_baselines=merge_baselines,
                                  valid_regions=valid_regions,
                                  merge_regions=merge_regions)

            if format_type is None:
                for page in evaluation_data:
                    val_set.add(**page)

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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['target']
        pred, _ = self.nn.nn(x)
        # scale target to output size
        y = F.interpolate(y, size=(pred.size(2), pred.size(3))).squeeze(0).bool()
        pred = pred.squeeze() > 0.3
        pred = pred.view(pred.size(0), -1)
        y = y.view(y.size(0), -1)

        return {'intersections': (y & pred).sum(dim=1, dtype=torch.double),
                'unions':  (y | pred).sum(dim=1, dtype=torch.double),
                'corrects': torch.eq(y, pred).sum(dim=1, dtype=torch.double),
                'cls_cnt': y.sum(dim=1, dtype=torch.double),
                'all_n': torch.tensor(y.size(1), dtype=torch.double, device=self.device)}

    def validation_epoch_end(self, outputs):
        smooth = torch.finfo(torch.float).eps

        intersections = torch.stack([x['intersections'] for x in outputs]).sum()
        unions = torch.stack([x['unions'] for x in outputs]).sum()
        corrects = torch.stack([x['corrects'] for x in outputs]).sum()
        cls_cnt = torch.stack([x['cls_cnt'] for x in outputs]).sum()
        all_n = torch.stack([x['all_n'] for x in outputs]).sum()

        # all_positives = tp + fp
        # actual_positives = tp + fn
        # true_positivies = tp
        pixel_accuracy = corrects.sum() / all_n.sum()
        mean_accuracy = torch.mean(corrects / all_n)
        iu = (intersections + smooth) / (unions + smooth)
        mean_iu = torch.mean(iu)
        freq_iu = torch.sum(cls_cnt / cls_cnt.sum() * iu)

        if mean_iu > self.best_metric:
            self.best_epoch = self.current_epoch
            self.best_metric = mean_iu

        self.log_dict({'val_accuracy': pixel_accuracy,
                       'val_mean_acc': mean_accuracy,
                       'val_mean_iu': mean_iu,
                       'val_freq_iu': freq_iu,
                       'val_metric': mean_iu}, prog_bar=True)

    def setup(self, stage: Optional[str] = None):
        # finalize models in case of appending/loading
        if stage in [None, 'fit']:
            if not self.model:
                self.spec = f'[{self.spec[1:-1]} O2l{self.train_set.dataset.num_classes}]'
                logger.info(f'Creating model {self.spec} with {self.train_set.dataset.num_classes} outputs ', nl=False)
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
                    elif self.resize == 'add':
                        new_bls = self.train_set.dataset.class_mapping['baselines'].keys() - self.nn.user_metadata['class_mapping']['baselines'].keys()
                        new_regions = self.train_set.dataset.class_mapping['regions'].keys() - self.nn.user_metadata['class_mapping']['regions'].keys()
                        cls_idx = max(max(self.nn.user_metadata['class_mapping']['baselines'].values()) if self.nn.user_metadata['class_mapping']['baselines'] else -1,
                                      max(self.nn.user_metadata['class_mapping']['regions'].values()) if self.nn.user_metadata['class_mapping']['regions'] else -1)
                        logger.info(f'Adding {len(new_bls) + len(new_regions)} missing types to network output layer.')
                        self.nn.resize_output(cls_idx + len(new_bls) + len(new_regions) + 1)
                        for c in new_bls:
                            cls_idx += 1
                            self.nn.user_metadata['class_mapping']['baselines'][c] = cls_idx
                        for c in new_regions:
                            cls_idx += 1
                            self.nn.user_metadata['class_mapping']['regions'][c] = cls_idx
                    elif self.resize == 'both':
                        logger.info('Fitting network exactly to training set.')
                        new_bls = self.train_set.dataset.class_mapping['baselines'].keys() - self.nn.user_metadata['class_mapping']['baselines'].keys()
                        new_regions = self.train_set.dataset.class_mapping['regions'].keys() - self.nn.user_metadata['class_mapping']['regions'].keys()
                        del_bls = self.nn.user_metadata['class_mapping']['baselines'].keys() - self.train_set.dataset.class_mapping['baselines'].keys()
                        del_regions = self.nn.user_metadata['class_mapping']['regions'].keys() - self.train_set.dataset.class_mapping['regions'].keys()

                        logger.info(f'Adding {len(new_bls) + len(new_regions)} missing '
                                    f'types and removing {len(del_bls) + len(del_regions)} to network output layer ')
                        cls_idx = max(max(self.nn.user_metadata['class_mapping']['baselines'].values()) if self.nn.user_metadata['class_mapping']['baselines'] else -1,
                                      max(self.nn.user_metadata['class_mapping']['regions'].values()) if self.nn.user_metadata['class_mapping']['regions'] else -1)

                        del_indices = [self.nn.user_metadata['class_mapping']['baselines'][x] for x in del_bls]
                        del_indices.extend(self.nn.user_metadata['class_mapping']['regions'][x] for x in del_regions)
                        self.nn.resize_output(cls_idx + len(new_bls) + len(new_regions) -
                                              len(del_bls) - len(del_regions) + 1, del_indices)

                        # delete old baseline/region types
                        cls_idx = min(min(self.nn.user_metadata['class_mapping']['baselines'].values()) if self.nn.user_metadata['class_mapping']['baselines'] else np.inf,
                                      min(self.nn.user_metadata['class_mapping']['regions'].values()) if self.nn.user_metadata['class_mapping']['regions'] else np.inf)

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
            self.nn.hyper_params = self.hparams

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

    def configure_optimizers(self):
        logger.debug(f'Constructing {self.hparams.optimizer} optimizer (lr: {self.hparams.lrate}, momentum: {self.hparams.momentum})')
        if self.hparams.optimizer == 'Adam':
            optim = torch.optim.Adam(self.nn.nn.parameters(), lr=self.hparams.lrate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Lamb':
            from pytorch_lamb import Lamb
            optim = Lamb(self.nn.nn.parameters(), lr=self.hparams.lrate, weight_decay=self.hparams.weight_decay)
        else:
            optim = getattr(torch.optim, self.hparams.optimizer)(self.nn.nn.parameters(),
                                                                 lr=self.hparams.lrate,
                                                                 momentum=self.hparams.momentum,
                                                                 weight_decay=self.hparams.weight_decay)

        lr_sched = {}
        if self.hparams.schedule == 'exponential':
            lr_sched['scheduler'] = lr_scheduler.ExponentialLR(optim, self.hparams.gamma)
        elif self.hparams.schedule == 'cosine':
            lr_sched['scheduler'] = lr_scheduler.CosineAnnealingLR(optim, self.hparams.cos_t_max)
        elif self.hparams.schedule == 'step':
            lr_sched['scheduler'] = lr_scheduler.StepLR(optim, self.hparams.step_size, self.hparams.gamma)
        elif self.hparams.schedule == 'reduceonplateau':
            lr_sched['scheduler'] = lr_scheduler.ReduceLROnPlateau(optim,
                                                                   mode='max',
                                                                   factor=self.hparams.rop_factor,
                                                                   patience=self.hparams.rop_patience)
        elif self.hparams.schedule == '1cycle':
            if self.hparams.epochs <= 0:
                raise ValueError('1cycle learning rate scheduler selected but '
                                 'number of epochs is less than 0 '
                                 f'({self.hparams.epochs}).')
            lr_sched['scheduler'] = lr_scheduler.OneCycleLR(optim, max_lr=1.0, epochs=self.hparams.epochs, steps_per_epoch=len(self.train_set))
            lr_sched['interval'] = 'step'
        elif self.hparams.schedule != 'constant':
            raise ValueError(f'Unsupported learning rate scheduler {self.hparams.schedule}.')

        if lr_sched:
            lr_sched['monitor'] = 'val_metric'

        return [optim], lr_sched if lr_sched else []

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
        if self.hparams.quit == 'early':
            callbacks.append(EarlyStopping(monitor='val_mean_iu',
                                           mode='max',
                                           patience=self.hparams.lag,
                                           stopping_threshold=1.0,
                                           check_on_train_epoch_end=True))

        return callbacks
