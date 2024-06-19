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
import logging
import re
import warnings
from typing import (TYPE_CHECKING, Any, Callable, Dict, Literal, Optional,
                    Sequence, Union)
from functools import partial

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import (BaseFinetuning, Callback,
                                         EarlyStopping, LearningRateMonitor)
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset, random_split
from torchmetrics.classification import (MultilabelAccuracy,
                                         MultilabelJaccardIndex)
from torchmetrics.text import CharErrorRate, WordErrorRate

from kraken.containers import Segmentation
from kraken.lib import default_specs, models, progress, vgsl
from kraken.lib.codec import PytorchCodec
from kraken.lib.dataset import (ArrowIPCRecognitionDataset, BaselineSet,
                                GroundTruthDataset, ImageInputTransforms,
                                PolygonGTDataset, collate_sequences)
from kraken.lib.exceptions import KrakenEncodeException, KrakenInputException
from kraken.lib.models import validate_hyper_parameters
from kraken.lib.util import make_printable, parse_gt_path
from kraken.lib.xml import XMLPage

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


def _star_fun(fun, kwargs):
    try:
        return fun(**kwargs)
    except FileNotFoundError as e:
        logger.warning(f'{e.strerror}: {e.filename}. Skipping.')
    except KrakenInputException as e:
        logger.warning(str(e))
    return None


def _validation_worker_init_fn(worker_id):
    """ Fix random seeds so that augmentation always produces the same
        results when validating. Temporarily increase the logging level
        for lightning because otherwise it will display a message
        at info level about the seed being changed. """
    from lightning.pytorch import seed_everything
    seed_everything(42)


class KrakenTrainer(L.Trainer):
    def __init__(self,
                 enable_progress_bar: bool = True,
                 enable_summary: bool = True,
                 min_epochs: int = 5,
                 max_epochs: int = 100,
                 freeze_backbone=-1,
                 pl_logger: Union[L.pytorch.loggers.logger.Logger, str, None] = None,
                 log_dir: Optional['PathLike'] = None,
                 *args,
                 **kwargs):
        kwargs['enable_checkpointing'] = False
        kwargs['enable_progress_bar'] = enable_progress_bar
        kwargs['min_epochs'] = min_epochs
        kwargs['max_epochs'] = max_epochs
        kwargs['callbacks'] = ([] if 'callbacks' not in kwargs else kwargs['callbacks'])
        if not isinstance(kwargs['callbacks'], list):
            kwargs['callbacks'] = [kwargs['callbacks']]

        if pl_logger:
            if 'logger' in kwargs and isinstance(kwargs['logger'], L.pytorch.loggers.logger.Logger):
                logger.debug('Experiment logger has been provided outside KrakenTrainer as `logger`')
            elif isinstance(pl_logger, L.pytorch.loggers.logger.Logger):
                logger.debug('Experiment logger has been provided outside KrakenTrainer as `pl_logger`')
                kwargs['logger'] = pl_logger
            elif pl_logger == 'tensorboard':
                logger.debug('Creating default experiment logger')
                kwargs['logger'] = L.pytorch.loggers.TensorBoardLogger(log_dir)
            else:
                logger.error('`pl_logger` was set, but %s is not an accepted value', pl_logger)
                raise ValueError(f'{pl_logger} is not acceptable as logger')
            kwargs['callbacks'].append(LearningRateMonitor(logging_interval='step'))
        else:
            kwargs['logger'] = False

        if enable_progress_bar:
            progress_bar_cb = progress.KrakenTrainProgressBar(leave=True)
            kwargs['callbacks'].append(progress_bar_cb)

        if enable_summary:
            from lightning.pytorch.callbacks import RichModelSummary
            summary_cb = RichModelSummary(max_depth=2)
            kwargs['callbacks'].append(summary_cb)
            kwargs['enable_model_summary'] = False

        if freeze_backbone > 0:
            kwargs['callbacks'].append(KrakenFreezeBackbone(freeze_backbone))

        kwargs['callbacks'].extend([KrakenSetOneChannelMode(), KrakenSaveModel()])
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def fit(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=UserWarning,
                                    message='The dataloader,')
            super().fit(*args, **kwargs)


class KrakenFreezeBackbone(BaseFinetuning):
    """
    Callback freezing all but the last layer for fixed number of iterations.
    """
    def __init__(self, unfreeze_at_iterations=10):
        super().__init__()
        self.unfreeze_at_iteration = unfreeze_at_iterations

    def freeze_before_training(self, pl_module):
        pass

    def finetune_function(self, pl_module, current_epoch, optimizer):
        pass

    def on_train_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.freeze(pl_module.net[:-1])

    def on_train_batch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch, batch_idx) -> None:
        """
        Called for each training batch.
        """
        if trainer.global_step == self.unfreeze_at_iteration:
            for opt_idx, optimizer in enumerate(trainer.optimizers):
                num_param_groups = len(optimizer.param_groups)
                self.unfreeze_and_add_param_group(modules=pl_module.net[:-1],
                                                  optimizer=optimizer,
                                                  train_bn=True,)
                current_param_groups = optimizer.param_groups
                self._store(pl_module, opt_idx, num_param_groups, current_param_groups)

    def on_train_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Called when the epoch begins."""
        pass


class KrakenSetOneChannelMode(Callback):
    """
    Callback that sets the one_channel_mode of the model after the first epoch.
    """
    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        # fill one_channel_mode after 1 iteration over training data set
        if not trainer.sanity_checking and trainer.current_epoch == 0 and trainer.model.nn.model_type == 'recognition':
            ds = getattr(pl_module, 'train_set', None)
            if not ds and trainer.datamodule:
                ds = trainer.datamodule.train_set
            im_mode = ds.dataset.im_mode
            if im_mode in ['1', 'L']:
                logger.info(f'Setting model one_channel_mode to {im_mode}.')
                trainer.model.nn.one_channel_mode = im_mode


class KrakenSaveModel(Callback):
    """
    Kraken's own serialization callback instead of pytorch's.
    """
    def on_validation_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if not trainer.sanity_checking:
            trainer.model.nn.hyper_params['completed_epochs'] += 1
            metric = float(trainer.logged_metrics['val_metric']) if 'val_metric' in trainer.logged_metrics else -1.0
            trainer.model.nn.user_metadata['accuracy'].append((trainer.global_step, metric))
            trainer.model.nn.user_metadata['metrics'].append((trainer.global_step, {k: float(v) for k, v in trainer.logged_metrics.items()}))

            logger.info('Saving to {}_{}.mlmodel'.format(trainer.model.output, trainer.current_epoch))
            trainer.model.nn.save_model(f'{trainer.model.output}_{trainer.current_epoch}.mlmodel')
            trainer.model.best_model = f'{trainer.model.output}_{trainer.model.best_epoch}.mlmodel'


class RecognitionModel(L.LightningModule):
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
            legacy_train_status = train_set.legacy_polygons_status
            if val_set and val_set.legacy_polygons_status != legacy_train_status:
                logger.warning('Train and validation set have different legacy '
                               f'polygon status: {legacy_train_status} and '
                               f'{val_set.legacy_polygons_status}. Train set '
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
                          collate_fn=collate_sequences,
                          worker_init_fn=_validation_worker_init_fn)

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


class SegmentationModel(L.LightningModule):
    def __init__(self,
                 hyper_params: Dict = None,
                 load_hyper_parameters: bool = False,
                 progress_callback: Callable[[str, int], Callable[[None], None]] = lambda string, length: lambda: None,
                 message: Callable[[str], None] = lambda *args, **kwargs: None,
                 output: str = 'model',
                 spec: str = default_specs.SEGMENTATION_SPEC,
                 model: Optional[Union['PathLike', str]] = None,
                 training_data: Union[Sequence[Union['PathLike', str]], Sequence[Segmentation]] = None,
                 evaluation_data: Optional[Union[Sequence[Union['PathLike', str]], Sequence[Segmentation]]] = None,
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

        validate_hyper_parameters(hyper_params_)
        self.hyper_params = hyper_params_
        self.save_hyperparameters()

        if format_type in ['xml', 'page', 'alto']:
            logger.info(f'Parsing {len(training_data)} XML files for training data')
            training_data = [XMLPage(file, format_type).to_container() for file in training_data]
            if evaluation_data:
                logger.info(f'Parsing {len(evaluation_data)} XML files for validation data')
                evaluation_data = [XMLPage(file, format_type).to_container() for file in evaluation_data]
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
                                          self.hparams.hyper_params['padding'],
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

        train_set = BaselineSet(line_width=self.hparams.hyper_params['line_width'],
                                im_transforms=transforms,
                                augmentation=self.hparams.hyper_params['augment'],
                                valid_baselines=valid_baselines,
                                merge_baselines=merge_baselines,
                                valid_regions=valid_regions,
                                merge_regions=merge_regions)

        for page in training_data:
            train_set.add(page)

        if evaluation_data:
            val_set = BaselineSet(line_width=self.hparams.hyper_params['line_width'],
                                  im_transforms=transforms,
                                  augmentation=False,
                                  valid_baselines=valid_baselines,
                                  merge_baselines=merge_baselines,
                                  valid_regions=valid_regions,
                                  merge_regions=merge_regions)

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
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
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
            self.nn.hyper_params = self.hparams.hyper_params

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
        if self.hparams.hyper_params['quit'] == 'early':
            callbacks.append(EarlyStopping(monitor='val_mean_iu',
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
                scheduler.step()


def _configure_optimizer_and_lr_scheduler(hparams, params, len_train_set=None, loss_tracking_mode='max'):
    optimizer = hparams.get("optimizer")
    lrate = hparams.get("lrate")
    momentum = hparams.get("momentum")
    weight_decay = hparams.get("weight_decay")
    schedule = hparams.get("schedule")
    gamma = hparams.get("gamma")
    cos_t_max = hparams.get("cos_t_max")
    cos_min_lr = hparams.get("cos_min_lr")
    step_size = hparams.get("step_size")
    rop_factor = hparams.get("rop_factor")
    rop_patience = hparams.get("rop_patience")
    epochs = hparams.get("epochs")
    completed_epochs = hparams.get("completed_epochs")

    # XXX: Warmup is not configured here because it needs to be manually done in optimizer_step()
    logger.debug(f'Constructing {optimizer} optimizer (lr: {lrate}, momentum: {momentum})')
    if optimizer == 'Adam':
        optim = torch.optim.Adam(params, lr=lrate, weight_decay=weight_decay)
    else:
        optim = getattr(torch.optim, optimizer)(params,
                                                lr=lrate,
                                                momentum=momentum,
                                                weight_decay=weight_decay)
    lr_sched = {}
    if schedule == 'exponential':
        lr_sched = {'scheduler': lr_scheduler.ExponentialLR(optim, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'cosine':
        lr_sched = {'scheduler': lr_scheduler.CosineAnnealingLR(optim,
                                                                cos_t_max,
                                                                cos_min_lr,
                                                                last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'step':
        lr_sched = {'scheduler': lr_scheduler.StepLR(optim, step_size, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'reduceonplateau':
        lr_sched = {'scheduler': lr_scheduler.ReduceLROnPlateau(optim,
                                                                mode=loss_tracking_mode,
                                                                factor=rop_factor,
                                                                patience=rop_patience),
                    'interval': 'step'}
    elif schedule == '1cycle':
        if epochs <= 0:
            raise ValueError('1cycle learning rate scheduler selected but '
                             'number of epochs is less than 0 '
                             f'({epochs}).')
        last_epoch = completed_epochs*len_train_set if completed_epochs else -1
        lr_sched = {'scheduler': lr_scheduler.OneCycleLR(optim,
                                                         max_lr=lrate,
                                                         epochs=epochs,
                                                         steps_per_epoch=len_train_set,
                                                         last_epoch=last_epoch),
                    'interval': 'step'}
    elif schedule != 'constant':
        raise ValueError(f'Unsupported learning rate scheduler {schedule}.')

    ret = {'optimizer': optim}
    if lr_sched:
        ret['lr_scheduler'] = lr_sched

    if schedule == 'reduceonplateau':
        lr_sched['monitor'] = 'val_metric'
        lr_sched['strict'] = False
        lr_sched['reduce_on_plateau'] = True

    return ret
