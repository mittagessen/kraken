# -*- coding: utf-8 -*-
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
import abc
import math
import torch
import shutil
import logging
import numpy as np
import torch.nn.functional as F

from itertools import cycle
from functools import partial
from typing import cast, Tuple, Callable, List, Dict, Any, Optional, Sequence

from kraken.lib import models, vgsl, segmentation, default_specs
from kraken.lib.util import make_printable
from kraken.lib.codec import PytorchCodec
from kraken.lib.dataset import BaselineSet, GroundTruthDataset, PolygonGTDataset, generate_input_transforms, preparse_xml_data, InfiniteDataLoader, compute_error, collate_sequences
from kraken.lib.models import validate_hyper_parameters
from kraken.lib.exceptions import KrakenInputException, KrakenEncodeException

from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class TrainStopper(object):

    def __init__(self):
        self.best_loss = -math.inf
        self.best_epoch = 0
        self.epochs = -1
        self.epoch = 0

    @abc.abstractmethod
    def update(self, val_loss: float) -> None:
        """
        Updates the internal state of the train stopper.
        """
        pass

    @abc.abstractmethod
    def trigger(self) -> bool:
        """
        Function that raises a KrakenStopTrainingException after if the abort
        condition is fulfilled.
        """
        pass

class annealing_step(object):
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

    def __call__(self, *args, **kwargs):
        self.scheduler.step()

    @property
    def call_frequency(self):
        return 'epoch'

class annealing_const(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    @property
    def call_frequency(self):
        return 'epoch'

class annealing_exponential(object):
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        self.step_size = step_size
        self.step = 0

    def __call__(self, *args, **kwargs):
        self.step += 1
        if not self.step % self.step_size:
            logger.info('Reducing learning rate exponentially.')
            self.scheduler.step()

    @property
    def call_frequency(self):
        return 'epoch'

class annealing_reduceonplateau(object):
    def __init__(self, optimizer, patience=5, factor=0.1, mode='max', min_lr=1e-7):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)

    def __call__(self, *args, **kwargs):
        self.scheduler.step(kwargs['val_loss'])

    @property
    def call_frequency(self):
        return 'epoch'

class annealing_cosine(object):
    def __init__(self, optimizer, t_max=50, eta_min=1e-7):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min)

    def __call__(self, *args, **kwargs):
        self.scheduler.step()

    @property
    def call_frequency(self):
        return 'epoch'


class annealing_onecycle(object):
    def __init__(self, optimizer, max_lr=1e-3, epochs=50, steps_per_epoch=None):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch)

    def __call__(self, *args, **kwargs):
        self.scheduler.step()

    @property
    def call_frequency(self):
        return 'batch'

class TrainScheduler(object):
    """
    Implements learning rate scheduling.
    """
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.steps = []
        self.lengths = []
        self.optimizer = optimizer
        self.iterations = 0
        self.epoch_iterations = 0
        self.t_steps = 0
        self.lr_sched = None
        self.cycle = False

    def add_phase(self,
                  steps: int,
                  annealing_fn: Callable = None) -> None:
        """
        Adds a new phase to the scheduler.

        Args:
            steps (int): Number of step for this scheduler. Can be epochs or
                         iteration depending on the scheduler.
            max_lr (float): Peak learning rate
            annealing_fn (Callable): LR change function.
        """
        self.lengths.append(steps)
        self.steps.append(annealing_fn(self.optimizer))

    def batch_step(self, loss = None) -> None:
        """
        Performs an optimization step.
        """
        if not self.cycle:
            logger.debug('First call to lr scheduler')
            self.lengths = cycle(self.lengths)
            self.steps = cycle(self.steps)
            self.cycle = True

        if self.lr_sched is None:
            self.t_steps = next(self.lengths)
            self.lr_sched = next(self.steps)

        self.iterations += 1
        if self.lr_sched.call_frequency == 'batch':
            logger.debug('Adjusting learning rate (batch)')
            self.lr_sched(loss=loss)
            if self.iterations == self.t_steps:
                logger.debug('Switching to next lr scheduler')
                self.iterations = 0
                self.t_steps = next(self.lengths)
                self.lr_sched = next(self.steps)

    def epoch_step(self, val_loss = None) -> None:
        """
        Performs an optimization step.
        """
        if not self.cycle:
            logger.debug('First call to lr scheduler')
            self.lengths = cycle(self.lengths)
            self.steps = cycle(self.steps)
            self.cycle = True

        if self.lr_sched is None:
            self.t_steps = next(self.lengths)
            self.lr_sched = next(self.steps)

        self.epoch_iterations += 1
        if self.lr_sched.call_frequency == 'epoch':
            logger.debug('Adjusting learning rate (epoch)')
            self.lr_sched(val_loss=val_loss)
            if self.epoch_iterations == self.t_steps:
                logger.debug('Switching to next lr scheduler')
                self.epoch_iterations = 0
                self.t_steps = next(self.lengths)
                self.lr_sched = next(self.steps)


class EarlyStopping(TrainStopper):
    """
    Early stopping to terminate training when validation loss doesn't improve
    over a certain time.
    """
    def __init__(self, min_delta: float = None, lag: int = 1000) -> None:
        """
        Args:
            it (torch.utils.data.DataLoader): training data loader
            min_delta (float): minimum change in validation loss to qualify as
                               improvement. If `None` then linear auto-scaling
                               of the delta is used with
                               min_delta = (1 - val_loss)/20.
            lag (int): Number of iterations to wait for improvement before
                       terminating.
        """
        super().__init__()
        self.min_delta = min_delta
        self.auto_delta = False if min_delta else True
        self.lag = lag
        self.wait = -1

    def update(self, val_loss: float) -> None:
        """
        Updates the internal validation loss state and increases counter by
        one.
        """
        self.epoch += 1
        self.wait += 1

        if self.auto_delta:
            self.min_delta = (1 - self.best_loss)/20
            logger.debug('Rescaling early stopping loss to {}'.format(self.min_delta))
        if (val_loss - self.best_loss) >= self.min_delta:
            logger.debug('Resetting early stopping counter')
            self.wait = 0
            self.best_loss = val_loss
            self.best_epoch = self.epoch

    def trigger(self) -> bool:
        return not self.wait >= self.lag


class EpochStopping(TrainStopper):
    """
    Dumb stopping after a fixed number of iterations.
    """
    def __init__(self, epochs: int) -> None:
        """
        Args:
            epochs (int): Number of epochs to train for
        """
        super().__init__()
        self.epochs = epochs

    def update(self, val_loss: float) -> None:
        """
        Only update internal best iteration
        """
        if val_loss > self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = self.epoch
        self.epoch += 1

    def trigger(self) -> bool:
        return self.epoch < self.epochs


class NoStopping(TrainStopper):
    """
    Never stops training.
    """
    def __init__(self) -> None:
        super().__init__()

    def update(self, val_loss: float) -> None:
        """
        Only update internal best iteration
        """
        if val_loss > self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = self.epoch
        self.epoch += 1

    def trigger(self) -> bool:
        return True


def recognition_loss_fn(criterion, output, target):
    if isinstance(output, tuple):
        seq_lens = output[1]
        output = output[0]
        target_lens = target[1]
        target = target[0]
    else:
        seq_lens = (output.size(2),)
        target_lens = (target.size(1),)
    # height should be 1 by now
    if output.size(2) != 1:
        raise KrakenInputException('Expected dimension 3 to be 1, actual {}'.format(output.size(2)))
    output = output.squeeze(2)
    # NCW -> WNC
    loss = criterion(output.permute(2, 0, 1),  # type: ignore
                     target,
                     seq_lens,
                     target_lens)
    return loss


def baseline_label_loss_fn(criterion, output, target):
    output, _ = output
    output = F.interpolate(output, size=(target.size(2), target.size(3)))
    loss = criterion(output, target)
    return loss


def recognition_evaluator_fn(model, val_loader, device):
    rec = models.TorchSeqRecognizer(model, device=device)
    chars, error = compute_error(rec, val_loader)
    chars = chars.item()
    model.train()
    accuracy = ((chars-error)/chars)
    return {'val_metric': accuracy, 'accuracy': accuracy, 'chars': chars, 'error': error}


def baseline_label_evaluator_fn(model, val_loader, device):
    smooth = np.finfo(np.float).eps
    val_set = val_loader.dataset
    corrects = torch.zeros(val_set.num_classes, dtype=torch.double).to(device)
    all_n = torch.zeros(val_set.num_classes, dtype=torch.double).to(device)
    intersections = torch.zeros(val_set.num_classes, dtype=torch.double).to(device)
    unions = torch.zeros(val_set.num_classes, dtype=torch.double).to(device)
    cls_cnt = torch.zeros(val_set.num_classes, dtype=torch.double).to(device)
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            x,y = batch['image'], batch['target']
            x = x.to(device)
            y = y.to(device)
            pred, _ = model.nn(x)
            # scale target to output size
            y = F.interpolate(y, size=(pred.size(2), pred.size(3))).squeeze(0).bool()
            pred = segmentation.denoising_hysteresis_thresh(pred.detach().squeeze().cpu().numpy(), 0.2, 0.3, 0)
            pred = torch.from_numpy(pred.astype('bool')).to(device)
            pred = pred.view(pred.size(0), -1)
            y = y.view(y.size(0), -1)
            intersections += (y & pred).sum(dim=1, dtype=torch.double)
            unions += (y | pred).sum(dim=1, dtype=torch.double)
            corrects += torch.eq(y, pred).sum(dim=1, dtype=torch.double)
            cls_cnt += y.sum(dim=1, dtype=torch.double)
            all_n += y.size(1)
    model.train()
    # all_positives = tp + fp
    # actual_positives = tp + fn
    # true_positivies = tp
    pixel_accuracy = corrects.sum()/all_n.sum()
    mean_accuracy = torch.mean(corrects/all_n)
    iu = (intersections+smooth)/(unions+smooth)
    mean_iu = torch.mean(iu)
    freq_iu = torch.sum(cls_cnt/cls_cnt.sum() * iu)
    return {'accuracy': pixel_accuracy, 'mean_acc': mean_accuracy, 'mean_iu': mean_iu, 'freq_iu': freq_iu, 'val_metric': mean_iu}


class KrakenTrainer(object):
    """
    Class encapsulating the recognition model training process.
    """
    def __init__(self,
                 model: vgsl.TorchVGSLModel,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cpu',
                 filename_prefix: str = 'model',
                 event_frequency: float = 1.0,
                 train_set: torch.utils.data.DataLoader = None,
                 val_set=None,
                 stopper=None,
                 loss_fn=recognition_loss_fn,
                 evaluator=recognition_evaluator_fn):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.filename_prefix = filename_prefix
        self.event_frequency = event_frequency
        self.event_it = int(len(train_set) * event_frequency)
        self.train_set = train_set
        self.val_set = val_set
        self.stopper = stopper if stopper else NoStopping()
        self.iterations = 0
        self.lr_scheduler = None
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        # fill training metadata fields in model files
        self.model.seg_type = train_set.dataset.seg_type

    def add_lr_scheduler(self, lr_scheduler: TrainScheduler):
        self.lr_scheduler = lr_scheduler

    def run(self, event_callback=lambda *args, **kwargs: None, iteration_callback=lambda *args, **kwargs: None):
        logger.debug('Moving model to device {}'.format(self.device))
        self.model.to(self.device)
        self.model.train()

        logger.debug('Starting up training...')

        if 'accuracy' not in self.model.user_metadata:
            self.model.user_metadata['accuracy'] = []

        while self.stopper.trigger():
            for _, batch in zip(range(self.event_it), self.train_set):
                input, target = batch['image'], batch['target']
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                input = input.requires_grad_()
                # sequence batch
                if 'seq_lens' in batch:
                    seq_lens, label_lens = batch['seq_lens'], batch['target_lens']
                    seq_lens = seq_lens.to(self.device, non_blocking=True)
                    label_lens = label_lens.to(self.device, non_blocking=True)
                    target = (target, label_lens)
                    o = self.model.nn(input, seq_lens)
                else:
                    o = self.model.nn(input)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model.criterion, o, target)
                if not torch.isinf(loss):
                    loss.backward()
                    self.optimizer.step()
                else:
                    logger.debug('infinite loss in trial')
                if self.lr_scheduler:
                    self.lr_scheduler.batch_step(loss=loss)
                iteration_callback()
                # prevent memory leak
                del loss, o
            self.iterations += self.event_it
            logger.debug('Starting evaluation run')
            eval_res = self.evaluator(self.model, self.val_set, self.device)
            if self.lr_scheduler:
                self.lr_scheduler.epoch_step(val_loss=eval_res['val_metric'])
            self.stopper.update(eval_res['val_metric'])
            self.model.user_metadata['accuracy'].append((self.iterations, float(eval_res['val_metric'])))
            logger.info('Saving to {}_{}'.format(self.filename_prefix, self.stopper.epoch))
            # fill one_channel_mode after 1 iteration over training data set
            im_mode = self.train_set.dataset.im_mode
            if im_mode in ['1', 'L']:
                self.model.one_channel_mode = im_mode
            try:
                self.model.hyper_params['completed_epochs'] = self.stopper.epoch
                self.model.save_model('{}_{}.mlmodel'.format(self.filename_prefix, self.stopper.epoch))
            except Exception as e:
                logger.error('Saving model failed: {}'.format(str(e)))
            event_callback(epoch=self.stopper.epoch, **eval_res)

    @classmethod
    def load_model(cls, model_path: str,
                   load_hyper_parameters: Optional[bool] = False,
                   message: Callable[[str], None] = lambda *args, **kwargs: None):
        logger.info(f'Loading existing model from {model_path} ')
        message(f'Loading existing model from {model_path} ', nl=False)
        nn = vgsl.TorchVGSLModel.load_model(model_path)
        if load_hyper_parameters:
            hyper_params = nn.hyper_params
        else:
            hyper_params = {}
        message('\u2713', fg='green', nl=False)
        return nn, hyper_params

    @classmethod
    def recognition_train_gen(cls,
                              hyper_params: Dict = None,
                              progress_callback: Callable[[str, int], Callable[[None], None]] = lambda string, length: lambda: None,
                              message: Callable[[str], None] = lambda *args, **kwargs: None,
                              output: str = 'model',
                              spec: str = default_specs.RECOGNITION_SPEC,
                              append: Optional[int] = None,
                              load: Optional[str] = None,
                              device: str = 'cpu',
                              reorder: bool = True,
                              training_data: Sequence[Dict] = None,
                              evaluation_data: Sequence[Dict] = None,
                              preload: Optional[bool] = None,
                              threads: int = 1,
                              load_hyper_parameters: bool = False,
                              repolygonize: bool = False,
                              force_binarization: bool = False,
                              format_type: str = 'path',
                              codec: Optional[Dict] = None,
                              resize: str = 'fail',
                              augment: bool = False):
        """
        This is an ugly constructor that takes all the arguments from the command
        line driver, finagles the datasets, models, and hyperparameters correctly
        and returns a KrakenTrainer object.

        Setup parameters (load, training_data, evaluation_data, ....) are named,
        model hyperparameters (everything in
        kraken.lib.default_specs.RECOGNITION_HYPER_PARAMS) are in in the
        `hyper_params` argument.

        Args:
            hyper_params (dict): Hyperparameter dictionary containing all fields
                                 from
                                 kraken.lib.default_specs.RECOGNITION_HYPER_PARAMS
            progress_callback (Callable): Callback for progress reports on various
                                          computationally expensive processes. A
                                          human readable string and the process
                                          length is supplied. The callback has to
                                          return another function which will be
                                          executed after each step.
            message (Callable): Messaging printing method for above log but below
                                warning level output, i.e. infos that should
                                generally be shown to users.
            **kwargs: Setup parameters, i.e. CLI parameters of the train() command.

        Returns:
            A KrakenTrainer object.
        """

        hyper_params_ = default_specs.RECOGNITION_HYPER_PARAMS

        # load model if given. if a new model has to be created we need to do that
        # after data set initialization, otherwise to output size is still unknown.
        if load:
            nn, hp = cls.load_model(load,
                                    load_hyper_parameters=load_hyper_parameters,
                                    message=message)
            hyper_params_.update(hp)
        else:
            nn = None

        if hyper_params:
            hyper_params_.update(hyper_params)

        validate_hyper_parameters(hyper_params_)

        hyper_params = hyper_params_

        DatasetClass = GroundTruthDataset
        valid_norm = True
        if format_type and format_type != 'path':
            logger.info(f'Parsing {len(training_data)} XML files for training data')
            if repolygonize:
                message('Repolygonizing data')
            training_data = preparse_xml_data(training_data, format_type, repolygonize)
            evaluation_data = preparse_xml_data(evaluation_data, format_type, repolygonize)
            DatasetClass = PolygonGTDataset
            valid_norm = False
        elif format_type == 'path':
            if force_binarization:
                logger.warning('Forced binarization enabled in `path` mode. Will be ignored.')
                force_binarization = False
            if repolygonize:
                logger.warning('Repolygonization enabled in `path` mode. Will be ignored.')
            training_data = [{'image': im} for im in training_data]
            if evaluation_data:
                evaluation_data = [{'image': im} for im in evaluation_data]
            valid_norm = True
        # format_type is None. Determine training type from length of training data entry
        else:
            if len(training_data[0]) >= 4:
                DatasetClass = PolygonGTDataset
                valid_norm = False
            else:
                if force_binarization:
                    logger.warning('Forced binarization enabled with box lines. Will be ignored.')
                    force_binarization = False
                if repolygonize:
                    logger.warning('Repolygonization enabled with box lines. Will be ignored.')


        # preparse input sizes from vgsl string to seed ground truth data set
        # sizes and dimension ordering.
        if not nn:
            spec = spec.strip()
            if spec[0] != '[' or spec[-1] != ']':
                raise KrakenInputException('VGSL spec {} not bracketed'.format(spec))
            blocks = spec[1:-1].split(' ')
            m = re.match(r'(\d+),(\d+),(\d+),(\d+)', blocks[0])
            if not m:
                raise KrakenInputException(f'Invalid input spec {blocks[0]}')
            batch, height, width, channels = [int(x) for x in m.groups()]
        else:
            batch, channels, height, width = nn.input

        transforms = generate_input_transforms(batch, height, width, channels, hyper_params['pad'], valid_norm, force_binarization)

        if len(training_data) > 2500 and not preload:
            logger.info('Disabling preloading for large (>2500) training data set. Enable by setting --preload parameter')
            preload = False
        # implicit preloading enabled for small data sets
        if preload is None:
            preload = True

        # set multiprocessing tensor sharing strategy
        if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
            logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
            torch.multiprocessing.set_sharing_strategy('file_system')

        gt_set = DatasetClass(normalization=hyper_params['normalization'],
                              whitespace_normalization=hyper_params['normalize_whitespace'],
                              reorder=reorder,
                              im_transforms=transforms,
                              preload=preload,
                              augmentation=hyper_params['augment'])
        bar = progress_callback('Building training set', len(training_data))
        for im in training_data:
            logger.debug(f'Adding line {im} to training set')
            try:
                gt_set.add(**im)
                bar()
            except FileNotFoundError as e:
                logger.warning(f'{e.strerror}: {e.filename}. Skipping.')
            except KrakenInputException as e:
                logger.warning(str(e))

        val_set = DatasetClass(normalization=hyper_params['normalization'],
                               whitespace_normalization=hyper_params['normalize_whitespace'],
                               reorder=reorder,
                               im_transforms=transforms,
                               preload=preload)
        bar = progress_callback('Building validation set', len(evaluation_data))
        for im in evaluation_data:
            logger.debug(f'Adding line {im} to validation set')
            try:
                val_set.add(**im)
                bar()
            except FileNotFoundError as e:
                logger.warning(f'{e.strerror}: {e.filename}. Skipping.')
            except KrakenInputException as e:
                logger.warning(str(e))

        if len(gt_set._images) == 0:
            logger.error('No valid training data was provided to the train command. Please add valid XML or line data.')
            return None

        logger.info(f'Training set {len(gt_set._images)} lines, validation set {len(val_set._images)} lines, alphabet {len(gt_set.alphabet)} symbols')
        alpha_diff_only_train = set(gt_set.alphabet).difference(set(val_set.alphabet))
        alpha_diff_only_val = set(val_set.alphabet).difference(set(gt_set.alphabet))
        if alpha_diff_only_train:
            logger.warning(f'alphabet mismatch: chars in training set only: {alpha_diff_only_train} (not included in accuracy test during training)')
        if alpha_diff_only_val:
            logger.warning(f'alphabet mismatch: chars in validation set only: {alpha_diff_only_val} (not trained)')
        logger.info('grapheme\tcount')
        for k, v in sorted(gt_set.alphabet.items(), key=lambda x: x[1], reverse=True):
            char = make_printable(k)
            if char == k:
                char = '\t' + char
            logger.info(f'{char}\t{v}')

        logger.debug('Encoding training set')

        # use model codec when given
        if append:
            # is already loaded
            nn = cast(vgsl.TorchVGSLModel, nn)
            gt_set.encode(codec)
            message('Slicing and dicing model ', nl=False)
            # now we can create a new model
            spec = '[{} O1c{}]'.format(spec[1:-1], gt_set.codec.max_label()+1)
            logger.info(f'Appending {spec} to existing model {nn.spec} after {append}')
            nn.append(append, spec)
            nn.add_codec(gt_set.codec)
            message('\u2713', fg='green')
            logger.info(f'Assembled model spec: {nn.spec}')
        elif load:
            # is already loaded
            nn = cast(vgsl.TorchVGSLModel, nn)

            # prefer explicitly given codec over network codec if mode is 'both'
            codec = codec if (codec and resize == 'both') else nn.codec

            try:
                gt_set.encode(codec)
            except KrakenEncodeException:
                message('Network codec not compatible with training set')
                alpha_diff = set(gt_set.alphabet).difference(set(codec.c2l.keys()))
                if resize == 'fail':
                    logger.error(f'Training data and model codec alphabets mismatch: {alpha_diff}')
                    return None
                elif resize == 'add':
                    message('Adding missing labels to network ', nl=False)
                    logger.info(f'Resizing codec to include {len(alpha_diff)} new code points')
                    codec = codec.add_labels(alpha_diff)
                    nn.add_codec(codec)
                    logger.info(f'Resizing last layer in network to {codec.max_label()+1} outputs')
                    nn.resize_output(codec.max_label()+1)
                    gt_set.encode(nn.codec)
                    message('\u2713', fg='green')
                elif resize == 'both':
                    message('Fitting network exactly to training set ', nl=False)
                    logger.info(f'Resizing network or given codec to {gt_set.alphabet} code sequences')
                    gt_set.encode(None)
                    ncodec, del_labels = codec.merge(gt_set.codec)
                    logger.info(f'Deleting {len(del_labels)} output classes from network ({len(codec)-len(del_labels)} retained)')
                    gt_set.encode(ncodec)
                    nn.resize_output(ncodec.max_label()+1, del_labels)
                    message('\u2713', fg='green')
                else:
                    logger.error(f'invalid resize parameter value {resize}')
                    return None
        else:
            gt_set.encode(codec)
            logger.info(f'Creating new model {spec} with {gt_set.codec.max_label()+1} outputs')
            spec = '[{} O1c{}]'.format(spec[1:-1], gt_set.codec.max_label()+1)
            nn = vgsl.TorchVGSLModel(spec)
            # initialize weights
            message('Initializing model ', nl=False)
            nn.init_weights()
            nn.add_codec(gt_set.codec)
            # initialize codec
            message('\u2713', fg='green')

        if nn.one_channel_mode and gt_set.im_mode != nn.one_channel_mode:
            logger.warning(f'Neural network has been trained on mode {nn.one_channel_mode} images, training set contains mode {gt_set.im_mode} data. Consider setting `force_binarization`')

        if format_type != 'path' and nn.seg_type == 'bbox':
            logger.warning('Neural network has been trained on bounding box image information but training set is polygonal.')

        # half the number of data loading processes if device isn't cuda and we haven't enabled preloading

        if device == 'cpu' and not preload:
            loader_threads = threads // 2
        else:
            loader_threads = threads
        train_loader = InfiniteDataLoader(gt_set, batch_size=hyper_params['batch_size'],
                                          shuffle=True,
                                          num_workers=loader_threads,
                                          pin_memory=True,
                                          collate_fn=collate_sequences)
        threads = max(threads - loader_threads, 1)

        # don't encode validation set as the alphabets may not match causing encoding failures
        val_set.no_encode()
        val_loader = DataLoader(val_set,
                                batch_size=hyper_params['batch_size'],
                                num_workers=loader_threads,
                                pin_memory=True,
                                collate_fn=collate_sequences)

        logger.debug('Constructing {} optimizer (lr: {}, momentum: {})'.format(hyper_params['optimizer'], hyper_params['lrate'], hyper_params['momentum']))

        # updates model's hyper params with users defined
        nn.hyper_params = hyper_params

        # set model type metadata field
        nn.model_type = 'recognition'

        # set mode to trainindg
        nn.train()

        # set number of OpenMP threads
        logger.debug(f'Set OpenMP threads to {threads}')
        nn.set_num_threads(threads)

        if hyper_params['optimizer'] == 'Adam':
            optim = torch.optim.Adam(nn.nn.parameters(), lr=hyper_params['lrate'], weight_decay=hyper_params['weight_decay'])
        else:
            optim = getattr(torch.optim, hyper_params['optimizer'])(nn.nn.parameters(),
                                                                    lr=hyper_params['lrate'],
                                                                    momentum=hyper_params['momentum'],
                                                                    weight_decay=hyper_params['weight_decay'])

        if 'seg_type' not in nn.user_metadata:
            nn.user_metadata['seg_type'] = 'baselines' if format_type != 'path' else 'bbox'

        tr_it = TrainScheduler(optim)
        if hyper_params['schedule'] == '1cycle':
            annealing_one = partial(annealing_onecycle, max_lr=hyper_params['lrate'], epochs=hyper_params['step_size'], steps_per_epoch=len(gt_set))
            tr_it.add_phase(int(len(gt_set) * hyper_params['epochs']),
                            annealing_one)
        elif hyper_params['schedule'] == 'exponential':
            annealing_exp = partial(annealing_exponential, step_size=hyper_params['step_size'], gamma=hyper_params['gamma'])
            tr_it.add_phase(int(len(gt_set) * hyper_params['epochs']),
                            annealing_exp)
        elif hyper_params['schedule'] == 'step':
            annealing_step = partial(annealing_step, step_size=hyper_params['step_size'], gamma=hyper_params['gamma'])
            tr_it.add_phase(int(len(gt_set) * hyper_params['epochs']),
                            annealing_step)
        elif hyper_params['schedule'] == 'reduceonplateau':
            annealing_red = partial(annealing_reduceonplateau, patience=hyper_params['rop_patience'], factor=hyper_params['gamma'])
            tr_it.add_phase(int(len(gt_set) * hyper_params['epochs']),
                            annealing_red)
        elif hyper_params['schedule'] == 'cosine':
            annealing_cos = partial(annealing_cosine, t_max=hyper_params['cos_t_max'])
            tr_it.add_phase(int(len(gt_set) * hyper_params['epochs']),
                            annealing_cos)
        else:
            # constant learning rate scheduler
            tr_it.add_phase(int(len(gt_set) * hyper_params['epochs']),
                            annealing_const)

        if hyper_params['quit'] == 'early':
            st_it = EarlyStopping(hyper_params['min_delta'], hyper_params['lag'])
        elif hyper_params['quit'] == 'dumb':
            st_it = EpochStopping(hyper_params['epochs'] - hyper_params['completed_epochs'])
        else:
            logger.error(f'Invalid training interruption scheme {quit}')
            return None

        trainer = cls(model=nn,
                      optimizer=optim,
                      device=device,
                      filename_prefix=output,
                      event_frequency=hyper_params['freq'],
                      train_set=train_loader,
                      val_set=val_loader,
                      stopper=st_it)

        trainer.add_lr_scheduler(tr_it)

        return trainer

    @classmethod
    def segmentation_train_gen(cls,
                               hyper_params: Dict = None,
                               load_hyper_parameters: bool = False,
                               progress_callback: Callable[[str, int], Callable[[None], None]] = lambda string, length: lambda: None,
                               message: Callable[[str], None] = lambda *args, **kwargs: None,
                               output: str = 'model',
                               spec: str = default_specs.SEGMENTATION_SPEC,
                               load: Optional[str] = None,
                               device: str = 'cpu',
                               training_data: Sequence[Dict] = None,
                               evaluation_data: Sequence[Dict] = None,
                               threads: int = 1,
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
                               augment: bool = False):
        """
        This is an ugly constructor that takes all the arguments from the command
        line driver, finagles the datasets, models, and hyperparameters correctly
        and returns a KrakenTrainer object.

        Setup parameters (load, training_data, evaluation_data, ....) are named,
        model hyperparameters (everything in
        kraken.lib.default_specs.SEGMENTATION_HYPER_PARAMS) are in in the
        `hyper_params` argument.

        Args:
            hyper_params (dict): Hyperparameter dictionary containing all fields
                                 from
                                 kraken.lib.default_specs.SEGMENTATION_HYPER_PARAMS
            progress_callback (Callable): Callback for progress reports on various
                                          computationally expensive processes. A
                                          human readable string and the process
                                          length is supplied. The callback has to
                                          return another function which will be
                                          executed after each step.
            message (Callable): Messaging printing method for above log but below
                                warning level output, i.e. infos that should
                                generally be shown to users.
            **kwargs: Setup parameters, i.e. CLI parameters of the train() command.

        Returns:
            A KrakenTrainer object.
        """
        # load model if given. if a new model has to be created we need to do that
        # after data set initialization, otherwise to output size is still unknown.
        nn = None

        hyper_params_ = default_specs.SEGMENTATION_HYPER_PARAMS

        if load:
            nn, hp = cls.load_model(load,
                                    load_hyper_parameters=load_hyper_parameters,
                                    message=message)
            hyper_params_.update(hp)
            batch, channels, height, width = nn.input
        else:
            # preparse input sizes from vgsl string to seed ground truth data set
            # sizes and dimension ordering.
            spec = spec.strip()
            if spec[0] != '[' or spec[-1] != ']':
                logger.error(f'VGSL spec "{spec}" not bracketed')
                return None
            blocks = spec[1:-1].split(' ')
            m = re.match(r'(\d+),(\d+),(\d+),(\d+)', blocks[0])
            if not m:
                logger.error(f'Invalid input spec {blocks[0]}')
                return None
            batch, height, width, channels = [int(x) for x in m.groups()]

        if hyper_params:
            hyper_params_.update(hyper_params)

        validate_hyper_parameters(hyper_params_)

        hyper_params = hyper_params_

        transforms = generate_input_transforms(batch, height, width, channels, 0, valid_norm=False)

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

        gt_set = BaselineSet(training_data,
                             line_width=hyper_params['line_width'],
                             im_transforms=transforms,
                             mode=format_type,
                             augmentation=hyper_params['augment'],
                             valid_baselines=valid_baselines,
                             merge_baselines=merge_baselines,
                             valid_regions=valid_regions,
                             merge_regions=merge_regions)
        val_set = BaselineSet(evaluation_data,
                              line_width=hyper_params['line_width'],
                              im_transforms=transforms,
                              mode=format_type,
                              augmentation=hyper_params['augment'],
                              valid_baselines=valid_baselines,
                              merge_baselines=merge_baselines,
                              valid_regions=valid_regions,
                              merge_regions=merge_regions)

        if format_type is None:
            for page in training_data:
                gt_set.add(**page)
            for page in evaluation_data:
                val_set.add(**page)

        # overwrite class mapping in validation set
        val_set.num_classes = gt_set.num_classes
        val_set.class_mapping = gt_set.class_mapping

        if not load:
            spec = f'[{spec[1:-1]} O2l{gt_set.num_classes}]'
            message(f'Creating model {spec} with {gt_set.num_classes} outputs ', nl=False)
            nn = vgsl.TorchVGSLModel(spec)
            message('\u2713', fg='green')
            if bounding_regions is not None:
                nn.user_metadata['bounding_regions'] = bounding_regions
        else:
            if gt_set.class_mapping['baselines'].keys() != nn.user_metadata['class_mapping']['baselines'].keys() or \
               gt_set.class_mapping['regions'].keys() != nn.user_metadata['class_mapping']['regions'].keys():

                bl_diff = set(gt_set.class_mapping['baselines'].keys()).symmetric_difference(set(nn.user_metadata['class_mapping']['baselines'].keys()))
                regions_diff = set(gt_set.class_mapping['regions'].keys()).symmetric_difference(set(nn.user_metadata['class_mapping']['regions'].keys()))

                if resize == 'fail':
                    logger.error(f'Training data and model class mapping differ (bl: {bl_diff}, regions: {regions_diff}')
                    raise KrakenInputException(f'Training data and model class mapping differ (bl: {bl_diff}, regions: {regions_diff}')
                elif resize == 'add':
                    new_bls = gt_set.class_mapping['baselines'].keys() - nn.user_metadata['class_mapping']['baselines'].keys()
                    new_regions = gt_set.class_mapping['regions'].keys() - nn.user_metadata['class_mapping']['regions'].keys()
                    cls_idx = max(max(nn.user_metadata['class_mapping']['baselines'].values()) if nn.user_metadata['class_mapping']['baselines'] else -1,
                                  max(nn.user_metadata['class_mapping']['regions'].values()) if nn.user_metadata['class_mapping']['regions'] else -1)
                    message(f'Adding {len(new_bls) + len(new_regions)} missing types to network output layer ', nl=False)
                    nn.resize_output(cls_idx + len(new_bls) + len(new_regions) + 1)
                    for c in new_bls:
                        cls_idx += 1
                        nn.user_metadata['class_mapping']['baselines'][c] = cls_idx
                    for c in new_regions:
                        cls_idx += 1
                        nn.user_metadata['class_mapping']['regions'][c] = cls_idx
                    message('\u2713', fg='green')
                elif resize == 'both':
                    message('Fitting network exactly to training set ', nl=False)
                    new_bls = gt_set.class_mapping['baselines'].keys() - nn.user_metadata['class_mapping']['baselines'].keys()
                    new_regions = gt_set.class_mapping['regions'].keys() - nn.user_metadata['class_mapping']['regions'].keys()
                    del_bls = nn.user_metadata['class_mapping']['baselines'].keys() - gt_set.class_mapping['baselines'].keys()
                    del_regions = nn.user_metadata['class_mapping']['regions'].keys() - gt_set.class_mapping['regions'].keys()

                    message(f'Adding {len(new_bls) + len(new_regions)} missing '
                            f'types and removing {len(del_bls) + len(del_regions)} to network output layer ',
                            nl=False)
                    cls_idx = max(max(nn.user_metadata['class_mapping']['baselines'].values()) if nn.user_metadata['class_mapping']['baselines'] else -1,
                                  max(nn.user_metadata['class_mapping']['regions'].values()) if nn.user_metadata['class_mapping']['regions'] else -1)

                    del_indices = [nn.user_metadata['class_mapping']['baselines'][x] for x in del_bls]
                    del_indices.extend(nn.user_metadata['class_mapping']['regions'][x] for x in del_regions)
                    nn.resize_output(cls_idx + len(new_bls) + len(new_regions) - len(del_bls) - len(del_regions) + 1, del_indices)

                    # delete old baseline/region types
                    cls_idx = min(min(nn.user_metadata['class_mapping']['baselines'].values()) if nn.user_metadata['class_mapping']['baselines'] else np.inf,
                                  min(nn.user_metadata['class_mapping']['regions'].values()) if nn.user_metadata['class_mapping']['regions'] else np.inf)

                    bls = {}
                    for k,v in sorted(nn.user_metadata['class_mapping']['baselines'].items(), key=lambda item: item[1]):
                        if k not in del_bls:
                            bls[k] = cls_idx
                            cls_idx += 1

                    regions = {}
                    for k,v in sorted(nn.user_metadata['class_mapping']['regions'].items(), key=lambda item: item[1]):
                        if k not in del_regions:
                            regions[k] = cls_idx
                            cls_idx += 1

                    nn.user_metadata['class_mapping']['baselines'] = bls
                    nn.user_metadata['class_mapping']['regions'] = regions

                    # add new baseline/region types
                    cls_idx -= 1
                    for c in new_bls:
                        cls_idx += 1
                        nn.user_metadata['class_mapping']['baselines'][c] = cls_idx
                    for c in new_regions:
                        cls_idx += 1
                        nn.user_metadata['class_mapping']['regions'][c] = cls_idx
                    message('\u2713', fg='green')
                else:
                    logger.error(f'invalid resize parameter value {resize}')
                    raise KrakenInputException(f'invalid resize parameter value {resize}')
            # backfill gt_set/val_set mapping if key-equal as the actual
            # numbering in the gt_set might be different
            gt_set.class_mapping = nn.user_metadata['class_mapping']
            val_set.class_mapping = nn.user_metadata['class_mapping']

        # updates model's hyper params with users defined
        nn.hyper_params = hyper_params

        message('Training line types:')
        for k, v in gt_set.class_mapping['baselines'].items():
            message(f'  {k}\t{v}\t{gt_set.class_stats["baselines"][k]}')
        message('Training region types:')
        for k, v in gt_set.class_mapping['regions'].items():
            message(f'  {k}\t{v}\t{gt_set.class_stats["regions"][k]}')

        if len(gt_set.imgs) == 0:
            logger.error('No valid training data was provided to the train command. Please add valid XML data.')
            return None

        if device == 'cpu':
            loader_threads = threads // 2
        else:
            loader_threads = threads

        train_loader = InfiniteDataLoader(gt_set, batch_size=1, shuffle=True, num_workers=loader_threads, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=loader_threads, pin_memory=True)
        threads = max((threads - loader_threads, 1))

        # set model type metadata field and dump class_mapping
        nn.model_type = 'segmentation'
        nn.user_metadata['class_mapping'] = val_set.class_mapping

        # set mode to training
        nn.train()

        logger.debug(f'Set OpenMP threads to {threads}')
        nn.set_num_threads(threads)

        if hyper_params['optimizer'] == 'Adam':
            optim = torch.optim.Adam(nn.nn.parameters(), lr=hyper_params['lrate'], weight_decay=hyper_params['weight_decay'])
        else:
            optim = getattr(torch.optim, hyper_params['optimizer'])(nn.nn.parameters(),
                                                                    lr=hyper_params['lrate'],
                                                                    momentum=hyper_params['momentum'],
                                                                    weight_decay=hyper_params['weight_decay'])

        tr_it = TrainScheduler(optim)
        tr_it = TrainScheduler(optim)
        if hyper_params['schedule'] == '1cycle':
            annealing_one = partial(annealing_onecycle, max_lr=hyper_params['lrate'], epochs=hyper_params['epochs'], steps_per_epoch=len(gt_set))
            tr_it.add_phase(int(len(gt_set) * hyper_params['epochs']),
                            annealing_one)
        elif hyper_params['schedule'] == 'exponential':
            annealing_exp = partial(annealing_exponential, step_size=hyper_params['step_size'], gamma=hyper_params['gamma'])
            tr_it.add_phase(int(len(gt_set) * hyper_params['epochs']),
                            annealing_exp)
        elif hyper_params['schedule'] == 'step':
            annealing_step = partial(annealing_step, step_size=hyper_params['step_size'], gamma=hyper_params['gamma'])
            tr_it.add_phase(int(len(gt_set) * hyper_params['epochs']),
                            annealing_step)
        elif hyper_params['schedule'] == 'reduceonplateau':
            annealing_red = partial(annealing_reduceonplateau, patience=hyper_params['rop_patience'], factor=hyper_params['gamma'])
            tr_it.add_phase(int(len(gt_set) * hyper_params['epochs']),
                            annealing_red)
        elif hyper_params['schedule'] == 'cosine':
            annealing_cos = partial(annealing_cosine, t_max=hyper_params['cos_t_max'])
            tr_it.add_phase(int(len(gt_set) * hyper_params['epochs']),
                            annealing_cos)
        else:
            # constant learning rate scheduler
            tr_it.add_phase(int(len(gt_set) * hyper_params['epochs']),
                            annealing_const)

        if hyper_params['quit'] == 'early':
            st_it = EarlyStopping(hyper_params['min_delta'], hyper_params['lag'])
        elif hyper_params['quit'] == 'dumb':
            st_it = EpochStopping(hyper_params['epochs'] - hyper_params['completed_epochs'])
        else:
            logger.error(f'Invalid training interruption scheme {quit}')
            return None

        trainer = cls(model=nn,
                      optimizer=optim,
                      device=device,
                      filename_prefix=output,
                      event_frequency=hyper_params['freq'],
                      train_set=train_loader,
                      val_set=val_loader,
                      stopper=st_it,
                      loss_fn=baseline_label_loss_fn,
                      evaluator=baseline_label_evaluator_fn)

        trainer.add_lr_scheduler(tr_it)

        return trainer
