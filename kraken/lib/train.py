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
import abc
import math
import torch
import logging
import numpy as np
import torch.nn.functional as F

from itertools import cycle
from typing import Tuple, Callable, List, Dict, Any

from kraken.lib import models, vgsl, segmentation
from kraken.lib.dataset import compute_error
from kraken.lib.exceptions import KrakenInputException

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


def annealing_const(start: float, end: float, pct: float) -> float:
    return start


def annealing_linear(start: float, end: float, pct: float) -> float:
    return start + pct * (end-start)


def annealing_cos(start: float, end: float, pct: float) -> float:
    co = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * co


class TrainScheduler(object):
    """
    Implements learning rate scheduling.
    """
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.steps: List[Dict[str, Any]] = []
        self.optimizer = optimizer
        self.cycle: Any = None

    def add_phase(self,
                  iterations: int,
                  lrate: Tuple[float, float] = (1e-4, 1e-4),
                  momentum: Tuple[float, float] = (0.9, 0.9),
                  wd: float = 0.0,
                  annealing_fn: Callable[[float, float, float], float] = annealing_const) -> None:
        """
        Adds a new phase to the scheduler.

        Args:
            sched (kraken.lib.train.Trainscheduler): TrainScheduler instance
            epochs (int): Number of iterations per cycle
            max_lr (float): Peak learning rate
            div (float): divisor to determine minimum learning rate (min_lr = max_lr / div)
            max_mon (float): Maximum momentum
            min_mon (float): Minimum momentum
            wd (float): Weight decay
            annealing_fn (Callable[[int, int, int], float]): LR change
            function. Can be one of `annealing_const` (keeping start value),
            `annealing_linear` (linear change), and `annealing_cos` (cosine
                    change).
        """
        self.steps.extend([{'lr': annealing_fn(*lrate, pct=x/iterations),
                            'momentum': annealing_fn(*momentum, pct=x/iterations),
                            'weight_decay': wd} for x in range(iterations)])

    def step(self) -> None:
        """
        Performs an optimization step.
        """
        if not self.cycle:
            self.cycle = cycle(self.steps)
        kwargs = next(self.cycle)
        for param_group in self.optimizer.param_groups:
            param_group.update(kwargs)


def add_1cycle(sched: TrainScheduler, iterations: int,
               max_lr: float = 1e-4, div: float = 25.0,
               max_mom: float = 0.95, min_mom: float = 0.85, wd: float = 0.0):
    """
    Adds 1cycle policy [0] phases to a learning rate scheduler.

    [0] Smith, Leslie N. "A disciplined approach to neural network
    hyper-parameters: Part 1--learning rate, batch size, momentum, and weight
    decay." arXiv preprint arXiv:1803.09820 (2018).

    Args:
        sched (kraken.lib.train.Trainscheduler): TrainScheduler instance
        iterations (int): Number of iterations per cycle
        max_lr (float): Peak learning rate
        div (float): divisor to determine minimum learning rate (min_lr = max_lr / div)
        max_mon (float): Maximum momentum
        min_mon (float): Minimum momentum
        wd (float): Weight decay
    """
    sched.add_phase(iterations//2, (max_lr/div, max_lr), (max_mom, min_mom), wd, annealing_linear)
    sched.add_phase(iterations//2, (max_lr, max_lr/div), (min_mom, max_mom), wd, annealing_cos)


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
    # height should be 1 by now
    if output.size(2) != 1:
        raise KrakenInputException('Expected dimension 3 to be 1, actual {}'.format(output.size(2)))
    output = output.squeeze(2)
    # NCW -> WNC
    loss = criterion(output.permute(2, 0, 1),  # type: ignore
                     target,
                     (output.size(2),),
                     (target.size(1),))
    return loss


def baseline_label_loss_fn(criterion, output, target):
    output = F.interpolate(output, size=(target.size(1), target.size(2)))
    loss = criterion(output, target)
    return loss


def recognition_evaluator_fn(model, val_set, device):
    rec = models.TorchSeqRecognizer(model, device=device)
    chars, error = compute_error(rec, list(val_set))
    model.train()
    accuracy = (chars-error)/chars
    return {'val_metric': accuracy, 'accuracy': accuracy, 'chars': chars, 'error': error}


def baseline_label_evaluator_fn(model, val_set, device):

    true_positives = 0
    all_positives = 0
    actual_positives = 0
    all_n = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_set:
            x = x.to(device)
            y = y.to(device).unsqueeze(0)
            pred = model.nn(x.unsqueeze(0))
            # scale output to target size
            pred = F.interpolate(pred, size=(y.size(1), y.size(2)))
            pred = segmentation.denoising_hysteresis_thresh(pred.detach().squeeze().cpu().numpy(), 0.4, 0.5, 0)
            # squash classes to indices
            pred = torch.from_numpy(pred.astype('f')).to(device)
            pred = pred.argmax(dim=0).view(-1)
            y = y.view(-1)
            correct = torch.eq(y, pred)
            correct[y == 0] = 0
            correct[pred == 0] = 0
            all_p = (pred != 0).sum(dim=0).type(torch.DoubleTensor)
            actual_p = (y != 0).sum(dim=0).type(torch.DoubleTensor)
            if correct.sum() == 0:
                tp = torch.zeros_like(all_p)
            else:
                tp = correct.sum(dim=0)
            tp = tp.type(torch.DoubleTensor)
            true_positives += tp
            all_positives += all_p
            actual_positives += actual_p
            all_n += len(y)
    model.train()
    # all_positives = tp + fp
    # actual_positives = tp + fn
    # true_positivies = tp
    precision = true_positives/(all_positives + 1e-20)
    recall = true_positives/(actual_positives + 1e-20)
    f1 = precision * recall * 2/(precision + recall + 1e-20)
    s = actual_positives/all_n
    p = all_positives/all_n
    mcc = ((true_positives/all_n) - s*p)/(torch.sqrt(p*s*(1-s)*(1-p)) + 1e-20)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'mcc': mcc, 'val_metric': mcc}


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
            for _, (input, target) in zip(range(self.event_it), self.train_set):
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                input = input.requires_grad_()
                o = self.model.nn(input)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model.criterion, o, target)
                if not torch.isinf(loss):
                    loss.backward()
                    self.optimizer.step()
                else:
                    logger.debug('infinite loss in trial')
                iteration_callback()
            self.iterations += self.event_it
            logger.debug('Starting evaluation run')
            eval_res = self.evaluator(self.model, self.val_set, self.device)
            self.stopper.update(eval_res['val_metric'])
            self.model.user_metadata['accuracy'].append((self.iterations, float(eval_res['val_metric'])))
            logger.info('Saving to {}_{}'.format(self.filename_prefix, self.stopper.epoch))
            event_callback(epoch=self.stopper.epoch, **eval_res)
            # fill one_channel_mode after 1 iteration over training data set
            print(self.train_set.dataset.im_mode)
            #self.model.one_channel_mode = self.train_set.dataset.one_channel_mode
            try:
                self.model.user_metadata['completed_epochs'] = self.stopper.epoch
                self.model.save_model('{}_{}.mlmodel'.format(self.filename_prefix, self.stopper.epoch))
            except Exception as e:
                logger.error('Saving model failed: {}'.format(str(e)))
