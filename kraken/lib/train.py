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

from itertools import cycle
from torch.utils import data
from functools import partial
from typing import Tuple, Union, Optional, Callable, List, Dict, Any
from collections.abc import Iterable

from kraken.lib.exceptions import KrakenStopTrainingException

logger = logging.getLogger(__name__)


class TrainStopper(Iterable):

    def __init__(self):
        self.best_loss = -math.inf
        self.best_iteration= 0

    @abc.abstractmethod
    def update(self, val_loss: float) -> None:
        """
        Updates the internal state of the train stopper.
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
            iterations (int): Number of iterations per cycle
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

    [0] Smith, Leslie N. "A disciplined approach to neural network hyper-parameters: Part 1--learning rate, batch size, momentum, and weight decay." arXiv preprint arXiv:1803.09820 (2018).

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
    def __init__(self, it: data.DataLoader, min_delta: float = None, lag: int = 1000) -> None:
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
        self.it = cycle(it)
        self.wait = -1
        self.iteration = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.wait >= self.lag:
            raise KrakenStopTrainingException
        self.iteration += 1
        self.wait += 1
        return next(self.it)

    def update(self, val_loss: float) -> None:
        """
        Updates the internal validation loss state
        """
        if self.auto_delta:
            self.min_delta = (1 - self.best_loss)/20
            logger.debug('Rescaling early stopping loss to {}'.format(self.min_delta))
        if (val_loss - self.best_loss) >= self.min_delta:
            logger.debug('Resetting early stopping counter')
            self.wait = 0
            self.best_loss = val_loss
            self.best_iteration = self.iteration


class EpochStopping(TrainStopper):
    """
    Dumb stopping after a fixed number of iterations.
    """
    def __init__(self, it: data.DataLoader, iterations: int) -> None:
        """
        Args:
            it (torch.utils.data.DataLoader): training data loader
            iterations (int): Number of iterations to train for
        """
        super().__init__()
        self.iterations = iterations
        self.iteration = -1
        self.it = cycle(it)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration < self.iterations - 1:
            self.iteration += 1
            return next(self.it)
        else:
            raise KrakenStopTrainingException

    def update(self, val_loss: float) -> None:
        """
        Only update internal best iteration
        """
        if val_loss > self.best_loss:
            self.best_loss = val_loss
            self.best_iteration = self.iteration
