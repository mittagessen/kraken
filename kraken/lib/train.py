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
import numpy as np

class EarlyStopping(object):
    """
    Early stopping to terminate training when validation loss doesn't improve
    over a certain time.
    """
    def __init__(self, it=None, min_delta=0.002, lag=5):
        """
        Args:
            it (torch.utils.data.DataLoader): training data loader
            min_delta (float): minimum change in validation loss to qualify as improvement.
            lag (int): Number of epochs to wait for improvement before
                       terminating.
        """
        self.min_delta = min_delta
        self.lag = lag
        self.it = it
        self.best_loss = 0
        self.wait = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.wait >= self.lag:
             raise StopIteration
        return self.it

    def update(self, val_loss):
        """
        Updates the internal validation loss state
        """
        if (val_loss - self.best_loss) < self.min_delta:
            self.wait += 1
        else:
            self.wait = 0
            self.best_loss = val_loss


class EpochStopping(object):
    """
    Dumb stopping after a fixed number of epochs.
    """
    def __init__(self, it=None, epochs=100):
        """
        Args:
            it (torch.utils.data.DataLoader): training data loader
            epochs (int): Number of epochs to train for
        """
        self.epochs = epochs
        self.compl_epochs = 0
        self.it = it

    def __iter__(self):
        return self

    def __next__(self):
        if self.compl_epochs < self.epochs:
            self.compl_epochs += 1
            return self.it
        else:
            raise StopIteration

    def update(self, val_loss):
        """
        No-Op for this stopper
        """
        pass
