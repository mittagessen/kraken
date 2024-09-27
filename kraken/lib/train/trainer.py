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
import warnings
import lightning as L

from typing import TYPE_CHECKING, Union, Optional
from lightning.pytorch.callbacks import (BaseFinetuning, Callback,
                                         LearningRateMonitor)

from kraken.lib import progress

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


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
