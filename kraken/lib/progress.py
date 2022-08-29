# Copyright Benjamin Kiessling
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Handlers for rich-based progress bars.
"""
from typing import Any, Dict, Optional, Union
from numbers import Number

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.base import ProgressBarBase

from rich.console import Console, RenderableType
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn, TimeRemainingColumn, TimeElapsedColumn, DownloadColumn
from rich.text import Text

__all__ = ['KrakenProgressBar', 'KrakenDownloadProgressBar', 'KrakenTrainProgressBar']


class BatchesProcessedColumn(ProgressColumn):
    def __init__(self):
        super().__init__()

    def render(self, task) -> RenderableType:
        total = task.total if task.total != float("inf") else "--"
        return Text(f"{int(task.completed)}/{total}", style='magenta')


class EarlyStoppingColumn(ProgressColumn):
    """
    A column containing text.
    """

    def __init__(self, trainer):
        self._trainer = trainer
        self._tasks = {}
        self._current_task_id = 0
        super().__init__()

    def render(self, task) -> Text:
        if (
            self._trainer.state.fn != "fit"
            or self._trainer.sanity_checking
            or self._trainer.progress_bar_callback.main_progress_bar_id != task.id
        ):
            return Text()
        if self._trainer.training and task.id not in self._tasks:
            self._tasks[task.id] = "None"
            if self._renderable_cache:
                self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
            self._current_task_id = task.id
        if self._trainer.training and task.id != self._current_task_id:
            return self._tasks[task.id]

        text = f'early_stopping: ' \
               f'{self._trainer.early_stopping_callback.wait_count}/{self._trainer.early_stopping_callback.patience} ' \
               f'{self._trainer.early_stopping_callback.best_score:.5f}'
        return Text(text, justify="left")


class MetricsTextColumn(ProgressColumn):
    """A column containing text."""

    def __init__(self, trainer):
        self._trainer = trainer
        self._tasks = {}
        self._current_task_id = 0
        self._metrics = {}
        super().__init__()

    def update(self, metrics):
        # Called when metrics are ready to be rendered.
        # This is to prevent render from causing deadlock issues by requesting metrics
        # in separate threads.
        self._metrics = metrics

    def render(self, task) -> Text:
        if (
            self._trainer.state.fn != "fit"
            or self._trainer.sanity_checking
            or self._trainer.progress_bar_callback.main_progress_bar_id != task.id
        ):
            return Text()
        if self._trainer.training and task.id not in self._tasks:
            self._tasks[task.id] = "None"
            if self._renderable_cache:
                self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
            self._current_task_id = task.id
        if self._trainer.training and task.id != self._current_task_id:
            return self._tasks[task.id]

        text = ""
        for k, v in self._metrics.items():
            if isinstance(v, Number):
                text += f"{k}: {v:.5f} "
            else:
                text += f"{k}: {v} "
        return Text(text, justify="left")


class KrakenProgressBar(Progress):
    """
    Adaptation of the default rich progress bar to fit with kraken/ketos output.
    """
    def __init__(self, *args, **kwargs):
        columns = [TextColumn("[progress.description]{task.description}"),
                   BarColumn(),
                   TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                   BatchesProcessedColumn(),
                   TimeRemainingColumn(),
                   TimeElapsedColumn()]
        kwargs['refresh_per_second'] = 1
        super().__init__(*columns, *args, **kwargs)


class KrakenDownloadProgressBar(Progress):
    """
    Adaptation of the default rich progress bar to fit with kraken/ketos download output.
    """
    def __init__(self, *args, **kwargs):
        columns = [TextColumn("[progress.description]{task.description}"),
                   BarColumn(),
                   TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                   DownloadColumn(),
                   TimeRemainingColumn(),
                   TimeElapsedColumn()]
        kwargs['refresh_per_second'] = 1
        super().__init__(*columns, *args, **kwargs)


class KrakenTrainProgressBar(ProgressBarBase):
    """
    Adaptation of the default ptl rich progress bar to fit with kraken (segtrain, train) output.

    Args:
        refresh_rate: Determines at which rate (in number of batches) the progress bars get updated.
            Set it to ``0`` to disable the display.
        leave: Leaves the finished progress bar in the terminal at the end of the epoch. Default: False
        console_kwargs: Args for constructing a `Console`
    """
    def __init__(self,
                 refresh_rate: int = 1,
                 leave: bool = True,
                 console_kwargs: Optional[Dict[str, Any]] = None,
                 ignored_metrics=('loss', 'val_metric')) -> None:
        super().__init__()
        self._refresh_rate: int = refresh_rate
        self._leave: bool = leave
        self._console_kwargs = console_kwargs or {}
        self._enabled: bool = True
        self.progress: Optional[Progress] = None
        self.val_sanity_progress_bar_id: Optional[int] = None
        self._reset_progress_bar_ids()
        self._metric_component = None
        self._progress_stopped: bool = False
        self.ignored_metrics = ignored_metrics

    @property
    def refresh_rate(self) -> float:
        return self._refresh_rate

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    @property
    def sanity_check_description(self) -> str:
        return "Validation Sanity Check"

    @property
    def validation_description(self) -> str:
        return "Validation"

    @property
    def test_description(self) -> str:
        return "Testing"

    def _init_progress(self, trainer):
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            self._console = Console(**self._console_kwargs)
            self._console.clear_live()
            columns = self.configure_columns(trainer)
            self._metric_component = MetricsTextColumn(trainer)
            columns.append(self._metric_component)

            if trainer.early_stopping_callback:
                self._early_stopping_component = EarlyStoppingColumn(trainer)
                columns.append(self._early_stopping_component)

            self.progress = Progress(*columns,
                                     auto_refresh=False,
                                     disable=self.is_disabled,
                                     console=self._console)
            self.progress.start()
            # progress has started
            self._progress_stopped = False

    def refresh(self) -> None:
        if self.progress:
            self.progress.refresh()

    def on_train_start(self, trainer, pl_module):
        self._init_progress(trainer)

    def on_test_start(self, trainer, pl_module):
        self._init_progress(trainer)

    def on_validation_start(self, trainer, pl_module):
        self._init_progress(trainer)

    def on_sanity_check_start(self, trainer, pl_module):
        self._init_progress(trainer)

    def on_sanity_check_end(self, trainer, pl_module):
        if self.progress is not None:
            self.progress.update(self.val_sanity_progress_bar_id, advance=0, visible=False)
        self.refresh()

    def on_train_epoch_start(self, trainer, pl_module):
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float("inf"):
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch

        total_batches = total_train_batches + total_val_batches

        train_description = f"stage {trainer.current_epoch}/{trainer.max_epochs if pl_module.hparams.quit == 'dumb' else '∞'}"
        if len(self.validation_description) > len(train_description):
            # Padding is required to avoid flickering due of uneven lengths of "Epoch X"
            # and "Validation" Bar description
            num_digits = len(str(trainer.current_epoch))
            required_padding = (len(self.validation_description) - len(train_description) + 1) - num_digits
            for _ in range(required_padding):
                train_description += " "

        if self.main_progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(trainer)
        if self.main_progress_bar_id is None:
            self.main_progress_bar_id = self._add_task(total_batches, train_description)
        elif self.progress is not None:
            self.progress.reset(
                self.main_progress_bar_id, total=total_batches, description=train_description, visible=True
            )
        self.refresh()

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            self.val_sanity_progress_bar_id = self._add_task(self.total_val_batches, self.sanity_check_description)
        else:
            self.val_progress_bar_id = self._add_task(
                self.total_val_batches, self.validation_description, visible=False
            )
        self.refresh()

    def _add_task(self, total_batches: int, description: str, visible: bool = True) -> Optional[int]:
        if self.progress is not None:
            return self.progress.add_task(
                f"{description}", total=total_batches, visible=visible
            )

    def _update(self, progress_bar_id: int, current: int, total: Union[int, float], visible: bool = True) -> None:
        if self.progress is not None and self._should_update(current, total):
            leftover = current % self.refresh_rate
            advance = leftover if (current == total and leftover != 0) else self.refresh_rate
            self.progress.update(progress_bar_id, advance=advance, visible=visible)
            self.refresh()

    def _should_update(self, current: int, total: Union[int, float]) -> bool:
        return self.is_enabled and (current % self.refresh_rate == 0 or current == total)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_progress_bar_id is not None and trainer.state.fn == "fit":
            self.progress.update(self.val_progress_bar_id, advance=0, visible=False)
            self.refresh()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.state.fn == "fit":
            self._update_metrics(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_progress_bar_id = self._add_task(self.total_test_batches, self.test_description)
        self.refresh()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._update(self.main_progress_bar_id, self.train_batch_idx, self.total_train_batches)
        self._update_metrics(trainer, pl_module)
        self.refresh()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._update_metrics(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.sanity_checking:
            self._update(self.val_sanity_progress_bar_id, self.val_batch_idx, self.total_val_batches)
        elif self.val_progress_bar_id is not None:
            # check to see if we should update the main training progress bar
            if self.main_progress_bar_id is not None:
                self._update(self.main_progress_bar_id, self.val_batch_idx, self.total_val_batches)
            self._update(self.val_progress_bar_id, self.val_batch_idx, self.total_val_batches)
        self.refresh()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._update(self.test_progress_bar_id, self.test_batch_idx, self.total_test_batches)
        self.refresh()

    def _stop_progress(self) -> None:
        if self.progress is not None:
            self.progress.stop()
            # # signals for progress to be re-initialized for next stages
            self._progress_stopped = True

    def _reset_progress_bar_ids(self):
        self.main_progress_bar_id: Optional[int] = None
        self.val_progress_bar_id: Optional[int] = None
        self.test_progress_bar_id: Optional[int] = None

    def _update_metrics(self, trainer, pl_module) -> None:
        metrics = self.get_metrics(trainer, pl_module)
        for x in self.ignored_metrics:
            metrics.pop(x, None)
        if self._metric_component:
            self._metric_component.update(metrics)

    def teardown(self, trainer, pl_module, stage: Optional[str] = None) -> None:
        self._stop_progress()

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        self._stop_progress()

    @property
    def val_progress_bar(self) -> Task:
        return self.progress.tasks[self.val_progress_bar_id]

    @property
    def val_sanity_check_bar(self) -> Task:
        return self.progress.tasks[self.val_sanity_progress_bar_id]

    @property
    def main_progress_bar(self) -> Task:
        return self.progress.tasks[self.main_progress_bar_id]

    @property
    def test_progress_bar(self) -> Task:
        return self.progress.tasks[self.test_progress_bar_id]

    def configure_columns(self, trainer) -> list:
        return [TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                BatchesProcessedColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn()]
