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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from lightning.pytorch.callbacks.progress.rich_progress import (
    CustomProgress, MetricsTextColumn, RichProgressBar)
from rich import get_console, reconfigure
from rich.default_styles import DEFAULT_STYLES
from rich.progress import (BarColumn, DownloadColumn, Progress, ProgressColumn,
                           TextColumn, TimeElapsedColumn, TimeRemainingColumn)
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.style import Style

__all__ = ['KrakenProgressBar', 'KrakenDownloadProgressBar', 'KrakenTrainProgressBar']


class BatchesProcessedColumn(ProgressColumn):
    def __init__(self):
        super().__init__()

    def render(self, task) -> 'RenderableType':
        total = task.total if task.total != float("inf") else "--"
        return Text(f"{int(task.completed)}/{total}", style='magenta')


class EarlyStoppingColumn(ProgressColumn):
    """
    A column containing text.
    """

    def __init__(self, trainer):
        self._trainer = trainer
        super().__init__()

    def render(self, task) -> Text:

        text = f'early_stopping: ' \
               f'{self._trainer.early_stopping_callback.wait_count}/{self._trainer.early_stopping_callback.patience} ' \
               f'{self._trainer.early_stopping_callback.best_score:.5f}'
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


class KrakenTrainProgressBar(RichProgressBar):
    """
    Adaptation of the default ptl rich progress bar to fit with kraken (segtrain, train) output.

    Args:
        refresh_rate: Determines at which rate (in number of batches) the progress bars get updated.
            Set it to ``0`` to disable the display.
        leave: Leaves the finished progress bar in the terminal at the end of the epoch. Default: False
        console_kwargs: Args for constructing a `Console`
    """
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs, theme=RichProgressBarTheme())

    def _init_progress(self, trainer):
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            self._console.clear_live()
            self._metric_component = MetricsTextColumn(trainer,
                                                       self.theme.metrics,
                                                       self.theme.metrics_text_delimiter,
                                                       self.theme.metrics_format)
            columns = self.configure_columns(trainer)
            columns.append(self._metric_component)

            if trainer.early_stopping_callback:
                self._early_stopping_component = EarlyStoppingColumn(trainer)
                columns.append(self._early_stopping_component)

            self.progress = CustomProgress(
                *columns,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False

    def _get_train_description(self, current_epoch: int) -> str:
        return f"stage {current_epoch}/" \
               f"{self.trainer.max_epochs if self.trainer.model.hparams.hyper_params['quit'] == 'fixed' else '∞'}"


@dataclass
class RichProgressBarTheme:
    """Styles to associate to different base components.

    Args:
        description: Style for the progress bar description. For eg., Epoch x, Testing, etc.
        progress_bar: Style for the bar in progress.
        progress_bar_finished: Style for the finished progress bar.
        progress_bar_pulse: Style for the progress bar when `IterableDataset` is being processed.
        batch_progress: Style for the progress tracker (i.e 10/50 batches completed).
        time: Style for the processed time and estimate time remaining.
        processing_speed: Style for the speed of the batches being processed.
        metrics: Style for the metrics

    https://rich.readthedocs.io/en/stable/style.html
    """

    description: Union[str, 'Style'] = DEFAULT_STYLES['progress.description']
    progress_bar: Union[str, 'Style'] = DEFAULT_STYLES['bar.complete']
    progress_bar_finished: Union[str, 'Style'] = DEFAULT_STYLES['bar.finished']
    progress_bar_pulse: Union[str, 'Style'] = DEFAULT_STYLES['bar.pulse']
    batch_progress: Union[str, 'Style'] = DEFAULT_STYLES['progress.description']
    time: Union[str, 'Style'] = DEFAULT_STYLES['progress.elapsed']
    processing_speed: Union[str, 'Style'] = DEFAULT_STYLES['progress.data.speed']
    metrics: Union[str, 'Style'] = DEFAULT_STYLES['progress.description']
    metrics_text_delimiter: str = ' '
    metrics_format: str = '.3f'
