#
# Copyright 2026 Benjamin Kiessling
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
Common base class for kraken LightningModule trainers.
"""
from typing import TYPE_CHECKING, Any, ClassVar, Union

import lightning as L

if TYPE_CHECKING:
    from os import PathLike
    from torch.nn import Module

__all__ = ['KrakenTrainerModule']


class KrakenTrainerModule(L.LightningModule):

    _task: ClassVar[str]
    _model_class: ClassVar[type]
    _config_class: ClassVar[type]

    @classmethod
    def load_from_weights(cls,
                          path: Union[str, 'PathLike'],
                          config: Any) -> 'KrakenTrainerModule':
        """
        Initializes the module from a model weights file by selecting the first
        model in the file whose type and class matches the trainer's task.
        """
        from kraken.models import load_models
        models = load_models(path, tasks=[cls._task])
        for model in models:
            if isinstance(model, cls._model_class):
                cls._post_load_weights(model, config)
                return cls(config=config, model=model)
        raise ValueError(f'No {cls._model_class.__name__} of type `{cls._task}` found in {path}.')

    @classmethod
    def _post_load_weights(cls, model: 'Module', config: Any) -> None:
        """Hook to mutate config or the loaded model before construction."""
        return None

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        self._save_checkpoint_extras(checkpoint)
        checkpoint['_module_config'] = self.hparams.config
        self._append_validation_metrics(checkpoint)

    def _save_checkpoint_extras(self, checkpoint: dict) -> None:
        """Hook to write task-specific keys/state to the checkpoint."""
        return None

    def _append_validation_metrics(self, checkpoint: dict) -> None:
        metrics = {k: v.item() if hasattr(v, 'item') else v
                   for k, v in self.trainer.callback_metrics.items()
                   if k.startswith('val_')}
        if metrics:
            self.net.user_metadata.setdefault('metrics', []).append((self.current_epoch, metrics))

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        config = checkpoint.get('_module_config')
        if config is None:
            config = checkpoint.get('hyper_parameters', {}).get('config')
        if not isinstance(config, self._config_class):
            raise ValueError(f'Checkpoint is not a {self._task} model.')
        self.net = self._build_net_from_checkpoint(checkpoint)
        self._post_load_checkpoint(checkpoint)

    def _build_net_from_checkpoint(self, checkpoint: dict) -> 'Module':
        """Reconstruct the underlying network from checkpoint state."""
        raise NotImplementedError

    def _post_load_checkpoint(self, checkpoint: dict) -> None:
        """Hook for additional state restoration after net reconstruction."""
        if hasattr(self.net, 'input'):
            self.batch, self.channels, self.height, self.width = self.net.input
