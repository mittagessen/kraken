#!/usr/bin/env python
"""
A script for a grid search over pretraining hyperparameters.
"""
import sys
from functools import partial

from ray import tune

from ray.tune.integration.pytorch_lightning import TuneReportCallback

from kraken.lib.default_spec import RECOGNITION_PRETRAIN_HYPER_PARAMS, RECOGNITION_SPEC
from kraken.lib.pretrain.model import PretrainDataModule, RecognitionPretrainModel
from ray.tune.schedulers import ASHAScheduler

import pytorch_lightning as pl


config = {'lrate': tune.loguniform(1e-8, 1e-2),
          'num_negatives': tune.qrandint(2, 100, 8),
          'mask_prob': tune.loguniform(0.01, 0.2),
          'mask_width': tune.qrandint(2, 8, 2)}

resources_per_trial = {"cpu": 8, "gpu": 0.5}


def train_tune(config, training_data=None, epochs=100):

    hyper_params = RECOGNITION_PRETRAIN_HYPER_PARAMS.copy()
    hyper_params.update(config)

    model = RecognitionPretrainModel(hyper_params=hyper_params,
                                     output='model',
                                     spec=RECOGNITION_SPEC)

    data_module = PretrainDataModule(batch_size=hyper_params.pop('batch_size'),
                                     pad=hyper_params.pop('pad'),
                                     augment=hyper_params.pop('augment'),
                                     training_data=training_data,
                                     num_workers=resources_per_trial['cpu'],
                                     height=model.height,
                                     width=model.width,
                                     channels=model.channels,
                                     format_type='binary')

    callback = TuneReportCallback({'loss': 'CE'}, on='validation_end')
    trainer = pl.Trainer(max_epochs=epochs,
                         gpus=1,
                         callbacks=[callback],
                         enable_progress_bar=False)
    trainer.fit(model)


analysis = tune.run(partial(train_tune, training_data=sys.argv[2:]), local_dir=sys.argv[1], num_samples=100, resources_per_trial=resources_per_trial, config=config)

print("Best hyperparameters found were: ", analysis.get_best_config(metric='accuracy', mode='max'))
