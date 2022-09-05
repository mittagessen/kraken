#!/usr/bin/env python
"""
A script for a grid search over pretraining hyperparameters.
"""
import click
from functools import partial

from ray import tune

from ray.tune.integration.pytorch_lightning import TuneReportCallback

from kraken.lib.default_specs import RECOGNITION_PRETRAIN_HYPER_PARAMS, RECOGNITION_SPEC
from kraken.lib.pretrain.model import PretrainDataModule, RecognitionPretrainModel
from kraken.ketos.util import _validate_manifests

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

config = {'lrate': tune.loguniform(1e-8, 1e-2),
          'num_negatives': tune.qrandint(1, 4, 1),
          'mask_prob': tune.loguniform(0.01, 0.2),
          'mask_width': tune.qrandint(2, 8, 2)}

resources_per_trial = {"cpu": 8, "gpu": 0.5}


def train_tune(config, training_data=None, epochs=100, spec=RECOGNITION_SPEC):

    hyper_params = RECOGNITION_PRETRAIN_HYPER_PARAMS.copy()
    hyper_params.update(config)

    model = RecognitionPretrainModel(hyper_params=hyper_params,
                                     output='./model',
                                     spec=spec)

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
                         accelerator='gpu',
                         devices=1,
                         callbacks=[callback],
                         enable_progress_bar=False)
    trainer.fit(model, datamodule=data_module)

@click.command()
@click.option('-v', '--verbose', default=0, count=True)
@click.option('-s', '--seed', default=42, type=click.INT,
              help='Seed for numpy\'s and torch\'s RNG. Set to a fixed value to '
                   'ensure reproducible random splits of data')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='pretrain_hyper', help='output directory')
@click.option('-n', '--num-samples', show_default=True, type=int, default=100, help='Number of samples to train')
@click.option('-N', '--epochs', show_default=True, type=int, default=10, help='Maximum number of epochs to train per sample')
@click.option('-s', '--spec', show_default=True, default=RECOGNITION_SPEC, help='VGSL spec of the network to train.')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.argument('files', nargs=-1)
def cli(verbose, seed, output, num_samples, epochs, spec, training_files, files):

    files = list(files)

    if training_files:
        files.extend(training_files)

    if not files:
        raise click.UsageError('No training data was provided to the search command. Use `-t` or the `files` argument.')

    seed_everything(seed, workers=True)

    analysis = tune.run(partial(train_tune,
                                training_data=files,
                                epochs=epochs,
                                spec=spec), local_dir=output, num_samples=num_samples, resources_per_trial=resources_per_trial, config=config)

    click.echo("Best hyperparameters found were: ", analysis.get_best_config(metric='accuracy', mode='max'))

if __name__ == '__main__':
    cli()
