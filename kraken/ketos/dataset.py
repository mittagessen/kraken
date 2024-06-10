#
# Copyright 2022 Benjamin Kiessling
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
kraken.ketos.dataset
~~~~~~~~~~~~~~~~~~~~

Command line driver for dataset compilation
"""
import click

from .util import _validate_manifests


@click.command('compile')
@click.pass_context
@click.option('-o', '--output', show_default=True, type=click.Path(), default='dataset.arrow', help='Output dataset file')
@click.option('--workers', show_default=True, default=1, help='Number of parallel workers for text line extraction.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page']), default='xml', show_default=True,
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information.')
@click.option('-F', '--files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data.')
@click.option('--random-split', type=float, nargs=3, default=None, show_default=True,
              help='Creates a fixed random split of the input data with the '
              'proportions (train, validation, test). Overrides the save split option.')
@click.option('--force-type', type=click.Choice(['bbox', 'baseline']), default=None, show_default=True,
              help='Forces the dataset type to a specific value. Can be used to '
                   '"convert" a line strip-type collection to a baseline-style '
                   'dataset, e.g. to disable centerline normalization.')
@click.option('--save-splits/--ignore-splits', show_default=True, default=True,
              help='Whether to serialize explicit splits contained in XML '
                   'files. Is ignored in `path` mode.')
@click.option('--skip-empty-lines/--keep-empty-lines', show_default=True, default=True,
              help='Whether to keep or skip empty text lines. Text-less '
                   'datasets are useful for unsupervised pretraining but '
                   'loading datasets with many empty lines for recognition '
                   'training is inefficient.')
@click.option('--recordbatch-size', show_default=True, default=100,
              help='Minimum number of records per RecordBatch written to the '
                   'output file. Larger batches require more transient memory '
                   'but slightly improve reading performance.')
@click.option('--legacy-polygons', show_default=True, default=False, is_flag=True,
                help='Use the old polygon extractor.')
@click.argument('ground_truth', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def compile(ctx, output, workers, format_type, files, random_split, force_type,
            save_splits, skip_empty_lines, recordbatch_size, ground_truth, legacy_polygons):
    """
    Precompiles a binary dataset from a collection of XML files.
    """
    from kraken.lib.progress import KrakenProgressBar

    from .util import message

    ground_truth = list(ground_truth)

    if files:
        ground_truth.extend(files)

    if not ground_truth:
        raise click.UsageError('No training data was provided to the compile command. Use the `ground_truth` argument.')

    from kraken.lib import arrow_dataset

    force_type = {'bbox': 'kraken_recognition_bbox',
                  'baseline': 'kraken_recognition_baseline',
                  None: None}[force_type]

    with KrakenProgressBar() as progress:
        extract_task = progress.add_task('Extracting lines', total=0, start=False, visible=True if not ctx.meta['verbose'] else False)

        def _update_bar(advance, total):
            if not progress.tasks[0].started:
                progress.start_task(extract_task)
            progress.update(extract_task, total=total, advance=advance)

        arrow_dataset.build_binary_dataset(ground_truth,
                                           output,
                                           format_type,
                                           workers,
                                           save_splits,
                                           random_split,
                                           force_type,
                                           recordbatch_size,
                                           skip_empty_lines,
                                           _update_bar,
                                           legacy_polygons=legacy_polygons)

    message(f'Output file written to {output}')
