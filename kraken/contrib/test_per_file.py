#!/usr/bin/env python
"""
Test GT with per-file detailed report.
"""
import click

import logging
from typing import List
from collections import defaultdict

from threadpoolctl import threadpool_limits
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate

from kraken.lib.default_specs import RECOGNITION_HYPER_PARAMS
from kraken.lib.exceptions import KrakenInputException
from kraken.lib import models
from kraken.lib.dataset import (ImageInputTransforms, PolygonGTDataset,
                                collate_sequences, global_align)
from kraken.lib.progress import KrakenProgressBar
from kraken.lib.xml import XMLPage
from kraken.ketos.util import _expand_gt, _validate_manifests, message

logging.captureWarnings(True)
logger = logging.getLogger('kraken')


@click.command()
@click.pass_context
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=RECOGNITION_HYPER_PARAMS['batch_size'], help='Batch sample size')
@click.option('-m', '--model', show_default=True, type=click.Path(exists=True, readable=True),
              multiple=True, help='Model(s) to evaluate')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--pad', show_default=True, type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('--workers', show_default=True, default=1,
              type=click.IntRange(0),
              help='Number of worker processes when running on CPU.')
@click.option('--threads', show_default=True, default=1,
              type=click.IntRange(1),
              help='Max size of thread pools for OpenMP/BLAS operations.')
@click.option('--reorder/--no-reorder', show_default=True, default=True, help='Reordering of code points to display order')
@click.option('--base-dir', show_default=True, default='auto',
              type=click.Choice(['L', 'R', 'auto']), help='Set base text '
              'direction.  This should be set to the direction used during the '
              'creation of the training data. If set to `auto` it will be '
              'overridden by any explicit value given in the input files.')
@click.option('-u', '--normalization', show_default=True, type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              default=None, help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace',
              show_default=True, default=True, help='Normalizes unicode whitespace')
@click.option('-f', '--format-type', type=click.Choice(['alto', 'page']), default='alto',
              help='Sets the input document format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines, polygons, and a '
              'link to source images.')
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def test(ctx, batch_size, model, evaluation_files, device, pad, workers,
         threads, reorder, base_dir, normalization, normalize_whitespace,
         format_type, test_set):
    """
    Evaluate on a test set and create a detailed report per file.
    """
    if not model:
        raise click.UsageError('No model to evaluate given.')

    logger.info('Building test set from {} line images'.format(len(test_set) + len(evaluation_files)))

    nn = {}
    for p in model:
        message('Loading model {}\t'.format(p), nl=False)
        nn[p] = models.load_any(p, device)
        message('\u2713', fg='green')

    pin_ds_mem = False
    if device != 'cpu':
        pin_ds_mem = True

    test_set = list(test_set)
    if evaluation_files:
        test_set.extend(evaluation_files)

    test_set = [{'page': XMLPage(file, filetype=format_type).to_container(),
                 'source': str(file)} for file in test_set]
    valid_norm = False

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    if reorder and base_dir != 'auto':
        reorder = base_dir

    grouped_files = defaultdict(list)
    for entry in test_set:
        grouped_files[entry['source']].append(entry['page'])

    with threadpool_limits(limits=threads):
        for p, net in nn.items():
            message('Evaluating {}'.format(p))
            logger.info('Evaluating {}'.format(p))
            batch, channels, height, width = net.nn.input

            aggregated_cer = []
            aggregated_wer = []

            ts = ImageInputTransforms(batch, height, width, channels, (pad, 0), valid_norm)

            # Process each file group separately.
            for source, pages in grouped_files.items():
                logger.info("Processing file: {}".format(source))

                # Create a new dataset
                ds = PolygonGTDataset(normalization=normalization,
                                      whitespace_normalization=normalize_whitespace,
                                      reorder=reorder,
                                      im_transforms=ts)

                for page in pages:
                    try:
                        ds.add(page=page)
                    except ValueError as e:
                        logger.info("Error adding page from {}: {}".format(source, e))

                ds.no_encode()
                ds_loader = DataLoader(ds,
                                       batch_size=batch_size,
                                       num_workers=workers,
                                       pin_memory=pin_ds_mem,
                                       collate_fn=collate_sequences)

                # Set up per-file metrics and counters.
                file_cer = CharErrorRate()
                file_wer = WordErrorRate()
                chars = 0
                error = 0
                algn_gt: List[str] = []
                algn_pred: List[str] = []

                # Evaluate with a per-file progress bar.
                with KrakenProgressBar() as progress:
                    total_batches = len(ds_loader)
                    pred_task = progress.add_task(f"Evaluating {source}",
                                                  total=total_batches,
                                                  visible=True if not ctx.meta.get('verbose', False) else False)
                    for batch in ds_loader:
                        im = batch['image']
                        text = batch['target']
                        lens = batch['seq_lens']
                        try:
                            pred = net.predict_string(im, lens)
                            for x, y in zip(pred, text):
                                chars += len(y)
                                c, a1, a2 = global_align(y, x)
                                algn_gt.extend(a1)
                                algn_pred.extend(a2)
                                error += c
                                file_cer.update(x, y)
                                file_wer.update(x, y)
                        except FileNotFoundError as e:
                            total_batches -= 1
                            progress.update(pred_task, total=total_batches)
                            logger.warning('{} {}. Skipping.'.format(e.strerror, e.filename))
                        except KrakenInputException as e:
                            total_batches -= 1
                            progress.update(pred_task, total=total_batches)
                            logger.warning(str(e))
                        progress.update(pred_task, advance=1)

                # Compute per-file accuracies (accuracy = 1 - error rate)
                cer_acc = 1.0 - file_cer.compute()
                wer_acc = 1.0 - file_wer.compute()
                aggregated_cer.append(cer_acc)
                aggregated_wer.append(wer_acc)

                report_line = "Char: {}, CER: {:.1f}%, WER: {:.1f}%".format(
                    chars, cer_acc * 100, wer_acc * 100
                )
                message(report_line.rjust(40))

            avg_cer = np.mean(aggregated_cer)
            avg_wer = np.mean(aggregated_wer)

            message('Aggregated results for model {}:'.format(p))
            message('Average character accuracy: {:0.2f}%'.format(avg_cer * 100))
            message('Average word accuracy: {:0.2f}%'.format(avg_wer * 100))


if __name__ == '__main__':
    test()
