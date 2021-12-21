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
Arrow IPC format dataset builders.
"""
__all__ = ['build_binary_dataset']

import io
import json
import pathlib
import pyarrow as pa
import tempfile

from PIL import Image
from collections import Counter
from typing import Optional, List, Union, Callable
from multiprocessing import Pool
from kraken.lib import functional_im_transforms as F_t
from kraken.lib.segmentation import extract_polygons
from kraken.lib.xml import parse_xml, parse_alto, parse_page
from kraken.lib.util import is_bitonal, make_printable
from kraken.lib.exceptions import KrakenInputException

import logging

logger = logging.getLogger(__name__)


def _extract_line(xml_record):
    lines = []
    try:
        im = Image.open(xml_record['image'])
    except FileNotFoundError:
        return lines, None, None
    if not line['text']:
        return lines, None, None
    if is_bitonal(im):
        im = im.convert('1')
    line_counts = Counter({'all': 0, 'train': 0, 'validation': 0, 'test': 0})
    for line_im, line in extract_polygons(im, xml_record):
        fp = io.BytesIO()
        line_im.save(fp, format='png')
        if line['split']:
            line_counts[line['split']] += 1
        else:
            line_counts['all'] += 1
        lines.append({'text': line['text'], 'im': fp.getvalue()})
    return lines, im.mode, dict(line_counts)


def _extract_path_line(xml_record):
    try:
        im = Image.open(xml_record['image'])
    except FileNotFoundError:
        return [], None, None
    if not xml_record['lines'][0]['text']:
        return [], None, None
    if is_bitonal(im):
        im = im.convert('1')
    fp = io.BytesIO()
    im.save(fp, format='png')
    line = {'text': xml_record['lines'][0]['text'], 'im': fp.getvalue()}
    return [line], im.mode, {'all': 1, 'train': 0, 'validation': 0, 'test': 0}


def parse_path(path: Union[str, pathlib.Path], suffix: str = '.gt.txt', split=F_t.default_split):
    with open(F_t.suffix_split(path, split=split, suffix=suffix), 'r', encoding='utf-8') as fp:
        gt = fp.read().strip('\n\r')
        if not gt:
            raise KrakenInputException(f'No text for ground truth line {path}.')
    return {'image': path, 'lines': [{'text': gt}]}


def build_binary_dataset(files: Optional[List[Union[str, pathlib.Path]]] = None,
                         output_file: Union[str, pathlib.Path] = None,
                         format_type: str = 'xml',
                         num_workers: int = 0,
                         ignore_splits: bool = False,
                         recordbatch_size: int = 100,
                         callback: Callable[[int, int], None] = lambda chunk, lines: None) -> None:
    """
    Parses XML files and dumps the baseline-style line images and text into a
    binary dataset.

    Args:
        files: List of XML input files.
        output_file: Path to the output file.
        format_type: One of `xml`, `alto`, `page`, or `path`.
        num_workers: Number of workers for parallelized extraction of line
                     images. Set to `0` to disable parallelism.
        ignore_splits: Switch to disable serialization of the explicit
                       train/validation/test splits contained in the source
                       files.
        recordbatch_size: Minimum number of records per RecordBatch written to
                          the output file. Larger batches require more
                          transient memory but slightly improve reading
                          performance.
        callback: Function called everytime a new recordbatch is flushed into
                  the Arrow IPC file.
    """

    logger.info('Parsing XML files')
    extract_fn = _extract_line
    if format_type == 'xml':
        parse_fn = parse_xml
    elif format_type == 'alto':
        parse_fn = parse_alto
    elif format_type == 'page':
        parse_fn = parse_page
    elif format_type == 'path':
        parse_fn = parse_path
        extract_fn = _extract_path_line
    else:
        raise ValueError(f'invalid format {format_type} for parse_(xml,alto,page,path)')

    docs = []
    for doc in files:
        try:
            data = parse_fn(doc)
        except KrakenInputException as e:
            logger.warning(f'Invalid input file {doc}')
            continue
        try:
            with open(data['image'], 'rb') as fp:
                Image.open(fp)
        except FileNotFoundError as e:
            logger.warning(f'Could not open file {e.filename} in {doc}')
            continue
        docs.append(data)
    logger.info(f'Parsed {len(docs)} files.')

    logger.info('Assembling dataset alphabet.')
    alphabet = Counter()
    num_lines = 0
    for doc in docs:
        for line in doc['lines']:
            num_lines += 1
            alphabet.update(line['text'])

    for k, v in sorted(alphabet.items(), key=lambda x: x[1], reverse=True):
        char = make_printable(k)
        if char == k:
            char = '\t' + char
        logger.info(f'{char}\t{v}')

    metadata = {'lines': {'type': 'kraken_recognition_baseline' if format_type != 'path' else 'kraken_recognition_bbox',
                          'alphabet': alphabet,
                          'text_type': 'raw',
                          'image_type': 'raw',
                          'splits': ['train', 'eval', 'test'],
                          'im_mode': '1',
                          'counts': Counter({'all': 0,
                                             'train': 0,
                                             'validation': 0,
                                             'test': 0
                                             }
                                            ),
                          }
                }

    ty = pa.struct([('text', pa.string()), ('im', pa.binary())])
    schema = pa.schema([('lines', ty), ('train', pa.bool_()), ('validation', pa.bool_()), ('test', pa.bool_())])

    def _make_record_batch(line_cache):
        ar = pa.array(line_cache, type=ty)
        train_mask = pa.array([False] * len(line_cache), type=pa.bool_())
        val_mask = pa.array([False] * len(line_cache), type=pa.bool_())
        test_mask = pa.array([False] * len(line_cache), type=pa.bool_())
        rbatch = pa.RecordBatch.from_arrays([ar, train_mask, val_mask, test_mask], schema=schema)
        return rbatch

    line_cache = []
    logger.info('Writing lines to temporary file.')
    with tempfile.TemporaryDirectory() as tmp_output_dir:
        tmp_file = tmp_output_dir + '/dataset.arrow'
        with pa.OSFile(tmp_file, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:

                if num_workers and num_workers > 1:
                    logger.info(f'Spinning up processing pool with {num_workers} workers.')
                    with Pool(num_workers) as pool:
                        for page_lines, im_mode, line_counts in pool.imap_unordered(extract_fn, docs):
                            if page_lines:
                                line_cache.extend(page_lines)
                                # comparison RGB(A) > L > 1
                                if im_mode > metadata['lines']['im_mode']:
                                    metadata['lines']['im_mode'] = im_mode
                                metadata['lines']['counts'].update(line_counts)

                            if len(line_cache) >= recordbatch_size:
                                logger.info(f'Flushing {len(line_cache)} lines into {tmp_file}.')
                                rbatch = _make_record_batch(line_cache)
                                writer.write(rbatch)
                                callback(len(line_cache), num_lines)
                                line_cache = []
                else:
                    for page_lines, im_mode, line_counts in map(extract_fn, docs):
                        if page_lines:
                            line_cache.extend(page_lines)
                            # comparison RGB(A) > L > 1
                            if im_mode > metadata['lines']['im_mode']:
                                metadata['lines']['im_mode'] = im_mode
                            metadata['lines']['counts'].update(line_counts)

                        if len(line_cache) >= recordbatch_size:
                            logger.info(f'Flushing {len(line_cache)} lines into {tmp_file}.')
                            rbatch = _make_record_batch(line_cache)
                            writer.write(rbatch)
                            callback(len(line_cache), num_lines)
                            line_cache = []

                if line_cache:
                    logger.info(f'Flushing last {len(line_cache)} lines into {tmp_file}.')
                    rbatch = _make_record_batch(line_cache)
                    writer.write(rbatch)
                    callback(len(line_cache), num_lines)

        logger.info('Dataset metadata')
        logger.info(f"type: {metadata['lines']['type']}\n"
                    f"text_type: {metadata['lines']['text_type']}\n"
                    f"image_type: {metadata['lines']['image_type']}\n"
                    f"splits: {metadata['lines']['splits']}\n"
                    f"im_mode: {metadata['lines']['im_mode']}\n"
                    f"lines: {metadata['lines']['counts']}\n")

        with pa.memory_map(tmp_file, 'rb') as source:
            logger.info(f'Rewriting output ({output_file}) to update metadata.')
            ds = pa.ipc.open_file(source).read_all()
            metadata['lines']['counts'] = dict(metadata['lines']['counts'])
            metadata['lines'] = json.dumps(metadata['lines'])
            schema = schema.with_metadata(metadata)
            with pa.OSFile(output_file, 'wb') as sink:
                with pa.ipc.new_file(sink, schema) as writer:
                    writer.write(ds)
