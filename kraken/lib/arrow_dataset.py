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
import numpy as np
import pyarrow as pa
import tempfile

from PIL import Image, UnidentifiedImageError
from functools import partial
from collections import Counter
from typing import Optional, List, Union, Callable, Tuple, Dict
from multiprocessing import Pool
from kraken.lib import functional_im_transforms as F_t
from kraken.lib.segmentation import extract_polygons
from kraken.lib.xml import parse_xml, parse_alto, parse_page
from kraken.lib.util import is_bitonal, make_printable
from kraken.lib.exceptions import KrakenInputException
from os import extsep, PathLike

import logging

logger = logging.getLogger(__name__)


def _extract_line(xml_record, skip_empty_lines: bool = True):
    lines = []
    try:
        im = Image.open(xml_record['image'])
    except (FileNotFoundError, UnidentifiedImageError):
        return lines, None, None
    if is_bitonal(im):
        im = im.convert('1')
    seg_key = 'lines' if 'lines' in xml_record else 'boxes'
    recs = xml_record.pop(seg_key)
    for idx, rec in enumerate(recs):
        try:
            line_im, line = next(extract_polygons(im, {**xml_record, seg_key: [rec]}))
        except KrakenInputException:
            logger.warning(f'Invalid line {idx} in {im.filename}')
            continue
        except Exception as e:
            logger.warning(f'Unexpected exception {e} from line {idx} in {im.filename}')
            continue
        if not line['text'] and skip_empty_lines:
            continue
        fp = io.BytesIO()
        line_im.save(fp, format='png')
        lines.append({'text': line['text'], 'im': fp.getvalue()})
    return lines, im.mode


def _extract_path_line(xml_record, skip_empty_lines: bool = True):
    try:
        im = Image.open(xml_record['image'])
    except (FileNotFoundError, UnidentifiedImageError):
        return [], None, None
    if not xml_record['lines'][0]['text'] and skip_empty_lines:
        return [], None, None
    if is_bitonal(im):
        im = im.convert('1')
    fp = io.BytesIO()
    im.save(fp, format='png')
    line = {'text': xml_record['lines'][0]['text'], 'im': fp.getvalue()}
    return [line], im.mode


def parse_path(path: Union[str, PathLike],
               suffix: str = '.gt.txt',
               split=F_t.default_split,
               skip_empty_lines: bool = True):
    with open(F_t.suffix_split(path, split=split, suffix=suffix), 'r', encoding='utf-8') as fp:
        gt = fp.read().strip('\n\r')
        if not gt and skip_empty_lines:
            raise KrakenInputException(f'No text for ground truth line {path}.')
    return {'image': path, 'lines': [{'text': gt}]}


def build_binary_dataset(files: Optional[List[Union[str, PathLike, Dict]]] = None,
                         output_file: Union[str, PathLike] = None,
                         format_type: str = 'xml',
                         num_workers: int = 0,
                         ignore_splits: bool = False,
                         random_split: Optional[Tuple[float, float, float]] = None,
                         force_type: Optional[str] = None,
                         recordbatch_size: int = 100,
                         skip_empty_lines: bool = True,
                         callback: Callable[[int, int], None] = lambda chunk, lines: None) -> None:
    """
    Parses XML files and dumps the baseline-style line images and text into a
    binary dataset.

    Args:
        files: List of XML input files.
        output_file: Path to the output file.
        format_type: One of `xml`, `alto`, `page`, `path`, or None. In `None`
                     mode, the files argument is expected to be a list of
                     dictionaries in the output format of the
                     `kraken.lib.xml.parse_{alto,page,xml}` functions.
        num_workers: Number of workers for parallelized extraction of line
                     images. Set to `0` to disable parallelism.
        ignore_splits: Switch to disable serialization of the explicit
                       train/validation/test splits contained in the source
                       files.
        random_split: Serializes a random split into the dataset with the
                       proportions (train, val, test).
        force_type: Forces a dataset type. Can be `kraken_recognition_baseline`
                    or `kraken_recognition_bbox`.
        recordbatch_size: Minimum number of records per RecordBatch written to
                          the output file. Larger batches require more
                          transient memory but slightly improve reading
                          performance.
        skip_empty_lines: Do not compile empty text lines into the dataset.
        callback: Function called every time a new recordbatch is flushed into
                  the Arrow IPC file.
    """

    logger.info('Parsing XML files')
    extract_fn = partial(_extract_line, skip_empty_lines=skip_empty_lines)
    parse_fn = None
    if format_type == 'xml':
        parse_fn = parse_xml
    elif format_type == 'alto':
        parse_fn = parse_alto
    elif format_type == 'page':
        parse_fn = parse_page
    elif format_type == 'path':
        if not ignore_splits:
            logger.warning('ignore_splits is False and format_type is path. Will not serialize splits.')
        parse_fn = partial(parse_path, skip_empty_lines=skip_empty_lines)
        extract_fn = partial(_extract_path_line, skip_empty_lines=skip_empty_lines)
    elif format_type is None:
        pass
    else:
        raise ValueError(f'invalid format {format_type} for parse_(xml,alto,page,path)')

    if force_type and force_type not in ['kraken_recognition_baseline', 'kraken_recognition_bbox']:
        raise ValueError(f'force_type set to invalid value {force_type}')

    docs = []
    if parse_fn:
        for doc in files:
            try:
                data = parse_fn(doc)
            except KrakenInputException:
                logger.warning(f'Invalid input file {doc}')
                continue
            try:
                name_ext = str(data['image']).split(extsep, 1)
                if name_ext[1] == 'gt.txt':
                    data['image'] = name_ext[0] + '.png'
                with open(data['image'], 'rb') as fp:
                    Image.open(fp)
            except (FileNotFoundError, UnidentifiedImageError) as e:
                logger.warning(f'Could not open file {e.filename} in {doc}')
                continue
            docs.append(data)
        logger.info(f'Parsed {len(docs)} files.')
    else:
        docs = files.copy()
        logger.info(f'Got {len(docs)} preparsed files.')

    logger.info('Assembling dataset alphabet.')
    alphabet = Counter()
    num_lines = 0
    for doc in docs:
        for line in doc['lines']:
            num_lines += 1
            alphabet.update(line['text'])

    callback(0, num_lines)

    for k, v in sorted(alphabet.items(), key=lambda x: x[1], reverse=True):
        char = make_printable(k)
        if char == k:
            char = '\t' + char
        logger.info(f'{char}\t{v}')

    if force_type:
        ds_type = force_type
    else:
        ds_type = 'kraken_recognition_baseline' if format_type != 'path' else 'kraken_recognition_bbox'

    metadata = {'lines': {'type': ds_type,
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
        if random_split:
            indices = np.random.choice(4, len(line_cache), p=(0.0,) + random_split)
        else:
            indices = np.zeros(len(line_cache))
        tr_ind = np.zeros(len(line_cache), dtype=bool)
        tr_ind[indices == 1] = True
        val_ind = np.zeros(len(line_cache), dtype=bool)
        val_ind[indices == 2] = True
        test_ind = np.zeros(len(line_cache), dtype=bool)
        test_ind[indices == 3] = True

        train_mask = pa.array(tr_ind)
        val_mask = pa.array(val_ind)
        test_mask = pa.array(test_ind)
        rbatch = pa.RecordBatch.from_arrays([ar, train_mask, val_mask, test_mask], schema=schema)
        return rbatch, (len(line_cache), int(sum(indices == 1)), int(sum(indices == 2)), int(sum(indices == 3)))

    line_cache = []
    logger.info('Writing lines to temporary file.')
    with tempfile.TemporaryDirectory() as tmp_output_dir:
        tmp_file = tmp_output_dir + '/dataset.arrow'
        with pa.OSFile(tmp_file, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:

                if num_workers and num_workers > 1:
                    logger.info(f'Spinning up processing pool with {num_workers} workers.')
                    with Pool(num_workers) as pool:
                        for page_lines, im_mode in pool.imap_unordered(extract_fn, docs):
                            if page_lines:
                                line_cache.extend(page_lines)
                                # comparison RGB(A) > L > 1
                                if im_mode > metadata['lines']['im_mode']:
                                    metadata['lines']['im_mode'] = im_mode

                            if len(line_cache) >= recordbatch_size:
                                logger.info(f'Flushing {len(line_cache)} lines into {tmp_file}.')
                                rbatch, counts = _make_record_batch(line_cache)
                                metadata['lines']['counts'].update({'all': counts[0],
                                                                    'train': counts[1],
                                                                    'validation': counts[2],
                                                                    'test': counts[3]})
                                writer.write(rbatch)
                                callback(len(line_cache), num_lines)
                                line_cache = []
                else:
                    for page_lines, im_mode in map(extract_fn, docs):
                        if page_lines:
                            line_cache.extend(page_lines)
                            # comparison RGB(A) > L > 1
                            if im_mode > metadata['lines']['im_mode']:
                                metadata['lines']['im_mode'] = im_mode

                        if len(line_cache) >= recordbatch_size:
                            logger.info(f'Flushing {len(line_cache)} lines into {tmp_file}.')
                            rbatch, counts = _make_record_batch(line_cache)
                            metadata['lines']['counts'].update({'all': counts[0],
                                                                'train': counts[1],
                                                                'validation': counts[2],
                                                                'test': counts[3]})
                            writer.write(rbatch)
                            callback(len(line_cache), num_lines)
                            line_cache = []

                if line_cache:
                    logger.info(f'Flushing last {len(line_cache)} lines into {tmp_file}.')
                    rbatch, counts = _make_record_batch(line_cache)
                    metadata['lines']['counts'].update({'all': counts[0],
                                                        'train': counts[1],
                                                        'validation': counts[2],
                                                        'test': counts[3]})
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
