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
kraken.rpred
~~~~~~~~~~~~

Generators for recognition on lines images.
"""
import logging
import bidi.algorithm as bd

from PIL import Image
from collections import defaultdict
from typing import List, Tuple, Optional, Generator, Union, Dict

from kraken.lib.util import get_im_str, is_bitonal
from kraken.lib.models import TorchSeqRecognizer
from kraken.lib.segmentation import extract_polygons, compute_polygon_section
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.dataset import generate_input_transforms


__all__ = ['ocr_record', 'bidi_record', 'mm_rpred', 'rpred']

logger = logging.getLogger(__name__)


class ocr_record(object):
    """
    A record object containing the recognition result of a single line
    """
    def __init__(self, prediction: str, cuts, confidences: List[float], line: Union[List, Dict[str, List]]) -> None:
        self.prediction = prediction
        self.cuts = cuts
        self.confidences = confidences
        self.script = None if 'script' not in line else line['script']
        self.type = 'baselines' if 'baseline' in line else 'box'
        if self.type == 'baselines':
            self.line = line['boundary']
            self.baseline = line['baseline']
        else:
            self.line = line

    def __len__(self) -> int:
        return len(self.prediction)

    def __str__(self) -> str:
        return self.prediction

    def __iter__(self):
        self.idx = -1
        return self

    def __next__(self) -> Tuple[str, int, float]:
        if self.idx + 1 < len(self):
            self.idx += 1
            return (self.prediction[self.idx], self.cuts[self.idx],
                    self.confidences[self.idx])
        else:
            raise StopIteration

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key >= len(self):
                raise IndexError('Index (%d) is out of range' % key)
            return (self.prediction[key], self.cuts[key],
                    self.confidences[key])
        else:
            raise TypeError('Invalid argument type')


def bidi_record(record: ocr_record) -> ocr_record:
    """
    Reorders a record using the Unicode BiDi algorithm.

    Models trained for RTL or mixed scripts still emit classes in LTR order
    requiring reordering for proper display.

    Args:
        record (kraken.rpred.ocr_record)

    Returns:
        kraken.rpred.ocr_record
    """
    storage = bd.get_empty_storage()
    base_level = bd.get_base_level(record.prediction)
    storage['base_level'] = base_level
    storage['base_dir'] = ('L', 'R')[base_level]

    bd.get_embedding_levels(record.prediction, storage)
    bd.explicit_embed_and_overrides(storage)
    bd.resolve_weak_types(storage)
    bd.resolve_neutral_types(storage, False)
    bd.resolve_implicit_levels(storage, False)
    for i, j in enumerate(record):
        storage['chars'][i]['record'] = j
    bd.reorder_resolved_levels(storage, False)
    bd.apply_mirroring(storage, False)
    prediction = ''
    cuts = []
    confidences = []
    for ch in storage['chars']:
        # code point may have been mirrored
        prediction = prediction + ch['ch']
        cuts.append(ch['record'][1])
        confidences.append(ch['record'][2])
    # carry over whole line information
    if record.type == 'baselines':
        line = {'boundary': record.line, 'baseline': record.baseline}
    else:
        line = record.line
    rec = ocr_record(prediction, cuts, confidences, line)
    rec.script = record.script
    return rec


class mm_rpred(object):
    """
    Multi-model version of kraken.rpred.rpred
    """
    def __init__(self,
                 nets: Dict[str, TorchSeqRecognizer],
                 im: Image.Image,
                 bounds: dict,
                 pad: int = 16,
                 bidi_reordering: bool = True,
                 script_ignore: Optional[List[str]] = None) -> Generator[ocr_record, None, None]:
        """
        Multi-model version of kraken.rpred.rpred.

        Takes a dictionary of ISO15924 script identifiers->models and an
        script-annotated segmentation to dynamically select appropriate models for
        these lines.

        Args:
            nets (dict): A dict mapping ISO15924 identifiers to TorchSegRecognizer
                         objects. Recommended to be an defaultdict.
            im (PIL.Image.Image): Image to extract text from
            bounds (dict): A dictionary containing a 'boxes' entry
                            with a list of lists of coordinates (script, (x0, y0,
                            x1, y1)) of a text line in the image and an entry
                            'text_direction' containing
                            'horizontal-lr/rl/vertical-lr/rl'.
            pad (int): Extra blank padding to the left and right of text line
            bidi_reordering (bool): Reorder classes in the ocr_record according to
                                    the Unicode bidirectional algorithm for correct
                                    display.
            script_ignore (list): List of scripts to ignore during recognition
        Yields:
            An ocr_record containing the recognized text, absolute character
            positions, and confidence values for each character.

        Raises:
            KrakenInputException if the mapping between segmentation scripts and
            networks is incomplete.
        """
        seg_types = set(recognizer.seg_type for recognizer in nets.values())
        if ('type' in bounds and bounds['type'] not in seg_types) or len(seg_types) > 1:
            logger.warning('Recognizers with segmentation types {} will be '
                           'applied to segmentation of type {}. This will likely result '
                           'in severely degraded performace'.format(seg_types,
                            bounds['type'] if 'type' in bounds else None))
        one_channel_modes = set(recognizer.nn.one_channel_mode for recognizer in nets.values())
        if '1' in one_channel_modes and len(one_channel_modes) > 1:
            raise KrakenInputException('Mixing binary and non-binary recognition models is not supported.')
        elif '1' in one_channel_modes and not is_bitonal(im):
            logger.warning('Running binary models on non-binary input image '
                           '(mode {}). This will result in severely degraded '
                           'performance'.format(im.mode))
        if 'type' in bounds and bounds['type'] == 'baselines':
            valid_norm = False
            self.len = len(bounds['lines'])
            self.seg_key = 'lines'
            self.next_iter = self._recognize_baseline_line
            self.line_iter = iter(bounds['lines'])
            scripts = [x['script'] for x in bounds['lines']]
        else:
            valid_norm = True
            self.len = len(bounds['boxes'])
            self.seg_key = 'boxes'
            self.next_iter = self._recognize_box_line
            self.line_iter = iter(bounds['boxes'])
            scripts = [x[0] for line in bounds['boxes'] for x in line]

        im_str = get_im_str(im)
        logger.info('Running {} multi-script recognizers on {} with {} lines'.format(len(nets), im_str, self.len))

        miss = [script for script in scripts if not nets.get(script)]
        if miss and not isinstance(nets, defaultdict):
            raise KrakenInputException('Missing models for scripts {}'.format(set(miss)))

        # build dictionary for line preprocessing
        self.ts = {}
        for script in scripts:
            logger.debug('Loading line transforms for {}'.format(script))
            network = nets[script]
            batch, channels, height, width = network.nn.input
            self.ts[script] = generate_input_transforms(batch, height, width, channels, pad, valid_norm)

        self.im = im
        self.nets = nets
        self.bidi_reordering = bidi_reordering
        self.pad = pad
        self.bounds = bounds
        self.script_ignore = script_ignore

    def _recognize_box_line(self, line):
        flat_box = [point for box in line['boxes'][0] for point in box[1]]
        xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
        ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
        rec = ocr_record('', [], [], [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
        for script, (box, coords) in zip(map(lambda x: x[0], line['boxes'][0]),
                                         extract_polygons(self.im, {'text_direction': line['text_direction'],
                                                                    'boxes': map(lambda x: x[1], line['boxes'][0])})):
            # skip if script is set to ignore
            if self.script_ignore is not None and script in self.script_ignore:
                logger.info('Ignoring {} line segment.'.format(script))
                continue
            # check if boxes are non-zero in any dimension
            if 0 in box.size:
                logger.warning('bbox {} with zero dimension. Emitting empty record.'.format(coords))
                continue
            # try conversion into tensor
            try:
                logger.debug('Preparing run.')
                line = self.ts[script](box)
            except Exception:
                logger.warning('Conversion of line {} failed. Skipping.'.format(coords))
                continue

            # check if line is non-zero
            if line.max() == line.min():
                logger.warning('Empty run. Skipping.')
                continue

            logger.debug('Forward pass with model {}'.format(script))
            preds = self.nets[script].predict(line.unsqueeze(0))[0]

            # calculate recognized LSTM locations of characters
            logger.debug('Convert to absolute coordinates')
            # calculate recognized LSTM locations of characters
            # scale between network output and network input
            net_scale = line.shape[2]/self.nets[script].outputs.shape[2]
            # scale between network input and original line
            in_scale = box.size[0]/(line.shape[2]-2*self.pad)

            def _scale_val(val, min_val, max_val):
                return int(round(min(max(((val*net_scale)-self.pad)*in_scale, min_val), max_val)))

            pred = ''.join(x[0] for x in preds)
            pos = []
            conf = []

            for _, start, end, c in preds:
                if self.bounds['text_direction'].startswith('horizontal'):
                    xmin = coords[0] + _scale_val(start, 0, box.size[0])
                    xmax = coords[0] + _scale_val(end, 0, box.size[0])
                    pos.append([[xmin, coords[1]], [xmin, coords[3]], [xmax, coords[3]], [xmax, coords[1]]])
                else:
                    ymin = coords[1] + _scale_val(start, 0, box.size[1])
                    ymax = coords[1] + _scale_val(start, 0, box.size[1])
                    pos.append([[coords[0], ymin], [coords[2], ymin], [coords[2], ymax], [coords[0], ymax]])
                conf.append(c)
            rec.prediction += pred
            rec.cuts.extend(pos)
            rec.confidences.extend(conf)
        if self.bidi_reordering:
            logger.debug('BiDi reordering record.')
            return bidi_record(rec)
        else:
            logger.debug('Emitting raw record')
            return rec

    def _recognize_baseline_line(self, line):
        try:
            box, coords = next(extract_polygons(self.im, line))
        except KrakenInputException as e:
            logger.warning(f'Extracting line failed: {e}')
            return ocr_record('', [], [], line['lines'][0])

        script = coords['script']
        # check if boxes are non-zero in any dimension
        if 0 in box.size:
            logger.warning('bbox {} with zero dimension. Emitting empty record.'.format(coords))
            return ocr_record('', [], [], coords)
        # try conversion into tensor
        try:
            line = self.ts[script](box)
        except Exception:
            return ocr_record('', [], [], coords)
        # check if line is non-zero
        if line.max() == line.min():
            return ocr_record('', [], [], coords)

        preds = self.nets[script].predict(line.unsqueeze(0))[0]
        # calculate recognized LSTM locations of characters
        # scale between network output and network input
        net_scale = line.shape[2]/self.nets[script].outputs.shape[2]
        # scale between network input and original line
        in_scale = box.size[0]/(line.shape[2]-2*self.pad)

        def _scale_val(val, min_val, max_val):
            return int(round(min(max(((val*net_scale)-self.pad)*in_scale, min_val), max_val-1)))

        # XXX: fix bounding box calculation ocr_record for multi-codepoint labels.
        pred = ''.join(x[0] for x in preds)
        pos = []
        conf = []
        for _, start, end, c in preds:
            pos.append(compute_polygon_section(coords['baseline'],
                                               coords['boundary'],
                                               _scale_val(start, 0, box.size[0]),
                                               _scale_val(end, 0, box.size[0])))
            conf.append(c)
        if self.bidi_reordering:
            logger.debug('BiDi reordering record.')
            rec = bidi_record(ocr_record(pred, pos, conf, coords))
            return rec
        else:
            logger.debug('Emitting raw record')
            return ocr_record(pred, pos, conf, coords)

    def __next__(self):
        bound = self.bounds
        bound[self.seg_key] = [next(self.line_iter)]
        o = self.next_iter(bound)
        return o

    def __iter__(self):
        return self

    def __len__(self):
        return self.len


def rpred(network: TorchSeqRecognizer,
          im: Image.Image,
          bounds: dict,
          pad: int = 16,
          bidi_reordering: bool = True) -> Generator[ocr_record, None, None]:
    """
    Uses a TorchSeqRecognizer and a segmentation to recognize text

    Args:
        network (kraken.lib.models.TorchSeqRecognizer): A TorchSegRecognizer
                                                        object
        im (PIL.Image.Image): Image to extract text from
        bounds (dict): A dictionary containing a 'boxes' entry with a list of
                       coordinates (x0, y0, x1, y1) of a text line in the image
                       and an entry 'text_direction' containing
                       'horizontal-lr/rl/vertical-lr/rl'.
        pad (int): Extra blank padding to the left and right of text line.
                   Auto-disabled when expected network inputs are incompatible
                   with padding.
        bidi_reordering (bool): Reorder classes in the ocr_record according to
                                the Unicode bidirectional algorithm for correct
                                display.
    Yields:
        An ocr_record containing the recognized text, absolute character
        positions, and confidence values for each character.
    """
    if 'boxes' in bounds:
        boxes = bounds['boxes']
        rewrite_boxes = []
        for box in boxes:
            rewrite_boxes.append([('default', box)])
        bounds['boxes'] = rewrite_boxes
        bounds['script_detection'] = True
    return mm_rpred({'default': network}, im, bounds, pad, bidi_reordering)
