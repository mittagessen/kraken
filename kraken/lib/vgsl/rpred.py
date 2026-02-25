#
# Copyright 2025 Benjamin Kiessling
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
VGSL recognition inference subclass
"""
import torch
import logging
import warnings
import dataclasses
import torch.nn.functional as F
from collections.abc import Generator
from typing import Optional, TYPE_CHECKING
from functools import partial

from kraken.containers import ocr_record, BBoxOCRRecord, BaselineOCRRecord
from kraken.lib.dataset import ImageInputTransforms
from kraken.lib.segmentation import extract_polygons

if TYPE_CHECKING:
    from PIL import Image
    from kraken.containers import Segmentation


__all__ = ['VGSLRecognitionInference']

logger = logging.getLogger(__name__)


def _extract_line(im, segmentation, line_idx, legacy: bool = False):
    line = segmentation.lines[line_idx]
    seg = dataclasses.replace(segmentation, lines=[line])
    try:
        im, _ = next(extract_polygons(im, seg, legacy=legacy))
        return im, line_idx
    except ValueError:
        return None, line_idx


class VGSLRecognitionInference:
    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def _recognition_pred(self,
                          im: 'Image.Image',
                          segmentation: 'Segmentation') -> Generator[ocr_record, None, None]:
        """
        Recognition inference.
        """
        self._len = len(segmentation.lines)
        # map of recognition results
        rec_results: list[Optional[ocr_record]] = [None] * self._len
        input_queue = []
        next_idx_to_emit = 0

        if segmentation.type == 'baselines':
            valid_norm = False
            self._line_iter = self._recognize_baseline_lines
            _empty_record_cls = BaselineOCRRecord
        else:
            valid_norm = True
            self._line_iter = self._recognize_box_lines
            _empty_record_cls = BBoxOCRRecord

        batch, channels, height, width = self.input
        transforms = ImageInputTransforms(batch,
                                          height,
                                          width,
                                          channels,
                                          (self._inf_config.padding, 0),
                                          valid_norm,
                                          dtype=self._m_dtype)

        if self.use_legacy_polygons and segmentation.type == 'baselines':
            if self._inf_config.no_legacy_polygons:
                warnings.warn('Enforcing use of the new polygon extractor for '
                              'models trained with old version. Accuracy may be '
                              'affected.')
                legacy = False
            else:
                warnings.warn('Using legacy polygon extractor, as the model was '
                              'not trained with the new method. Please retrain '
                              'your model to get speed improvement.')
                legacy = True
        else:
            legacy = False

        _exl = partial(_extract_line, im, segmentation, legacy=legacy)

        with self._fabric.init_tensor():
            for im, line_idx in self._line_extraction_pool.imap_unordered(_exl, range(self._len)):
                if im is None or 0 in im.size:
                    rec_results[line_idx] = _empty_record_cls('', [], [], segmentation.lines[line_idx])
                else:
                    try:
                        ts_im = transforms(im)
                    except Exception:
                        rec_results[line_idx] = _empty_record_cls('', [], [], segmentation.lines[line_idx])
                    else:
                        if ts_im.max() == ts_im.min():
                            rec_results[line_idx] = _empty_record_cls('', [], [], segmentation.lines[line_idx])
                        else:
                            input_queue.append((ts_im, im, line_idx))
                            # feed the input queue through the network
                            if len(input_queue) == self._inf_config.batch_size or len(input_queue) == rec_results.count(None):
                                for rec, idx in self._line_iter(input_queue, segmentation):
                                    logger.debug(f'Inserting batch result at index {idx}: {rec}')
                                    rec_results[idx] = rec
                                input_queue.clear()
                while next_idx_to_emit < self._len and rec_results[next_idx_to_emit] is not None:
                    yield rec_results[next_idx_to_emit]
                    next_idx_to_emit += 1

    def _recognize_box_lines(self,
                             lines: list[tuple['torch.Tensor', 'Image.Image', int]],
                             segmentation: 'Segmentation') -> Generator[tuple[BBoxOCRRecord, int], None, None]:
        max_len = max([seq.shape[2] for seq, *_ in lines])
        seqs = torch.stack([F.pad(seq, pad=(0, max_len - seq.shape[2])) for seq, *_ in lines])
        seq_lens = torch.LongTensor([seq.shape[2] for seq, *_ in lines])

        preds, olens = self._rec_predict(seqs, seq_lens)

        for idx, (pred, olen) in enumerate(zip(preds, olens)):
            # calculate recognized LSTM locations of characters
            # scale between network output and network input
            self.net_scale = lines[idx][0].shape[2] / olen.item()
            # scale between network input and original line
            self.in_scale = lines[idx][1].width / (lines[idx][0].shape[2] - 2 * self._inf_config.padding)

            # XXX: fix bounding box calculation ocr_record for multi-codepoint labels.
            pred_str = ''.join(x[0] for x in pred)
            pos = []
            conf = []
            for _, start, end, c in pred:
                if segmentation.text_direction.startswith('horizontal'):
                    x, ymin, _, ymax = segmentation.lines[lines[idx][2]].bbox
                    xmin = x + self._scale_val(start, 0, lines[idx][1].width)
                    xmax = x + self._scale_val(end, 0, lines[idx][1].width)
                    pos.append([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
                else:
                    xmin, y, xmax, _ = segmentation.lines[lines[idx][2]].bbox
                    ymin = y + self._scale_val(start, 0, lines[idx][1].height)
                    ymax = y + self._scale_val(end, 0, lines[idx][1].height)
                    pos.append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
                conf.append(c)
            rec = BBoxOCRRecord(pred_str,
                                pos,
                                conf,
                                segmentation.lines[lines[idx][2]],
                                logits=pred if self._inf_config.return_logits else None,
                                image=lines[idx][1] if self._inf_config.return_line_image else None)
            if self._inf_config.bidi_reordering:
                logger.debug('BiDi reordering record.')
                yield rec.logical_order(base_dir=self._inf_config.bidi_reordering if self._inf_config.bidi_reordering in ('L', 'R') else None), lines[idx][2]
            else:
                logger.debug('Emitting raw record')
                yield rec.display_order(None), lines[idx][2]

    def _recognize_baseline_lines(self,
                                  lines: list[tuple['torch.Tensor', 'Image.Image', int]],
                                  segmentation: 'Segmentation') -> Generator[tuple[BaselineOCRRecord, int], None, None]:
        max_len = max([seq.shape[2] for seq, *_ in lines])
        seqs = torch.stack([F.pad(seq, pad=(0, max_len - seq.shape[2])) for seq, *_ in lines])
        seq_lens = torch.LongTensor([seq.shape[2] for seq, *_ in lines])

        preds, olens = self._rec_predict(seqs, seq_lens)

        for idx, (pred, olen) in enumerate(zip(preds, olens)):
            # calculate recognized LSTM locations of characters scale between
            # network output and network input. It should stay fixed, as the
            # reduction factor is constant, but non-divisibility can cause
            # slight differences between lines.
            self.net_scale = lines[idx][0].shape[2] / olen.item()
            # scale between network input and original line
            self.in_scale = lines[idx][1].width / (lines[idx][0].shape[2] - 2 * self._inf_config.padding)
            # XXX: fix bounding box calculation ocr_record for multi-codepoint labels.
            pred_str = ''.join(x[0] for x in pred)
            pos = []
            conf = []
            for _, start, end, c in pred:
                pos.append([self._scale_val(start, 0, lines[idx][1].width),
                            self._scale_val(end, 0, lines[idx][1].width)])
                conf.append(c)
            rec = BaselineOCRRecord(pred_str,
                                    pos,
                                    conf,
                                    segmentation.lines[lines[idx][2]],
                                    logits=self.outputs[idx, ..., :olen].clone() if self._inf_config.return_logits else None,
                                    image=lines[idx][1] if self._inf_config.return_line_image else None)

            if self._inf_config.bidi_reordering:
                logger.debug('BiDi reordering record.')
                yield rec.logical_order(base_dir=self._inf_config.bidi_reordering if self._inf_config.bidi_reordering in ('L', 'R') else None), lines[idx][2]
            else:
                logger.debug('Emitting raw record')
                yield rec.display_order(None), lines[idx][2]

    def _rec_predict(self,
                     line: torch.Tensor,
                     lens: Optional[torch.Tensor] = None) -> list[list[tuple[str, int, int, float]]]:
        """
        Performs a forward pass on a torch tensor of a line with shape (N, C, H, W)
        and returns the decoding as a list of tuples (string, start, end,
        confidence).

        Args:
            line: NCHW line tensor
            lens: Optional tensor containing sequence lengths if N > 1

        Returns:
            List of decoded sequences.
        """
        logits, olens = self.nn(line, lens)
        probs = (logits / self._inf_config.temperature).softmax(1)
        self.outputs = probs.detach().squeeze(2)
        dec_seqs = [self.codec.decode(locs) for locs in self._inf_config.decoder(self.outputs, olens)]
        return dec_seqs, olens

    def _scale_val(self, val, min_val, max_val):
        return int(round(min(max(((val * self.net_scale) - self._inf_config.padding) * self.in_scale, min_val), max_val - 1)))
