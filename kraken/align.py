#
# Copyright 2021 Teklia
# Copyright 2021 Benjamin Kiessling
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
align
~~~~~

A character alignment module using a network output lattice and ground truth to
accuractely determine grapheme locations in input data.
"""
import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import torch
from bidi.algorithm import get_display
from PIL import Image

from kraken import rpred
from kraken.containers import BaselineOCRRecord, Segmentation

if TYPE_CHECKING:
    from kraken.lib.models import TorchSeqRecognizer

logger = logging.getLogger('kraken')


def forced_align(doc: Segmentation, model: 'TorchSeqRecognizer', base_dir: Optional[Literal['L', 'R']] = None) -> Segmentation:
    """
    Performs a forced character alignment of text with recognition model
    output activations.

    Argument:
        doc: Parsed document.
        model: Recognition model to use for alignment.

    Returns:
        A Segmentation object where the record's contain the aligned text.
    """
    im = Image.open(doc.imagename)
    predictor = rpred.rpred(model, im, doc)

    records = []

    # enable training mode in last layer to get log_softmax output
    model.nn.nn[-1].training = True

    for idx, line in enumerate(doc.lines):
        # convert text to display order
        do_text = get_display(line.text, base_dir=base_dir)
        # encode into labels, ignoring unencodable sequences
        labels = model.codec.encode(do_text).long()
        next(predictor)
        if model.outputs.shape[2] < 2*len(labels):
            logger.warning(f'Could not align line {idx}. Output sequence length {model.outputs.shape[2]} < '
                           f'{2*len(labels)} (length of "{line.text}" after encoding).')
            records.append(BaselineOCRRecord('', [], [], line))
            continue
        emission = torch.tensor(model.outputs).squeeze().T
        trellis = get_trellis(emission, labels)
        path = backtrack(trellis, emission, labels)
        path = merge_repeats(path, do_text)
        pred = []
        pos = []
        conf = []
        for seg in path:
            pred.append(seg.label)
            pos.append((predictor._scale_val(seg.start, 0, predictor.box.size[0]),
                        predictor._scale_val(seg.end, 0, predictor.box.size[0])))
            conf.append(seg.score)
        records.append(BaselineOCRRecord(pred, pos, conf, line, display_order=True))
    return dataclasses.replace(doc, lines=records)


"""
Copied from the forced alignment with Wav2Vec2 tutorial of pytorch available
at:
https://github.com/pytorch/audio/blob/main/examples/tutorials/forced_alignment_tutorial.py
"""


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def get_trellis(emission, tokens):
    # width x labels in log domain
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra dimensions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, 0],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, 0]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


def merge_repeats(path, ground_truth):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                ground_truth[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments
