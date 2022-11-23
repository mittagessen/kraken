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
accuractely determine grapheme locations in input data. Requires OpenFST and
either the official python bindings or pywrapfst.
"""
import logging
import numpy as np

from PIL import Image
from bidi.algorithm import get_display

from typing import List, Dict, Any

from kraken import rpred
from kraken.lib.codec import PytorchCodec
from kraken.lib.models import TorchSeqRecognizer
from kraken.lib.exceptions import KrakenInputException, KrakenEncodeException
from kraken.lib.segmentation import compute_polygon_section

logger = logging.getLogger('kraken')

try:
    import pywrapfst as fst

    _get_fst = lambda: fst.VectorFst() # NOQA
    _get_best_weight = lambda x: fst.Weight.one(x.weight_type()) # NOQA
except ImportError:
    logger.info('pywrapfst not available. Falling back to openfst_python.')
    try:
        import openfst_python as fst
        _get_fst = lambda: fst.Fst() # NOQA
        _get_best_weight = lambda x: 0 # NOQA
    except ImportError:
        logger.error('Neither pywrapfst nor openfst_python bindings available.')
        raise


def _get_arc(input_label, output_label, weight, state):
    return fst.Arc(input_label, output_label, weight, state)


def _compose(first_graph, second_graph):
    return fst.compose(first_graph, second_graph)


def _shortest_path(graph):
    return fst.shortestpath(graph)


def fst_from_lattice(outputs: np.ndarray):
    """
    Generates an FST from the network output tensor.

    Arguments:
        outputs: Output tensor of shape (batch, labels, length).

    Returns:
        An OpenFST FST object.
    """
    f = _get_fst()
    state = f.add_state()
    f.set_start(state)
    for idx, frame in enumerate(np.transpose(outputs[0], [1, 0])):
        next_state = f.add_state()
        for label, conf in enumerate(frame):
            f.add_arc(state, _get_arc(idx, label+1, -np.log(conf), next_state))
        state = next_state

    f.set_final(state)

    if not f.verify():
        raise KrakenInputException('Invalid FST')

    return f


def fst_from_text(line: str, codec: PytorchCodec):
    """
    Creates an FST of a text line in label space.

    In order to match the Kraken output, we take into account the fact that
    Kraken will output a large number of frames and therefore each character
    of our line must be able to be detected between several non-characters "#".

    In the same way, Kraken can predict several times the same character on
    neighboring frames, if this is the case it doesn't mean that there is
    several times the same letter but only that it appears on several frames.

    To handle this particular case we use Ɛ:Ɛ transitions between each character
    of our line and we add a Ɛ:mychar loop to eliminate identical neighboring
    characters.

    To detect the word "loop", we'll have something like:

       Ɛ:Ɛ    l:l    Ɛ:Ɛ    o:o    Ɛ:Ɛ    o:o    Ɛ:Ɛ    p:p
    [0]───>(1)───>(2)───>(3)───>(4)───>(5)───>(6)───>(7)───>((8))
     ↺      ↺      ↺      ↺      ↺      ↺      ↺      ↺       ↺
    #:Ɛ    l:Ɛ    #:Ɛ    o:Ɛ    #:Ɛ    o:Ɛ    #:Ɛ    p:Ɛ     #:Ɛ
    """
    f = _get_fst()
    one = _get_best_weight(f)
    state = f.add_state()
    f.set_start(state)

    codec.strict = True

    def _detect_char(graph, one, start_state, char, codec):
        # Empty char loop
        graph.add_arc(start_state, _get_arc(1, 0, one, start_state))
        epsilon_state = graph.add_state()
        # Epsilon transition
        graph.add_arc(start_state, _get_arc(0, 0, one, epsilon_state))
        # Adding one to expand neural network codec with epsilon at pos 0
        # We will end with the empty char # at pos 1 (see: kraken.lib.codec.py)
        try:
            encoded_char = codec.encode(char)[0] + 1
        except KrakenEncodeException:
            encoded_char = 0
            logger.warning(f'Error while encoding code point {char}. Skipping.')
        # Espilon loop
        graph.add_arc(epsilon_state, _get_arc(encoded_char, 0, one, epsilon_state))
        end_state = graph.add_state()
        # Char transition
        graph.add_arc(epsilon_state, _get_arc(encoded_char, encoded_char, one, end_state))
        return end_state

    for char in line:
        state = _detect_char(f, one, state, char, codec)

    f.add_arc(state, _get_arc(1, 0, one, state))
    f.set_final(state)

    if not f.verify():
        raise KrakenInputException('Invalid FST')

    return f


def _generate_line_record(graph, final_step):
    next_state = graph.start()
    labels = []
    activations = [0]

    while next_state:
        state = next_state
        next_state = None
        for arc in graph.arcs(state):
            next_state = arc.nextstate
            olabel = arc.olabel - 1
            if olabel == -1:
                continue
            labels.append(olabel)
            activations.append(arc.ilabel)
    activations.append(final_step)
    # spread activations to halfway point between activations
    bounds = []
    for prev_ac, cur_ac, next_ac in zip(activations[:-2], activations[1:-1], activations[2:]):
        bounds.append((prev_ac + (cur_ac-prev_ac)//2, cur_ac + (next_ac-cur_ac)//2))
    return zip(bounds, labels)


def forced_align(doc: Dict[str, Any], model: TorchSeqRecognizer) -> List[rpred.ocr_record]:
    """
    Performs a forced character alignment of text with recognition model
    output activations.

    Argument:
        doc: Parsed document.
        model: Recognition model to use for alignment.

    Returns:
        A list of kraken.rpred.ocr_record.
    """
    im = Image.open(doc['image'])
    predictor = rpred.rpred(model, im, doc)
    records = []
    for idx, line in enumerate(doc['lines']):
        bidi_text = get_display(line['text'])
        gt_fst = fst_from_text(bidi_text, model.codec)
        next(predictor)
        model.codec.strict = False
        if model.outputs.shape[2] < 2*len(model.codec.encode(bidi_text)):
            logger.warning(f'Could not align line {idx}. Output sequence length {model.outputs.shape[2]} < '
                           f'{2*len(model.codec.encode(bidi_text))} (length of "{line["text"]}" after encoding).')
            records.append(rpred.ocr_record('', [], [], line))
            model.codec.strict = True
            continue
        model.codec.strict = True

        lat_fst = fst_from_lattice(model.outputs)
        composed_graph = _compose(lat_fst, gt_fst)
        short_graph = _shortest_path(composed_graph)
        pred = []
        pos = []
        conf = []
        for act, label in _generate_line_record(short_graph, model.outputs.shape[2]-1):
            pos.append(compute_polygon_section(line['baseline'],
                                               line['boundary'],
                                               predictor._scale_val(act[0], 0, predictor.box.size[0]),
                                               predictor._scale_val(act[1], 0, predictor.box.size[0])))
            conf.append(1.0)
            pred.append(model.codec.decode([(label, 0, 0, 0)])[0][0])
        records.append(rpred.bidi_record(rpred.ocr_record(pred, pos, conf, line)))
    return records
