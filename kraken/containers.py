#
# Copyright 2023 Benjamin Kiessling
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
kraken.containers
~~~~~~~~~~~~~~~~~

Container classes replacing the old dictionaries returned by kraken's
functional blocks.
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple,
                    Union)

import bidi.algorithm as bd
import numpy as np

from kraken.lib.segmentation import compute_polygon_section

if TYPE_CHECKING:
    from os import PathLike


__all__ = ['BaselineLine',
           'BBoxLine',
           'Segmentation',
           'Region',
           'ocr_record',
           'BaselineOCRRecord',
           'BBoxOCRRecord',
           'ProcessingStep']


@dataclass
class ProcessingStep:
    """
    A processing step in the recognition pipeline.

    Attributes:
        id: Unique identifier
        category: Category of processing step that has been performed.
        description: Natural-language description of the process.
        settings: Dict describing the parameters of the processing step.
    """
    id: str
    category: Literal['preprocessing', 'processing', 'postprocessing']
    description: str
    settings: Dict[str, Union[Dict, str, float, int, bool]]


@dataclass
class BaselineLine:
    """
    Baseline-type line record.

    A container class for a single line in baseline + bounding polygon format,
    optionally containing a transcription, tags, or associated regions.

    Attributes:
        id: Unique identifier
        baseline: List of tuples `(x_n, y_n)` defining the baseline.
        boundary: List of tuples `(x_n, y_n)` defining the bounding polygon of
                  the line. The first and last points should be identical.
        text: Transcription of this line.
        base_dir: An optional string defining the base direction (also called
                  paragraph direction) for the BiDi algorithm. Valid values are
                  'L' or 'R'. If None is given the default auto-resolution will
                  be used.
        imagename: Path to the image associated with the line.
        tags: A dict mapping types to values.
        split: Defines whether this line is in the `train`, `validation`, or
               `test` set during training.
        regions: A list of identifiers of regions the line is associated with.
    """
    id: str
    baseline: List[Tuple[int, int]]
    boundary: List[Tuple[int, int]]
    text: Optional[str] = None
    base_dir: Optional[Literal['L', 'R']] = None
    type: str = 'baselines'
    imagename: Optional[Union[str, 'PathLike']] = None
    tags: Optional[Dict[str, str]] = None
    split: Optional[Literal['train', 'validation', 'test']] = None
    regions: Optional[List[str]] = None


@dataclass
class BBoxLine:
    """
    Bounding box-type line record.

    A container class for a single line in axis-aligned bounding box format,
    optionally containing a transcription, tags, or associated regions.

    Attributes:
        id: Unique identifier
        bbox: Tuple in form `(xmin, ymin, xmax, ymax)` defining
              the bounding box.
        text: Transcription of this line.
        base_dir: An optional string defining the base direction (also called
                  paragraph direction) for the BiDi algorithm. Valid values are
                  'L' or 'R'. If None is given the default auto-resolution will
                  be used.
        imagename: Path to the image associated with the line..
        tags: A dict mapping types to values.
        split: Defines whether this line is in the `train`, `validation`, or
               `test` set during training.
        regions: A list of identifiers of regions the line is associated with.
        text_direction: Sets the principal orientation (of the line) and
                        reading direction (of the document).
    """
    id: str
    bbox: Tuple[int, int, int, int]
    text: Optional[str] = None
    base_dir: Optional[Literal['L', 'R']] = None
    type: str = 'bbox'
    imagename: Optional[Union[str, 'PathLike']] = None
    tags: Optional[Dict[str, str]] = None
    split: Optional[Literal['train', 'validation', 'test']] = None
    regions: Optional[List[str]] = None
    text_direction: Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'] = 'horizontal-lr'


@dataclass
class Region:
    """
    Container class of a single polygonal region.

    Attributes:
        id: Unique identifier
        boundary: List of tuples `(x_n, y_n)` defining the bounding polygon of
                  the region. The first and last points should be identical.
        imagename: Path to the image associated with the region.
        tags: A dict mapping types to values.
    """
    id: str
    boundary: List[Tuple[int, int]]
    imagename: Optional[Union[str, 'PathLike']] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class Segmentation:
    """
    A container class for segmentation or recognition results.

    In order to allow easy JSON de-/serialization, nested classes for lines
    (BaselineLine/BBoxLine) and regions (Region) are reinstantiated from their
    dictionaries.

    Attributes:
        type: Field indicating if baselines
              (:class:`kraken.containers.BaselineLine`) or bbox
              (:class:`kraken.containers.BBoxLine`) line records are in the
              segmentation.
        imagename: Path to the image associated with the segmentation.
        text_direction: Sets the principal orientation (of the line), i.e.
                        horizontal/vertical, and reading direction (of the
                        document), i.e. lr/rl.
        script_detection: Flag indicating if the line records have tags.
        lines: List of line records. Records are expected to be in a valid
               reading order.
        regions: Dict mapping types to lists of regions.
        line_orders: List of alternative reading orders for the segmentation.
                     Each reading order is a list of line indices.
    """
    type: Literal['baselines', 'bbox']
    imagename: Union[str, 'PathLike']
    text_direction: Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl']
    script_detection: bool
    lines: Optional[List[Union[BaselineLine, BBoxLine]]] = None
    regions: Optional[Dict[str, List[Region]]] = None
    line_orders: Optional[List[List[int]]] = None

    def __post_init__(self):
        if not self.regions:
            self.regions = {}
        if not self.lines:
            self.lines = []
        if not self.line_orders:
            self.line_orders = []
        if len(self.lines) and not isinstance(self.lines[0], BBoxLine) and not isinstance(self.lines[0], BaselineLine):
            line_cls = BBoxLine if self.type == 'bbox' else BaselineLine
            self.lines = [line_cls(**line) for line in self.lines]
        if len(self.regions):
            for regs in self.regions.values():
                if regs and not isinstance(regs[0], Region):
                    regs = {}
                    for k, v in self.regions.items():
                        regs[k] = [Region(**reg) for reg in v]
                    self.regions = regs
                    break


class ocr_record(ABC):
    """
    A record object containing the recognition result of a single line
    """
    base_dir = None

    def __init__(self,
                 prediction: str,
                 cuts: List[Union[Tuple[int, int], Tuple[Tuple[int, int],
                                                         Tuple[int, int],
                                                         Tuple[int, int],
                                                         Tuple[int, int]]]],
                 confidences: List[float],
                 display_order: bool = True) -> None:
        self._prediction = prediction
        self._cuts = cuts
        self._confidences = confidences
        self._display_order = display_order

    @property
    @abstractmethod
    def type(self):
        pass

    def __len__(self) -> int:
        return len(self._prediction)

    def __str__(self) -> str:
        return self._prediction

    @property
    def prediction(self) -> str:
        return self._prediction

    @property
    def cuts(self) -> List:
        return self._cuts

    @property
    def confidences(self) -> List[float]:
        return self._confidences

    def __iter__(self):
        self.idx = -1
        return self

    @abstractmethod
    def __next__(self) -> Tuple[str,
                                Union[List[Tuple[int, int]],
                                      Tuple[Tuple[int, int],
                                            Tuple[int, int],
                                            Tuple[int, int],
                                            Tuple[int, int]]],
                                float]:
        pass

    @abstractmethod
    def __getitem__(self, key: Union[int, slice]):
        pass

    @abstractmethod
    def display_order(self, base_dir) -> 'ocr_record':
        pass

    @abstractmethod
    def logical_order(self, base_dir) -> 'ocr_record':
        pass


class BaselineOCRRecord(ocr_record, BaselineLine):
    """
    A record object containing the recognition result of a single line in
    baseline format.

    Attributes:
        type: 'baselines' to indicate a baseline record
        prediction: The text predicted by the network as one continuous string.
        cuts: The absolute bounding polygons for each code point in prediction
              as a list of tuples [(x0, y0), (x1, y2), ...].
        confidences: A list of floats indicating the confidence value of each
                     code point.
        base_dir: An optional string defining the base direction (also called
                  paragraph direction) for the BiDi algorithm. Valid values are
                  'L' or 'R'. If None is given the default auto-resolution will
                  be used.
        display_order: Flag indicating the order of the code points in the
                       prediction. In display order (`True`) the n-th code
                       point in the string corresponds to the n-th leftmost
                       code point, in logical order (`False`) the n-th code
                       point corresponds to the n-th read code point. See [UAX
                       #9](https://unicode.org/reports/tr9) for more details.

    Notes:
        When slicing the record the behavior of the cuts is changed from
        earlier versions of kraken. Instead of returning per-character bounding
        polygons a single polygons section of the line bounding polygon
        starting at the first and extending to the last code point emitted by
        the network is returned. This aids numerical stability when computing
        aggregated bounding polygons such as for words. Individual code point
        bounding polygons are still accessible through the `cuts` attribute or
        by iterating over the record code point by code point.
    """
    type = 'baselines'

    def __init__(self, prediction: str,
                 cuts: List[Tuple[int, int]],
                 confidences: List[float],
                 line: Union[BaselineLine, Dict[str, Any]],
                 base_dir: Optional[Literal['L', 'R']] = None,
                 display_order: bool = True) -> None:
        if not isinstance(line, dict):
            line = asdict(line)
        if line['type'] != 'baselines':
            raise TypeError('Invalid argument type (non-baseline line)')
        BaselineLine.__init__(self, **line)
        self._line_base_dir = self.base_dir
        self.base_dir = base_dir
        ocr_record.__init__(self, prediction, cuts, confidences, display_order)

    def __repr__(self) -> str:
        return f'pred: {self.prediction} baseline: {self.baseline} boundary: {self.boundary} confidences: {self.confidences}'

    def __next__(self) -> Tuple[str, int, float]:
        if self.idx + 1 < len(self):
            self.idx += 1
            return (self.prediction[self.idx],
                    compute_polygon_section(self.baseline,
                                            self.boundary,
                                            self.cuts[self.idx][0],
                                            self.cuts[self.idx][1]),
                    self.confidences[self.idx])
        else:
            raise StopIteration

    def _get_raw_item(self, key: int):
        if key < 0:
            key += len(self)
        if key >= len(self):
            raise IndexError('Index (%d) is out of range' % key)
        return (self.prediction[key],
                self._cuts[key],
                self.confidences[key])

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, slice):
            recs = [self._get_raw_item(i) for i in range(*key.indices(len(self)))]
            prediction = ''.join([x[0] for x in recs])
            flat_offsets = sum((tuple(x[1]) for x in recs), ())
            cut = compute_polygon_section(self.baseline,
                                          self.boundary,
                                          min(flat_offsets),
                                          max(flat_offsets))
            confidence = np.mean([x[2] for x in recs])
            return (prediction, cut, confidence)
        elif isinstance(key, int):
            pred, cut, confidence = self._get_raw_item(key)
            return (pred,
                    compute_polygon_section(self.baseline, self.boundary, cut[0], cut[1]),
                    confidence)
        else:
            raise TypeError('Invalid argument type')

    @property
    def cuts(self) -> List[Tuple[int, int]]:
        return tuple([compute_polygon_section(self.baseline, self.boundary, cut[0], cut[1]) for cut in self._cuts])

    def logical_order(self, base_dir: Optional[Literal['L', 'R']] = None) -> 'BaselineOCRRecord':
        """
        Returns the OCR record in Unicode logical order, i.e. in the order the
        characters in the line would be read by a human.

        Args:
            base_dir: An optional string defining the base direction (also
                      called paragraph direction) for the BiDi algorithm. Valid
                      values are 'L' or 'R'. If None is given the default
                      auto-resolution will be used.
        """
        if self._display_order:
            return self._reorder(base_dir)
        else:
            return self

    def display_order(self, base_dir: Optional[Literal['L', 'R']] = None) -> 'BaselineOCRRecord':
        """
        Returns the OCR record in Unicode display order, i.e. ordered from left
        to right inside the line.

        Args:
            base_dir: An optional string defining the base direction (also
                      called paragraph direction) for the BiDi algorithm. Valid
                      values are 'L' or 'R'. If None is given the default
                      auto-resolution will be used.
        """
        if self._display_order:
            return self
        else:
            return self._reorder(base_dir)

    def _reorder(self, base_dir: Optional[Literal['L', 'R']] = None) -> 'BaselineOCRRecord':
        """
        Reorder the record using the BiDi algorithm.
        """
        storage = bd.get_empty_storage()

        if base_dir not in ('L', 'R'):
            base_level = bd.get_base_level(self._prediction)
        else:
            base_level = {'L': 0, 'R': 1}[base_dir]

        storage['base_level'] = base_level
        storage['base_dir'] = ('L', 'R')[base_level]
        bd.get_embedding_levels(self._prediction, storage)
        bd.explicit_embed_and_overrides(storage)
        bd.resolve_weak_types(storage)
        bd.resolve_neutral_types(storage, False)
        bd.resolve_implicit_levels(storage, False)
        for i, j in enumerate(zip(self._prediction, self._cuts, self._confidences)):
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
        line = BaselineLine(id=self.id,
                            baseline=self.baseline,
                            boundary=self.boundary,
                            text=self.text,
                            base_dir=self._line_base_dir,
                            imagename=self.imagename,
                            tags=self.tags,
                            split=self.split,
                            regions=self.regions)

        rec = BaselineOCRRecord(prediction=prediction,
                                cuts=cuts,
                                confidences=confidences,
                                line=line,
                                base_dir=base_dir,
                                display_order=not self._display_order)
        return rec


class BBoxOCRRecord(ocr_record, BBoxLine):
    """
    A record object containing the recognition result of a single line in
    bbox format.

    Attributes:
        type: 'bbox' to indicate a bounding box record
        prediction: The text predicted by the network as one continuous string.
        cuts: The absolute bounding polygons for each code point in prediction
              as a list of 4-tuples `((x0, y0), (x1, y0), (x1, y1), (x0, y1))`.
        confidences: A list of floats indicating the confidence value of each
                     code point.
        base_dir: An optional string defining the base direction (also called
                  paragraph direction) for the BiDi algorithm. Valid values are
                  'L' or 'R'. If None is given the default auto-resolution will
                  be used.
        display_order: Flag indicating the order of the code points in the
                       prediction. In display order (`True`) the n-th code
                       point in the string corresponds to the n-th leftmost
                       code point, in logical order (`False`) the n-th code
                       point corresponds to the n-th read code point. See [UAX
                       #9](https://unicode.org/reports/tr9) for more details.

    Notes:
        When slicing the record the behavior of the cuts is changed from
        earlier versions of kraken. Instead of returning per-character bounding
        polygons a single polygons section of the line bounding polygon
        starting at the first and extending to the last code point emitted by
        the network is returned. This aids numerical stability when computing
        aggregated bounding polygons such as for words. Individual code point
        bounding polygons are still accessible through the `cuts` attribute or
        by iterating over the record code point by code point.
    """
    type = 'bbox'

    def __init__(self,
                 prediction: str,
                 cuts: List[Tuple[Tuple[int, int],
                                  Tuple[int, int],
                                  Tuple[int, int],
                                  Tuple[int, int]]],
                 confidences: List[float],
                 line: Union[BBoxLine, Dict[str, Any]],
                 base_dir: Optional[Literal['L', 'R']] = None,
                 display_order: bool = True) -> None:
        if not isinstance(line, dict):
            line = asdict(line)
        if line['type'] != 'bbox':
            raise TypeError('Invalid argument type (non-bbox line)')
        BBoxLine.__init__(self, **line)
        self._line_base_dir = self.base_dir
        self.base_dir = base_dir
        ocr_record.__init__(self, prediction, cuts, confidences, display_order)

    def __repr__(self) -> str:
        return f'pred: {self.prediction} bbox: {self.bbox} confidences: {self.confidences}'

    def __next__(self) -> Tuple[str, int, float]:
        if self.idx + 1 < len(self):
            self.idx += 1
            return (self.prediction[self.idx],
                    self.cuts[self.idx],
                    self.confidences[self.idx])
        else:
            raise StopIteration

    def _get_raw_item(self, key: int):
        if key < 0:
            key += len(self)
        if key >= len(self):
            raise IndexError('Index (%d) is out of range' % key)
        return (self.prediction[key],
                self.cuts[key],
                self.confidences[key])

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, slice):
            recs = [self._get_raw_item(i) for i in range(*key.indices(len(self)))]
            prediction = ''.join([x[0] for x in recs])
            box = [x[1] for x in recs]
            flat_box = [point for pol in box for point in pol]
            flat_box = [x for point in flat_box for x in point]
            min_x, max_x = min(flat_box[::2]), max(flat_box[::2])
            min_y, max_y = min(flat_box[1::2]), max(flat_box[1::2])
            cut = ((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y))
            confidence = np.mean([x[2] for x in recs])
            return (prediction, cut, confidence)
        elif isinstance(key, int):
            return self._get_raw_item(key)
        else:
            raise TypeError('Invalid argument type')

    def logical_order(self, base_dir: Optional[Literal['L', 'R']] = None) -> 'BBoxOCRRecord':
        """
        Returns the OCR record in Unicode logical order, i.e. in the order the
        characters in the line would be read by a human.

        Args:
            base_dir: An optional string defining the base direction (also
                      called paragraph direction) for the BiDi algorithm. Valid
                      values are 'L' or 'R'. If None is given the default
                      auto-resolution will be used.
        """
        if self._display_order:
            return self._reorder(base_dir)
        else:
            return self

    def display_order(self, base_dir: Optional[Literal['L', 'R']] = None) -> 'BBoxOCRRecord':
        """
        Returns the OCR record in Unicode display order, i.e. ordered from left
        to right inside the line.

        Args:
            base_dir: An optional string defining the base direction (also
                      called paragraph direction) for the BiDi algorithm. Valid
                      values are 'L' or 'R'. If None is given the default
                      auto-resolution will be used.
        """
        if self._display_order:
            return self
        else:
            return self._reorder(base_dir)

    def _reorder(self, base_dir: Optional[Literal['L', 'R']] = None) -> 'BBoxOCRRecord':
        storage = bd.get_empty_storage()

        if base_dir not in ('L', 'R'):
            base_level = bd.get_base_level(self.prediction)
        else:
            base_level = {'L': 0, 'R': 1}[base_dir]

        storage['base_level'] = base_level
        storage['base_dir'] = ('L', 'R')[base_level]

        bd.get_embedding_levels(self.prediction, storage)
        bd.explicit_embed_and_overrides(storage)
        bd.resolve_weak_types(storage)
        bd.resolve_neutral_types(storage, False)
        bd.resolve_implicit_levels(storage, False)
        for i, j in enumerate(zip(self.prediction, self.cuts, self.confidences)):
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
        line = BBoxLine(id=self.id,
                        bbox=self.bbox,
                        text=self.text,
                        base_dir=self._line_base_dir,
                        imagename=self.imagename,
                        tags=self.tags,
                        split=self.split,
                        regions=self.regions)
        rec = BBoxOCRRecord(prediction=prediction,
                            cuts=cuts,
                            confidences=confidences,
                            line=line,
                            base_dir=base_dir,
                            display_order=not self._display_order)
        return rec
