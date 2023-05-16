
import PIL.Image

from typing import Literal, List, Dict, Sequence, Union, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

__all__ = ['BaselineLine',
           'BBoxLine',
           'Segmentation',
           'ocr_record',
           'BaselineOCRRecord',
           'BBoxOCRRecord']


@dataclass
class BaselineLine:
    """
    """
    id: str
    baseline: List[Tuple[int, int]]
    boundary: List[Tuple[int, int]]
    text: Optional[str] = None
    base_dir: Optional[Literal['L', 'R']] = None
    type: str = 'baselines'
    image: Optional[PIL.Image.Image] = None
    tags: Optional[Dict[str, str]] = None
    split: Optional[Literal['train', 'validation', 'test']] = None
    regions: Optional[List[str]] = None

@dataclass
class BBoxLine:
    """
    """
    id: str
    bbox: Tuple[Tuple[int, int],
                Tuple[int, int],
                Tuple[int, int],
                Tuple[int, int]]
    text: Optional[str] = None
    base_dir: Optional[Literal['L', 'R']] = None
    type: str = 'bbox'
    image: Optional[PIL.Image.Image] = None
    tags: Optional[Dict[str, str]] = None
    split: Optional[Literal['train', 'validation', 'test']] = None
    regions: Optional[List[str]] = None
    text_direction: Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'] = 'horizontal-lr'

@dataclass
class Region:
    """

    """
    id: str
    boundary: List[Tuple[int, int]]
    image: Optional[PIL.Image.Image] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class Segmentation:
    """

    """
    type: Literal['baselines', 'bbox']
    imagename: str
    text_direction: Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl']
    script_detection: bool
    lines: Sequence[Union[BaselineLine, BBoxLine]]
    regions: Dict[str, List[Region]]
    line_orders: Optional[List[List[int]]] = None


class ocr_record(ABC):
    """
    A record object containing the recognition result of a single line
    """
    base_dir = None

    def __init__(self,
                 prediction: str,
                 cuts: Sequence[Union[Tuple[int, int], Tuple[Tuple[int, int],
                                                             Tuple[int, int],
                                                             Tuple[int, int],
                                                             Tuple[int, int]]]],
                 confidences: Sequence[float],
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
    def cuts(self) -> Sequence:
        return self._cuts

    @property
    def confidences(self) -> List[float]:
        return self._confidences

    def __iter__(self):
        self.idx = -1
        return self

    @abstractmethod
    def __next__(self) -> Tuple[str,
                                Union[Sequence[Tuple[int, int]],
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
                 cuts: Sequence[Tuple[int, int]],
                 confidences: Sequence[float],
                 line: BaselineLine,
                 base_dir: Optional[Literal['L', 'R']] = None,
                 display_order: bool = True) -> None:
        if line.type != 'baselines':
            raise TypeError('Invalid argument type (non-baseline line)')
        BaselineLine.__init__(self, **asdict(line))
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
                                            self.line,
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
                                          self.line,
                                          min(flat_offsets),
                                          max(flat_offsets))
            confidence = np.mean([x[2] for x in recs])
            return (prediction, cut, confidence)
        elif isinstance(key, int):
            pred, cut, confidence = self._get_raw_item(key)
            return (pred,
                    compute_polygon_section(self.baseline, self.line, cut[0], cut[1]),
                    confidence)
        else:
            raise TypeError('Invalid argument type')

    @property
    def cuts(self) -> Sequence[Tuple[int, int]]:
        return tuple([compute_polygon_section(self.baseline, self.line, cut[0], cut[1]) for cut in self._cuts])

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
                            image=self.image,
                            tags=self.tags,
                            split=self.split,
                            region=self.region)
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
    """
    type = 'bbox'

    def __init__(self,
                 prediction: str,
                 cuts: Sequence[Tuple[Tuple[int, int],
                                      Tuple[int, int],
                                      Tuple[int, int],
                                      Tuple[int, int]]],
                 confidences: Sequence[float],
                 line: BBoxLine,
                 base_dir: Optional[Literal['L', 'R']],
                 display_order: bool = True) -> None:
        if line.type != 'bbox':
            raise TypeError('Invalid argument type (non-bbox line)')
        BBoxLine.__init__(self, **asdict(line))
        self._line_base_dir = self.base_dir
        self.base_dir = base_dir
        ocr_record.__init__(self, prediction, cuts, confidences, display_order)

    def __repr__(self) -> str:
        return f'pred: {self.prediction} line: {self.line} confidences: {self.confidences}'

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
                        image=self.image,
                        tags=self.tags,
                        split=self.split,
                        region=self.region)
        rec = BBoxOCRRecord(prediction=prediction,
                                cuts=cuts,
                                confidences=confidences,
                                line=line,
                                base_dir=base_dir,
                                display_order=not self._display_order)
        return rec


