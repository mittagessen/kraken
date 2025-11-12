"""
kraken.lib.tasks.segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrappers around models for specific tasks.
"""
import torch
import shapely.geometry as geom

from torch import nn
from importlib import resources
from dataclasses import replace
from typing import TYPE_CHECKING, Union, Optional

from kraken.containers import Segmentation, BaselineLine
from kraken.lib.segmentation import is_in_region
from kraken.models import load_models
from kraken.configs import SegmentationInferenceConfig

if TYPE_CHECKING:
    from os import PathLike
    from PIL import Image

__all__ = ['SegmentationTaskModel']

import logging

logger = logging.getLogger(__name__)


class SegmentationTaskModel:
    """
    A wrapper class collecting one or more models that perform segmentation.

    It deals with the following tasks:
        * region segmentation
        * line detection
        * line reading order

    If no neural reading order model is part of the model collection handed to
    the task, a simple heuristic will be used.

    Args:
        models: A collection of models performing segmentation tasks.

    Raises:
        ValueError: Is raised when no segmentation models are in the model list.
    """
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        # only use recognition models.
        self.seg_models = [net for net in models if 'segmentation' in net.model_type]
        self.ro_models = [net for net in models if 'reading_order' in net.model_type]

        if not len(self.seg_models):
            raise ValueError('No segmentation models in model list {models}.')

    @torch.inference_mode()
    def predict(self,
                im: 'Image.Image',
                config: SegmentationInferenceConfig) -> Segmentation:
        """
        Runs all models associated with the task to produce a segmentation for
        the input page.

        Args:
            im: Input image with an arbitrary color mode and size
            config: A configuration object for the segmentation task.

        Returns:
            A single Segmentation object that contains the merged output of all
            associated segmentation models.
        """
        segs = []
        for net in self.seg_models:
            logger.info(f'Applying model {net}.')
            net.prepare_for_inference(config)
            segs.append(net.predict(im=im))
        logger.info(f'Merging {len(segs)} segmentations.')
        segmentation = self._merge_segmentations(segs, config)
        logger.info('Computing reading order(s)')
        return self._compute_additional_line_orders(segmentation, config)

    @classmethod
    def load_model(cls, path: Optional[Union[str, 'PathLike']] = None) -> 'SegmentationTaskModel':
        """
        Loads a collection from layout analysis models from the given file path.

        Args:
            path: Path to model weights file. If `None`, the default BLLA
                  segmentation model will be loaded.
        """
        if not path:
            path = resources.files('kraken').joinpath('blla.mlmodel')
            logger.info(f'No segmentation model given. Loading default model from {path}.')
        models = load_models(path)
        return cls(models)

    @staticmethod
    def _merge_segmentations(segmentations: list[Segmentation],
                             config: SegmentationInferenceConfig) -> Segmentation:
        lines = []
        regions = {}
        script_detection = False
        languages = set()
        _shp_regs = {}
        for seg in segmentations:
            script_detection = script_detection or seg.script_detection
            languages.update(seg.language if seg.language is not None else [])
            if lines and seg.lines:
                logger.warning('Multiple models produced line output. This is '
                               'likely unintended.')
            lines.extend(seg.lines)
            for reg_type, regs in seg.regions.items():
                regions.setdefault(reg_type, [])
                regions[reg_type].extend(regs)
                for reg in regs:
                    _shp_regs[reg.id] = geom.Polygon(reg.boundary)

        _lines = []
        for line in lines:
            line_regs = []
            for reg_id, reg in _shp_regs.items():
                line_ls = geom.LineString(line.baseline)
                if is_in_region(line_ls, reg):
                    line_regs.append(reg_id)
            _lines.append(replace(line, regions=line_regs))

        if len(ltypes := set(type(line) for line in _lines)) > 1:
            raise ValueError(f'Mixed line data models in one segmentation task are not supported. Got {ltypes}')

        if isinstance(_lines[0], BaselineLine):
            ro_fn = config.baseline_ro_fn
        else:
            ro_fn = config.bbox_ro_fn

        basic_lo = ro_fn(lines=_lines,
                         regions=_shp_regs.values(),
                         text_direction=segmentations[0].text_direction[-2:])

        _lines = [_lines[idx] for idx in basic_lo]

        return replace(segmentations[0],
                       script_detection=script_detection,
                       language=list(languages),
                       type='baselines' if isinstance(_lines[0], BaselineLine) else 'bbox',  # Region-only model can have an arbitrary type.
                       lines=_lines,
                       regions=regions)

    def _compute_additional_line_orders(self,
                                        segmentation: Segmentation,
                                        config: SegmentationInferenceConfig) -> Segmentation:
        """
        Computes additional reading orders with neural models.

        Not implemented yet.
        """
        return segmentation
