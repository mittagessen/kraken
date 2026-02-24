"""
kraken.lib.tasks.segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A wrapper around models for layout analysis and reading order determination.
"""
import torch
import shapely.geometry as geom

from collections import defaultdict
from torch import nn
from importlib import resources
from dataclasses import replace
from typing import TYPE_CHECKING, Union, Optional

from kraken.containers import Segmentation, BaselineLine
from kraken.lib.segmentation import is_in_region, neural_reading_order
from kraken.models import load_models
from kraken.configs import SegmentationInferenceConfig

if TYPE_CHECKING:
    from os import PathLike
    from PIL import Image

__all__ = ['SegmentationTaskModel']

import logging

logger = logging.getLogger(__name__)


class SegmentationTaskModel(nn.Module):
    """
    A wrapper class collecting one or more models that perform segmentation.

    A segmentation task is the process of identifying the regions and lines of
    text in an image. This class provides a high-level interface for running
    segmentation models on an image.

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

        # validate RO models: no duplicates at the same level, class
        # mappings match the segmentation model's.
        seg_class_mapping = self.seg_models[0].user_metadata.get('class_mapping', {})
        ro_levels = set()
        for m in self.ro_models:
            level = m.user_metadata.get('level', 'baselines')
            if level in ro_levels:
                raise ValueError(f'Multiple reading order models at level `{level}`.')
            ro_levels.add(level)
            ro_cm = m.user_metadata.get('class_mapping', {})
            seg_cm = seg_class_mapping.get(level, {})
            diff = set(ro_cm.keys()).symmetric_difference(set(seg_cm.keys()))
            diff.discard('default')
            if diff:
                raise ValueError(f'Reading order model class mapping at level '
                                 f'`{level}` does not match segmentation model: {diff}')

    @torch.inference_mode()
    def predict(self,
                im: 'Image.Image',
                config: SegmentationInferenceConfig) -> Segmentation:
        """
        Runs all models associated with the task to produce a segmentation for
        the input page.

        Args:
            im: Input image with an arbitrary color mode and size
            config: A configuration object for the segmentation task, such as
                    the batch size and the precision.

        Returns:
            A single Segmentation object that contains the merged output of all
            associated segmentation models.

        Example:
            >>> from PIL import Image
            >>> from kraken.tasks import SegmentationTaskModel
            >>> from kraken.configs import SegmentationInferenceConfig

            >>> model = SegmentationTaskModel.load_model()
            >>> im = Image.open('image.png')
            >>> config = SegmentationInferenceConfig()

            >>> segmentation = model.predict(im, config)
        """
        segs = []
        for net in self.seg_models:
            logger.info(f'Applying model {net}.')
            net.prepare_for_inference(config)
            segs.append(net.predict(im=im))
        logger.info(f'Merging {len(segs)} segmentations.')
        segmentation = self._merge_segmentations(segs, config)
        logger.info('Computing reading order(s)')
        im_size = im.size
        return self._compute_additional_line_orders(segmentation, config, im_size=im_size)

    @classmethod
    def load_model(cls, path: Optional[Union[str, 'PathLike']] = None) -> 'SegmentationTaskModel':
        """
        Loads a collection from layout analysis models from the given file path.

        If no path is provided, the default BLLA segmentation model will be
        loaded.

        Args:
            path: Path to model weights file.
        """
        if not path:
            path = resources.files('kraken').joinpath('blla.mlmodel')
            logger.info(f'No segmentation model given. Loading default model from {path}.')
        models = load_models(path)
        return cls(models)

    @staticmethod
    def _merge_segmentations(segmentations: list[Segmentation],
                             config: SegmentationInferenceConfig) -> Segmentation:
        if len(segmentations) == 1:
            return segmentations[0]
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
            if hasattr(line, 'baseline') and line.baseline:
                line_geom = geom.LineString(line.baseline)
            elif hasattr(line, 'bbox') and line.bbox:
                xmin, ymin, xmax, ymax = line.bbox
                line_geom = geom.box(xmin, ymin, xmax, ymax)
            else:
                _lines.append(line)
                continue
            for reg_id, reg in _shp_regs.items():
                if is_in_region(line_geom, reg):
                    line_regs.append(reg_id)
            _lines.append(replace(line, regions=line_regs))

        if len(ltypes := set(type(line) for line in _lines)) > 1:
            raise ValueError(f'Mixed line data models in one segmentation task are not supported. Got {ltypes}')

        all_regions = [reg for rgs in regions.values() for reg in rgs]
        if _lines:
            if isinstance(_lines[0], BaselineLine):
                ro_fn = config.baseline_ro_fn
            else:
                ro_fn = config.bbox_ro_fn

            basic_lo = ro_fn(lines=_lines,
                             regions=all_regions,
                             text_direction=segmentations[0].text_direction[-2:])

            _lines = [_lines[idx] for idx in basic_lo]

        if _lines:
            seg_type = 'baselines' if isinstance(_lines[0], BaselineLine) else 'bbox'
        else:
            seg_type = segmentations[0].type

        return replace(segmentations[0],
                       script_detection=script_detection,
                       language=list(languages),
                       type=seg_type,
                       lines=_lines,
                       regions=regions)

    def _compute_additional_line_orders(self,
                                        segmentation: Segmentation,
                                        config: SegmentationInferenceConfig,
                                        im_size: Optional[tuple[int, int]] = None) -> Segmentation:
        """
        Computes additional reading orders with neural models.

        Args:
            segmentation: Segmentation container to augment with reading orders.
            config: Inference configuration.
            im_size: Dimensions of the source image as a (width, height) tuple.

        Neural reading order models are separated by level (baselines/regions).
        If a region-level model is present, regions are ordered first. If a
        line-level model is present, lines within each region are ordered. The
        result is appended as an additional entry in `line_orders`.
        """
        if not self.ro_models:
            return segmentation

        line_ro = None
        region_ro = None
        for model in self.ro_models:
            level = model.user_metadata.get('level', 'baselines')
            if level == 'regions':
                region_ro = model
            else:
                line_ro = model

        seg_class_mapping = self.seg_models[0].user_metadata.get('class_mapping', {})
        if not segmentation.lines or not isinstance(segmentation.lines[0], BaselineLine):
            logger.warning('Neural reading order only supports baselines. Skipping.')
            return segmentation

        if im_size is None:
            logger.warning('No image size available. Cannot compute neural reading order.')
            return segmentation

        all_regions = [reg for rgs in segmentation.regions.values() for reg in rgs]

        # Order regions if region model present
        if region_ro and all_regions:
            region_order = neural_reading_order(lines=all_regions,
                                                model=region_ro,
                                                im_size=im_size,
                                                class_mapping=seg_class_mapping.get('regions', {}))
            if region_order is not None:
                ordered_regions = [all_regions[i] for i in region_order]
            else:
                ordered_regions = all_regions
        else:
            ordered_regions = all_regions

        # Order lines within each region if line model present
        if line_ro:
            line_class_mapping = seg_class_mapping.get('baselines', {})
            ordered_lines = []
            region_line_map = defaultdict(list)
            region_ids = {reg.id for reg in ordered_regions}
            for line in segmentation.lines:
                if line.regions and line.regions[0] in region_ids:
                    region_line_map[line.regions[0]].append(line)
                else:
                    region_line_map[None].append(line)

            if region_ro and ordered_regions:
                for region in ordered_regions:
                    rlines = region_line_map.get(region.id, [])
                    if len(rlines) > 1:
                        lo = neural_reading_order(lines=rlines,
                                                  model=line_ro,
                                                  im_size=im_size,
                                                  class_mapping=line_class_mapping)
                        if lo is not None:
                            ordered_lines.extend([rlines[i] for i in lo])
                        else:
                            ordered_lines.extend(rlines)
                    else:
                        ordered_lines.extend(rlines)
                # Orphan lines (no region)
                orphans = region_line_map.get(None, [])
                if len(orphans) > 1:
                    lo = neural_reading_order(lines=orphans,
                                              model=line_ro,
                                              im_size=im_size,
                                              class_mapping=line_class_mapping)
                    if lo is not None:
                        ordered_lines.extend([orphans[i] for i in lo])
                    else:
                        ordered_lines.extend(orphans)
                else:
                    ordered_lines.extend(orphans)
            else:
                lo = neural_reading_order(lines=segmentation.lines,
                                          model=line_ro,
                                          im_size=im_size,
                                          class_mapping=line_class_mapping)
                if lo is not None:
                    ordered_lines = [segmentation.lines[i] for i in lo]
                else:
                    ordered_lines = list(segmentation.lines)
        elif region_ro:
            # Only region model — group lines by region order, keep intra-region order
            ordered_lines = []
            used = set()
            for region in ordered_regions:
                for line in segmentation.lines:
                    if line.regions and line.regions[0] == region.id and id(line) not in used:
                        ordered_lines.append(line)
                        used.add(id(line))
            for line in segmentation.lines:
                if id(line) not in used:
                    ordered_lines.append(line)
        else:
            return segmentation

        # Build index mapping for line_orders
        old_to_new = {id(line): idx for idx, line in enumerate(segmentation.lines)}
        neural_order = [old_to_new[id(line)] for line in ordered_lines]

        line_orders = list(segmentation.line_orders) if segmentation.line_orders else []
        line_orders.append(neural_order)
        return replace(segmentation, line_orders=line_orders)
