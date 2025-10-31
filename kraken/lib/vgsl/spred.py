"""
VGSL plumbing
"""
import logging
from collections.abc import Generator

from kraken.containers import Segmentation


__all__ = ['VGSLSegmentationinference']

logger = logging.getLogger(__name__)


class VGSLSegmentationInference:
    def __init__(self):
        super().__init__()

    def _segmentation_pred(self, im) -> Generator[Segmentation, None, None]:
        """
        Segmentation inference.
        """
        pass
