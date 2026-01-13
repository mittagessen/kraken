"""
kraken.lib.tasks.align
~~~~~~~~~~~~~~~~~~~~~~

Forced alignment of CTC output.
"""
import torch
from torch import nn
from bidi.algorithm import get_display
from dataclasses import replace, dataclass

from typing import TYPE_CHECKING, Union

from kraken.models import load_models
from kraken.lib.vgsl import TorchVGSLModel
from kraken.containers import Segmentation, BaselineOCRRecord
from kraken.configs import RecognitionInferenceConfig

if TYPE_CHECKING:
    from os import PathLike
    from PIL import Image

__all__ = ['ForcedAlignmentTaskModel']

import logging

logger = logging.getLogger(__name__)


class ForcedAlignmentTaskModel(nn.Module):
    """
    A wrapper for forced alignment of CTC output.

    Using a text recognition model the existing transcription of a page will be
    aligned to the character positions of the network output.

    Raises:
        ValueError: Is raised when the model type is not a sequence recognizer.
    """
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        # only use recognition models.
        models = [net for net in models if 'recognition' in net.model_type]

        if not len(models):
            raise ValueError('No recognition model in model list {models}.')
        if len(models) > 1:
            logger.warning('More than one recognition model in model collection. Using first model.')
        if not isinstance(models[0], TorchVGSLModel):
            raise ValueError('Forced alignment is only supported by TorchVGSLModel networks.')

        self.net = models[0]
        self.one_channel_mode = self.net.one_channel_mode
        self.seg_type = self.net.seg_type

    @torch.inference_mode()
    def predict(self,
                im: 'Image.Image',
                segmentation: 'Segmentation',
                config: RecognitionInferenceConfig) -> Segmentation:
        """
        Aligns the transcription of an image with the output of the text
        recognition model, producing approximate character locations.

        When the character sets of transcription and recognition model differ,
        the affected code points in the furnished transcription will silently
        be ignored. In case inference fails on a line, a record without
        cuts/confidences is returned.

        Args:
            im: The input image
            segmentation: A segmentation with transcriptions to align.
            config: A recognition inference configuration. The task model will
                    automatically set some required configuration flags on it.

        Returns:
            A single segmentation that contains the aligned `ocr_record` objects.

        Example:
            >>> from PIL import Image
            >>> from kraken.tasks import ForcedAlignmentTaskModel
            >>> from kraken.containers import Segmentation, BaselineLine
            >>> from kraken.configs import RecognitionInferenceConfig

            >>> # Assume `model.mlmodel` is a recognition model
            >>> model = ForcedAlignmentTaskModel.load_model('model.mlmodel')
            >>> im = Image.open('image.png')
            >>> # Create a dummy segmentation with a line and a transcription
            >>> line = BaselineLine(baseline=[(0,0), (100,0)], boundary=[(0,-10), (100,-10), (100,10), (0,10)], text='Hello World')
            >>> segmentation = Segmentation(lines=[line])
            >>> config = RecognitionInferenceConfig()

            >>> aligned_segmentation = model.predict(im, segmentation, config)
            >>> record = aligned_segmentation.lines[0]:
            >>> print(record.prediction)
            >>> print(record.cuts)
        """
        if not config.return_logits:
            logger.info('Forced alignment requires logits in output records. Enabling.')
            config.return_logits = True
        if not config.return_line_image:
            logger.info('Forced alignment requires line images in output records. Enabling.')
            config.return_line_image = True

        self.net.prepare_for_inference(config)

        records = []
        for idx, record in enumerate(self.net.predict(im=im, segmentation=segmentation)):
            # convert text to display order
            do_text = get_display(record.text, base_dir=config.bidi_reordering if config.bidi_reordering in ('L', 'R') else None)
            # encode into labels, ignoring unencodable sequences
            labels = self.net.codec.encode(do_text).long()

            if record.logits.shape[-1] < 2*len(labels):
                logger.warning(f'Could not align line {idx}. Output sequence length {record.logits.shape[-1]} < '
                               f'{2*len(labels)} (length of "{record.text}" after encoding).')
                records.append(record.__class__('', [], [], segmentation.lines[idx]))
                continue
            emission = record.logits.squeeze().log_softmax(0).T
            trellis = get_trellis(emission, labels)
            path = backtrack(trellis, emission, labels)
            path = merge_repeats(path, do_text)
            pred = []
            pos = []
            conf = []
            # net_scale should stay fairly static, but in_scale changes between
            # lines so we just set it here again.
            self.net.in_scale = record.image.width/(record.logits.shape[-1]*self.net.net_scale-2*config.padding)
            for seg in path:
                pred.append(seg.label)
                pos.append((self.net._scale_val(seg.start, 0, record.image.width),
                            self.net._scale_val(seg.end, 0, record.image.width)))
                conf.append(seg.score)
            rec = BaselineOCRRecord(pred, pos, conf, segmentation.lines[idx], display_order=True)
            if config.bidi_reordering:
                logger.debug('BiDi reordering record.')
                rec.logical_order(base_dir=config.bidi_reordering if config.bidi_reordering in ('L', 'R') else None)
            records.append(rec)
        return replace(segmentation, lines=records)

    @classmethod
    def load_model(cls, path: Union[str, 'PathLike']):
        models = load_models(path)
        return cls(models)


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
