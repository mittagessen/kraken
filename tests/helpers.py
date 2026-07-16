# -*- coding: utf-8 -*-
"""
Shared helpers for the kraken test suite.
"""
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from kraken.containers import (BaselineLine, BaselineOCRRecord, BBoxOCRRecord,
                               Segmentation)

RESOURCES = Path(__file__).resolve().parent / 'resources'

WORDS = ['kraken', 'paddle', 'ocrv6', 'recognition', 'model', 'text', 'line',
         'image', 'training', 'ctc', 'svtr', 'backbone', 'neck', 'head',
         'pytorch', 'lightning']

# logical (reading) order text of the Arabic test line fixture (arabic.webp /
# arabic_*_records.json), with combining characters as explicit escapes
# (maddah above U+0653, hamza above U+0654)
ARABIC_LINE_LOGICAL = ('\u0639\u0646\u062f \u0639\u062f\u0645 \u0627\u0644\u0639\u0635\u0628\u0627\u062a '
                    '\u0627\u0630\u0627 \u0644\u0645 \u064a\u0643\u0646 \u0644\u0644\u0635\u063a\u064a\u0631\u0629 '
                    '\u0627\u0654\u0645 \u0627\u0654\u064a\u0636\u0627\u064b \u0644\u0645\u0627\u0630 '
                    '\u0643\u0631. . \u0648\u0644\u0646\u0627 \u0627\u0654\u0646 \u0646\u0642\u0648\u0644 '
                    '\u0627\u0646 \u0627\u0644\u0627\u0653\u0645')


def render_line(text, path, height=48):
    """Render a single text line to a grayscale PNG."""
    font = ImageFont.load_default()
    width = max(32, 14 * len(text))
    im = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(im)
    draw.text((4, height // 2 - 6), text, fill=0, font=font)
    im.save(path)


def build_path_dataset(tmp_path, words=WORDS):
    """Renders words as image/gt.txt pairs and returns the image paths."""
    img_paths = []
    for i, word in enumerate(words):
        p = Path(tmp_path) / f'line_{i:03d}.png'
        render_line(word, str(p))
        (Path(tmp_path) / f'line_{i:03d}.gt.txt').write_text(word)
        img_paths.append(str(p))
    return img_paths


def make_baseline_segmentation(imagename=RESOURCES / 'bw.png',
                               width=2543,
                               height=155,
                               text=None,
                               tags=None,
                               line_id='foo'):
    """
    A single-line baseline Segmentation covering a full line strip image.
    """
    return Segmentation(type='baselines',
                        imagename=imagename,
                        lines=[BaselineLine(id=line_id,
                                            baseline=[[0, 10], [width, 10]],
                                            boundary=[[0, 0], [width, 0], [width, height], [0, height]],
                                            text=text,
                                            tags=tags)],
                        text_direction='horizontal-lr',
                        script_detection=False)


def make_ocr_record(data, **kwargs):
    """
    Constructs a Baseline/BBoxOCRRecord from a JSON fixture dict of raw
    constructor arguments.
    """
    data = {**data, **kwargs}
    line = data['line']
    cls = BaselineOCRRecord if line['type'] == 'baselines' else BBoxOCRRecord
    return cls(prediction=data['prediction'],
               cuts=data['cuts'],
               confidences=data['confidences'],
               line=line,
               base_dir=data.get('base_dir'),
               display_order=data.get('display_order', True))


def load_record_dicts(name):
    """
    Loads a JSON fixture of raw OCR record constructor-argument dicts.
    """
    with open(RESOURCES / name) as fp:
        return json.load(fp)


def load_records(name):
    """
    Loads a JSON fixture as a list of constructed OCR records.
    """
    return [make_ocr_record(d) for d in load_record_dicts(name)]


def load_segmentation(name):
    """
    Loads a Segmentation from a JSON fixture, constructing OCR records for
    lines carrying recognition results.
    """
    with open(RESOURCES / name) as fp:
        data = json.load(fp)
    data['lines'] = [make_ocr_record(line) if 'prediction' in line else line
                     for line in data['lines']]
    return Segmentation(**data)


@contextmanager
def temp_output(suffix=''):
    """
    Yields the path of a closed temporary file that is removed on exit.
    """
    fp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    fp.close()
    try:
        yield fp.name
    finally:
        os.unlink(fp.name)


def make_smoke_trainer(**kwargs):
    """
    A minimal CPU KrakenTrainer for single-epoch training smoke tests.
    """
    from kraken.train.utils import KrakenTrainer
    defaults = dict(accelerator='cpu',
                    devices=1,
                    max_epochs=1,
                    min_epochs=1,
                    enable_progress_bar=False,
                    enable_summary=False,
                    logger=False,
                    num_sanity_val_steps=0)
    defaults.update(kwargs)
    if 'callbacks' not in defaults:
        defaults.setdefault('enable_checkpointing', False)
    return KrakenTrainer(**defaults)
