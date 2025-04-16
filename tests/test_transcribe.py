# -*- coding: utf-8 -*-
import json
import os
import unittest
from io import BytesIO
from pathlib import Path

from lxml import etree
from PIL import Image

from kraken import containers
from kraken.transcribe import TranscriptionInterface

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestTranscriptionInterface(unittest.TestCase):

    """
    Test of the transcription interface generation
    """
    def setUp(self):
        with open(resources /'records.json', 'r') as fp:
            self.box_records = [containers.BBoxOCRRecord(**x) for x in json.load(fp)]

        self.box_segmentation = containers.Segmentation(type='bbox',
                                                        imagename='foo.png',
                                                        text_direction='horizontal-lr',
                                                        lines=self.box_records,
                                                        script_detection=True,
                                                        regions={})

        self.im = Image.open(resources / 'input.jpg')

    def test_transcription_generation(self):
        """
        Tests creation of transcription interfaces with segmentation.
        """
        tr = TranscriptionInterface()
        tr.add_page(im = self.im, segmentation=self.box_segmentation)
        fp = BytesIO()
        tr.write(fp)
        # this will not throw an exception ever so we need a better validator
        etree.HTML(fp.getvalue())
