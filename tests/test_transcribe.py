# -*- coding: utf-8 -*-
import os
import json
import unittest

from PIL import Image
from io import BytesIO
from lxml import etree
from pathlib import Path

from kraken.transcribe import TranscriptionInterface

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'

class TestTranscriptionInterface(unittest.TestCase):

    """
    Test of the transcription interface generation
    """

    def test_transcription_generation(self):
        """
        Tests creation of transcription interfaces with segmentation.
        """
        tr = TranscriptionInterface()
        with open(resources / 'segmentation.json') as fp:
            seg = json.load(fp)
        with Image.open(resources / 'input.jpg') as im:
            tr.add_page(im, seg)
        fp = BytesIO()
        tr.write(fp)
        # this will not throw an exception ever so we need a better validator
        etree.HTML(fp.getvalue())
