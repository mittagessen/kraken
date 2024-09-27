# -*- coding: utf-8 -*-
import json
import os
import unittest
from io import BytesIO
from pathlib import Path

from lxml import etree
from PIL import Image

from kraken.containers import Segmentation, BBoxLine
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


        seg = Segmentation(type='bbox',
                           imagename = resources / 'bw.png',
                           lines=[BBoxLine(id='foo',
                                           bbox=[200, 10, 400, 156])],
                           text_direction='horizontal-lr',
                           script_detection=False
                          )

        with Image.open(resources / 'input.jpg') as im:
            tr.add_page(im, seg)
        fp = BytesIO()
        tr.write(fp)
        # this will not throw an exception ever so we need a better validator
        etree.HTML(fp.getvalue())
