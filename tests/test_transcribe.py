# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import json
import unittest

from PIL import Image
from lxml import etree
from io import BytesIO 
from kraken.transcribe import TranscriptionInterface

thisfile = os.path.abspath(os.path.dirname(__file__))
resources = os.path.abspath(os.path.join(thisfile, 'resources'))

class TestTranscriptionInterface(unittest.TestCase):

    """
    Test of the transcription interface generation
    """

    def test_transcription_generation(self):
        """
        Tests creation of transcription interfaces with segmentation.
        """
        tr = TranscriptionInterface()
        with open(os.path.join(resources, 'segmentation.json')) as fp:
            seg = json.load(fp)
        with Image.open(os.path.join(resources, 'input.jpg')) as im:
            tr.add_page(im, seg)
        fp = BytesIO()
        tr.write(fp)
        # this will not throw an exception ever so we need a better validator
        etree.HTML(fp.getvalue())
