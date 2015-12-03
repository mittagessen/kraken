# -*- coding: utf-8 -*-
"""
Utility functions for ground truth transcription.
"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

from kraken.html import max_bbox
from kraken.lib.exceptions import KrakenInputException

from jinja2 import Environment, PackageLoader
from itertools import izip_longest
from StringIO import StringIO

import regex
import base64
import os

class TranscriptionInterface(object):

    def __init__(self):
        env = Environment(loader=PackageLoader('kraken', 'templates'))
        self.tmpl = env.get_template('layout.html')
        self.pages = []
        self.page_idx = 1
        self.line_idx = 1
        self.seg_idx = 1

    def add_page(self, im, segmentation=None, records=None):
        """
        Adds an image to the transcription interface, optionally filling in
        information from a list of ocr_record objects.

        Args:
            im (PIL.Image): Input image
            records (lis): A list of ocr_record objects.
        """
        page = {}
        fd = StringIO()
        im.save(fd, format='png')
        page['index'] = self.page_idx
        self.page_idx += 1
        page['img'] = 'data:image/png;base64,' + base64.b64encode(fd.getvalue())
        page['lines'] = []
        if records:
            for record in records:
                splits = regex.split(u'(\s+)', record.prediction)
                bbox = max_bbox(record.cuts)
                line_offset = 0
                segments = []
                for segment, whitespace in izip_longest(splits[0::2], splits[1::2]):
                    if len(segment):
                        seg_bbox = max_bbox(record.cuts[line_offset:line_offset + len(segment)])
                        segments.append({'bbox': '{}, {}. {}, {}'.format(*seg_bbox), 'text': segment, 'index': self.seg_idx})
                        self.seg_idx += 1
                        line_offset += len(segment)
                    if whitespace:
                        line_offset += len(whitespace)
                page['lines'].append({'index': self.line_idx, 'recognition': segments, 
                                      'bbox': '{}, {}, {}, {}'.format(int(bbox[0]), 
                                                                      int(bbox[1]),
                                                                      int(bbox[2]),
                                                                      int(bbox[3]))})
                self.line_idx += 1
        elif segmentation:
            for bbox in segmentation:
                page['lines'].append({'index': self.line_idx, 'bbox': '{}, {}, {}, {}'.format(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))})
                self.line_idx += 1
        else:
            raise KrakenInputException('Neither segmentations nor records given')
        self.pages.append(page)

    def write(self, fd):
        """
        Writes the HTML file to a file descriptor.

        Args:
            fd (File): File descriptor to write to.
        """
        fd.write(self.tmpl.render(pages=self.pages).encode('utf-8'))
