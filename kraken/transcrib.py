# -*- coding: utf-8 -*-
#
# Copyright 2015 Benjamin Kiessling
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

# -*- coding: utf-8 -*-
"""
Utility functions for ground truth transcription.
"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

from future.standard_library import install_aliases
install_aliases()

from kraken.serialization import max_bbox
from kraken.lib.exceptions import KrakenInputException

from jinja2 import Environment, PackageLoader
from itertools import zip_longest
from io import BytesIO 

import os
import uuid
import regex
import base64

class TranscriptionInterface(object):

    def __init__(self, font=None, font_style=None):
        env = Environment(loader=PackageLoader('kraken', 'templates'))
        self.tmpl = env.get_template('layout.html')
        self.pages = []
        self.font = {'font': font, 'style': font_style}
        self.text_direction = 'horizontal-tb'
        self.page_idx = 1
        self.line_idx = 1
        self.seg_idx = 1

    def add_page(self, im, segmentation=None, records=None):
        """
        Adds an image to the transcription interface, optionally filling in
        information from a list of ocr_record objects.

        Args:
            im (PIL.Image): Input image
            records (list): A list of ocr_record objects.
        """
        page = {}
        fd = BytesIO()
        im.save(fd, format='png', optimize=True)
        page['index'] = self.page_idx
        self.page_idx += 1
        page['img'] = 'data:image/png;base64,' + base64.b64encode(fd.getvalue()).decode('ascii')
        page['lines'] = []
        if records:
            for record in records:
                splits = regex.split(u'(\s+)', record.prediction)
                bbox = max_bbox(record.cuts)
                line_offset = 0
                segments = []
                for segment, whitespace in zip_longest(splits[0::2], splits[1::2]):
                    if len(segment):
                        seg_bbox = max_bbox(record.cuts[line_offset:line_offset + len(segment)])
                        segments.append({'bbox': '{}, {}, {}, {}'.format(*seg_bbox), 'text': segment, 'index': self.seg_idx})
                        self.seg_idx += 1
                        line_offset += len(segment)
                    if whitespace:
                        line_offset += len(whitespace)
                page['lines'].append({'index': self.line_idx, 'recognition': segments,
                                      'left': 100*int(bbox[0]) / im.size[0],
                                      'top': 100*int(bbox[1]) / im.size[1],
                                      'width': 100*(bbox[2] - bbox[0])/im.size[0],
                                      'height': 100*(int(bbox[3]) - int(bbox[1]))/im.size[1],
                                      'bbox': '{}, {}, {}, {}'.format(int(bbox[0]),
                                                                      int(bbox[1]),
                                                                      int(bbox[2]),
                                                                      int(bbox[3]))})

                self.line_idx += 1
        elif segmentation:
            self.text_direction = segmentation['text_direction']
            for bbox in segmentation['boxes']:
                page['lines'].append({'index': self.line_idx, 
                                      'left': 100*int(bbox[0]) / im.size[0],
                                      'top': 100*int(bbox[1]) / im.size[1],
                                      'width': 100*(bbox[2] - bbox[0])/im.size[0],
                                      'height': 100*(int(bbox[3]) - int(bbox[1]))/im.size[1],
                                      'bbox': '{}, {}, {}, {}'.format(int(bbox[0]),
                                                                      int(bbox[1]),
                                                                      int(bbox[2]),
                                                                      int(bbox[3]))})
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
        fd.write(self.tmpl.render(uuid=str(uuid.uuid4()), pages=self.pages,
                                  font=self.font,
                                  text_direction=self.text_direction).encode('utf-8'))
