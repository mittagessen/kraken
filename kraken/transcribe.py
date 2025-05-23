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
"""
Utility functions for ground truth transcription.
"""
import base64
import logging
import uuid
from io import BytesIO
from typing import Any, Dict, List

from jinja2 import Environment, PackageLoader

from kraken.lib.util import get_im_str

logger = logging.getLogger()


class TranscriptionInterface(object):

    def __init__(self, font=None, font_style=None):
        logging.info('Initializing transcription object.')
        logger.debug('Initializing jinja environment.')
        env = Environment(loader=PackageLoader('kraken', 'templates'), autoescape=True)
        logger.debug('Loading transcription template.')
        self.tmpl = env.get_template('layout.html')
        self.pages: List[Dict[Any, Any]] = []
        self.font = {'font': font, 'style': font_style}
        self.text_direction = 'horizontal-tb'
        self.page_idx = 1
        self.line_idx = 1
        self.seg_idx = 1

    def add_page(self, im, segmentation = None):
        """
        Adds an image to the transcription interface, optionally filling in
        information from a list of ocr_record objects.

        Args:
            im: Input image
            segmentation: Output of the segment method.
        """
        im_str = get_im_str(im)
        logger.info(f'Adding page {im_str} with {len(segmentation.lines)} lines')
        page = {}
        fd = BytesIO()
        im.save(fd, format='png', optimize=True)
        page['index'] = self.page_idx
        self.page_idx += 1
        logger.debug('Base64 encoding image')
        page['img'] = 'data:image/png;base64,' + base64.b64encode(fd.getvalue()).decode('ascii')
        page['lines'] = []
        logger.debug('Adding segmentation.')
        self.text_direction = segmentation.text_direction
        for line in segmentation.lines:
            bbox = line.bbox
            page['lines'].append({'index': self.line_idx,
                                  'left': 100*int(bbox[0]) / im.size[0],
                                  'top': 100*int(bbox[1]) / im.size[1],
                                  'width': 100*(bbox[2] - bbox[0])/im.size[0],
                                  'height': 100*(int(bbox[3]) - int(bbox[1]))/im.size[1],
                                  'bbox': '{}, {}, {}, {}'.format(int(bbox[0]),
                                                                  int(bbox[1]),
                                                                  int(bbox[2]),
                                                                  int(bbox[3]))})
            if line.text:
                page['lines'][-1]['text'] = line.prediction
            self.line_idx += 1
        self.pages.append(page)

    def write(self, fd):
        """
        Writes the HTML file to a file descriptor.

        Args:
            fd (File): File descriptor (mode='rb') to write to.
        """
        logger.info('Rendering and writing transcription.')
        fd.write(self.tmpl.render(uuid=f'_{uuid.uuid4()}', pages=self.pages,
                                  font=self.font,
                                  text_direction=self.text_direction).encode('utf-8'))
