# -*- coding: utf-8 -*-
#
# Copyright 2014 Google Inc. All rights reserved.
# Copyright 2015 Benjamin Kiessling
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""= 
linegen
~~~~~~~

An advanced line generation tool using Pango for proper text shaping. The
actual drawing code was adapted from the create_image utility from nototools
available at [0].

[0] https://github.com/googlei18n/nototools
"""

from __future__ import absolute_import, division, print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object

from jinja2 import Environment, PackageLoader

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import affine_transform, geometric_transform
from PIL import Image

import numpy as np
import pangocairo
import tempfile
import shutil
import cairo
import pango

from kraken.lib.util import pil2array, array2pil


def set_fonts(font_file):
    """
    Activates a temporary fontconfig environment and loads pango.

    Writes a temporary fontconfig configuration with ``font_file`` being the
    only font in the cache. It is then activated by setting the FONTCONFIG_FILE
    environment variable and loading pango/cairo.

    .. warning::
        This function can only be executed once as letting pango/cairo
        reinitialize fontconfig doesn't seem to be possible.

    Args:
        font_file (unicode): Location of an font file understood by pango
    """
    global cairo
    global pango
    global pangocairo
    font_dir = tempfile.mkdtemp()
    shutil.copy(font_file, font_dir)

    env = Environment(loader=PackageLoader('kraken', 'templates'))
    template = env.get_template('fonts.conf')

    fp = tempfile.NamedTemporaryFile(delete=False)
    fp.write(template.render(font_dir=font_dir, font_file=fp.name).encode('utf-8'))
    os.putenv("FONTCONFIG_FILE", fp.name)
    

def draw_on_surface(surface, text, family, font_size, language, rtl, vertical):
    pangocairo_ctx = pangocairo.CairoContext(cairo.Context(surface))
    layout = pangocairo_ctx.create_layout()

    pango_ctx = layout.get_context()
    if language is not None:
        pango_ctx.set_language(pango.Language(language))

    if rtl:
        if vertical:
            base_dir = pango.DIRECTION_TTB_RTL
        else:
            base_dir = pango.DIRECTION_RTL
    else:
        if vertical:
            base_dir = pango.DIRECTION_TTB_LTR
        else:
            base_dir = pango.DIRECTION_LTR

    pango_ctx.set_base_dir(base_dir)

    font = pango.FontDescription()
    font.set_family(family)
    font.set_size(font_size * pango.SCALE)

    layout.set_font_description(font)
    layout.set_text(text)

    extents = layout.get_pixel_extents()
    top_usage = min(extents[0][1], extents[1][1], 0)
    bottom_usage = max(extents[0][3], extents[1][3])

    width = max(extents[0][2], extents[1][2])

    pangocairo_ctx.set_antialias(cairo.ANTIALIAS_GRAY)
    pangocairo_ctx.set_source_rgb(1, 1, 1)  # White background
    pangocairo_ctx.paint()

    pangocairo_ctx.translate(0, -top_usage)
    pangocairo_ctx.set_source_rgb(0, 0, 0)  # Black text color
    pangocairo_ctx.show_layout(layout)

    return bottom_usage - top_usage, width


def render_line(text, family, font_size=32, language=None, rtl=False, vertical=False):
    """
    Renders ``text`` into a PIL Image using pango and cairo.

    Args:
        text (unicode): A unicode string to be rendered
        family (unicode): Font family to use for rendering
        font_size (unicode): Font size in points
        language (unicode): RFC-3066 language tag
        rtl (bool): Set base horizontal text direction. The BiDi algorithm will
                    still apply so it's usually not necessary to touch this
                    option.
        vertical (bool): Set vertical text direction (True = Top-to-Bottom)

    Returns:
        (B/W) PIL.Image in RGBA mode
    """
    temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0)
    height, width = draw_on_surface(temp_surface, text, family,
                                    font_size,language, rtl, vertical)
    
    real_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    draw_on_surface(real_surface, text, family, font_size, language, rtl, vertical)
    return Image.frombuffer("RGBA", (width, height), real_surface.get_data(), "raw", "BGRA", 0, 1)


def degrade_line(im, mean=0.0, sigma=0.001, density=0.02):
    """
    Degrades a line image by adding several kinds of noise.

    Args:
        im (PIL.Image): Input image
        mean (float): Mean of distribution for Gaussian noise
        sigma (float): Standard deviation for Gaussian noise
        density (float): Noise density for Salt and Pepper noise

    Returns:
        PIL.Image in mode 'L'
    """
    def add_gaussian(im, mean, sigma):
        """
        Adds Gaussian white noise.
    
        Args:
            im (PIL.Image): Input image
            mean (float): Mean value
            sigma (float): Standard deviation
    
        Returns:
            PIL.Image in 'L' mode.
        """
        a = pil2array(im)
        b = a.astype('f')/np.amax(a) + np.random.normal(mean, sigma, a.shape)
        return array2pil(np.clip(b * np.amax(a), 0, 255).astype('uint8'))
    
    
    def add_salt_and_pepper(im, d):
        """
        Adds (symmetric) salt and pepper noise.
    
        Args:
            im (PIL.Image): Input image
            d (float): Noise density
    
        Returns:
            PIL.Image in 'L' mode.
        """
        a = pil2array(im.convert('L'))
        flipped = np.ceil(d/2 * a.size)
        coords = [np.random.randint(0, i - 1, int(flipped)) for i in a.shape]
        a[coords] = 255
        coords = [np.random.randint(0, i - 1, int(flipped)) for i in a.shape]
        a[coords] = 0
        return array2pil(a)

    im = add_gaussian(im, mean, sigma)
    return add_salt_and_pepper(im, density)


def distort_line(im, distort=3.0, sigma=10.0):
    """
    Distorts a line image.

    Args:
        im (PIL.Image): Input image
        distort (float):
        sigma (float):
    """
    w, h = im.size
    line = pil2array(im.convert('L'))
    hs = gaussian_filter(np.random.randn(h, w), sigma)
    ws = gaussian_filter(np.random.randn(h, w), sigma)
    hs *= distort/np.amax(hs)
    ws *= distort/np.amax(ws)

    def f(p):
        return (p[0]+hs[p[0],p[1]],p[1]+ws[p[0],p[1]])

    return array2pil(geometric_transform(line, f, output_shape=(h, w), order=1, mode='nearest'))
