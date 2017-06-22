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

"""
linegen
~~~~~~~

An advanced line generation tool using Pango for proper text shaping. The
actual drawing code was adapted from the create_image utility from nototools
available at [0].

[0] https://github.com/googlei18n/nototools
"""

from __future__ import absolute_import, division, print_function
from future import standard_library
from builtins import object

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.interpolation import affine_transform, geometric_transform
from PIL import Image, ImageOps

import numpy as np
import ctypes.util
import ctypes

from kraken.lib.exceptions import KrakenCairoSurfaceException
from kraken.lib.util import pil2array, array2pil

standard_library.install_aliases()

pangocairo = ctypes.CDLL(ctypes.util.find_library('pangocairo-1.0'))
pango = ctypes.CDLL(ctypes.util.find_library('pango-1.0'))
cairo = ctypes.CDLL(ctypes.util.find_library('cairo'))

__all__ = ['LineGenerator', 'ocropy_degrade', 'degrade_line', 'distort_line']

class PangoFontDescription(ctypes.Structure):
    pass


class PangoLayout(ctypes.Structure):
    pass


class PangoContext(ctypes.Structure):
    pass


class PangoRectangle(ctypes.Structure):
    _fields_ = [('x', ctypes.c_int),
                ('y', ctypes.c_int),
                ('width', ctypes.c_int),
                ('height', ctypes.c_int)]


class ensureBytes(object):
    """
    Simple class ensuring the arguments of type char * are actually a series of
    bytes.
    """
    @classmethod
    def from_param(cls, value):
        if isinstance(value, bytes):
            return value
        else:
            return value.encode('utf-8')


cairo.cairo_image_surface_get_data.restype = ctypes.c_void_p

pango.pango_language_from_string.argtypes = [ensureBytes]
pango.pango_context_set_language.argtypes = [ctypes.POINTER(PangoContext), ensureBytes]
pangocairo.pango_cairo_create_context.restype = ctypes.POINTER(PangoContext)

pango.pango_layout_new.restype = ctypes.POINTER(PangoLayout)
pango.pango_font_description_new.restype = ctypes.POINTER(PangoFontDescription)
pango.pango_font_description_set_family.argtypes = [ctypes.POINTER(PangoFontDescription), ensureBytes]
pango.pango_layout_set_markup.argtypes = [ctypes.POINTER(PangoLayout), ensureBytes, ctypes.c_int]


class LineGenerator(object):
    """
    Produces degraded line images using a single collection of font families.
    """
    def __init__(self, family='Sans', font_size=32, font_weight=400, language=None):
        self.language = language
        self.font = pango.pango_font_description_new()
        # XXX: get PANGO_SCALE programatically from somewhere
        pango.pango_font_description_set_size(self.font, font_size * 1024)
        pango.pango_font_description_set_family(self.font, family)
        pango.pango_font_description_set_weight(self.font, font_weight)

    def render_line(self, text):
        """
        Draws a line onto a Cairo surface which will be converted to an pillow
        Image.

        Args:
            text (unicode): A string which will be rendered as a single line.

        Returns:
            PIL.Image of mode 'L'.

        Raises:
            KrakenCairoSurfaceException if the Cairo surface couldn't be created
            (usually caused by invalid dimensions.
        """
        temp_surface = cairo.cairo_image_surface_create(0, 0, 0)
        width, height = _draw_on_surface(temp_surface, self.font, self.language, text)
        cairo.cairo_surface_destroy(temp_surface)
        if width == 0 or height == 0:
            raise KrakenCairoSurfaceException('Surface zero pixels in at least one dimension', width, height)
        real_surface = cairo.cairo_image_surface_create(0, width, height)
        _draw_on_surface(real_surface, self.font, self.language, text)
        data = cairo.cairo_image_surface_get_data(real_surface)
        size = int(4 * width * height)
        buffer = ctypes.create_string_buffer(size)
        ctypes.memmove(buffer, data, size)
        im = Image.frombuffer("RGBA", (width, height), buffer, "raw", "BGRA", 0, 1)
        cairo.cairo_surface_destroy(real_surface)
        im = im.convert('L')
        im = ImageOps.expand(im, 5, 255)
        return im


def _draw_on_surface(surface, font, language, text):

    cr = cairo.cairo_create(surface)
    pangocairo_ctx = pangocairo.pango_cairo_create_context(cr)
    layout = pango.pango_layout_new(pangocairo_ctx)

    pango_ctx = pango.pango_layout_get_context(layout)
    if language is not None:
        pango_language = pango.pango_language_from_string(language)
        pango.pango_context_set_language(pango_ctx, pango_language)

    pango.pango_layout_set_font_description(layout, font)

    cairo.cairo_set_source_rgb(cr, ctypes.c_double(1.0), ctypes.c_double(1.0), ctypes.c_double(1.0))
    cairo.cairo_paint(cr)

    pango.pango_layout_set_markup(layout, text, -1)

    cairo.cairo_set_source_rgb(cr, ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0))
    pangocairo.pango_cairo_update_layout(cr, layout)
    pangocairo.pango_cairo_show_layout(cr, layout)

    cairo.cairo_destroy(cr)

    ink_rect = PangoRectangle()
    logical_rect = PangoRectangle()
    pango.pango_layout_get_pixel_extents(layout, ctypes.byref(ink_rect), ctypes.byref(logical_rect))

    return max(ink_rect.width, logical_rect.width), max(ink_rect.height, logical_rect.height)


def ocropy_degrade(im, distort=1.0, dsigma=20.0, eps=0.03, delta=0.3, degradations=[(0.5, 0.0, 0.5, 0.0)]):
    """
    Degrades and distorts a line using the same noise model used by ocropus.

    Args:
        im (PIL.Image): Input image
        distort (float):
        dsigma (float):
        eps (float):
        delta (float):
        degradations (list): list returning 4-tuples corresponding to
                             the degradations argument of ocropus-linegen.

    Returns:
        PIL.Image in mode 'L'
    """
    w, h = im.size
    # XXX: determine correct output shape from transformation matrices instead
    # of guesstimating.
    image = Image.new('L', (int(1.5*w), 4*h), 255)
    image.paste(im, (int((image.size[0] - w) / 2), int((image.size[1] - h) / 2)))
    a = pil2array(image.convert('L'))
    (sigma, ssigma, threshold, sthreshold) = degradations[np.random.choice(len(degradations))]
    sigma += (2 * np.random.rand() - 1) * ssigma
    threshold += (2 * np.random.rand() - 1) * sthreshold
    a = a * 1.0 / np.amax(a)
    if sigma > 0.0:
        a = gaussian_filter(a, sigma)
    a += np.clip(np.random.randn(*a.shape) * 0.2, -0.25, 0.25)
    m = np.array([[1 + eps * np.random.randn(), 0.0], [eps * np.random.randn(), 1.0 + eps * np.random.randn()]])
    w, h = a.shape
    c = np.array([w / 2.0, h / 2])
    d = c - np.dot(m, c) + np.array([np.random.randn() * delta, np.random.randn() * delta])
    a = affine_transform(a, m, offset=d, order=1, mode='constant', cval=a[0, 0])
    a = np.array(a > threshold, 'f')
    [[r, c]] = find_objects(np.array(a == 0, 'i'))
    r0 = r.start
    r1 = r.stop
    c0 = c.start
    c1 = c.stop
    a = a[r0 - 5:r1 + 5, c0 - 5:c1 + 5]
    if distort > 0:
        h, w = a.shape
        hs = np.random.randn(h, w)
        ws = np.random.randn(h, w)
        hs = gaussian_filter(hs, dsigma)
        ws = gaussian_filter(ws, dsigma)
        hs *= distort / np.amax(hs)
        ws *= distort / np.amax(ws)

        def f(p):
            return (p[0] + hs[p[0], p[1]], p[1] + ws[p[0], p[1]])

        a = geometric_transform(a, f, output_shape=(h, w), order=1, mode='constant', cval=np.amax(a))
    im = array2pil(a).convert('L')
    return im


def degrade_line(im, mean=0.0, sigma=0.001, density=0.002):
    """
    Degrades a line image by adding several kinds of noise.

    Args:
        im (PIL.Image): Input image
        mean (float): Mean of distribution for Gaussian noise
        sigma (float): Standard deviation for Gaussian noise
        density (float): Noise density for Salt and Pepper noiase

    Returns:
        PIL.Image in mode 'L'
    """
    im = pil2array(im)
    m = np.amax(im)
    im = gaussian_filter(im.astype('f')/m, 0.5)
    im += np.random.normal(mean, sigma, im.shape)
    flipped = np.ceil(density/2 * im.size)
    coords = [np.random.randint(0, i - 1, int(flipped)) for i in im.shape]
    im[coords] = 255
    coords = [np.random.randint(0, i - 1, int(flipped)) for i in im.shape]
    im[coords] = 0
    return array2pil(np.clip(im * m, 0, 255).astype('uint8'))


def distort_line(im, distort=3.0, sigma=10, eps=0.03, delta=0.3):
    """
    Distorts a line image.

    Run BEFORE degrade_line as a white border of 5 pixels will be added.

    Args:
        im (PIL.Image): Input image
        distort (float):
        sigma (float):
        eps (float):
        delta (float):

    Returns:
        PIL.Image in mode 'L'
    """
    w, h = im.size
    # XXX: determine correct output shape from transformation matrices instead
    # of guesstimating.
    image = Image.new('L', (int(1.5*w), 4*h), 255)
    image.paste(im, (int((image.size[0] - w) / 2), int((image.size[1] - h) / 2)))
    line = pil2array(image.convert('L'))

    # shear in y direction with factor eps * randn(), scaling with 1 + eps *
    # randn() in x/y axis (all offset at d)
    m = np.array([[1 + eps * np.random.randn(), 0.0], [eps * np.random.randn(), 1.0 + eps * np.random.randn()]])
    c = np.array([w/2.0, h/2])
    d = c - np.dot(m, c) + np.array([np.random.randn() * delta, np.random.randn() * delta])
    line = affine_transform(line, m, offset=d, order=1, mode='constant', cval=255)

    hs = gaussian_filter(np.random.randn(4*h, int(1.5*w)), sigma)
    ws = gaussian_filter(np.random.randn(4*h, int(1.5*w)), sigma)
    hs *= distort/np.amax(hs)
    ws *= distort/np.amax(ws)

    def f(p):
        return (p[0] + hs[p[0], p[1]], p[1] + ws[p[0], p[1]])

    im = array2pil(geometric_transform(line, f, order=1, mode='nearest'))
    im = im.crop(ImageOps.invert(im).getbbox())
    im = ImageOps.expand(im, 5, 255)
    return im
