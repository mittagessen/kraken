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

Line degradation uses a local model described in [1].

[0] https://github.com/googlei18n/nototools
[1] Kanungo, Tapas, et al. "A statistical, nonparametric methodology for document degradation model validation." IEEE Transactions on Pattern Analysis and Machine Intelligence 22.11 (2000): 1209-1223.

"""

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import distance_transform_cdt, binary_closing

from scipy.ndimage.interpolation import affine_transform, geometric_transform
from PIL import Image, ImageOps

from typing import AnyStr

import logging
import ctypes
import ctypes.util
import numpy as np

from kraken.lib.exceptions import KrakenCairoSurfaceException
from kraken.lib.util import pil2array, array2pil

logger = logging.getLogger(__name__)

pc_lib = ctypes.util.find_library('pangocairo-1.0')
p_lib = ctypes.util.find_library('pango-1.0')
c_lib = ctypes.util.find_library('cairo')
if pc_lib is None:
    raise ImportError('Couldnt load pangocairo line generator dependency. Please install pangocairo, pango, and cairo.')
if p_lib is None:
    raise ImportError('Couldnt load pango line generator dependency. Please install pangocairo, pango, and cairo.')
if c_lib is None:
    raise ImportError('Couldnt load cairo line generator dependency. Please install pangocairo, pango, and cairo.')
pangocairo = ctypes.CDLL(pc_lib)
pango = ctypes.CDLL(p_lib)
cairo = ctypes.CDLL(c_lib)


__all__ = ['LineGenerator', 'ocropy_degrade', 'degrade_line', 'distort_line']


class CairoSurface(ctypes.Structure):
    pass


class CairoContext(ctypes.Structure):
    pass


class PangoFontDescription(ctypes.Structure):
    pass


class PangoLanguage(ctypes.Structure):
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
    def from_param(cls, value: AnyStr) -> bytes:
        if isinstance(value, bytes):
            return value
        else:
            return value.encode('utf-8')


cairo.cairo_create.argtypes = [ctypes.POINTER(CairoSurface)]
cairo.cairo_create.restype = ctypes.POINTER(CairoContext)

cairo.cairo_destroy.argtypes = [ctypes.POINTER(CairoContext)]

cairo.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
cairo.cairo_image_surface_create.restype = ctypes.POINTER(CairoSurface)

cairo.cairo_surface_destroy.argtypes = [ctypes.POINTER(CairoSurface)]

cairo.cairo_image_surface_get_data.restype = ctypes.c_void_p

cairo.cairo_set_source_rgb.argtypes = [ctypes.POINTER(CairoContext), ctypes.c_double, ctypes.c_double, ctypes.c_double]

cairo.cairo_paint.argtypes = [ctypes.POINTER(CairoContext)]


pangocairo.pango_cairo_create_context.argtypes = [ctypes.POINTER(CairoContext)]
pangocairo.pango_cairo_create_context.restype = ctypes.POINTER(PangoContext)

pangocairo.pango_cairo_update_layout.argtypes = [ctypes.POINTER(CairoContext), ctypes.POINTER(PangoLayout)]
pangocairo.pango_cairo_show_layout.argtypes = [ctypes.POINTER(CairoContext), ctypes.POINTER(PangoLayout)]

pango.pango_language_from_string.argtypes = [ensureBytes]  # type: ignore
pango.pango_language_from_string.restype = ctypes.POINTER(PangoLanguage)

pango.pango_context_set_language.argtypes = [ctypes.POINTER(PangoContext), ctypes.POINTER(PangoLanguage)]

pango.pango_font_description_new.restype = ctypes.POINTER(PangoFontDescription)
pango.pango_font_description_set_family.argtypes = [ctypes.POINTER(PangoFontDescription), ensureBytes]  # type: ignore
pango.pango_font_description_set_size.argtypes = [ctypes.POINTER(PangoFontDescription), ctypes.c_int]
pango.pango_font_description_set_weight.argtypes = [ctypes.POINTER(PangoFontDescription), ctypes.c_uint]

pango.pango_layout_new.restype = ctypes.POINTER(PangoLayout)
pango.pango_layout_set_markup.argtypes = [ctypes.POINTER(PangoLayout), ensureBytes, ctypes.c_int]  # type: ignore
pango.pango_layout_set_font_description.argtypes = [ctypes.POINTER(PangoLayout), ctypes.POINTER(PangoFontDescription)]
pango.pango_layout_get_context.argtypes = [ctypes.POINTER(PangoLayout)]
pango.pango_layout_get_context.restype = ctypes.POINTER(PangoContext)

pango.pango_layout_get_pixel_extents.argtypes = [ctypes.POINTER(PangoLayout), ctypes.POINTER(PangoRectangle), ctypes.POINTER(PangoRectangle)]


class LineGenerator(object):
    """
    Produces degraded line images using a single collection of font families.
    """
    def __init__(self, family='Sans', font_size=32, font_weight=400, language=None):
        self.language = language
        self.font = pango.pango_font_description_new()
        # XXX: get PANGO_SCALE programatically from somewhere
        logger.debug('Setting font {}, size {}, weight {}'.format(family, font_size, font_weight))
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
        logger.info('Rendering line \'{}\''.format(text))
        logger.debug('Creating temporary cairo surface')
        temp_surface = cairo.cairo_image_surface_create(0, 0, 0)
        width, height = _draw_on_surface(temp_surface, self.font, self.language, text)
        cairo.cairo_surface_destroy(temp_surface)
        if width == 0 or height == 0:
            logger.error('Surface for \'{}\' zero pixels in at least one dimension'.format(text))
            raise KrakenCairoSurfaceException('Surface zero pixels in at least one dimension', width, height)
        logger.debug('Creating sized cairo surface')
        real_surface = cairo.cairo_image_surface_create(0, width, height)
        _draw_on_surface(real_surface, self.font, self.language, text)
        logger.debug('Extracing data from real surface')
        data = cairo.cairo_image_surface_get_data(real_surface)
        size = int(4 * width * height)
        buffer = ctypes.create_string_buffer(size)
        ctypes.memmove(buffer, data, size)
        logger.debug('Loading data into PIL image')
        im = Image.frombuffer("RGBA", (width, height), buffer, "raw", "BGRA", 0, 1)
        cairo.cairo_surface_destroy(real_surface)
        logger.debug('Expand and grayscale image')
        im = im.convert('L')
        im = ImageOps.expand(im, 5, 255)
        return im


def _draw_on_surface(surface, font, language, text):

    logger.debug('Creating cairo and pangocairo contexts')
    cr = cairo.cairo_create(surface)
    pangocairo_ctx = pangocairo.pango_cairo_create_context(cr)
    logger.debug('Creating pangocairo layout')
    layout = pango.pango_layout_new(pangocairo_ctx)

    pango_ctx = pango.pango_layout_get_context(layout)
    if language is not None:
        logger.debug('Setting language {} on context'.format(language))
        pango_language = pango.pango_language_from_string(language)
        pango.pango_context_set_language(pango_ctx, pango_language)

    logger.debug('Setting font description on layout')
    pango.pango_layout_set_font_description(layout, font)

    logger.debug('Filling background of surface')
    cairo.cairo_set_source_rgb(cr, 1.0, 1.0, 1.0)
    cairo.cairo_paint(cr)

    logger.debug('Typsetting text')
    pango.pango_layout_set_markup(layout, text, -1)

    logger.debug('Drawing text')
    cairo.cairo_set_source_rgb(cr, 0.0, 0.0, 0.0)
    pangocairo.pango_cairo_update_layout(cr, layout)
    pangocairo.pango_cairo_show_layout(cr, layout)

    cairo.cairo_destroy(cr)

    logger.debug('Getting pixel extents')
    ink_rect = PangoRectangle()
    logical_rect = PangoRectangle()
    pango.pango_layout_get_pixel_extents(layout, ctypes.byref(ink_rect), ctypes.byref(logical_rect))

    return max(ink_rect.width, logical_rect.width), max(ink_rect.height, logical_rect.height)


def ocropy_degrade(im, distort=1.0, dsigma=20.0, eps=0.03, delta=0.3, degradations=((0.5, 0.0, 0.5, 0.0))):
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
    logger.debug('Pasting source image into canvas')
    image = Image.new('L', (int(1.5*w), 4*h), 255)
    image.paste(im, (int((image.size[0] - w) / 2), int((image.size[1] - h) / 2)))
    a = pil2array(image.convert('L'))
    logger.debug('Selecting degradations')
    (sigma, ssigma, threshold, sthreshold) = degradations[np.random.choice(len(degradations))]
    sigma += (2 * np.random.rand() - 1) * ssigma
    threshold += (2 * np.random.rand() - 1) * sthreshold
    a = a * 1.0 / np.amax(a)
    if sigma > 0.0:
        logger.debug('Apply Gaussian filter')
        a = gaussian_filter(a, sigma)
    logger.debug('Adding noise')
    a += np.clip(np.random.randn(*a.shape) * 0.2, -0.25, 0.25)
    logger.debug('Perform affine transformation and resize')
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
        logger.debug('Perform geometric transformation')
        h, w = a.shape
        hs = np.random.randn(h, w)
        ws = np.random.randn(h, w)
        hs = gaussian_filter(hs, dsigma)
        ws = gaussian_filter(ws, dsigma)
        hs *= distort / np.amax(hs)
        ws *= distort / np.amax(ws)

        def _f(p):
            return (p[0] + hs[p[0], p[1]], p[1] + ws[p[0], p[1]])

        a = geometric_transform(a, _f, output_shape=(h, w), order=1, mode='constant', cval=np.amax(a))
    im = array2pil(a).convert('L')
    return im


def degrade_line(im, eta=0.0, alpha=1.5, beta=1.5, alpha_0=1.0, beta_0=1.0):
    """
    Degrades a line image by adding noise.

    For parameter meanings consult [1].

    Args:
        im (PIL.Image): Input image
        eta (float):
        alpha (float):
        beta (float):
        alpha_0 (float):
        beta_0 (float):

    Returns:
        PIL.Image in mode '1'
    """
    logger.debug('Inverting and normalizing input image')
    im = pil2array(im)
    im = np.amax(im)-im
    im = im*1.0/np.amax(im)

    logger.debug('Calculating foreground distance transform')
    fg_dist = distance_transform_cdt(1-im, metric='taxicab')
    logger.debug('Calculating flip to white probability')
    fg_prob = alpha_0 * np.exp(-alpha * (fg_dist**2)) + eta
    fg_prob[im == 1] = 0
    fg_flip = np.random.binomial(1, fg_prob)

    logger.debug('Calculating background distance transform')
    bg_dist = distance_transform_cdt(im, metric='taxicab')
    logger.debug('Calculating flip to black probability')
    bg_prob = beta_0 * np.exp(-beta * (bg_dist**2)) + eta
    bg_prob[im == 0] = 0
    bg_flip = np.random.binomial(1, bg_prob)

    # flip
    logger.debug('Flipping')
    im -= bg_flip
    im += fg_flip

    logger.debug('Binary closing')
    sel = np.array([[1, 1], [1, 1]])
    im = binary_closing(im, sel)
    logger.debug('Converting to image')
    return array2pil(255-im.astype('B')*255)


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
    logger.debug('Pasting source image into canvas')
    image = Image.new('L', (int(1.5*w), 4*h), 255)
    image.paste(im, (int((image.size[0] - w) / 2), int((image.size[1] - h) / 2)))
    line = pil2array(image.convert('L'))

    # shear in y direction with factor eps * randn(), scaling with 1 + eps *
    # randn() in x/y axis (all offset at d)
    logger.debug('Performing affine transformation')
    m = np.array([[1 + eps * np.random.randn(), 0.0], [eps * np.random.randn(), 1.0 + eps * np.random.randn()]])
    c = np.array([w/2.0, h/2])
    d = c - np.dot(m, c) + np.array([np.random.randn() * delta, np.random.randn() * delta])
    line = affine_transform(line, m, offset=d, order=1, mode='constant', cval=255)

    hs = gaussian_filter(np.random.randn(4*h, int(1.5*w)), sigma)
    ws = gaussian_filter(np.random.randn(4*h, int(1.5*w)), sigma)
    hs *= distort/np.amax(hs)
    ws *= distort/np.amax(ws)

    def _f(p):
        return (p[0] + hs[p[0], p[1]], p[1] + ws[p[0], p[1]])

    logger.debug('Performing geometric transformation')
    im = array2pil(geometric_transform(line, _f, order=1, mode='nearest'))
    logger.debug('Cropping canvas to content box')
    im = im.crop(ImageOps.invert(im).getbbox())
    return im
