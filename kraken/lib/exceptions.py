# -*- coding: utf-8 -*-
"""
kraken.lib.exceptions
~~~~~~~~~~~~~~~~~~~~~

All custom exceptions raised by kraken's modules and packages. Packages should
always define their exceptions here.
"""

class KrakenCodecException(Exception):

    def __init__(self, message=None):
        Exception.__init__(self, message)


class KrakenStopTrainingException(Exception):

    def __init__(self, message=None):
        Exception.__init__(self, message)


class KrakenEncodeException(Exception):

    def __init__(self, message=None):
        Exception.__init__(self, message)


class KrakenRecordException(Exception):

    def __init__(self, message=None):
        Exception.__init__(self, message)


class KrakenInvalidModelException(Exception):

    def __init__(self, message=None):
        Exception.__init__(self, message)


class KrakenInputException(Exception):

    def __init__(self, message=None):
        Exception.__init__(self, message)


class KrakenRepoException(Exception):

    def __init__(self, message=None):
        Exception.__init__(self, message)


class KrakenCairoSurfaceException(Exception):
    """
    Raised when the Cairo surface couldn't be created.

    Attributes:
        message (str): Error message
        width (int): Width of the surface
        height (int): Height of the surface
    """
    def __init__(self, message: str, width: int, height: int) -> None:
        self.message = message
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return repr(self.message)
