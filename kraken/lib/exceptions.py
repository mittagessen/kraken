# -*- coding: utf-8 -*-
"""
kraken.lib.exceptions
~~~~~~~~~~~~~~~~~~~~~

All custom exceptions raised by kraken's modules and packages. Packages should
always define their exceptions here.
"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals


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
