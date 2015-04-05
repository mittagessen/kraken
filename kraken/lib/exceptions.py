# -*- coding: utf-8 -*-
"""
kraken.lib.exceptions
~~~~~~~~~~~~~~~~~~~~~

All custom exceptions raised by kraken's modules and packages. Packages should
always define their exceptions here.
"""

class KrakenRecordException(Exception):

    def __init__(self, message=None):
        Exception.__init__(self, message)

