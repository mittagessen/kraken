# -*- coding: utf-8 -*-
#
# Copyright 2018 Benjamin Kiessling
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
kraken.lib.log
~~~~~~~~~~~~~~~~~

Handlers and formatters for logging.
"""
import time
import click
import logging


class LogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        level = record.levelname.lower()
        err = level in ('warning', 'error', 'exception', 'critical')
        click.echo(msg, err=err)


class LogFormatter(logging.Formatter):
    colors = {
        'error': dict(fg='red'),
        'exception': dict(fg='red'),
        'critical': dict(fg='red'),
        'warning': dict(fg='yellow'),
    }

    st_time = time.time()

    def format(self, record):
        if not record.exc_info:
            level = record.levelname.lower()
            msg = record.msg
            if level in self.colors:
                style = self.colors[level]
            else:
                style = {}
            msg = click.style('[{:2.4f}] {} '.format(time.time() - self.st_time, str(msg)), **style)
            return msg
        return logging.Formatter.format(self, record)


def progressbar(*args, **kwargs):
    """
    Slight extension to click's progressbar disabling output on when log level
    is set below 30.
    """
    import logging
    logger = logging.getLogger(__name__)
    bar = click.progressbar(*args, **kwargs)
    if logger.getEffectiveLevel() < 30:
        bar.is_hidden = True  # type: ignore
    return bar


def set_logger(logger=None, level=logging.ERROR):
    handler = LogHandler()
    handler.setFormatter(LogFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
