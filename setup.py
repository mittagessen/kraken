#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

from setuptools import setup

setup(
    include_package_data=True,
    test_suite="nose.collector",
    tests_require=['nose', 'hocr-spec'],
    setup_requires=['pbr'],
    pbr=True,
)
