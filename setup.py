#!/usr/bin/env python

from setuptools import setup

setup(
    include_package_data=True,
    test_suite="nose.collector",
    tests_require="nose",
    setup_requires=['pbr'],
    pbr=True,
)
