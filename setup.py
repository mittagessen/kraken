#!/usr/bin/env python
from setuptools import setup, find_packages

requirements = []
try:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
except IOError as e:
    print(e)

setup(
    name='krakenocr',
    zip_safe=False,
    packages=find_packages(),
    package_dir={},
    include_package_data=True,
    author="Shreejan Shrestha",
    # entry_points={"console_scripts": ["paddleocr= paddleocr.paddleocr:main"]},
    version="1.0",
    install_requires=requirements,
    license='Apache License 2.0',
    # long_description=readme(),
    long_description_content_type='text/markdown',
    keywords=[
        'krakenOCR textdetection textrecognition krakenocr clstm'
    ],
    classifiers=[
        'Intended Audience :: Developers', 'Operating System :: OS Independent',
        'Natural Language :: English (Simplified)',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ], )
