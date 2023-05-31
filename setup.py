#!/usr/bin/env python
from setuptools import setup, find_packages

requirements = []
try:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
except IOError as e:
    print(e)

def readme():
    with open('doc/doc_en/whl_en.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README


setup(
    name='krakenocr',
    packages=find_packages(),
    package_dir={'kraken': ''},
    include_package_data=True,
    author="Shreejan Shrestha",
    # entry_points={"console_scripts": ["paddleocr= paddleocr.paddleocr:main"]},
    version="1.0",
    install_requires=requirements,
    license='Apache License 2.0',
    long_description=readme(),
    long_description_content_type='text/markdown',
    keywords=[
        'ocr textdetection textrecognition krakenocr clstm'
    ],
    classifiers=[
        'Intended Audience :: Developers', 'Operating System :: OS Independent',
        'Natural Language :: English (Simplified)',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ], )
