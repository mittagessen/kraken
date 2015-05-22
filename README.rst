Description
===========

.. image:: https://travis-ci.org/mittagessen/kraken.svg
    :target: https://travis-ci.org/mittagessen/kraken

kraken is a fork of ocropus intended to rectify a number of issues while
preserving (mostly) functional equivalence. Its main goals are:

  - Explicit input/output handling ✓
  - Clean public API 
  - Word bounding boxes in hOCR ✓
  - Tests
  - Removal of runtime dependency on gcc ✓
  - Removal of unused spaghetti code ✓

Ticked of goals have been realized while some others still require further
work. Pull requests and code contributions are always welcome.

Installation
============

While kraken does not require a working C compiler on run-time anymore numpy
and scipy compilation still requires build-essential or your distributions
equivalent. Because the build behavior of pip versions older than 6.1.0
interferes with the scipy build process numpy has to be installed before doing
the actual install:

::

  # pip install numpy

Install kraken either from pypi:

::

  $ pip install kraken

or by running pip in the git repository:

::

  $ pip install .

Finally you'll have to scrounge up an RNN to do the actual recognition of
characters. To download ocropus' default RNN and place it in the kraken
directory for the current user:

::

  $ kraken download

Quickstart
==========

To binarize a single image using the nlbin algorithm:

::

  $ kraken binarize grey.png bw.png

To segment a binarized image into reading-order sorted lines:

::

  $ kraken segment bw.png lines.txt

To OCR a binarized image using the default RNN and the previously generated
page segmentation:

::

  $ kraken ocr --lines lines.txt bw.png

Documentation
=============

Have a look at the `documentation <http://mittagessen.github.io/kraken>`_
