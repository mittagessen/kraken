Description
===========

.. image:: https://travis-ci.org/mittagessen/kraken.svg?branch=master
    :target: https://travis-ci.org/mittagessen/kraken

kraken is a fork of ocropus intended to rectify a number of issues while
preserving (mostly) functional equivalence. Its main goals are:

  - Explicit input/output handling ✓
  - Word and character bounding boxes in hOCR ✓
  - Removal of runtime dependency on gcc ✓
  - `clstm <https://github.com/tmbdev/clstm>`_ compatibility ✓
  - Right-to-left/BiDi support ✓
  - Clean public API 
  - Tests

Ticked of goals have been realized while some others still require further
work. Pull requests and code contributions are always welcome.

Installation
============

kraken does not require a working C compiler on run-time anymore. When using a
recent version of pip all dependencies will be installed from binary wheel
packages, so installing build-essential or your distributions equivalent is
often unnecessary.

``clstm`` is supported through automatically installed binary wheels now, that
should work on most Linux systems except for non-x86 architectures. If the
install process fails because the fallback source compilation does not work
refer to the `documentation
<https://github.com/tmbdev/clstm/blob/master/README.md>`_ to install build
dependencies.

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

  $ kraken get default

A list of libre models available in the central repository can be retrieved by
running:

::

  $ kraken list

Quickstart
==========

Recognizing text on an image using the default parameters including the
prerequisite steps of binarization and page segmentation:

::

  $ kraken -i image.tif image.txt binarize segment ocr

To binarize a single image using the nlbin algorithm:

::

  $ kraken -i image.tif bw.tif binarize

To segment a binarized image into reading-order sorted lines:

::

  $ kraken -i bw.tif lines.txt segment bw.png

To OCR a binarized image using the default RNN and the previously generated
page segmentation:

::

  $ kraken -i bw.tif image.txt ocr --lines lines.txt

All subcommands and options are documented. Use the ``help`` option to get more
information.

Documentation
=============

Have a look at the `documentation <http://kraken.re>`_
