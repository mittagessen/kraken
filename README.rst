Description
===========

.. image:: https://travis-ci.org/mittagessen/kraken.svg?branch=master
    :target: https://travis-ci.org/mittagessen/kraken

kraken is a fork of ocropus intended to rectify a number of issues while
preserving (mostly) functional equivalence. Its main features are:

  - Script detection and multiscript recognition support
  - `Right-to-Left <https://en.wikipedia.org/wiki/Right-to-left>`_, `BiDi
    <https://en.wikipedia.org/wiki/Bi-directional_text>`_, and Top-to-Bottom
    script support
  - `ALTO <https://www.loc.gov/standards/alto/>`_, abbyXML, and hOCR output
  - Word bounding boxes and character cuts
  - `Public repository <https://github.com/mittagessen/kraken-models>`_ of model files
  - Dynamic recognition model architectures and GPU acceleration
  - Clean public API 

Installation
============

When using a recent version of pip all dependencies will be installed from
binary wheel packages, so installing build-essential or your distributions
equivalent is often unnecessary. kraken only runs on **Linux or Mac OS X**.
Windows is not supported.

Install the latest 1.0 release through `conda <https://anaconda.org>`_:

::

  $ wget https://raw.githubusercontent.com/mittagessen/kraken/master/environment.yml
  $ conda env create -f environment.yml

or:

::

  $ wget https://raw.githubusercontent.com/mittagessen/kraken/master/environment_cuda.yml
  $ conda env create -f environment_cuda.yml

for CUDA acceleration with the appropriate hardware.

It is also possible to install the same version from pypi:

::

  $ pip install kraken

Finally you'll have to scrounge up a model to do the actual recognition of
characters. To download the default model for printed English text and place it
in the kraken directory for the current user:

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

  $ kraken -i image.tif bw.png binarize

To segment a binarized image into reading-order sorted lines:

::

  $ kraken -i bw.png lines.json segment

To OCR a binarized image using the default RNN and the previously generated
page segmentation:

::

  $ kraken -i bw.png image.txt ocr --lines lines.json

All subcommands and options are documented. Use the ``help`` option to get more
information.

Documentation
=============

Have a look at the `docs <http://kraken.re>`_

Funding
=======

kraken is developed at `Université PSL <http://www.psl.eu>`_.
