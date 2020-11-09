Description
===========

.. image:: https://travis-ci.org/mittagessen/kraken.svg?branch=master
    :target: https://travis-ci.org/mittagessen/kraken

kraken is a turn-key OCR system optimized for historical and non-Latin script
material.

kraken's main features are:

  - Fully trainable layout analysis and character recognition
  - `Right-to-Left <https://en.wikipedia.org/wiki/Right-to-left>`_, `BiDi
    <https://en.wikipedia.org/wiki/Bi-directional_text>`_, and Top-to-Bottom
    script support
  - `ALTO <https://www.loc.gov/standards/alto/>`_, PageXML, abbyXML, and hOCR
    output
  - Word bounding boxes and character cuts
  - Multi-script recognition support
  - `Public repository <https://zenodo.org/communities/ocr_models>`_ of model files
  - Lightweight model files
  - Variable recognition network architectures

Installation
============

When using a recent version of pip all dependencies will be installed from
binary wheel packages, so installing build-essential or your distributions
equivalent is often unnecessary. kraken only runs on **Linux or Mac OS X**.
Windows is not supported.

Install the latest development version through `conda <https://anaconda.org>`_:

::

  $ wget https://raw.githubusercontent.com/mittagessen/kraken/master/environment.yml
  $ conda env create -f environment.yml

or:

::

  $ wget https://raw.githubusercontent.com/mittagessen/kraken/master/environment_cuda.yml
  $ conda env create -f environment_cuda.yml

for CUDA acceleration with the appropriate hardware.

It is also possible to install the latest stable release from pypi:

::

  $ pip install kraken

Finally you'll have to scrounge up a model to do the actual recognition of
characters. To download the default model for printed English text and place it
in the kraken directory for the current user:

::

  $ kraken get 10.5281/zenodo.2577813 

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

To segment an image (binarized or not) with the new baseline segmenter:

::

  $ kraken -i image.tif lines.json segment -bl
 

To segment and OCR an image using the default model(s):

::

  $ kraken -i image.tif image.txt segment -bl ocr

All subcommands and options are documented. Use the ``help`` option to get more
information.

Documentation
=============

Have a look at the `docs <http://kraken.re>`_

Funding
=======

kraken is developed at the `École Pratique des Hautes Études <http://ephe.fr>`_, `Université PSL <http://www.psl.eu>`_.
