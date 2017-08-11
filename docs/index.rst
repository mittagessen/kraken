kraken
======

.. toctree::
   :hidden:
   :maxdepth: 2

   advanced
   Training <ketos>
   API <api>
   Models <models>

kraken is a turn-key OCR system forked from `ocropus
<https://github.com/tmbdev/ocropy>`_. It is intended to rectify a number of
issues while preserving (mostly) functional equivalence. 

If you already got a model trained for ocropus you can always expect it to work
with kraken without all the fuss of the original ocropus tools.

Features
========

kraken's main features are:

  - Script detection and multi-script recognition support
  - `Right-to-Left <https://en.wikipedia.org/wiki/Right-to-left>`_, `BiDi
    <https://en.wikipedia.org/wiki/Bi-directional_text>`_, and Top-to-Bottom
    script support
  - `ALTO <https://www.loc.gov/standards/alto/>`_ and hOCR output
  - Word bounding boxes and character cuts
  - `Public repository <https://github.com/mittagessen/kraken-models>`_ of model files
  - :ref:`Lightweight model files <models>`
  - Support for `clstm <https://github.com/tmbdev/clstm>`_ models

Currently missing or underdocumented are:

  - Tests
  - New training interface

All functionality not pertaining to OCR and prerequisite steps has been
removed, i.e. no more error rate measuring, etc.

Pull requests and code contributions are always welcome. 

Installation
============

.. note::

  A vagrant virtual machine is also available on the hashicorp atlas. Run
  ``vagrant init openphilology/kraken && vagrant up`` to install without doing
  any compiling yourself.

kraken requires some external libraries to run. On Debian/Ubuntu they may be
installed using:

.. code-block:: console

        # apt install libpangocairo-1.0 libxml2 libblas3 liblapack3 python-dev python-pip

If `clstm <https://github.com/tmbdev/clstm>`_ support is desired (highly
recommended) the associated python extension has to be build and installed.

Because the build behavior of pip versions older than 6.1.0 interferes with the
scipy build process numpy has to be installed before doing the actual install:

.. code-block:: console

  $ pip install numpy

Install kraken either from pypi:

.. code-block:: console

  $ pip install kraken

or by running pip in the git repository:

.. code-block:: console

  $ pip install .

.. note::

  While kraken is Python 2/3 compliant, there are limits to its compatibility.
  For various reasons it is not possible to use :ref:`pickled models
  <pyrnn>` under Python 3. As the vast majority of models are still in
  the legacy format it is recommended to use Python 2.7. On the other hand all
  models in the central repository are converted to the fully upward compatible
  pronn format.

Finally you'll have to scrounge up an RNN to do the actual recognition of
characters. To download ocropus' default RNN converted to the new format and
place it in the kraken directory for the current user:

.. code-block:: console

  $ kraken get default

A list of libre models available in the central repository can be retrieved by
running:

::

  $ kraken list

Model metadata can be extracted using:

::

  $ kraken show arabic-alam-al-kutub
  name: arabic-alam-al-kutub.clstm

  An experimental model for Classical Arabic texts.

  Network trained on 889 lines of [0] as a test case for a general Classical
  Arabic model. Ground truth was prepared by Sarah Savant
  <sarah.savant@aku.edu> and Maxim Romanov <maxim.romanov@uni-leipzig.de>.

  Vocalization was omitted in the ground truth. Training was stopped at ~35000
  iterations with an accuracy of 97%.

  [0] Ibn al-Faqīh (d. 365 AH). Kitāb al-buldān. Edited by Yūsuf al-Hādī, 1st
  edition. Bayrūt: ʿĀlam al-kutub, 1416 AH/1996 CE.
  alphabet:  !()-.0123456789:[] «»،؟ءابةتثجحخدذرزسشصضطظعغفقكلمنهوىي ARABIC
  MADDAH ABOVE, ARABIC HAMZA ABOVE, ARABIC HAMZA BELOW

Quickstart
==========

Recognizing text on an image using the default parameters including the
prerequisite steps of binarization and page segmentation:

::

  $ kraken -i image.tif image.txt binarize segment ocr
  Loading RNN     ✓
  Processing      ⣻

To binarize a single image using the nlbin algorithm:

::

  $ kraken -i image.tif bw.tif binarize

To segment a binarized image into reading-order sorted lines:

::

  $ kraken -i bw.tif lines.json segment

To OCR a binarized image using the default RNN and the previously generated
page segmentation:

::

  $ kraken -i bw.tif image.txt ocr --lines lines.json

All commands and their parameters are documented, just add the standard
``--help`` flag for further information.

Training Tutorial
=================

There is a training tutorial at :doc:`training`.

.. _license:

License
=======

``Kraken`` is provided under the terms and conditions of the `Apache 2.0
License <https://github.com/mittagessen/kraken/blob/master/LICENSE>`_ retained
from the original ``ocropus`` distribution.
