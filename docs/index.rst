kraken
======

.. toctree::
   :hidden:
   :maxdepth: 2

   advanced
   Training <ketos>
   API Tutorial <api>
   API Reference <api_docs>
   Models <models>

kraken is a turn-key OCR system optimized for historical and non-Latin script
material.

Features
========

kraken's main features are:

  - Fully trainable :ref:`layout analysis <segtrain>`, :ref:`reading order <rotrain>`, and :ref:`character recognition <predtrain>`
  - `Right-to-Left <https://en.wikipedia.org/wiki/Right-to-left>`_, `BiDi
    <https://en.wikipedia.org/wiki/Bi-directional_text>`_, and Top-to-Bottom
    script support
  - `ALTO <https://www.loc.gov/standards/alto/>`_, PageXML, abbyyXML, and hOCR
    output
  - Word bounding boxes and character cuts
  - Multi-script recognition support
  - :ref:`Public repository <repo>` of model files
  - :ref:`Variable recognition network architectures <vgsl>`

Pull requests and code contributions are always welcome.

Installation
============

Kraken can be run on Linux or Mac OS X (both x64 and ARM). Installation through
the on-board *pip* utility and the `anaconda <https://anaconda.org>`_
scientific computing python are supported.

Installation using Pip
----------------------

.. code-block:: console

  $ pip install kraken

or by running pip in the git repository:

.. code-block:: console

  $ pip install .

If you want direct PDF and multi-image TIFF/JPEG2000 support it is necessary to
install the `pdf` extras package for PyPi:

.. code-block:: console

   $ pip install kraken[pdf]

or

.. code-block:: console

   $ pip install .[pdf]

respectively.

Installation using Conda
------------------------

To install the stable version through `conda <https://anaconda.org>`_:

.. code-block:: console

   $ conda install -c conda-forge -c mittagessen kraken

Again PDF/multi-page TIFF/JPEG2000 support requires some additional dependencies:

.. code-block:: console

   $ conda install -c conda-forge pyvips

The git repository contains some environment files that aid in setting up the latest development version:

.. code-block:: console

  $ git clone https://github.com/mittagessen/kraken.git
  $ cd kraken
  $ conda env create -f environment.yml

or:

.. code-block:: console

  $ git clone https://github.com/mittagessen/kraken.git
  $ cd kraken
  $ conda env create -f environment_cuda.yml

for CUDA acceleration with the appropriate hardware.

Finding Recognition Models
--------------------------

Finally you'll have to scrounge up a model to do the actual recognition of
characters. To download the default model for printed French text and place it
in the kraken directory for the current user:

::

  $ kraken get 10.5281/zenodo.10592716


A list of libre models available in the central repository can be retrieved by
running:

.. code-block:: console

  $ kraken list

Model metadata can be extracted using:

.. code-block:: console

  $ kraken show 10.5281/zenodo.10592716
  name: 10.5281/zenodo.10592716

  CATMuS-Print (Large, 2024-01-30) - Diachronic model for French prints and other languages

  <p><strong>CATMuS-Print (Large) - Diachronic model for French prints and other West European languages</strong></p>
  <p>CATMuS (Consistent Approach to Transcribing ManuScript) Print is a Kraken HTR model trained on data produced by several projects, dealing with different languages (French, Spanish, German, English, Corsican, Catalan, Latin, Italian&hellip;) and different centuries (from the first prints of the 16th c. to digital documents of the 21st century).</p>
  <p>Transcriptions follow graphematic principles and try to be as compatible as possible with guidelines previously published for French: no ligature (except those that still exist), no allographetic variants (except the long s), and preservation of the historical use of some letters (u/v, i/j). Abbreviations are not resolved. Inconsistencies might be present, because transcriptions have been done over several years and the norms have slightly evolved.</p>
  <p>The model is trained with NFKD Unicode normalization: each diacritic (including superscripts) are transcribed as their own characters, separately from the "main" character.</p>
  <p>This model is the result of the collaboration from researchers from the University of Geneva and Inria Paris and will be consolidated under the CATMuS Medieval Guidelines in an upcoming paper.</p>
  scripts: Latn
  alphabet: !"#$%&'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz|}~¡£¥§«¬°¶·»¿ÆßæđłŒœƀǝɇΑΒΓΔΕΖΘΙΚΛΜΝΟΠΡΣΤΥΦΧΩαβγδεζηθικλμνξοπρςστυφχωϛחלרᑕᗅᗞᚠẞ–—‘’‚“”„‟†•⁄⁊⁋℟←▽◊★☙✠✺✻⟦⟧⬪ꝑꝓꝗꝙꝟꝯꝵ SPACE, COMBINING GRAVE ACCENT, COMBINING ACUTE ACCENT, COMBINING CIRCUMFLEX ACCENT, COMBINING TILDE, COMBINING MACRON, COMBINING DOT ABOVE, COMBINING DIAERESIS, COMBINING RING ABOVE, COMBINING COMMA ABOVE, COMBINING REVERSED COMMA ABOVE, COMBINING CEDILLA, COMBINING OGONEK, COMBINING GREEK PERISPOMENI, COMBINING GREEK YPOGEGRAMMENI, COMBINING LATIN SMALL LETTER I, COMBINING LATIN SMALL LETTER U, 0xe682, 0xe68b, 0xe8bf, 0xf1a7
  accuracy: 98.56%
  license: cc-by-4.0
  author(s): Gabay, Simon; Clérice, Thibault
  date: 2024-01-30

Quickstart
==========

The structure of an OCR software consists of multiple steps, primarily
preprocessing, segmentation, and recognition, each of which takes the output of
the previous step and sometimes additional files such as models and templates
that define how a particular transformation is to be performed.

In kraken these are separated into different subcommands that can be chained or
ran separately:

.. raw:: html
    :file: _static/kraken_workflow.svg

Recognizing text on an image using the default parameters including the
prerequisite step of page segmentation:

.. code-block:: console

  $ kraken -i image.tif image.txt segment -bl ocr -m catmus-print-fondue-large.mlmodel
  Loading RNN     ✓
  Processing      ⣻

To segment an image into reading-order sorted baselines and regions:

.. code-block:: console

  $ kraken -i bw.tif lines.json segment -bl

To OCR an image using the previously downloaded model:

.. code-block:: console

  $ kraken -i bw.tif image.txt segment -bl ocr -m catmus-print-fondue-large.mlmodel

To OCR an image using the default model and serialize the output using the ALTO
template:

.. code-block:: console

  $ kraken -a -i bw.tif image.txt segment -bl ocr -m catmus-print-fondue-large.mlmodel

All commands and their parameters are documented, just add the standard
``--help`` flag for further information.

Training Tutorial
=================

There is a training tutorial at :doc:`training`.

Related Software
================

These days kraken is quite closely linked to the `eScriptorium
<https://gitlab.com/scripta/escriptorium/>`_ project developed in the same eScripta research
group. eScriptorium provides a user-friendly interface for annotating data,
training models, and inference (but also much more). There is a `gitter channel
<https://gitter.im/escripta/escriptorium>`_ that is mostly intended for
coordinating technical development but is also a spot to find people with
experience on applying kraken on a wide variety of material.

.. _license:

License
=======

``Kraken`` is provided under the terms and conditions of the `Apache 2.0
License <https://github.com/mittagessen/kraken/blob/main/LICENSE>`_.

Funding
=======

kraken is developed at the `École Pratique des Hautes Études <https://www.ephe.psl.eu>`_, `Université PSL <https://www.psl.eu>`_.


.. container:: twocol

   .. container:: leftside

        .. image:: _static/normal-reproduction-low-resolution.jpg
          :width: 100
          :alt: Co-financed by the European Union

   .. container:: rightside

        This project was partially funded through the RESILIENCE project, funded from
        the European Union’s Horizon 2020 Framework Programme for Research and
        Innovation.


.. container:: twocol

   .. container:: leftside

      .. image:: https://projet.biblissima.fr/sites/default/files/2021-11/biblissima-baseline-sombre-ia.png
         :width: 300
         :alt: Received funding from the Programme d’investissements d’Avenir

   .. container:: rightside

        Ce travail a bénéficié d’une aide de l’État gérée par l’Agence Nationale de la
        Recherche au titre du Programme d’Investissements d’Avenir portant la référence
        ANR-21-ESRE-0005 (Biblissima+).


