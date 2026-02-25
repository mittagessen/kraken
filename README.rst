Description
===========

.. image:: https://github.com/mittagessen/kraken/actions/workflows/test.yml/badge.svg
    :target: https://github.com/mittagessen/kraken/actions/workflows/test.yml

kraken is a turn-key OCR system optimized for historical and non-Latin script
material.

kraken's main features are:

  - Fully trainable layout analysis, reading order, and character recognition
  - `Right-to-Left <https://en.wikipedia.org/wiki/Right-to-left>`_, `BiDi
    <https://en.wikipedia.org/wiki/Bi-directional_text>`_, and Top-to-Bottom
    script support
  - `ALTO <https://www.loc.gov/standards/alto/>`_, PageXML, abbyyXML, and hOCR
    output
  - Word bounding boxes and character cuts
  - Multi-script recognition support
  - `Public repository <https://zenodo.org/communities/ocr_models>`_ of model files
  - Variable recognition network architecture

Installation
============

Kraken can be run on Linux or Mac OS X (both x64 and ARM). Installation is
through the on-board *pip* utility. To not pollute the global state of your
distribution's package manager it is recommended to use virtual environments.
If you do not have a setup or do not wish to handle virtual environments
yourself you can use `pipx`.

.. code-block:: console

   $ sudo apt install pipx
   $ pipx install kraken

kraken works both on Linux and Mac OS X and with any python interpreter between
3.10 and 3.13. It is possible the installation fails because `pipx` defaults to
an unsupported interpreter version. In that case you need to install a
compatible interpreter version such as 3.11 and then specify this version
explicitly:

.. code-block:: console

   $ sudo apt install python3.13-full
   $ pipx install --python python3.13 kraken


Installation using pip
----------------------

Create and activate a separate virtual environment using whatever tool you
like.

.. code-block:: console

  $ pip install kraken

or by running pip in the git repository:

.. code-block:: console

  $ pip install .

If you want direct PDF and multi-image TIFF/JPEG2000 support it is necessary to
install the `pdf` extras package for PyPi:

.. code-block:: console

   $ pip install kraken[pdf]

Finally you'll have to scrounge up a model to do the actual recognition of
characters. To download the default model for printed French text and place it
in the kraken directory for the current user:

::

  $ kraken get 10.5281/zenodo.10592716

A list of libre models available in the central repository can be retrieved by
running:

::

  $ kraken list

Tests can be run with `pytest`. This requires additional installations:

.. code-block:: console

  $ pip install ".[augment,test]"
  $ pytest

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

  $ kraken -i image.tif image.txt segment -bl ocr -m catmus-print-fondue-large.mlmodel

All subcommands and options are documented. Use the ``help`` option to get more
information.

Documentation
=============

Have a look at the `docs <https://kraken.re>`_.

Related Software
================

These days kraken is quite closely linked to the `eScriptorium
<https://gitlab.com/scripta/escriptorium/>`_ project developed in the same eScripta research
group. eScriptorium provides a user-friendly interface for annotating data,
training models, and inference (but also much more). There is a `gitter channel
<https://gitter.im/escripta/escriptorium>`_ that is mostly intended for
coordinating technical development but is also a spot to find people with
experience on applying kraken on a wide variety of material.

Funding
=======

kraken is developed at the `École Pratique des Hautes Études <https://www.ephe.psl.eu>`_, `Université PSL <https://www.psl.eu>`_.

.. container:: twocol

   .. container::

        .. image:: https://raw.githubusercontent.com/mittagessen/kraken/main/docs/_static/normal-reproduction-low-resolution.jpg
          :width: 100
          :alt: Co-financed by the European Union

   .. container::

        This project was funded in part by the European Union (ATRIUM, project
        number 101132163). This project was funded in part by the European Union
        (ERC, MiDRASH, project number 101071829). This project was partially
        funded through the RESILIENCE project, funded from the European Union’s
        Horizon 2020 Framework Programme for Research and Innovation.

.. container:: twocol

   .. container::

      .. image:: https://projet.biblissima.fr/sites/default/files/2021-11/biblissima-baseline-sombre-ia.png
         :width: 400
         :alt: Received funding from the Programme d’investissements d’Avenir

   .. container::

        Ce travail a bénéficié d’une aide de l’État gérée par l’Agence Nationale de la
        Recherche au titre du Programme d’Investissements d’Avenir portant la référence
        ANR-21-ESRE-0005 (Biblissima+).


