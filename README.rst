Description
===========

.. image:: https://github.com/mittagessen/kraken/actions/workflows/test.yml/badge.svg
    :target: https://github.com/mittagessen/kraken/actions/workflows/test.yml

kraken is a turn-key OCR system optimized for historical and non-Latin script
material.

kraken's main features are:

  - Fully trainable layout analysis and character recognition
  - `Right-to-Left <https://en.wikipedia.org/wiki/Right-to-left>`_, `BiDi
    <https://en.wikipedia.org/wiki/Bi-directional_text>`_, and Top-to-Bottom
    script support
  - `ALTO <https://www.loc.gov/standards/alto/>`_, PageXML, abbyyXML, and hOCR
    output
  - Word bounding boxes and character cuts
  - Multi-script recognition support
  - `Public repository <https://zenodo.org/communities/ocr_models>`_ of model files
  - Variable recognition network architectures

Installation
============

kraken only runs on **Linux or Mac OS X**. Windows is not supported.

The latest stable releases can be installed either from `PyPi <https://pypi.org>`_:

::

  $ pip install kraken

or through `conda <https://anaconda.org>`_:

::

  $ conda install -c conda-forge -c mittagessen kraken

If you want direct PDF and multi-image TIFF/JPEG2000 support it is necessary to
install the `pdf` extras package for PyPi:

::

  $ pip install kraken[pdf]

or install `pyvips` manually with conda:

::

  $ conda install -c conda-forge pyvips

Conda environment files are provided which for the seamless installation of the
master branch as well:

::

  $ git clone https://github.com/mittagessen/kraken.git 
  $ cd kraken
  $ conda env create -f environment.yml

or:

::

  $ git clone https://github.com/mittagessen/kraken.git 
  $ cd kraken
  $ conda env create -f environment_cuda.yml

for CUDA acceleration with the appropriate hardware.

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

Related Software
================

These days kraken is quite closely linked to the `escriptorium
<https://escriptorium.fr>`_ project developed in the same eScripta research
group. eScriptorium provides a user-friendly interface for annotating data,
training models, and inference (but also much more). There is a `gitter channel
<https://gitter.im/escripta/escriptorium>`_ that is mostly intended for
coordinating technical development but is also a spot to find people with
experience on applying kraken on a wide variety of material.

Funding
=======

kraken is developed at the `École Pratique des Hautes Études <http://ephe.fr>`_, `Université PSL <http://www.psl.eu>`_.

.. container:: twocol

   .. container::

        .. image:: https://raw.githubusercontent.com/mittagessen/kraken/master/docs/_static/normal-reproduction-low-resolution.jpg
          :width: 100
          :alt: Co-financed by the European Union

   .. container::

        This project was partially funded through the RESILIENCE project, funded from
        the European Union’s Horizon 2020 Framework Programme for Research and
        Innovation.


.. container:: twocol

   .. container::

      .. image:: https://www.gouvernement.fr/sites/default/files/styles/illustration-centre/public/contenu/illustration/2018/10/logo_investirlavenir_rvb.png
         :width: 100
         :alt: Received funding from the Programme d’investissements d’Avenir

   .. container::

        Ce travail a bénéficié d’une aide de l’État gérée par l’Agence Nationale de la
        Recherche au titre du Programme d’Investissements d’Avenir portant la référence
        ANR-21-ESRE-0005.


