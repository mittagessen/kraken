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

Kraken can be run on Linux or Mac OS X (both x64 and ARM). Installation is
through the on-board *pip* utility.

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

Development Branch Installation using Pip
-----------------------------------------

To install the latest development branch through clone the kraken git
repository and perform an editable install:

.. code-block:: console

  $ git clone https://github.com/mittagessen/kraken.git
  $ cd kraken
  $ pip install --editable . 

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
                                   
  Retrieving model list ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 65/65 0:00:00 0:00:18
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃ DOI                         ┃ summary                                                                                                                                                                                                                     ┃ model type   ┃ keywords                                                                                 ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │ 10.5281/zenodo.7051645      │                                                                                                                                                                                                                             │              │                                                                                          │
  │ ├── 10.5281/zenodo.14585602 │ Printed Urdu Base Model Trained on the OpenITI Corpus                                                                                                                                                                       │ recognition  │ automatic-text-recognition                                                               │
  │ ├── 10.5281/zenodo.14574660 │ Printed Urdu Base Model Trained on the OpenITI Corpus                                                                                                                                                                       │ recognition  │ kraken_pytorch                                                                           │
  │ └── 10.5281/zenodo.7051646  │ Printed Urdu Base Model Trained on the OpenITI Corpus                                                                                                                                                                       │ recognition  │ kraken_pytorch                                                                           │
  │ 10.5281/zenodo.10066218     │                                                                                                                                                                                                                             │              │                                                                                          │
  │ ├── 10.5281/zenodo.12743230 │ CATMuS Medieval 1.5.0                                                                                                                                                                                                       │ recognition  │ kraken_pytorch; handwritten text recognition; htr; middle ages                           │
  │ └── 10.5281/zenodo.10066219 │ CATMuS Medieval                                                                                                                                                                                                             │ recognition  │ kraken_pytorch; handwritten text recognition; htr; middle ages                           │
  │ 10.5281/zenodo.13788176     │                                                                                                                                                                                                                             │              │                                                                                          │
  │ └── 10.5281/zenodo.13788177 │ McCATMuS - Transcription model for handwritten, printed and typewritten documents from the 16th century to the 21st century                                                                                                 │ recognition  │ kraken_pytorch; HTR; OCR; generic model                                                  │
  │ 10.5281/zenodo.5468572      │                                                                                                                                                                                                                             │              │                                                                                          │
  │ └── 10.5281/zenodo.5468573  │ Medieval Hebrew manuscripts in Italian bookhand version 1.0                                                                                                                                                                 │ recognition  │ kraken_pytorch                                                                           │
  │ 10.5281/zenodo.13741956     │                                                                                                                                                                                                                             │              │                                                                                          │
  │ └── 10.5281/zenodo.13741957 │ Model trained on 11th century manuscripts to produce graphematic transcription (Latin).                                                                                                                                     │ recognition  │ kraken_pytorch                                                                           │
  ...  

Model metadata can be extracted using:

.. code-block:: console

  $ kraken show 10.5281/zenodo.10592716
                                               CATMuS-Print (Large, 2024-01-30) - Diachronic model for French prints and other languages                                             
  ┌──────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ DOI              │ 10.5281/zenodo.10592716                                                                                                                                      │
  │ concept DOI      │ 10.5281/zenodo.10592715                                                                                                                                      │
  │ publication date │ 2024-01-31T21:24:10+00:00                                                                                                                                    │
  │ model type       │ recognition                                                                                                                                                  │
  │ script           │ Latin                                                                                                                                                        │
  │ alphabet         │ ! " # $ % & ' ( ) * + , - . / 0 1 2 3 4 5 6 7 8 9 : ; < = > ? A B C D E F G H I J K L M N O P Q R S T U V W X Y Z [ \ ] ^ _ a b c d e f g h i j k l m n o p  │
  │                  │ q r s t u v w x y z | } ~ ¡ £ ¥ § « ¬ ° ¶ · » ¿ Æ ß æ đ ł Œ œ ƀ ǝ ɇ Α Β Γ Δ Ε Ζ Θ Ι Κ Λ Μ Ν Ο Π Ρ Σ Τ Υ Φ Χ Ω α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ ς σ τ υ φ χ  │
  │                  │ ω ϛ ח ל ר ᑕ ᗅ ᗞ ᚠ ẞ – — ‘ ’ ‚ “ ” „ ‟ † • ⁄ ⁊ ⁋ ℟ ← ▽ ◊ ★ ☙ ✠ ✺ ✻ ⟦ ⟧ ⬪ ꝑ ꝓ ꝗ ꝙ ꝟ ꝯ ꝵ                                                                        │
  │                  │ SPACE, COMBINING GRAVE ACCENT, COMBINING ACUTE ACCENT, COMBINING CIRCUMFLEX ACCENT, COMBINING TILDE, COMBINING MACRON, COMBINING DOT ABOVE, COMBINING        │
  │                  │ DIAERESIS, COMBINING RING ABOVE, COMBINING COMMA ABOVE, COMBINING REVERSED COMMA ABOVE, COMBINING CEDILLA, COMBINING OGONEK, COMBINING GREEK PERISPOMENI,    │
  │                  │ COMBINING GREEK YPOGEGRAMMENI, COMBINING LATIN SMALL LETTER I, COMBINING LATIN SMALL LETTER U, 0xe682, 0xe68b, 0xe8bf, 0xf1a7                                │
  │ keywords         │ kraken_pytorch                                                                                                                                               │
  │                  │ optical text recognition                                                                                                                                     │
  │ metrics          │ cer: 1.44                                                                                                                                                    │
  │ license          │ CC-BY-4.0                                                                                                                                                    │
  │ creators         │ Gabay, Simon (https://orcid.org/0000-0001-9094-4475) (University of Geneva)                                                                                  │
  │                  │ Clérice, Thibault (https://orcid.org/0000-0003-1852-9204) (Institut national de recherche en informatique et en automatique)                                 │
  │                  │ Simon Gabay (Université de Genève)                                                                                                                           │
  │ description      │ Training data come from different projects, dealing with different languages (French, Spanish, German, English, Corsican, Catalan, Latin…) and different     │
  │                  │ centuries (from the first prints of the 16th c. to digital documents of the 21st century). Transcriptions follow graphematic principles and try to be as     │
  │                  │ compatible as possible with guidelines previously published for French: no ligature (except those that still exist), no allographetic variants (except the   │
  │                  │ long s), and preservation of the historical use of some letters (u/v, i/j). Inconsistencies might be present, because transcriptions have been done over     │
  │                  │ several years and the norms have slightly evolved.                                                                                                           │
  └──────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

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
  Loading ANN /home/mittagessen/git/kraken/kraken/blla.mlmodel	✓
  Loading ANN catmus-print-fondue-large.mlmodel	✓
  Segmenting image.tif	✓
  Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 65/65 0:00:00 0:00:06
  Writing recognition results for image.tif	✓

To segment an image into reading-order sorted baselines and regions serialized into an ALTO document:

.. code-block:: console

  $ kraken -a -i image.tif segmentation.xml segment -bl
  Loading ANN /home/mittagessen/git/kraken/kraken/blla.mlmodel	✓
  Segmenting image.tif	✓

To OCR an image using the previously downloaded model and output a plain text file:

.. code-block:: console

  $ kraken -i image .tif image.txt segment -bl ocr -m catmus-print-fondue-large.mlmodel
  Loading ANN /home/mittagessen/git/kraken/kraken/blla.mlmodel	✓
  Loading ANN catmus-print-fondue-large.mlmodel	✓
  Segmenting image.tif	✓
  Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 65/65 0:00:00 0:00:06
  Writing recognition results for image.tif	✓

To OCR an image using the default model and serialize the output using the ALTO
template:

.. code-block:: console

  $ kraken -a -i image.tif image.txt segment -bl ocr -m catmus-print-fondue-large.mlmodel
  Loading ANN /home/mittagessen/git/kraken/kraken/blla.mlmodel	✓
  Loading ANN catmus-print-fondue-large.mlmodel	✓
  Segmenting image.tif	✓
  Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 65/65 0:00:00 0:00:06
  Writing recognition results for image.tif	✓

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


