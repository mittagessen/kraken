kraken
======

.. toctree::
   :hidden:
   :maxdepth: 2

   introduction_to_atr
   getting_started
   user_guide/index
   api/index


kraken is an open-source, turn-key automatic text recognition (ATR) system
optimized for historical and non-Latin script writing. Designed as a universal
text recognizer for the humanities, it directly addresses the unique challenges
of digitizing historical documents.

Highly adaptable and trainable, kraken excels in scenarios often overlooked by
commercial ATR engines. It is particularly well-suited for the "long tail" of
digitization, supporting low-resource languages and diverse, non-conventional
scripts common in humanities research.

Kraken offers two primary interfaces for its powerful functionality:

* A flexible and customizable **command-line interface**, intended for most users, enabling modular, chainable workflows for all tasks in the recognition pipeline.
* A comprehensive **API**, designed for developers building custom workflows, integrating kraken into other projects, or requiring fine-grained control over the ATR process.

This documentation is structured to guide you through kraken's capabilities:

* The :doc:`introduction_to_atr` is a basic primer on the core concepts intended for readers without prior ATR or machine learning experience.
* The :doc:`getting_started` guide provides a concise introduction to installation and basic usage.
* The :doc:`user_guide/index` offers detailed information on utilizing the CLI tools for various tasks.
* The :doc:`api/index` serves as a comprehensive reference for integrating kraken into your python projects.

Features
========

kraken's main features are:

  - Fully trainable layout analysis, reading order, and character recognition.
  - `Right-to-Left <https://en.wikipedia.org/wiki/Right-to-left>`_, `BiDi
    <https://en.wikipedia.com/wiki/Bi-directional_text>`_, and Top-to-Bottom
    script support
  - `ALTO <https://www.loc.gov/standards/alto/>`_, PageXML, abbyyXML, and hOCR
    output
  - Word bounding boxes and character cuts
  - Public repository of model files via `HTRMoPo <https://zenodo.org/communities/ocr_models/records>`_
  - Configurable recognition through a network specification language and a plugin system

Integrations
============

Through its API, kraken has also been integrated into various frontend applications and larger processing suites that offer graphical user interfaces or scaffolding for large-scale digitization:

*   `eScriptorium <https://www.escriptorium.fr/>`_: A web-based platform for annotating, transcribing, and training ATR models, tightly integrating kraken.
*   `OCR4all <https://ocr4all.org/>`_: A comprehensive ATR framework designed for historical documents, offering a user-friendly interface for various ATR tasks.
*   `OCR-D suite <https://ocr-d.de/>`_: A collection of tools for ATR-related tasks, aiming to build a full-stack ATR workflow for historical prints.
*   `arkindex by Teklia <https://arkindex.teklia.com/>`_: A platform for large-scale document analysis and indexing with kraken support through a plugin.


Community & Support
===================

Kraken is an open-source project driven by community contributions. We warmly
welcome feedback, pull requests, bug reports, and feature suggestions on our
`gitHub repository <https://github.com/mittagessen/kraken>`_.

If you are looking for help, want to discuss features, or need support regarding
integrations (particularly with **eScriptorium**), please join our community
chat: `eScriptorium Gitter Channel <https://app.gitter.im/#/room/#escripta_escriptorium:gitter.im>`_

.. _license:

License
=======

``Kraken`` is provided under the terms and conditions of the `Apache 2.0
License <https://github.com/mittagessen/kraken/blob/main/LICENSE>`_.

Funding
=======

kraken is developed at `Inria <https://inria.fr>`_ and the `École Pratique des
Hautes Études <https://www.ephe.psl.eu>`_, `Université PSL
<https://www.psl.eu>`_.


.. container:: twocol

   .. container:: leftside

        .. image:: https://raw.githubusercontent.com/mittagessen/kraken/main/docs/_static/normal-reproduction-low-resolution.jpg
          :width: 100
          :alt: Co-financed by the European Union

   .. container:: rightside

        This project was funded in part by the European Union. (ERC, MiDRASH,
        project number 101071829). This project was partially funded through
        the RESILIENCE project, funded from the European Union’s Horizon 2020
        Framework Programme for Research and Innovation.

.. container:: twocol

   .. container:: leftside

      .. image:: https://projet.biblissima.fr/sites/default/files/2021-11/biblissima-baseline-sombre-ia.png
         :width: 300
         :alt: Received funding from the Programme d’investissements d’Avenir

   .. container:: rightside

        Ce travail a bénéficié d’une aide de l’État gérée par l’Agence Nationale de la
        Recherche au titre du Programme d’Investissements d’Avenir portant la référence
        ANR-21-ESRE-0005 (Biblissima+).
