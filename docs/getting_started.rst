.. _getting_started:

Getting Started
===============

This guide provides a brief overview of how to install and use kraken.

Installation
------------

Kraken can be run on Linux or Mac OS X (both x64 and ARM). Installation is
through the on-board *pip* utility. To not pollute the global state of your
distribution's package manager it is recommended to use virtual environments.
If you do not have a setup or do not wish to handle virtual environments
yourself you can use `pipx`.

.. code-block:: console

   $ sudo apt install pipx
   $ pipx install kraken

kraken works both on Linux and Mac OS X and with any python interpreter between
3.9 and 3.12. It is possible the installation fails because `pipx` defaults to
an unsupported interpreter version. In that case you need to install a
compatible interpreter version such as 3.12 and then specify this version
explicitly:

.. code-block:: console

   $ sudo apt install python3.12-full
   $ pipx install --python python3.12 kraken


Installation using pip
~~~~~~~~~~~~~~~~~~~~~~

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

or

.. code-block:: console

   $ pip install .[pdf]

respectively.

Development branch installation using pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install the latest development branch through clone the kraken git
repository and perform an editable install:

.. code-block:: console

  $ git clone https://github.com/mittagessen/kraken.git
  $ cd kraken
  $ pip install --editable . 

Model Retrieval
---------------

After installation, you'll need a model to process your documents. In kraken,
models are pre-trained files that contain the knowledge for a specific task,
such as identifying the layout of a page or recognizing characters in a
particular script.

Kraken provides a public repository of freely available models that can be accessed
from the command line. To list all available models, run:

.. code-block:: console

    $ kraken list

To download a model, use the `get` command with the model's DOI. For example,
to download the default model for printed French text, run:

.. code-block:: console

  $ kraken get 10.5281/zenodo.10592716

For more information on how to interact with the model repository, please refer
to the :doc:`user_guide/repo` section of the user guide.

The ATR Workflow
----------------

Automatic text recognition is a multi-step process that transforms an image of
a document into a text file. In kraken, this process is broken down into a
sequence of chainable commands, each performing a specific task.

The three main steps in a typical ATR workflow are:

1.  **Layout Analysis (Segmentation):** This step identifies the regions and
    lines of text on the page. In kraken, this is done with the `segment`
    command.
2.  **Text Recognition (ATR):** This step transcribes the text from the line
    images identified in the previous step. In kraken, this is done with the
    `ocr` command.
3.  **Serialization:** This step saves the output of the previous steps in a
    structured format, such as plain text, ALTO, or PageXML. This is handled
    by the output options of the `kraken` command.

Models are essential to this workflow, as they provide the specific knowledge
for layout analysis and text recognition. They are integrated into the kraken
workflow as parameters for the `segment` and `ocr` commands. The choice of
model is crucial for achieving good results, as a model trained on a specific
type of material will perform best on similar material.

Here is a quick example of a complete workflow:

.. raw:: html
    :file: _static/kraken_workflow.svg

Recognizing text on an image using the default parameters, including page
segmentation:

.. code-block:: console

  $ kraken -i image.tif image.txt segment -bl ocr -m catmus-print-fondue-large.mlmodel

In this example, `segment` performs the layout analysis, and `ocr` performs the
text recognition using the `catmus-print-fondue-large.mlmodel`. The final
transcription is saved to `image.txt`.
