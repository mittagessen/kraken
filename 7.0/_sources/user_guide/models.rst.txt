.. _models:

Model Management
================

kraken relies on trained neural network models for both segmentation and
recognition. While you can train your own, there is a semi-curated repository
of freely licensed models that can be interacted with directly from the command
line.

Since the 5.x releases, this repository follows the `HTRMoPo
<https://github.com/mittagessen/HTRMoPo>`_ schema, which adds model card
support and allows for the publishing of arbitrary models.

.. note::
   Models are typically stored in ``~/.local/share/kraken`` (or ``~/.local/share/htrmopo`` depending on version configuration).

Retrieving Models
-----------------

You can search, query, and download models using the CLI tools.

Listing Available Models
~~~~~~~~~~~~~~~~~~~~~~~~

The ``list`` subcommand retrieves all available models and prints them
alongside their most important metadata.

.. code-block:: bash

   kraken list

Understanding DOIs
^^^^^^^^^^^^^^^^^^

Models in the repository are identified by a unique, persistent identifier
called a DOI. Because the repository supports versioning, the output displays a
tree structure:

* **Concept DOI:** The root of the tree. It represents the model as a whole and always points to the latest version.
* **Version DOI:** The branches beneath the root. These refer to specific historical versions of the model.

Filtering the List
~~~~~~~~~~~~~~~~~~

The full list output can be overwhelming. You can filter the output using
specific flags for model type, language, script, and keywords.

Filter Flags
^^^^^^^^^^^^

* ``--recognition``: Lists only recognition models.
* ``--segmentation``: Lists only segmentation models.
* ``--language [code]``: Lists models claiming support for a specific language.
* ``--script [code]``: Lists models claiming support for a specific script (e.g., ``Arab``).
* ``--keyword [string]``: Lists models containing a specific keyword (e.g., ``"middle ages"``).

Defining multiple filters of different types will **AND** them, while multiple filters of the same type will **OR** them.

Inspecting Model Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~

To view the full metadata record of a particular model—such as its description,
accuracy, or alphabet—use the ``show`` command with the model's DOI.

.. code-block:: bash

   kraken show 10.5281/zenodo.14585602

Downloading Models
~~~~~~~~~~~~~~~~~~

Once you have selected a model, use the ``get`` subcommand to download it.

.. code-block:: bash

   kraken get 10.5281/zenodo.14585602

Models are placed in the local storage directory and can be accessed for
inference using the filename printed in the last line of the output.

.. code-block:: bash

   # Example usage after download
   kraken -i input.jpg output.txt ocr -m urdu_best.mlmodel

Publishing Models
-----------------

Users can share models on the repository, making them discoverable for others.
The process involves two stages: creating a deposit on Zenodo, followed by
community approval.

Requirements
~~~~~~~~~~~~

1.  **Zenodo Account:** Required to host the files.
2.  **Personal Access Token:** Required for the upload tool to authenticate. Tokens can be created under your Zenodo account settings.
3.  **Metadata File:** A file consisting of a YAML header (machine-readable record) and a Markdown body (model card).

Metadata File Structure
~~~~~~~~~~~~~~~~~~~~~~~

The metadata file combines a structured YAML header with a Markdown body that
serves as the Model Card. The following example demonstrates the general
structure of a metadata file for a recognition model:

.. code-block:: yaml

   ---
   # Example metadata to be added to a model card.
   summary: Pretrained multilingual model
   authors:
     - name: Benjamin Kiessling
       affiliation: École Pratique des Hautes Études, PSL University
   license: Apache-2.0
   language:
   - ara
   - fas
   - urd
   script:
   - Arab
   tags:
   - automatic-text-recognition
   - multilingual
   metrics:
     cer: 0.05
     wer: 0.15
   base_model:
   - https://zenodo.org/records/15030337 
   ---
   # Model Card Title

   This is the Markdown body where you provide a detailed description of the model.

   ## Architecture
   Describe the neural network architecture (e.g., standard Kraken recognition model).

   ## Training Data
   Describe the datasets used for training (e.g., OpenITI Arabic MS Data).

   ## Accuracy
   Provide detailed accuracy metrics or limitations (e.g., "Best performance on Naskh script").

.. tip::
   It is not necessary to fill in the ID field of the metadata header; kraken will automatically request a new DOI and add it.

Uploading a New Model
~~~~~~~~~~~~~~~~~~~~~

Use the ``ketos publish`` command to upload a model. You must provide your
access token, the metadata file, and the model file.

.. code-block:: bash

   # Syntax: ketos publish -a $ACCESS_TOKEN -i metadata.yaml model_file.mlmodel
   ketos publish -a $ACCESS_TOKEN -i metadata.yaml aaebv2-2.mlmodel

Kraken will verify that the metadata conforms to the schema before depositing
the model at Zenodo.

.. warning::
   This deposit is persistent. It cannot be changed or deleted later, so ensure all information is correct before publishing.

Updating an Existing Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create a new version of an existing record (e.g., to update metadata or
release a retrained model) by referencing the existing DOI.

.. code-block:: bash

   # Use the -d flag to specify the existing record DOI
   ketos publish -a $ACCESS_TOKEN -i metadata.yaml aaebv2-2.mlmodel -d 10.5281/zenodo.5617783

Once deposited, a request for inclusion in the repository is automatically
created. After manual approval, the model becomes discoverable to other users.
