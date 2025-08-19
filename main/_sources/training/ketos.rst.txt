.. _ketos:

Training
========

This page describes the training utilities available through the ``ketos``
command line utility in depth. For a gentle introduction on model training
please refer to the :ref:`tutorial <training>`.

There are currently three trainable components in the kraken processing pipeline:

* :ref:`Segmentation <segtrain>`: finding lines and regions in images
* :ref:`Reading Order <rotrain>`: ordering lines found in the previous segmentation step. Reading order models are closely linked to segmentation models and both are usually trained on the same dataset.
* :ref:`Recognition <rectrain>`: recognition models transform images of lines into text.

Depending on the use case it is not necessary to manually train new models for
each material. The default segmentation model works well on quite a variety of
handwritten and printed documents, a reading order model might not perform
better than the default heuristic for simple text flows, and there are
recognition models for some types of material available in the repository.

The main ``ketos`` command has some common options that configure how most of
its sub-commands use computational resources:

======================================================= ======
option                                                  action
======================================================= ======
-d, \--device                                           Select device to use (cpu, cuda:0, cuda:1,...). GPU acceleration requires in addition to hardware the requisite libraries (CUDA, ROCM, ...). It can be set to `auto` to select the best device available.
\--precision                                            The precision to run neural network operations in. On GPU, bfloat16 (`bf16-true` or `bf16-mixed`) can lead to considerable speedup.
\--workers                                              Number of workers processes to spawn for data loading, transformation, or compilation.
\--threads                                              Number of threads to use for intra-op parallelization like OpenMP, BLAS, ...
======================================================= ======

Training data formats
---------------------

.. _ketos_format:

The training tools accept a variety of training data formats, usually some kind
of custom low level format, the XML-based formats that are commony used for
archival of annotation and transcription data, and in the case of recognizer
training a precompiled binary format. It is recommended to use the XML formats
for segmentation and reading order training and the binary format for
recognition training.

ALTO
~~~~

Kraken parses ALTO files using schema version 4.2 or upwards and produces files
according to ALTO 4.4. An example showing the attributes necessary for
segmentation, recognition, and reading order training follows:

.. literalinclude:: /_static/alto.xml
   :language: xml
   :force:

Importantly, the parser only works with measurements in the pixel domain, i.e.
an unset `MeasurementUnit` or one with an element value of `pixel`. 

For a more in-depth description of how kraken parses and writes ALTO files see
:ref:`here <alto>`.

PAGE XML
~~~~~~~~

PAGE XML is parsed and produced according to the 2019-07-15 version of the
schema, although the parser is not strict and works with non-conformant output
from a variety of tools. As with ALTO, PAGE XML files can be used to train
segmentation, reading order, and recognition models.

.. literalinclude:: /_static/pagexml.xml
   :language: xml
   :force:

For a more in-depth description of how kraken parses and writes ALTO files see
:ref:`here <pagexml>`.

Binary Datasets
~~~~~~~~~~~~~~~

.. _binary_datasets:

In addition to training recognition models directly from XML and image files, a
binary dataset format offering a couple of advantages is supported for
recognition training. Binary datasets drastically improve loading performance
allowing the saturation of most GPUs with minimal computational overhead while
also allowing training with datasets that are larger than the systems main
memory. A minor drawback is a ~30% increase in dataset size in comparison to
the raw images + XML approach.

To realize this speedup the dataset has to be compiled first:

.. code-block:: console

   $ ketos compile -f xml -o dataset.arrow file_1.xml file_2.xml ...

if there are a lot of individual lines containing many lines this process can
take a long time. It can easily be parallelized by specifying the number of
separate parsing workers with the ``--workers`` option:

.. code-block:: console

   $ ketos --workers 8 compile -f xml ...

In addition, binary datasets can contain fixed splits which allow
reproducibility and comparability between training and evaluation runs.
Training, validation, and test splits can be pre-defined from multiple sources.
Per default they are sourced from tags defined in the source XML files unless
the option telling kraken to ignore them is set:

.. code-block:: console

   $ ketos compile --ignore-splits -f xml ...

Alternatively fixed-proportion random splits can be created ad-hoc during
compile time:

.. code-block:: console

   $ ketos compile --random-split 0.8 0.1 0.1 ...


The above line splits assigns 80% of the source lines to the training set, 10%
to the validation set, and 10% to the test set. The training and validation
sets in the dataset file are used automatically by `ketos train` (unless told
otherwise) while the remaining 10% of the test set is selected by `ketos test`.

.. warning::
    Fixed splits in datasets are ignored during training and testing per
    default as they require loading the entire dataset into main memory at
    once, drastically increasing memory consumption and causing initial delays.
    Use the `\-\-fixed-splits` option in `ketos train` and `ketos test` to
    respect fixed splits.
