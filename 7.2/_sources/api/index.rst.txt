.. _api:

Python API
==========

For a gentle introduction to the python API, please refer to the
:doc:`../user_guide/api` in the user guide.

The API is built around a few core concepts, primarily the data containers
that are used to pass information between different processing steps. These
containers are defined in the :py:mod:`kraken.containers` module.

The three primary containers are:

* :py:class:`~kraken.containers.Segmentation`: Represents the segmentation of a page,
  including baselines, bounding boxes, and regions.
* :py:class:`~kraken.containers.BaselineLine`: Represents the positional and
  typology information of a line in a segmentation.
* :py:class:`~kraken.containers.BaselineOCRRecord`: Represents a line of text that has
  been recognized, including the transcription and confidence scores.

API Reference
-------------

.. toctree::
   :maxdepth: 1

   reference
