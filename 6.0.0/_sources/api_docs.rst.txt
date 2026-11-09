*************
API Reference
*************

Segmentation
============

kraken.blla module
------------------

.. note::

    `blla` provides the interface to the fully trainable segmenter. For the
    legacy segmenter interface refer to the `pageseg` module. Note that
    recognition models are not interchangeable between segmenters.

.. autoapifunction:: kraken.blla.segment

kraken.pageseg module
---------------------

.. note::

    `pageseg` is the legacy bounding box-based segmenter. For the trainable
    baseline segmenter interface refer to the `blla` module. Note that
    recognition models are not interchangeable between segmenters.

.. autoapifunction:: kraken.pageseg.segment

Recognition
===========

kraken.rpred module
-------------------

.. autoapiclass:: kraken.rpred.mm_rpred
        :members:

.. autoapifunction:: kraken.rpred.rpred

Serialization
=============

kraken.serialization module
---------------------------

.. autoapifunction:: kraken.serialization.render_report

.. autoapifunction:: kraken.serialization.serialize

.. autoapifunction:: kraken.serialization.serialize_segmentation

Default templates
-----------------

.. _templates:

ALTO 4.4
^^^^^^^^

.. literalinclude:: ../kraken/templates/alto
        :language: xml+jinja

PageXML
^^^^^^^

.. literalinclude:: ../kraken/templates/alto
        :language: xml+jinja

hOCR
^^^^

.. literalinclude:: ../kraken/templates/alto
        :language: xml+jinja

ABBYY XML
^^^^^^^^^

.. literalinclude:: ../kraken/templates/abbyyxml
        :language: xml+jinja

Containers and Helpers
======================

kraken.lib.codec module
-----------------------

.. autoapiclass:: kraken.lib.codec.PytorchCodec
    :members:

kraken.containers module
------------------------

.. autoapiclass:: kraken.containers.Segmentation
        :members:

.. autoapiclass:: kraken.containers.BaselineLine
        :members:

.. autoapiclass:: kraken.containers.BBoxLine
        :members:

.. autoapiclass:: kraken.containers.Region
        :members:

.. autoapiclass:: kraken.containers.ocr_record
        :members:

.. autoapiclass:: kraken.containers.BaselineOCRRecord
        :members:

.. autoapiclass:: kraken.containers.BBoxOCRRecord
        :members:

.. autoapiclass:: kraken.containers.ProcessingStep
        :members:

kraken.lib.ctc_decoder
----------------------

.. autoapifunction:: kraken.lib.ctc_decoder.beam_decoder

.. autoapifunction:: kraken.lib.ctc_decoder.greedy_decoder

.. autoapifunction:: kraken.lib.ctc_decoder.blank_threshold_decoder

kraken.lib.exceptions
---------------------

.. autoapiclass:: kraken.lib.exceptions.KrakenCodecException
    :members:

.. autoapiclass:: kraken.lib.exceptions.KrakenStopTrainingException
    :members:

.. autoapiclass:: kraken.lib.exceptions.KrakenEncodeException
    :members:

.. autoapiclass:: kraken.lib.exceptions.KrakenRecordException
    :members:

.. autoapiclass:: kraken.lib.exceptions.KrakenInvalidModelException
    :members:

.. autoapiclass:: kraken.lib.exceptions.KrakenInputException
    :members:

.. autoapiclass:: kraken.lib.exceptions.KrakenRepoException
    :members:

.. autoapiclass:: kraken.lib.exceptions.KrakenCairoSurfaceException
    :members:

kraken.lib.models module
------------------------

.. autoapiclass:: kraken.lib.models.TorchSeqRecognizer
    :members:

.. autoapifunction:: kraken.lib.models.load_any

kraken.lib.segmentation module
------------------------------

.. autoapifunction:: kraken.lib.segmentation.reading_order

.. autoapifunction:: kraken.lib.segmentation.neural_reading_order

.. autoapifunction:: kraken.lib.segmentation.polygonal_reading_order

.. autoapifunction:: kraken.lib.segmentation.vectorize_lines

.. autoapifunction:: kraken.lib.segmentation.calculate_polygonal_environment

.. autoapifunction:: kraken.lib.segmentation.scale_polygonal_lines

.. autoapifunction:: kraken.lib.segmentation.scale_regions

.. autoapifunction:: kraken.lib.segmentation.compute_polygon_section

.. autoapifunction:: kraken.lib.segmentation.extract_polygons

kraken.lib.vgsl module
----------------------

.. autoapiclass:: kraken.lib.vgsl.TorchVGSLModel
    :members:

kraken.lib.xml module
---------------------

.. autoapiclass:: kraken.lib.xml.XMLPage

Training
========

kraken.lib.train module
-----------------------

Loss and Evaluation Functions
-----------------------------

.. autoapifunction:: kraken.lib.train.recognition_loss_fn

.. autoapifunction:: kraken.lib.train.baseline_label_loss_fn

.. autoapifunction:: kraken.lib.train.recognition_evaluator_fn

.. autoapifunction:: kraken.lib.train.baseline_label_evaluator_fn

Trainer
-------

.. autoapiclass:: kraken.lib.train.KrakenTrainer
    :members:


kraken.lib.dataset module
-------------------------

Recognition datasets
^^^^^^^^^^^^^^^^^^^^

.. autoapiclass:: kraken.lib.dataset.ArrowIPCRecognitionDataset
    :members:

.. autoapiclass:: kraken.lib.dataset.BaselineSet
    :members:

.. autoapiclass:: kraken.lib.dataset.GroundTruthDataset
    :members:

Segmentation datasets
^^^^^^^^^^^^^^^^^^^^^

.. autoapiclass:: kraken.lib.dataset.PolygonGTDataset
    :members:

Reading order datasets
^^^^^^^^^^^^^^^^^^^^^^

.. autoapiclass:: kraken.lib.dataset.PairWiseROSet
    :members:

.. autoapiclass:: kraken.lib.dataset.PageWiseROSet
    :members:

Helpers
^^^^^^^

.. autoapiclass:: kraken.lib.dataset.ImageInputTransforms
   :members:

.. autoapifunction:: kraken.lib.dataset.collate_sequences

.. autoapifunction:: kraken.lib.dataset.global_align

.. autoapifunction:: kraken.lib.dataset.compute_confusions

Legacy modules
==============

These modules are retained for compatibility reasons or highly specialized use
cases. In most cases their use is not necessary and they aren't further
developed for interoperability with new functionality, e.g. the transcription
and line generation modules do not work with the baseline segmenter.

kraken.binarization module
--------------------------

.. autoapifunction:: kraken.binarization.nlbin

kraken.transcribe module
------------------------

.. autoapiclass:: kraken.transcribe.TranscriptionInterface
    :members:

kraken.linegen module
---------------------

.. autoapiclass:: kraken.transcribe.LineGenerator
    :members:

.. autoapifunction:: kraken.transcribe.ocropy_degrade

.. autoapifunction:: kraken.transcribe.degrade_line

.. autoapifunction:: kraken.transcribe.distort_line
