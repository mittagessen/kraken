*************
API Reference
*************

kraken.blla module
==================

.. note::

    `blla` provides the interface to the fully trainable segmenter. For the
    legacy segmenter interface refer to the `pageseg` module. Note that
    recognition models are not interchangeable between segmenters.

.. autoapifunction:: kraken.blla.segment

kraken.pageseg module
=====================

.. note::

    `pageseg` is the legacy bounding box-based segmenter. For the trainable
    baseline segmenter interface refer to the `blla` module. Note that
    recognition models are not interchangeable between segmenters.

.. autoapifunction:: kraken.pageseg.segment

kraken.rpred module
===================

.. autoapifunction:: kraken.rpred.bidi_record

.. autoapiclass:: kraken.rpred.mm_rpred
        :members:

.. autoapiclass:: kraken.rpred.ocr_record
        :members:

.. autoapifunction:: kraken.rpred.rpred


kraken.serialization module
===========================

.. autoapifunction:: kraken.serialization.render_report

.. autoapifunction:: kraken.serialization.serialize

.. autoapifunction:: kraken.serialization.serialize_segmentation

kraken.lib.models module
========================

.. autoapiclass:: kraken.lib.models.TorchSeqRecognizer
    :members:

.. autoapifunction:: kraken.lib.models.load_any

kraken.lib.vgsl module
======================

.. autoapiclass:: kraken.lib.vgsl.TorchVGSLModel
    :members:

kraken.lib.xml module
=====================

.. autoapifunction:: kraken.lib.xml.parse_xml

.. autoapifunction:: kraken.lib.xml.parse_page

.. autoapifunction:: kraken.lib.xml.parse_alto

kraken.lib.codec module
=======================

.. autoapiclass:: kraken.lib.codec.PytorchCodec
    :members:

kraken.lib.train module
=======================

Training Schedulers
-------------------

.. autoapiclass:: kraken.lib.train.TrainScheduler
    :members:

.. autoapiclass:: kraken.lib.train.annealing_step
    :members:

.. autoapiclass:: kraken.lib.train.annealing_const
    :members:

.. autoapiclass:: kraken.lib.train.annealing_exponential
    :members:

.. autoapiclass:: kraken.lib.train.annealing_reduceonplateau
    :members:

.. autoapiclass:: kraken.lib.train.annealing_cosine
    :members:

.. autoapiclass:: kraken.lib.train.annealing_onecycle
    :members:

Training Stoppers
-----------------

.. autoapiclass:: kraken.lib.train.TrainStopper
    :members:

.. autoapiclass:: kraken.lib.train.EarlyStopping
    :members:

.. autoapiclass:: kraken.lib.train.EpochStopping
    :members:

.. autoapiclass:: kraken.lib.train.NoStopping
    :members:

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
=========================

Datasets
--------

.. autoapiclass:: kraken.lib.dataset.BaselineSet
    :members:

.. autoapiclass:: kraken.lib.dataset.PolygonGTDataset
    :members:

.. autoapiclass:: kraken.lib.dataset.GroundTruthDataset
    :members:

Helpers
-------

.. autoapifunction:: kraken.lib.dataset.compute_error

.. autoapifunction:: kraken.lib.dataset.preparse_xml_data

.. autoapifunction:: kraken.lib.dataset.generate_input_transforms

kraken.lib.segmentation module
------------------------------

.. autoapifunction:: kraken.lib.segmentation.reading_order

.. autoapifunction:: kraken.lib.segmentation.polygonal_reading_order

.. autoapifunction:: kraken.lib.segmentation.denoising_hysteresis_thresh

.. autoapifunction:: kraken.lib.segmentation.vectorize_lines

.. autoapifunction:: kraken.lib.segmentation.calculate_polygonal_environment

.. autoapifunction:: kraken.lib.segmentation.scale_polygonal_lines

.. autoapifunction:: kraken.lib.segmentation.scale_regions

.. autoapifunction:: kraken.lib.segmentation.compute_polygon_section

.. autoapifunction:: kraken.lib.segmentation.extract_polygons


kraken.lib.ctc_decoder
======================

.. autoapifunction:: kraken.lib.ctc_decoder.beam_decoder

.. autoapifunction:: kraken.lib.ctc_decoder.greedy_decoder

.. autoapifunction:: kraken.lib.ctc_decoder.blank_threshold_decoder

kraken.lib.exceptions
=====================

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
