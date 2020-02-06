kraken API
==========

.. module:: kraken

Kraken provides routines which are usable by third party tools. In general
you can expect function in the ``kraken`` package to remain stable. We will try
to keep these backward compatible, but as kraken is still in an early
development stage and the API is still quite rudimentary nothing can be
garantueed.

Simple use cases of the API are contained in the `contrib` directory of the
kraken git repository. For specialized use cases it is recommended to look at
these instead of the command line drivers which contain lots of boilerplate to
enable all use cases.

kraken.binarization module
--------------------------

.. automodule:: kraken.binarization
    :members:
    :show-inheritance:

kraken.serialization module
---------------------------

.. automodule:: kraken.serialization
    :members:
    :show-inheritance:


kraken.blla module
------------------

.. note::

    `blla` provides the interface to the fully trainable segmenter. For the
    legacy segmenter interface refer to the `pageseg` module. Note that
    recognition models are not interchangeable between segmenters.

.. automodule:: kraken.blla
    :members:
    :show-inheritance:

kraken.pageseg module
---------------------

.. note::

    `pageseg` is the legacy bounding box-based segmenter. For the trainable
    baseline segmenter interface refer to the `blla` module. Note that
    recognition models are not interchangeable between segmenters.

.. automodule:: kraken.pageseg
    :members:
    :show-inheritance:

kraken.rpred module
-------------------

.. automodule:: kraken.rpred
    :members:
    :show-inheritance:


kraken.transcribe module
------------------------

.. automodule:: kraken.transcribe
    :members:
    :show-inheritance:

kraken.linegen module
---------------------

.. automodule:: kraken.linegen
    :members:
    :show-inheritance:

kraken.lib.models module
------------------------

.. automodule:: kraken.lib.models
    :members:
    :show-inheritance:

kraken.lib.vgsl module
----------------------

.. automodule:: kraken.lib.vgsl
    :members:
    :show-inheritance:

kraken.lib.codec
----------------

.. automodule:: kraken.lib.codec
    :members:
    :show-inheritance:

kraken.lib.train module
-----------------------

.. automodule:: kraken.lib.train
    :members:
    :show-inheritance:

kraken.lib.dataset module
-------------------------

.. automodule:: kraken.lib.dataset
    :members:
    :show-inheritance:

kraken.lib.segmentation module
------------------------------

.. automodule:: kraken.lib.vgsl
    :members:
    :show-inheritance:

kraken.lib.ctc_decoder
----------------------

.. automodule:: kraken.lib.ctc_decoder
    :members:
    :show-inheritance:
