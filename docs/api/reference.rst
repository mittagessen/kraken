.. _api_reference:

*************
API Reference
*************

This reference is automatically generated from the docstrings in the kraken
source code.

High-Level API
==============

These modules provide the main entry points for using kraken programmatically.

kraken.containers
-----------------

.. autoapimodule:: kraken.containers
   :members:

kraken.lib.xml
--------------

.. autoapimodule:: kraken.lib.xml
  :members:

kraken.tasks
------------

.. autoapimodule:: kraken.tasks
   :members:

kraken.train
------------

.. autoapimodule:: kraken.train
  :members:

Low-Level API
=============

These modules provide lower-level access to the core components of kraken. In
most cases, it is recommended to use the high-level API instead.

.. toctree::
   :maxdepth: 1
   :hidden:

   kraken.ketos
   kraken.lib

.. autoapimodule:: kraken.ketos.pretrain
.. autoapimodule:: kraken.ketos.train
.. autoapimodulesummary:: kraken.lib
.. autoapimodule:: kraken.lib.dataset
.. autoapimodule:: kraken.lib.codec
.. autoapimodule:: kraken.lib.exceptions
.. autoapimodule:: kraken.lib.segmentation
.. autoapimodule:: kraken.lib.lineest
.. autoapimodule:: kraken.lib.ctc_decoder
.. autoapimodule:: kraken.lib.vgsl

Legacy Modules
==============

These modules are retained for compatibility reasons or highly specialized use
cases. Their use is not recommended.

.. autoapimodule:: kraken.binarization
.. autoapimodule:: kraken.pageseg
.. autoapimodule:: kraken.rpred
.. autoapimodule:: kraken.lib.models
