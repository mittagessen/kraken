.. _models:

Models
======

There are currently three kinds of models containing the recurrent neural
networks doing all the character recognition supported by kraken: ``pronn``
files serializing old pickled ``pyrnn`` models as protobuf, clstm's native
serialization, and versatile `Core ML
<https://developer.apple.com/documentation/coreml>`_ models.

CoreML
------

Core ML allows arbitrary network architectures in a compact serialization with
metadata. This is the default format in pytorch-based kraken.

