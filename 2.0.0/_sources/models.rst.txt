.. _models:

Models
======

There are currently three kinds of models containing the recurrent neural
networks doing all the character recognition supported by kraken: ``pronn``
files serializing old pickled ``pyrnn`` models as protobuf, clstm's native
serialization, and versatile `Core ML
<https://developer.apple.com/documentation/coreml>`_ models.

.. _pyrnn:

pyrnn
-----

These are serialized instances of python ``lstm.SeqRecognizer`` objects. Using
such a model just entails loading the pickle and calling the appropriate
functions to perform recognition much like a shared library in other
programming languages.

Support for these models has been dropped with kraken 1.0 as python 2.7 is
phased out.

pronn
-----

Legacy python models can be converted to a protobuf based serialization. These
are loadable by kraken 1.0 and will be automatically converted to Core ML.

Protobuf models have several advantages over pickled ones. They are noticeably
smaller (80Mb vs 1.8Mb for the default model), don't allow arbitrary code
execution, and are upward compatible with python 3. Because they are so much
more lightweight they are also loaded much faster. 

clstm
-----

`clstm <https://github.com/tmbdev/clstm>`_, a small and fast implementation of
LSTM networks that was used in previous kraken versions. The model files can be
loaded with pytorch-based kraken and will be converted to Core ML.

CoreML
------

Core ML allows arbitrary network architectures in a compact serialization with
metadata. This is the default format in pytorch-based kraken.

Conversion
----------

Per default pronn/clstm models are automatically converted to the new Core ML
format when explicitely defined using the ``-m`` option to the ``ocr`` utility
on the command line. They are stored in the user kraken directory (default is
~/.kraken) and will be automatically substituted in future runs.

If conversion is not desired, e.g. because there is a bug in the conversion
routine, it can be disabled using the ``--disable-autoconversion`` switch.
