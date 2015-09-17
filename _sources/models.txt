.. _models:

Models
======

There are currently three kinds of models containing the recurrent neural
networks doing all the character recognition supported by kraken: traditional
``pyrnn`` models that are just pickled instances of python objects, a new
format serializing data in ``pronn`` files, and clstm's native `protocol buffer
<https://developers.google.com/protocol-buffers/>`_ serialization.

.. _pyrnn:

pyrnn
-----

These are serialized instances of python ``lstm.SeqRecognizer`` objects. Using
such a model just entails loading the pickle and calling the appropriate
functions to perform recognition much like a shared library in other
programming languages.

Several drawbacks exist when using pickled models. First they inherently allow
arbitrary code execution with relative ease. Additionally, they are not upward
compatible between python 2.x and 3.x and significantly larger than the newer
HDF5 models (roughly 6.5Mb per state).

pronn
-----

Legacy python models can be converted to a protobuf based serialization. By
default this conversion step happens automatically every time a pickled model
is used. 

Protobuf models have several advantages over pickled ones. They are noticeably
smaller (80Mb vs 1.8Mb for the default model), don't allow arbitrary code
execution, and are upward compatible with python 3. Because they are so much
more lightweight they are also loaded much faster. 

HDF5
----

`HDF5 <https://www.hdfgroup.org/HDF5/>`_ is a file format designed to store
large amounts of numerical data efficiently. It was used in the past to handle
``pyrnn`` models without incurring the deserialization and code execution
penalty of pickled objects. As no training facility was included in kraken at
the time, hence no HDF5-only models should exist, support for them has been
removed.

clstm
-----

`clstm <https://github.com/tmbdev/clstm>`_, a small and fast implementation of
LSTM networks, creates neural networks serialized as protocol buffers. These
are usually slightly smaller than converted models and require clstm's python
extension to load and run. While they are significantly faster than the native
python models clstm is still in early development and there aren't many trained
models available, yet.

Conversion
----------

Per default pyrnn models are automatically converted to the new protobuf format
when explicitely defined using the ``-m`` option to the ``ocr`` utility on the
command line. They are stored in the user kraken directory (default is
~/.kraken) and will be automatically substituted in future runs.

This substitution process is extremely fast, in fact loading the pickle is
usually several magnitudes slower than converting it once loaded.

If conversion is not desired, e.g. because there is a bug in the conversion
routine, it can be disabled using the ``--disable-autoconversion`` switch.
