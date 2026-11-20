.. _api_introduction:

Introduction to the Python API
==============================

kraken provides a powerful python API for programmatic access to all its
functionality. This guide provides a basic introduction to the most important
parts of the API.

High-Level API
--------------

The easiest way to use kraken programmatically is through the high-level API
in the :py:mod:`kraken.tasks` module. This API provides a set of task-oriented
classes for segmentation, recognition, and forced alignment.

Segmentation
~~~~~~~~~~~~

To segment an image, you can use the :py:class:`~kraken.tasks.SegmentationTaskModel` class.
It returns a :py:class:`~kraken.containers.Segmentation` object containing the
segmentation results.

The lines within the `Segmentation` object can be of two types, depending on
the model used:

* :py:class:`~kraken.containers.BaselineLine` for models that output baselines and polygons.
* :py:class:`~kraken.containers.BBoxLine` for models that output bounding boxes.

.. code-block:: python

    from PIL import Image
    from kraken.tasks import SegmentationTaskModel
    from kraken.configs import SegmentationInferenceConfig

    # Load the default segmentation model
    model = SegmentationTaskModel.load_model()

    im = Image.open('image.png')

    config = SegmentationInferenceConfig()

    segmentation = model.predict(im, config)

    for line in segmentation.lines:
        print(line.baseline)

Recognition
~~~~~~~~~~~

To recognize the text in an image, you can use the
:py:class:`~kraken.tasks.RecognitionTaskModel` class. This class takes a
:py:class:`~kraken.containers.Segmentation` object, a PIL image, and a
configuration as inputs and returns an iterator of `ocr_record` objects.

Similar to segmentation, the returned records can be of two types:

* :py:class:`~kraken.containers.BaselineOCRRecord` gor baseline-based recognition.
* :py:class:`~kraken.containers.BBoxOCRRecord` for bounding box-based recognition.

.. code-block:: python

    from PIL import Image
    from kraken.tasks import RecognitionTaskModel
    from kraken.configs import RecognitionInferenceConfig

    # Load a recognition model
    model = RecognitionTaskModel.load_model('model.safetensors')

    im = Image.ope('image.png')

    config = RecognitionInferenceConfig()

    # segmentation is a Segmentation object created by loading an XML file o
    # running segmentation manually.
    for record in model.predict(im, segmentation, config):
        print(record.prediction)

Forced Alignment
~~~~~~~~~~~~~~~~

Forced alignment is the process of aligning a given transcription to the output of a text recognition model, producing approximate character locations. This is a specialized operation outside a normal ATR workflow and can be used, e.g., to produce word bounding boxes for a known good transcription.

You can use the :py:class:`~kraken.tasks.ForcedAlignmentTaskModel` class to perform forced alignment:

.. code-block:: python

    from PIL import Image
    from kraken.tasks import ForcedAlignmentTaskModel
    from kraken.containers import Segmentation, BaselineLine
    from kraken.configs import RecognitionInferenceConfig

    # `model.safetensor` is a recognition model
    model = ForcedAlignmentTaskModel.load_model('model.safetensor')
    im = Image.open('image.png')
    line = BaselineLine(baseline=[(0,0), (100,0)], boundary=[(0,-10), (100,-10), (100,10), (0,10)], text='Hello World')
    segmentation = Segmentation(lines=[line])
    config = RecognitionInferenceConfig()

    aligned_segmentation = model.predict(im, segmentation, config)
    record = aligned_segmentation.lines[0]
    print(record.prediction)
    print(record.cuts)

Parsing XML
~~~~~~~~~~~

kraken can parse ALTO and PageXML files into :py:class:`~kraken.containers.Segmentation` objects. This is useful for loading ground truth data or the results of other OCR engines. The :py:class:`~kraken.lib.xml.XMLPage` class handles this.

.. note::
   The parser has been refactored in kraken 7.0 with changes to reading order parsing and robustness improvements. In particular, if the XML dimension field is invalid, kraken falls back to reading the source image to determine dimensions.

.. code-block:: python

    from kraken.lib.xml import XMLPage

    xml_page = XMLPage('input.xml')
    segmentation = xml_page.to_container()

Serialization
~~~~~~~~~~~~~

After segmentation and recognition, you can serialize the results into various formats, such as ALTO or PageXML, with the :py:func:`kraken.serialization.serialize` function.

.. code-block:: python

    from kraken.serialization import serialize

    # Assume `segmentation` is a Segmentation object from a previous step
    # and `im` is the PIL image object.
    
    # Serialize to ALTO
    alto_xml = serialize(segmentation, image_size=im.size, template='alto')

    with open('output.alto.xml', 'w') as f:
        f.write(alto_xml)

    # Serialize to PageXML
    page_xml = serialize(segmentation, image_size=im.size, template='page')

    with open('output.page.xml', 'w') as f:
        f.write(page_xml)

Plugin System
-------------

kraken features a plugin system that allows developers to extend its functionality with new commands, model types, and tasks. This system is based on python's entry points mechanism and primarily targets pytorch-based implementations.

To create a plugin, you need to:

1.  Create a new python package that depends on `kraken`.
2.  In your package, create a class that implements the required interface.
3.  Register your class as an entry point in your package's `pyproject.toml` or `setup.cfg`.

Entry Point Groups
~~~~~~~~~~~~~~~~~~

kraken provides several entry point groups for different types of plugins:

*   ``kraken.cli``: Adds new subcommands to the `kraken` command-line
    interface.
*   ``ketos.cli``: Adds new subcommands to the `ketos` command-line
    interface.
*   ``kraken.models``: Registers new model architectures.
*   ``kraken.lightning_modules``: Registers new PyTorch Lightning modules for training and model conversion.
*   ``kraken.loaders``: Registers new model loaders.
*   ``kraken.writers``: Registers new model writers.
*   ``kraken.tasks``: Registers new high-level tasks.

Model Plugins
~~~~~~~~~~~~~

The most common use case for plugins is to add new machine learning architectures for an already existing task type, such as defining a new segmentation method. This typically involves:

1.  Implementing a class that inherits from the requisite base model interface in :py:mod:`kraken.models.base`, such as :py:class:`~kraken.models.base.RecognitionBaseModel` for text recognition or :py:class:`~kraken.models.base.SegmentationBaseModel` for layout analysis.
2.  Registering this class in your plugin's `pyproject.toml` or `setup.cfg` under the `kraken.models` entry point.
3.  Implement a checkpoint container that provides a `load_from_checkpoint` method and is registered under the `kraken.lightning_modules` entrypoint. The easiest way to ensure correct behavior is to implement this class as a `lightning`_ LightningModule.
4.  Optionally, adding a training command to `ketos` by creating a `click`
    command and registering it under the `ketos.cli` entry point.

For a complete example of a layout analysis model plugin, refer to the
`dfine_kraken`_ project, which implements a D-FINE based segmentation method.

.. _`dfine_kraken`: https://github.com/mittagessen/dfine_kraken
.. _`lightning`: https://lightning.ai/docs/pytorch/stable/

Low-Level API
-------------

For more fine-grained control, you can use the low-level API in the :py:mod:`kraken.lib` module. This API provides direct access to the core components of kraken, such as the neural network models and the CTC decoders.

For more information, please refer to the :ref:`api_reference`.
