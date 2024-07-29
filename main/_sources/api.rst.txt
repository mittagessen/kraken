API Quickstart
==============

Kraken provides routines which are usable by third party tools to access all
functionality of the OCR engine. Most functional blocks, binarization,
segmentation, recognition, and serialization are encapsulated in one high
level method each.

Simple use cases of the API which are mostly useful for debugging purposes are
contained in the `contrib` directory. In general it is recommended to look at
this tutorial, these scripts, or the API reference. The command line drivers
are unnecessarily complex for straightforward applications as they contain lots
of boilerplate to enable all use cases.

Basic Concepts
--------------

The fundamental modules of the API are similar to the command line drivers.
Image inputs and outputs are generally `Pillow <https://python-pillow.org/>`_
objects and numerical outputs numpy arrays.

Top-level modules implement high level functionality while :mod:`kraken.lib`
contains loaders and low level methods that usually should not be used if
access to intermediate results is not required.

Preprocessing and Segmentation
------------------------------

The primary preprocessing function is binarization although depending on the
particular setup of the pipeline and the models utilized it can be optional.
For the non-trainable legacy bounding box segmenter binarization is mandatory
although it is still possible to feed color and grayscale images to the
recognizer. The trainable baseline segmenter can work with black and white,
grayscale, and color images, depending on the training data and network
configuration utilized; though grayscale and color data are used in almost all
cases.

.. code-block:: python

        >>> from PIL import Image

        >>> from kraken import binarization

        # can be any supported image format and mode
        >>> im = Image.open('foo.png')
        >>> bw_im = binarization.nlbin(im)

Legacy segmentation
~~~~~~~~~~~~~~~~~~~

The basic parameter of the legacy segmenter consists just of a b/w image
object, although some additional parameters exist, largely to change the
principal text direction (important for column ordering and top-to-bottom
scripts) and explicit masking of non-text image regions:

.. code-block:: python

        >>> from kraken import pageseg

        >>> seg = pageseg.segment(bw_im)
        >>> seg
        Segmentation(type='bbox',
                     imagename='foo.png',
                     text_direction='horizontal-lr',
                     script_detection=False,
                     lines=[BBoxLine(id='0ce11ad6-1f3b-4f7d-a8c8-0178e411df69',
                                     bbox=[74, 61, 136, 101],
                                     text=None,
                                     base_dir=None,
                                     type='bbox',
                                     imagename=None,
                                     tags=None,
                                     split=None,
                                     regions=None,
                                     text_direction='horizontal-lr'),
                            BBoxLine(id='c4a751dc-6731-4eea-a287-d4b57683f5b0', ...),
                            ....],
                     regions={},
                     line_orders=[])

All segmentation methods return a :class:`kraken.containers.Segmentation`
object that contains all elements of the segmentation: its type, a list of
lines (either :class:`kraken.containers.BBoxLine` or
:class:`kraken.containers.BaselineLine`), a dictionary mapping region types to
lists of regions (:class:`kraken.containers.Region`), and one or more line
reading orders.
                  
Baseline segmentation
~~~~~~~~~~~~~~~~~~~~~

The baseline segmentation method is based on a neural network that classifies
image pixels into baselines and regions. Because it is trainable, a
segmentation model is required in addition to the image to be segmented and
it has to be loaded first:

.. code-block:: python

        >>> from kraken import blla
        >>> from kraken.lib import vgsl

        >>> model_path = 'path/to/model/file'
        >>> model = vgsl.TorchVGSLModel.load_model(model_path)

A segmentation model contains a basic neural network and associated metadata
defining the available line and region types, bounding regions, and an
auxiliary baseline location flag for the polygonizer:

.. raw:: html
    :file: _static/kraken_segmodel.svg

Afterwards they can be fed into the segmentation method
:func:`kraken.blla.segment` with image objects:

.. code-block:: python

        >>> from kraken import blla
        >>> from kraken import serialization

        >>> baseline_seg = blla.segment(im, model=model)
        >>> baseline_seg
        Segmentation(type='baselines',
                     imagename='foo.png',
                     text_direction='horizontal-lr',
                     script_detection=False,
                     lines=[BaselineLine(id='22fee3d1-377e-4130-b9e5-5983a0c50ce8',
                                         baseline=[[71, 93], [145, 92]],
                                         boundary=[[71, 93], ..., [71, 93]], 
                                         text=None,
                                         base_dir=None,
                                         type='baselines',
                                         imagename=None,
                                         tags={'type': 'default'},
                                         split=None,
                                         regions=['f17d03e0-50bb-4a35-b247-cb910c0aaf2b']),
                            BaselineLine(id='539eadce-f795-4bba-a785-c7767d10c407', ...), ...],
                     regions={'text': [Region(id='f17d03e0-50bb-4a35-b247-cb910c0aaf2b',
                                              boundary=[[277, 54], ..., [277, 54]],
                                              imagename=None,
                                              tags={'type': 'text'})]},
                     line_orders=[])                     
        >>> alto = serialization.serialize(baseline_seg,
                                           image_size=im.size,
                                           template='alto')
        >>> with open('segmentation_output.xml', 'w') as fp:
                fp.write(alto)

A default segmentation model is supplied and will be used if none is specified
explicitly as an argument.  Optional parameters are largely the same as for the
legacy segmenter, i.e. text direction and masking.

Images are automatically converted into the proper mode for recognition, except
in the case of models trained on binary images as there is a plethora of
different algorithms available, each with strengths and weaknesses. For most
material the kraken-provided binarization should be sufficient, though. This
does not mean that a segmentation model trained on RGB images will have equal
accuracy for B/W, grayscale, and RGB inputs. Nevertheless the drop in quality
will often be modest or non-existent for color models while non-binarized
inputs to a binary model will cause severe degradation (and a warning to that
notion).

Per default segmentation is performed on the CPU although the neural network
can be run on a GPU with the `device` argument. As the vast majority of the
processing required is postprocessing the performance gain will most likely
modest though.

The above API is the most simple way to perform a complete segmentation. The
process consists of multiple steps such as pixel labelling, separate region and
baseline vectorization, and bounding polygon calculation:

.. raw:: html
    :file: _static/kraken_segmentation.svg

It is possible to only run a subset of the functionality depending on one's
needs by calling the respective functions in :mod:`kraken.lib.segmentation`. As
part of the sub-library the API is not guaranteed to be stable but it generally
does not change much. Examples of more fine-grained use of the segmentation API
can be found in `contrib/repolygonize.py
<https://github.com/mittagessen/kraken/blob/main/kraken/contrib/repolygonize.py>`_
and `contrib/segmentation_overlay.py
<https://github.com/mittagessen/kraken/blob/main/kraken/contrib/segmentation_overlay.py>`_.

Recognition
-----------

Recognition itself is a multi-step process with a neural network producing a
matrix with a confidence value for possible outputs at each time step. This
matrix is decoded into a sequence of integer labels (*label domain*) which are
subsequently mapped into Unicode code points using a codec. Labels and code
points usually correspond one-to-one, i.e. each label is mapped to exactly one
Unicode code point, but if desired more complex codecs can map single labels to
multiple code points, multiple labels to single code points, or multiple labels
to multiple code points (see the :ref:`Codec <codecs>` section for further
information).

.. _recognition_steps:

.. raw:: html
    :file: _static/kraken_recognition.svg

As the customization of this two-stage decoding process is usually reserved
for specialized use cases, sensible defaults are chosen by default: codecs are
part of the model file and do not have to be supplied manually; the preferred
CTC decoder is an optional parameter of the recognition model object.

To perform text line recognition a neural network has to be loaded first. A
:class:`kraken.lib.models.TorchSeqRecognizer` is returned which is a wrapper
around the :class:`kraken.lib.vgsl.TorchVGSLModel` class seen above for
segmentation model loading.

.. code-block:: python

        >>> from kraken.lib import models

        >>> rec_model_path = '/path/to/recognition/model'
        >>> model = models.load_any(rec_model_path)

The sequence recognizer wrapper combines the neural network itself, a
:ref:`codec <codecs>`, metadata such as if the input is supposed to be
grayscale or binarized, and an instance of a CTC decoder that performs the
conversion of the raw output tensor of the network into a sequence of labels:

.. raw:: html
    :file: _static/kraken_torchseqrecognizer.svg

Afterwards, given an image, a segmentation and the model one can perform text
recognition. The code is identical for both legacy and baseline segmentations.
Like for segmentation input images are auto-converted to the correct color
mode, except in the case of binary models for which a warning will be raised if
there is a mismatch.

There are two methods for recognition, a basic single model call
:func:`kraken.rpred.rpred` and a multi-model recognizer
:func:`kraken.rpred.mm_rpred`. The latter is useful for recognizing
multi-scriptal documents, i.e. applying different models to different parts of
a document.

.. code-block:: python

        >>> from kraken import rpred
        # single model recognition
        >>> pred_it = rpred(network=model,
                            im=im,
                            segmentation=baseline_seg)
        >>> for record in pred_it:
                print(record)

The output isn't just a sequence of characters but, depending on the type of
segmentation supplied, a :class:`kraken.containers.BaselineOCRRecord` or
:class:`kraken.containers.BBoxOCRRecord` record object containing the character
prediction, cuts (approximate locations), and confidences.

.. code-block:: python

        >>> record.cuts
        >>> record.prediction
        >>> record.confidences

it is also possible to access the original line information:

.. code-block:: python

        # for baselines
        >>> record.type
        'baselines'
        >>> record.line
        >>> record.baseline
        >>> record.script

        # for box lines
        >>> record.type
        'bbox'
        >>> record.line
        >>> record.script

Sometimes the undecoded raw output of the network is required. The :math:`C
\times W` softmax output matrix is accessible as the `outputs` attribute on the
:class:`kraken.lib.models.TorchSeqRecognizer` after each step of the
:func:`kraken.rpred.rpred` iterator. To get a mapping from the label space
:math:`C` the network operates in to Unicode code points a codec is used. An
arbitrary sequence of labels can generate an arbitrary number of Unicode code
points although usually the relation is one-to-one.

.. code-block:: python

        >>> pred_it = rpred(model, im, baseline_seg)
        >>> next(pred_it)
        >>> model.output
        >>> model.codec.l2c
        {'\x01': ' ',
         '\x02': '"',
         '\x03': "'",
         '\x04': '(',
         '\x05': ')',
         '\x06': '-',
         '\x07': '/',
         ...
        }

There are several different ways to convert the output matrix to a sequence of
labels that can be decoded into a character sequence. These are contained in
:mod:`kraken.lib.ctc_decoder` with
:func:`kraken.lib.ctc_decoder.greedy_decoder` being the default.

XML Parsing
-----------

Sometimes it is desired to take the data in an existing XML serialization
format like PageXML or ALTO and apply an OCR function on it. The
:mod:`kraken.lib.xml` module includes parsers extracting information into data
structures processable with minimal transformation by the functional blocks:

Parsing is accessed is through the :class:`kraken.lib.xml.XMLPage` class.

.. code-block:: python

        >>> from kraken.lib import xml

        >>> alto_doc = '/path/to/alto'
        >>> parsed_doc = xml.XMLPage(alto_doc)
        >>> parsed_doc
        XMLPage(filename='/path/to/alto', filetype=alto)
        >>> parsed_doc.lines
        {'line_1469098625593_463': BaselineLine(id='line_1469098625593_463',
                                                baseline=[(2337, 226), (2421, 239)],
                                                boundary=[(2344, 182), (2428, 195), (2420, 244), (2336, 231)],
                                                text='$pag:39',
                                                base_dir=None,
                                                type='baselines',
                                                imagename=None,
                                                tags={'type': '$pag'},
                                                split=None,
                                                regions=['region_1469098609000_462']),
 
         'line_1469098649515_464': BaselineLine(id='line_1469098649515_464',
                                                baseline=[(789, 269), (2397, 304)],
                                                boundary=[(790, 224), (2398, 259), (2397, 309), (789, 274)],
                                                text='$-nor su hijo, De todos sus bienes, con los pactos',
                                                base_dir=None,
                                                type='baselines',
                                                imagename=None,
                                                tags={'type': '$pac'},
                                                split=None,
                                                regions=['region_1469098557906_461']),
         ....}
        >>> parsed_doc.regions
        {'$pag': [Region(id='region_1469098609000_462',
                         boundary=[(2324, 171), (2437, 171), (2436, 258), (2326, 237)],
                         imagename=None,
                         tags={'type': '$pag'})],
         '$pac': [Region(id='region_1469098557906_461',
                         boundary=[(738, 203), (2339, 245), (2398, 294), (2446, 345), (2574, 469), (2539, 1873), (2523, 2053), (2477, 2182), (738, 2243)],
                         imagename=None,
                         tags={'type': '$pac'})],
         '$tip': [Region(id='TextRegion_1520586482298_194',
                         boundary=[(687, 2428), (688, 2422), (107, 2420), (106, 2264), (789, 2256), (758, 2404)],
                         imagename=None,
                         tags={'type': '$tip'})],
         '$par': [Region(id='TextRegion_1520586482298_193',
                         boundary=[(675, 3772), (687, 2428), (758, 2404), (789, 2256), (2542, 2236), (2581, 3748)], 
                         imagename=None,
                         tags={'type': '$par'})]
        }

The parser is aware of reading order(s), thus the basic properties accessing
lines and regions are unordered dictionaries. Reading orders can be accessed
separately through the `reading_orders` property:

.. code-block:: python

        >>> parsed_doc.region_orders
        {'line_implicit': {'order': ['line_1469098625593_463',
                                     'line_1469098649515_464',
                                     ...
                                    'line_1469099255968_508'],
                           'is_total': True,
                           'description': 'Implicit line order derived from element sequence'},
        'region_implicit': {'order': ['region_1469098609000_462',
                                      ...
                                     'TextRegion_1520586482298_193'],
                            'is_total': True,
                            'description': 'Implicit region order derived from element sequence'},
        'region_transkribus': {'order': ['region_1469098609000_462',
                                         ...
                                        'TextRegion_1520586482298_193'],
                            'is_total': True,
                            'description': 'Explicit region order from `custom` attribute'},
        'line_transkribus': {'order': ['line_1469098625593_463',
                                       ...
                                       'line_1469099255968_508'],
                             'is_total': True,
                             'description': 'Explicit line order from `custom` attribute'},
        'o_1530717944451': {'order': ['region_1469098609000_462',
                                      ...
                                      'TextRegion_1520586482298_193'],
                           'is_total': True,
                           'description': 'Regions reading order'}}

Reading orders are created from different sources, depending on the content of
the XML file. Every document will contain at least implicit orders for lines
and regions (`line_implicit` and `region_implicit`) sourced from the sequence
of line and region elements. There can also be explicit additional orders
defined by the standard reading order elements, for example `o_1530717944451`
in the above example. In Page XML files reading orders defined with the
Transkribus style custom attribute are also recognized.

To access the lines or regions of a document in a particular order:

.. code-block:: python

        >>> parsed_doc.get_sorted_lines(ro='line_implicit')
        [BaselineLine(id='line_1469098625593_463',
                      baseline=[(2337, 226), (2421, 239)],
                      boundary=[(2344, 182), (2428, 195), (2420, 244), (2336, 231)],
                      text='$pag:39',
                      base_dir=None,
                      type='baselines',
                      imagename=None,
                      tags={'type': '$pag'},
                      split=None,
                      regions=['region_1469098609000_462']),
         BaselineLine(id='line_1469098649515_464',
                      baseline=[(789, 269), (2397, 304)],
                      boundary=[(790, 224), (2398, 259), (2397, 309), (789, 274)],
                      text='$-nor su hijo, De todos sus bienes, con los pactos',
                      base_dir=None,
                      type='baselines',
                      imagename=None,
                      tags={'type': '$pac'},
                      split=None,
                      regions=['region_1469098557906_461'])
        ...]

The recognizer functions do not accept :class:`kraken.lib.xml.XMLPage` objects
directly which means that for most practical purposes these need to be
converted into :class:`container <kraken.containers.Segmentation>` objects:

.. code-block:: python

        >>> segmentation = parsed_doc.to_container()
        >>> pred_it = rpred(network=model,
                            im=im,
                            segmentation=segmentation)
        >>> for record in pred_it:
                print(record)


Serialization
-------------


The serialization module can be used to transform results returned by the
segmenter or recognizer into a text based (most often XML) format for archival.
The module renders `jinja2 <https://jinja.palletsprojects.com>`_ templates,
either ones :ref:`packaged <templates>` with kraken or supplied externally,
through the :func:`kraken.serialization.serialize` function.

.. code-block:: python

        >>> import dataclasses
        >>> from kraken.lib import serialization

        >>> alto_seg_only = serialization.serialize(baseline_seg, image_size=im.size, template='alto')

        >>> records = [record for record in pred_it]
        >>> results = dataclasses.replace(pred_it.bounds, lines=records)
        >>> alto = serialization.serialize(results, image_size=im.size, template='alto')
        >>> with open('output.xml', 'w') as fp:
                fp.write(alto)

The serialization function accepts arbitrary
:class:`kraken.containers.Segmentation` objects, which may contain textual or
only segmentation information. As the recognizer returns
:class:`ocr_records <kraken.containers.ocr_record>` which cannot be serialized
directly it is necessary to either construct a new
:class:`kraken.containers.Segmentation` from scratch or insert them into the
segmentation fed into the recognizer (:class:`ocr_records
<kraken.containers.ocr_record>` subclass :class:`BaselineLine
<kraken.containers.BaselineLine>`/:class:`BBoxLine
<kraken.containers.BBoxLine>` The container classes are immutable data classes,
therefore it is necessary for simple insertion of the records to use
`dataclasses.replace` to create a new segmentation with a changed lines
attribute.

Training
--------

Training is largely implemented with the `pytorch lightning
<https://www.pytorchlightning.ai/>`_ framework. There are separate
`LightningModule`s for recognition and segmentation training and a small
wrapper around the lightning's `Trainer` class that mainly sets up model
handling and verbosity options for the CLI.


.. code-block:: python

        >>> from kraken.lib.train import RecognitionModel, KrakenTrainer
        >>> ground_truth = glob.glob('training/*.xml')
        >>> training_files = ground_truth[:250] # training data is shuffled internally
        >>> evaluation_files = ground_truth[250:]
        >>> model = RecognitionModel(training_data=training_files, evaluation_data=evaluation_files, format_type='xml', augment=True)
        >>> trainer = KrakenTrainer()
        >>> trainer.fit(model)

Likewise for a baseline and region segmentation model:

.. code-block:: python

        >>> from kraken.lib.train import SegmentationModel, KrakenTrainer
        >>> ground_truth = glob.glob('training/*.xml')
        >>> training_files = ground_truth[:250] # training data is shuffled internally
        >>> evaluation_files = ground_truth[250:]
        >>> model = SegmentationModel(training_data=training_files, evaluation_data=evaluation_files, format_type='xml', augment=True)
        >>> trainer = KrakenTrainer()
        >>> trainer.fit(model)

When the `fit()` method is called the dataset is initialized and the training
commences. Both can take quite a bit of time. To get insight into what exactly
is happening the standard `lightning callbacks
<https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#callbacks>`_
can be attached to the trainer object:

.. code-block:: python

        >>> from pytorch_lightning.callbacks import Callback
        >>> from kraken.lib.train import RecognitionModel, KrakenTrainer
        >>> class MyPrintingCallback(Callback):
            def on_init_start(self, trainer):
                print("Starting to init trainer!")

            def on_init_end(self, trainer):
                print("trainer is init now")

            def on_train_end(self, trainer, pl_module):
                print("do something when training ends")
        >>> ground_truth = glob.glob('training/*.xml')
        >>> training_files = ground_truth[:250] # training data is shuffled internally
        >>> evaluation_files = ground_truth[250:]
        >>> model = RecognitionModel(training_data=training_files, evaluation_data=evaluation_files, format_type='xml', augment=True)
        >>> trainer = KrakenTrainer(enable_progress_bar=False, callbacks=[MyPrintingCallback])
        >>> trainer.fit(model)
        Starting to init trainer!
        trainer is init now

This is only a small subset of the training functionality. It is suggested to
have a closer look at the command line parameters for features as transfer
learning, region and baseline filtering, training continuation, and so on.
