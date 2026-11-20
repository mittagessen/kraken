API Quickstart 
==============

Kraken provides routines which are usable by third party tools to access all
functionality of the OCR engine. Most functional blocks, binarization,
segmentation, recognition, and serialization  are encapsulated in one high
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
grayscale, and color images, depending on the training data and netork
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
        {'text_direction': 'horizontal-lr',
         'boxes': [[0, 29, 232, 56],
                   [28, 54, 121, 84],
                   [9, 73, 92, 117],
                   [103, 76, 145, 131],
                   [7, 105, 119, 230],
                   [10, 228, 126, 345],
                   ...
                  ],
         'script_detection': False}

Baseline segmentation
~~~~~~~~~~~~~~~~~~~~~

The baseline segmentation method is based on a neural network that classifies
image pixels into baselines and regions. Because it is trainable, a
segmentation model is required in addition to the image to be segmentation and
it has to be loaded first:

.. code-block:: python

        >>> from kraken import blla
        >>> from kraken.lib import vgsl

        >>> model_path = 'path/to/model/file'
        >>> model = vgsl.TorchVGSLModel.load_model(model_path)
               
Afterwards they can be fed into the segmentation method
:func:`kraken.blla.segment` with image objects:

.. code-block:: python

        >>> from kraken import blla

        >>> baseline_seg = blla.segment(im, model=model)
        >>> baseline_seg
        {'text_direction': 'horizontal-lr',
         'type': 'baselines',
         'script_detection': False,
         'lines': [{'script': 'default',
                    'baseline': [[471, 1408], [524, 1412], [509, 1397], [1161, 1412], [1195, 1412]],
                    'boundary': [[471, 1408], [491, 1408], [515, 1385], [562, 1388], [575, 1377], ... [473, 1410]]},
                   ...],
         'regions': {'$tip':[[[536, 1716], ... [522, 1708], [524, 1716], [536, 1716], ...]
                     '$par': ...
                     '$nop':  ...}}

Optional parameters are largely the same as for the legacy segmenter, i.e. text
direction and masking.

Images are automatically converted into the proper mode for recognition, except
in the case of models trained on binary images as there is a plethora of
different algorithms available, each with strengths and weaknesses. For most
material the kraken-provided binarization should be sufficient, though. This
does not mean that a segmentation model trained on RGB images will have equal
accuracy for B/W, grayscale, and RGB inputs. Nevertheless the drop in quality
will often be modest or non-existant in for color models while non-binarized
inputs to a binary model will cause severe degradation (and a warning to that
notion).

Per default segmentation is performed on the CPU although the neural network
can be run on a GPU with the `device` argument. As the vast majority of the
processing required is postprocessing the performance gain will most likely
modest though.

Recognition
-----------

The character recognizer is equally based on a neural network which has to be
loaded first. 

.. code-block:: python

        >>> from kraken.lib import models

        >>> rec_model_path = '/path/to/recognition/model'
        >>> model = models.load_any(rec_model_path)
        
Afterwards, given an image, a segmentation and the model one can perform text
recognition. The code is identical for both legacy and baseline segmentations.
Like for segmentation input images are auto-converted to the correct color
mode, except in the case of binary models and a warning will be raised if there
is a mismatch for binary input models.

There are two methods for recognition, a basic single model call
:func:`kraken.rpred.rpred` and a multi-model recognizer
:func:`kraken.rpred.mm_rpred`. The latter is useful for recognizing
multi-scriptal documents, i.e. applying different models to different parts of
a document.

.. code-block:: python

        >>> from kraken import rpred
        # single model recognition
        >>> pred_it = rpred(model, im, baseline_seg)
        >>> for record in pred_it:
                print(record)

The output isn't just a sequence of characters but a record object containing
the character prediction, cuts (approximate locations), and confidences.

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
        'box'
        >>> record.line
        >>> record.script

Sometimes the undecoded raw output of the network is required. The :math:`C
\times W` softmax output matrix is accessible as an attribute on the
:class:`kraken.lib.models.TorchSeqRecognizer` after each step of the :func:`kraken.rpred.rpred` iterator. To get a mapping
from the label space :math:`C` the network operates in to Unicode code points a
codec is used. An arbitrary sequence of labels can generate an arbitrary number
of Unicode code points although usually the relation is one-to-one.

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
structures processable with minimal transformtion by the functional blocks:

.. code-block:: python

        >>> from kraken.lib import xml

        >>> alto_doc = '/path/to/alto'
        >>> xml.parse_alto(alto_doc)
        {'image': '/path/to/image/file',
         'type': 'baselines',
         'lines': [{'baseline': [(24, 2017), (25, 2078)],
                    'boundary': [(69, 2016), (70, 2077), (20, 2078), (19, 2017)],
                    'text': '',
                    'script': 'default'},
                   {'baseline': [(79, 2016), (79, 2041)],
                    'boundary': [(124, 2016), (124, 2041), (74, 2041), (74, 2016)],
                    'text': '',
                    'script': 'default'}, ...],
         'regions': {'Image/Drawing/Figure': [[(-5, 3398), (207, 3398), (207, 2000), (-5, 2000)],
                                              [(253, 3292), (668, 3292), (668, 3455), (253, 3455)],
                                              [(216, -4), (1015, -4), (1015, 534), (216, 534)]],
                     'Handwritten text': [[(2426, 3367), (2483, 3367), (2483, 3414), (2426, 3414)],
                                          [(1824, 3437), (2072, 3437), (2072, 3514), (1824, 3514)]],
                     ...}
        }

        >>> page_doc = '/path/to/page'
        >>> xml.parse_page(page_doc)
        {'image': '/path/to/image/file',
         'type': 'baselines',
         'lines': [{'baseline': [(24, 2017), (25, 2078)],
                    'boundary': [(69, 2016), (70, 2077), (20, 2078), (19, 2017)],
                    'text': '',
                    'script': 'default'},
                   {'baseline': [(79, 2016), (79, 2041)],
                    'boundary': [(124, 2016), (124, 2041), (74, 2041), (74, 2016)],
                    'text': '',
                    'script': 'default'}, ...],
         'regions': {'Image/Drawing/Figure': [[(-5, 3398), (207, 3398), (207, 2000), (-5, 2000)],
                                              [(253, 3292), (668, 3292), (668, 3455), (253, 3455)],
                                              [(216, -4), (1015, -4), (1015, 534), (216, 534)]],
                     'Handwritten text': [[(2426, 3367), (2483, 3367), (2483, 3414), (2426, 3414)],
                                          [(1824, 3437), (2072, 3437), (2072, 3514), (1824, 3514)]],
                     ...}


Serialization
-------------

The serialization module can be used to transform the :class:`ocr_records
<kraken.rpred.ocr_record>` returned by the prediction iterator into a text
based (most often XML) format for archival. The module renders `jinja2
<https://jinja.palletsprojects.com>`_ templates in `kraken/templates` through
the :func:`kraken.serialization.serialize` function.

.. code-block:: python

        >>> from kraken.lib import serialization

        >>> records = [record for record in pred_it]
        >>> alto = serialization.serialize(records, image_name='path/to/image', image_size=im.size, template='alto')
        >>> with open('output.xml', 'w') as fp:
                fp.write(alto)


Training
--------

There are catch-all constructors for quickly setting up
:cls:`kraken.lib.train.KrakenTrainer` instances for all training needs. They
largely map the comand line utils `ketos train` and `ketos segtrain` to a
programmatic interface. The arguments are identical, apart from a
differentiation between general arguments (data sources and setup, file names,
devices, ...) and hyperparameters (optimizers, learning rate schedules,
augmentation.

Training a recognition model from a number of xml files in ALTO or PAGE XML:

.. code-block:: python

        >>> from kraken.lib.train import KrakenTrainer
        >>> ground_truth = glob.glob('training/*.xml')
        >>> training_files = ground_truth[:250] # training data is shuffled internally
        >>> evaluation_files = ground_truth[250:]
        >>> trainer = KrakenTrainer.recognition_train_gen(training_data=training_files, evaluation_data=evaluation_files, format_type='xml', augment=True)
        >>> trainer.run()

Likewise for a baseline and region segmentation model:

.. code-block:: python

        >>> from kraken.lib.train import KrakenTrainer
        >>> ground_truth = glob.glob('training/*.xml')
        >>> training_files = ground_truth[:250] # training data is shuffled internally
        >>> evaluation_files = ground_truth[250:]
        >>> trainer = KrakenTrainer.segmentation_train_gen(training_data=training_files, evaluation_data=evaluation_files, format_type='xml', augment=True)
        >>> trainer.run()

Both constructing the trainer object and the training itself can take quite a
bit of time. The constructor provides a callback for each iterative process
during object initialization that is intended to set up a progress bar:

.. code-block:: python

        >>> from kraken.lib.train import KrakenTrainer

        >>> def progress_callback(string, length):
                print(f'starting process "{string}" of length {length}')
                return lambda: print('.', end='')
        >>> ground_truth = glob.glob('training/*.xml')
        >>> training_files = ground_truth[:25] # training data is shuffled internally
        >>> evaluation_files = ground_truth[25:95]
        >>> trainer = KrakenTrainer.segmentation_train_gen(training_data=training_files, evaluation_data=evaluation_files, format_type='xml', progress_callback=progress_callback, augment=True)
        starting process "Building training set" of length 25
        .........................
        starting process "Building validation set" of length 70
        ......................................................................      
        >>> trainer.run()

Executing the trainer object has two callbacks as arguments, one called after
each iteration and one returning the evaluation metrics after the end of each
epoch:

.. code-block:: python

        >>> from kraken.lib.train import KrakenTrainer
        >>> ground_truth = glob.glob('training/*.xml')
        >>> training_files = ground_truth[:250] # training data is shuffled internally
        >>> evaluation_files = ground_truth[250:]
        >>> trainer = KrakenTrainer.segmentation_train_gen(training_data=training_files, evaluation_data=evaluation_files, format_type='xml', augment=True)
        >>> def _update_progress():
                print('.', end='')
        >>> def _print_eval(epoch, accuracy, **kwargs):
                print(accuracy)
        >>> trainer.run(_print_eval, _update_progress)
        .........................0.0
        .........................0.0
        .........................0.0
        .........................0.0
        .........................0.0
        ...

The metrics differ for recognition
(:func:`kraken.lib.train.recognition_evaluator_fn`) and segmentation
(:func:`kraken.lib.train.baseline_label_evaluator_fn`).

Depending on the stopping method chosen the last model file might not be the
one with the best accuracy. Per default early stopping is used which aborts
training after a certain number of epochs without improvement. In that case the
best model and evaluation loss can be determined through:

.. code-block:: python

        >>> trainer.stopper.best_epoch
        >>> trainer.stopper.best_loss
        >>> best_model_path = f'{trainer.filename_prefix}_{trainer.stopper.best_epoch}.mlmodel'

This is only a small subset of the training functionality. It is suggested to
have a closer look at the command line parameters for features as transfer
learning, region and baseline filtering, training continuation, and so on.
