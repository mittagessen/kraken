.. _rectrain:

Recognition Training
====================

This page describes the recognition training command line utility in depth. For
a gentle introduction on model training please refer to the :ref:`tutorial
<training>`.

The main ``ketos`` command has some common options that configure how most of
its sub-commands use computational resources:

======================================================= ======
option                                                  action
======================================================= ======
-d, \--device                                           Select device to use (cpu, cuda:0, cuda:1,...). GPU acceleration requires CUDA.
\--precision                                            The precision to run neural network operations in. On GPU, bfloat16 can lead to considerable speedup.
\--workers                                              Number of workers processes to spawn for data loading, transformation, or compilation.
\--threads                                              Number of threads to use for intra-op parallelization like OpenMP, BLAS, ...
======================================================= ======


Best practices
--------------

Here is a list of best practices in recognition model training that will be
helpful both for experienced users and beginners having read through the
tutorial:

* The default architecture works well for decently sized datasets but is limited in its generalization capabilities. Generalized model that are intended to recognize very diverse material or a number of different languages can benefit from scaling up the depth and width of the network using the built-in :ref:`VGSL <vgsl>` language.
* Use precompiled binary datasets and put them in a place where they can be memory mapped during training (local storage, not NFS or similar).
* Fixed splits in precompiled datasets increase memory use and slow down startup as the dataset needs to be loaded once into the dataset. It is recommended to create explicit splits by compiling source XML files into separate datasets.
* Use the ``--logger`` flag to track your training metrics across experiments using Tensorboard.
* If the network doesn't converge before the early stopping aborts training, increase ``--min-epochs``. Training losses are logged on the progress bar and should decrease during training. Stable or oscillating losses can point to erroneous data preparation or wrong hyperparameter choice.
* Use the flag ``--augment`` to activate data augmentation.
* Increase the amount of ``--workers`` to speedup data loading. This is essential when you use the ``--augment`` option.
* When using an Nvidia GPU, set the ``--precision`` option to ``bf16-mixed`` to use automatic mixed precision (AMP). This can provide significant speedup without any loss in accuracy.
* Use option ``-B`` to scale batch size until GPU utilization reaches 100%. When using a larger batch size, it is recommended to use option -r to scale the learning rate by the square root of the batch size (1e-3 * sqrt(batch_size)).
* When fine-tuning, it is recommended to use ``new`` mode not ``union`` as the network will rapidly unlearn missing labels in the new dataset.
* If the new dataset is fairly dissimilar or your base model has been pretrained with ketos pretrain, use ``--warmup`` in conjunction with ``--freeze-backbone`` for one 1 or 2 epochs.
* Upload your models to the :ref:`model repository <repo>`.

Training data formats
---------------------

Recognition training supports the XML formats described in the :ref:`data
format <ketos_format>` section, compiled binary dataset files created from XML
files, and a legacy format of line strip images with separate transcription
text files. For performance reasons it is recommended to use binary datasets
whenever possible.

Legacy Line Strip Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~

Before the advent of trainable segmenters using the baseline paradigm a common
format for text recognition training consisted of individual line images cut
out of the page and parallel text files containing the transcription, sharing a
common file name prefix. An example of such a dataset can be found `here
<https://github.com/chreul/OCR_Testdata_EarlyPrintedBooks>`_. 

Kraken still supports training models from data in this format, albeit the
models trained on this kind of data are not compatible with baseline
segmentation so little is to be gained using such datasets.

Files are expected to share a prefix until the last extension of the image file
with the corresponding ground truth transcription having a ``.gt.txt`` suffix,
for example ``line_img_0.train.png``` and ``line_img_0.train.gt.txt``. To
enable this legacy format use the ``path`` value with the ``--format-type``
option as with other support formats.

Training
--------

The training utility allows training of :ref:`VGSL <vgsl>` specified models
both from scratch and from existing models. Here are its most important command
line options:

======================================================= ======
option                                                  action
======================================================= ======
-o, \--output                                           Output model file prefix. Defaults to model.
-s, \--spec                                             VGSL spec of the network to train. CTC layer will be added automatically. default: [1,48,0,1 Cr3,3,32 Do0.1,2 Mp2,2 Cr3,3,64 Do0.1,2 Mp2,2 S1(1x12)1,3 Lbx100 Do]
-a, \--append                                           Removes layers before argument and then appends spec. Only works when loading an existing model
-i, \--load                                             Load existing file to continue training
-F, \--savefreq                                         Model save frequency in epochs during training
-q, \--quit                                             Stop condition for training. Set to `early` for early stopping (default) or `fixed` for fixed number of epochs.
-N, \--epochs                                           Number of epochs to train for.
\--min-epochs                                           Minimum number of epochs to train for when using early stopping.
\--lag                                                  Number of epochs to wait before stopping training without improvement. Only used when using early stopping.
\--optimizer                                            Select optimizer (Adam, AdamW, SGD, RMSprop).
-r, \--lrate                                            Learning rate  [default: 0.001]
-m, \--momentum                                         Momentum used with SGD optimizer. Ignored otherwise.
-w, \--weight-decay                                     Weight decay.
\--schedule                                             Sets the learning rate scheduler. May be either constant, 1cycle, exponential, cosine, step, or reduceonplateau. For 1cycle the cycle length is determined by the `--epoch` option.
-p, \--partition                                        Ground truth data partition ratio between train/validation set
-u, \--normalization                                    Ground truth Unicode normalization. One of NFC, NFKC, NFD, NFKD.
-c, \--codec                                            Load a codec JSON definition (invalid if loading existing model)
\--resize                                               Codec/output layer resizing option. If set to `union` code points will be added, `new` will set the layer to match exactly the training data, `fail` will abort if training data and model codec do not match. Only valid when refining an existing model.
-n, \--reorder / \--no-reorder                          Reordering of code points to display order.
-t, \--training-files                                   File(s) with additional paths to training data. Used to enforce an explicit train/validation set split and deal with training sets with more lines than the command line can process. Can be used more than once.
-e, \--evaluation-files                                 File(s) with paths to evaluation data. Overrides the `-p` parameter.
-f, \--format-type                                      Sets the training and evaluation data format. Valid choices are 'path', 'xml' (default), 'alto', 'page', or binary. In `alto`, `page`, and xml mode all data is extracted from XML files containing both baselines and a link to source images. In `path` mode arguments are image files sharing a prefix up to the last extension with `.gt.txt` text files containing the transcription. In binary mode files are datasets files containing pre-extracted text lines.
\--augment / \--no-augment                              Enables/disables data augmentation.
======================================================= ======

From Scratch
~~~~~~~~~~~~

The absolute minimal example to train a new recognition model from a number of
ALTO or PAGE XML documents is similar to the segmentation training:

.. code-block:: console

        $ ketos train -f xml training_data/*.xml

Training will continue until the validation metric does not improve anymore and
the best model (among intermediate results) will be saved in the current
directory; this approach is called early stopping.

In some cases changing the network architecture might be useful. One such
example would be material that is not well recognized in the grayscale domain,
as the default architecture definition converts images into grayscale. The
input definition can be changed quite easily to train on color data (RGB) instead:

.. code-block:: console

        $ ketos train -f page -s '[1,120,0,3 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]]' syr/*.xml

Complete documentation for the network description language can be found on the
:ref:`VGSL <vgsl>` page.

Sometimes the early stopping default parameters might produce suboptimal
results such as stopping training too soon, for example when training 

.. code-block:: console

        $ ketos train -f page --lag 10 syr/*.xml

To switch optimizers from Adam to any other supported algorithm just set the
option:

.. code-block:: console

        $ ketos train --optimizer SGD syr/*.png

It is possible to resume training from a previously saved model:

.. code-block:: console

        $ ketos train -i model_25.mlmodel syr/*.png

A good configuration for a small precompiled print dataset and GPU acceleration
would be:

.. code-block:: console

        $ ketos -d cuda train -f binary dataset.arrow

A better configuration for large and complicated datasets such as handwritten texts:

.. code-block:: console

        $ ketos --workers 4 -d cuda train --augment-f binary --min-epochs 20 -w 0 -s '[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do.1,2 Lbx200 Do]' -r 0.0001 dataset_large.arrow

This configuration is slower to train and often requires a couple of epochs to
output any sensible text at all. Therefore we tell ketos to train for at least
20 epochs so the early stopping algorithm doesn't prematurely interrupt the
training process.

Fine Tuning
~~~~~~~~~~~

Fine tuning an existing model for another typeface or new characters is also
possible with the same syntax as resuming regular training:

.. code-block:: console

        $ ketos train -f page -i model_best.mlmodel syr/*.xml

The caveat is that the alphabet of the base model and training data have to be
an exact match. Otherwise an error will be raised:

.. code-block:: console

        $ ketos train -i model_5.mlmodel kamil/*.xml
        Building training set  [####################################]  100%
        Building validation set  [####################################]  100%
        [0.8616] alphabet mismatch {'~', '»', '8', '9', 'ـ'}
        Network codec not compatible with training set
        [0.8620] Training data and model codec alphabets mismatch: {'ٓ', '؟', '!', 'ص', '،', 'ذ', 'ة', 'ي', 'و', 'ب', 'ز', 'ح', 'غ', '~', 'ف', ')', 'د', 'خ', 'م', '»', 'ع', 'ى', 'ق', 'ش', 'ا', 'ه', 'ك', 'ج', 'ث', '(', 'ت', 'ظ', 'ض', 'ل', 'ط', '؛', 'ر', 'س', 'ن', 'ء', 'ٔ', '«', 'ـ', 'ٕ'}

There are two modes dealing with mismatching alphabets, ``union`` and ``new``.
``union`` resizes the output layer and codec of the loaded model to include all
characters in the new training set without removing any characters. ``new``
will make the resulting model an exact match with the new training set by both
removing unused characters from the model and adding new ones.

.. code-block:: console

        $ ketos -v train --resize union -i model_5.mlmodel syr/*.xml
        ...
        [0.7943] Training set 788 lines, validation set 88 lines, alphabet 50 symbols
        ...
        [0.8337] Resizing codec to include 3 new code points
        [0.8374] Resizing last layer in network to 52 outputs
        ...

In this example 3 characters were added for a network that is able to
recognize 52 different characters after sufficient additional training.

.. code-block:: console

        $ ketos -v train --resize new -i model_5.mlmodel syr/*.xml
        ...
        [0.7593] Training set 788 lines, validation set 88 lines, alphabet 49 symbols
        ...
        [0.7857] Resizing network or given codec to 49 code sequences
        [0.8344] Deleting 2 output classes from network (46 retained)
        ...

In ``new`` mode 2 of the original characters were removed and 3 new ones were added.

Slicing
~~~~~~~

Refining on mismatched alphabets has its limits. If the alphabets are highly
different the modification of the final linear layer to add/remove character
will destroy the inference capabilities of the network. Even when this is the
case fine-tuning from a good base model will often produce better results, as
the model will not have to learn good features from the input data from
scratch, so there is usually no need to not utilize the standard fine-tuning
capabilities offered by `--resize`. Nevertheless, it is possible to
reinitialize layers of the network completely using a slicing mechanism.

Taking the default network definition as printed in the debug log (`ketos -vvv
train ...`) we can see the layer indices of the model:

.. code-block:: console

        [0.8760] Creating new model [1,48,0,1 Cr3,3,32 Do0.1,2 Mp2,2 Cr3,3,64 Do0.1,2 Mp2,2 S1(1x12)1,3 Lbx100 Do] with 48 outputs
        [0.8762] layer		type	params
        [0.8790] 0		conv	kernel 3 x 3 filters 32 activation r
        [0.8795] 1		dropout	probability 0.1 dims 2
        [0.8797] 2		maxpool	kernel 2 x 2 stride 2 x 2
        [0.8802] 3		conv	kernel 3 x 3 filters 64 activation r
        [0.8804] 4		dropout	probability 0.1 dims 2
        [0.8806] 5		maxpool	kernel 2 x 2 stride 2 x 2
        [0.8813] 6		reshape from 1 1 x 12 to 1/3
        [0.8876] 7		rnn	direction b transposed False summarize False out 100 legacy None
        [0.8878] 8		dropout	probability 0.5 dims 1
        [0.8883] 9		linear	augmented False out 48

To remove everything after the initial convolutional stack and add untrained
layers we define a network stub and index for appending:

.. code-block:: console

        $ ketos train -i model_1.mlmodel --append 7 -s '[Lbx256 Do]' syr/*.xml
        Building training set  [####################################]  100%
        Building validation set  [####################################]  100%
        [0.8014] alphabet mismatch {'8', '3', '9', '7', '܇', '݀', '݂', '4', ':', '0'}
        Slicing and dicing model ✓

The new model will behave exactly like a new one, except potentially training a
lot faster.

Text Normalization and Unicode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _text_norm:

.. note:

   The description of the different behaviors of Unicode text below are highly
   abbreviated. If confusion arrises it is recommended to take a look at the
   linked documents which are more exhaustive and include visual examples.

Text can be encoded in multiple different ways when using Unicode. For many
scripts characters with diacritics can be encoded either as a single code point
or a base character and the diacritic, `different types of whitespace
<https://jkorpela.fi/chars/spaces.html>`_ exist, and mixed bidirectional text
can be written differently depending on the `base line direction
<https://www.w3.org/International/articles/inline-bidi-markup/uba-basics#context>`_.

Ketos provides options to largely normalize input into normalized forms that
make processing of data from multiple sources possible. Principally, two
options are available: one for `Unicode normalization
<https://unicode.org/reports/tr15/>`_ and one for whitespace normalization. The
Unicode normalization (disabled per default) switch allows one to select one of
the 4 normalization forms:

.. code-block:: console

   $ ketos train --normalization NFD -f xml training_data/*.xml
   $ ketos train --normalization NFC -f xml training_data/*.xml
   $ ketos train --normalization NFKD -f xml training_data/*.xml
   $ ketos train --normalization NFKC -f xml training_data/*.xml

Whitespace normalization is enabled per default and converts all Unicode
whitespace characters into a simple space (U+0020). It is highly recommended to
leave this function enabled as the variation of space width, resulting either
from text justification or the irregularity of handwriting, is difficult for a
recognition model to accurately model and map onto the different space code
points. Nevertheless it can be disabled through:

.. code-block:: console

   $ ketos train --no-normalize-whitespace -f xml training_data/*.xml

Further the behavior of the `BiDi algorithm
<https://unicode.org/reports/tr9/>`_ can be influenced through two options. The
configuration of the algorithm is important as the recognition network is
trained to output characters (or rather labels which are mapped to code points
by a :ref:`codec <codecs>`) in the order a line is fed into the network, i.e.
left-to-right also called display order. Unicode text is encoded as a stream of
code points in logical order, i.e. the order the characters in a line are read
in by a human reader, for example (mostly) right-to-left for a text in Hebrew.
The BiDi algorithm resolves this logical order to the display order expected by
the network and vice versa. The primary parameter of the algorithm is the base
direction which is just the default direction of the input fields of the user
when the ground truth was initially transcribed. Base direction will be
automatically determined by kraken when using PAGE XML or ALTO files that
contain it, otherwise it will have to be supplied if it differs from the
default when training a model:

.. code-block:: console

   $ ketos train --base-dir R -f xml rtl_training_data/*.xml

It is also possible to disable BiDi processing completely, e.g. when the text
has been brought into display order already:

.. code-block:: console

   $ ketos train --no-reorder -f xml rtl_display_data/*.xml

Codecs
~~~~~~

.. _codecs:

Codecs map between the label decoded from the raw network output and Unicode
code points (see :ref:`this <recognition_steps>` diagram for the precise steps
involved in text line recognition). Codecs are attached to a recognition model
and are usually defined once at initial training time, although they can be
adapted either explicitly (with the API) or implicitly through domain adaptation.

The default behavior of kraken is to auto-infer this mapping from all the
characters in the training set and map each code point to one separate label.
This is usually sufficient for alphabetic scripts, abjads, and abugidas apart
from very specialised use cases. Logographic writing systems with a very large
number of different graphemes, such as all the variants of Han characters or
Cuneiform, can be more problematic as their large inventory makes recognition
both slow and error-prone. In such cases it can be advantageous to decompose
each code point into multiple labels to reduce the output dimensionality of the
network. During decoding valid sequences of labels will be mapped to their
respective code points as usual.

There are multiple approaches one could follow constructing a custom codec:
*randomized block codes*, i.e. producing random fixed-length labels for each code
point, *Huffmann coding*, i.e. variable length label sequences depending on the
frequency of each code point in some text (not necessarily the training set),
or *structural decomposition*, i.e. describing each code point through a
sequence of labels that describe the shape of the grapheme similar to how some
input systems for Chinese characters function.

Custom codecs can be supplied as simple JSON files that contain a dictionary
mapping between strings and integer sequences, e.g.:

.. code-block:: console

   $ ketos train -c sample.codec -f xml training_data/*.xml

with `sample.codec` containing:

.. code-block:: json

   {"S": [50, 53, 74, 23],
    "A": [95, 60, 19, 95],
    "B": [2, 96, 28, 29],
    "\u1f05": [91, 14, 95, 90]}

Unsupervised recognition pretraining
------------------------------------

.. _pretrain:

Text recognition models can be pretrained in an unsupervised fashion from text
line images, both in bounding box and baseline format. The pretraining is
performed through a contrastive surrogate task aiming to distinguish in-painted
parts of the input image features from randomly sampled distractor slices.

All data sources accepted by the supervised trainer are valid for pretraining
but for performance reasons it is recommended to use pre-compiled binary
datasets. One thing to keep in mind is that compilation filters out empty
(non-transcribed) text lines per default which is undesirable for pretraining.
With the ``--keep-empty-lines`` option all valid lines will be written to the
dataset file:

.. code-block:: console

   $ ketos compile --keep-empty-lines -f xml -o foo.arrow *.xml

The basic pretraining call is very similar to a training one:

.. code-block:: console

   $ ketos pretrain -f binary foo.arrow

There are a couple of hyperparameters that are specific to pretraining: the
mask width (at the subsampling level of the last convolutional layer), the
probability of a particular position being the start position of a mask, and
the number of negative distractor samples.

.. code-block:: console

   $ ketos pretrain -o pretrain --mask-width 4 --mask-probability 0.2 --num-negatives 3 -f binary foo.arrow

Once a model has been pretrained it has to be adapted to perform actual
recognition with a standard labelled dataset, although training data
requirements will usually be much reduced:

.. code-block:: console

   $ ketos train -i pretrain_best.mlmodel --warmup 5000 --freeze-backbone 1000 -f binary labelled.arrow

It is necessary to use learning rate warmup (`--warmup`) for at least a couple
of epochs in addition to freezing the backbone (all but the last fully
connected layer performing the classification) to have the model converge
during fine-tuning. Fine-tuning models from pre-trained weights is quite a bit
less stable than training from scratch or fine-tuning an existing model. As
such it can be necessary to run a couple of trials with different
hyperparameters (principally learning rate) to find workable ones. It is
entirely possible that pretrained models do not converge at all even with
reasonable hyperparameter configurations.

Recognition testing
-------------------

Picking a particular model from a pool or getting a more detailed look on the
recognition accuracy can be done with the `test` command. It uses transcribed
lines, the test set, in the same formats supported by the `train` command,
recognizes the line images with one or more models, and creates a detailed
report of the differences from the ground truth for each of them.

======================================================= ======
option                                                  action
======================================================= ======
-f, \--format-type                                      Sets the test set data format.
                                                        Valid choices are 'path', 'xml' (default), 'alto', 'page', or binary.
                                                        In `alto`, `page`, and xml mode all data is extracted from XML files
                                                        containing both baselines and a link to source images.
                                                        In `path` mode arguments are image files sharing a prefix up to the last
                                                        extension with JSON `.path` files containing the baseline information.
                                                        In `binary` mode arguments are precompiled binary dataset files.
-m, \--model                                            Model(s) to evaluate.
-e, \--evaluation-files                                 File(s) with paths to evaluation data.
======================================================= ======

The `test` command supports the same text normalization options as `train`
which allows adapting unnormalized test sets to the normalization(s) used
during training.

Transcriptions are handed to the command in the same way as for the `train`
command, either through a manifest with ``-e/--evaluation-files`` or by just
adding a number of image files as the final argument:

.. code-block:: console

   $ ketos test -m $model -e test.txt test/*.xml
   Evaluating $model
   Evaluating  [####################################]  100%
   === report test_model.mlmodel ===

   7012	Characters
   6022	Errors
   14.12%	Accuracy

   5226	Insertions
   2	Deletions
   794	Substitutions

   Count Missed   %Right
   1567  575	63.31%	Common
   5230	 5230	0.00%	Arabic
   215	 215	0.00%	Inherited

   Errors	Correct-Generated
   773	{ ا } - {  }
   536	{ ل } - {  }
   328	{ و } - {  }
   274	{ ي } - {  }
   266	{ م } - {  }
   256	{ ب } - {  }
   246	{ ن } - {  }
   241	{ SPACE } - {  }
   207	{ ر } - {  }
   199	{ ف } - {  }
   192	{ ه } - {  }
   174	{ ع } - {  }
   172	{ ARABIC HAMZA ABOVE } - {  }
   144	{ ت } - {  }
   136	{ ق } - {  }
   122	{ س } - {  }
   108	{ ، } - {  }
   106	{ د } - {  }
   82	{ ك } - {  }
   81	{ ح } - {  }
   71	{ ج } - {  }
   66	{ خ } - {  }
   62	{ ة } - {  }
   60	{ ص } - {  }
   39	{ ، } - { - }
   38	{ ش } - {  }
   30	{ ا } - { - }
   30	{ ن } - { - }
   29	{ ى } - {  }
   28	{ ذ } - {  }
   27	{ ه } - { - }
   27	{ ARABIC HAMZA BELOW } - {  }
   25	{ ز } - {  }
   23	{ ث } - {  }
   22	{ غ } - {  }
   20	{ م } - { - }
   20	{ ي } - { - }
   20	{ ) } - {  }
   19	{ : } - {  }
   19	{ ط } - {  }
   19	{ ل } - { - }
   18	{ ، } - { . }
   17	{ ة } - { - }
   16	{ ض } - {  }
   ...
   Average accuracy: 14.12%, (stddev: 0.00)

The report(s) contains character accuracy measured per script and a detailed
list of confusions. When evaluating multiple models the last line of the output
will the average accuracy and the standard deviation across all of them.
