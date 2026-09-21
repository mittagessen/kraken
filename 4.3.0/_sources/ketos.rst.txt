.. _ketos:

Training
========

This page describes the training utilities available through the ``ketos``
command line utility in depth. For a gentle introduction on model training
please refer to the :ref:`tutorial <training>`. 

Both segmentation and recognition are trainable in kraken. The segmentation
model finds baselines and regions on a page image. Recognition models convert
text image lines found by the segmenter into digital text.

Training data formats
---------------------

The training tools accept a variety of training data formats, usually some kind
of custom low level format, the XML-based formats that are commony used for
archival of annotation and transcription data, and in the case of recognizer
training a precompiled binary format. It is recommended to use the XML formats
for segmentation training and the binary format for recognition training.

ALTO
~~~~

Kraken parses and produces files according to ALTO 4.2. An example showing the
attributes necessary for segmentation and recognition training follows:

.. literalinclude:: alto.xml
   :language: xml
   :force:

Importantly, the parser only works with measurements in the pixel domain, i.e.
an unset `MeasurementUnit` or one with an element value of `pixel`. In
addition, as the minimal version required for ingestion is quite new it is
likely that most existing ALTO documents will not contain sufficient
information to be used with kraken out of the box.

PAGE XML
~~~~~~~~

PAGE XML is parsed and produced according to the 2019-07-15 version of the
schema, although the parser is not strict and works with non-conformant output
of a variety of tools.

.. literalinclude:: pagexml.xml
   :language: xml
   :force:

Binary Datasets
~~~~~~~~~~~~~~~

.. _binary_datasets:

In addition to training recognition models directly from XML and image files, a
binary dataset format offering a couple of advantages is supported. Binary
datasets drastically improve loading performance allowing the saturation of
most GPUs with minimal computational overhead while also allowing training with
datasets that are larger than the systems main memory. A minor drawback is a
~30% increase in dataset size in comparison to the raw images + XML approach.

To realize this speedup the dataset has to be compiled first:

.. code-block:: console

   $ ketos compile -f xml -o dataset.arrow file_1.xml file_2.xml ...

if there are a lot of individual lines containing many lines this process can
take a long time. It can easily be parallelized by specifying the number of
separate parsing workers with the `--workers` option:

.. code-block:: console

   $ ketos compile --workers 8 -f xml ...

In addition, binary datasets can contain fixed splits which allow
reproducibility and comparability between training and evaluation runs.
Training, validation, and test splits can be pre-defined from multiple sources.
Per default they are sourced from tags defined in the source XML files unless
the option telling kraken to ignore them is set:

.. code-block:: console

   $ ketos compile --ignore-splits -f xml ...

Alternatively fixed-proportion random splits can be created ad-hoc during
compile time:

.. code-block:: console

   $ ketos compile --random-split 0.8 0.1 0.1 ...

The above line splits assigns 80% of the source lines to the training set, 10%
to the validation set, and 10% to the test set. The training and validation
sets in the dataset file are used automatically by `ketos train` (unless told
otherwise) while the remaining 10% of the test set is selected by `ketos test`.

Recognition training
--------------------

The training utility allows training of :ref:`VGSL <vgsl>` specified models
both from scratch and from existing models. Here are its most important command line options:

======================================================= ======
option                                                  action
======================================================= ======
-o, \--output                                           Output model file prefix. Defaults to model.
-s, \--spec                                             VGSL spec of the network to train. CTC layer
                                                        will be added automatically. default:
                                                        [1,48,0,1 Cr3,3,32 Do0.1,2 Mp2,2 Cr3,3,64
                                                        Do0.1,2 Mp2,2 S1(1x12)1,3 Lbx100 Do]
-a, \--append                                           Removes layers before argument and then
                                                        appends spec. Only works when loading an
                                                        existing model
-i, \--load                                             Load existing file to continue training
-F, \--savefreq                                         Model save frequency in epochs during
                                                        training
-q, \--quit                                             Stop condition for training. Set to `early`
                                                        for early stopping (default) or `dumb` for fixed
                                                        number of epochs.
-N, \--epochs                                           Number of epochs to train for.
\--min-epochs                                           Minimum number of epochs to train for when using early stopping.
\--lag                                                  Number of epochs to wait before stopping
                                                        training without improvement. Only used when using early stopping.
-d, \--device                                           Select device to use (cpu, cuda:0, cuda:1,...). GPU acceleration requires CUDA.
\--optimizer                                            Select optimizer (Adam, SGD, RMSprop).
-r, \--lrate                                            Learning rate  [default: 0.001]
-m, \--momentum                                         Momentum used with SGD optimizer. Ignored otherwise.
-w, \--weight-decay                                     Weight decay.
\--schedule                                             Sets the learning rate scheduler. May be either constant, 1cycle, exponential, cosine, step, or
                                                        reduceonplateau. For 1cycle the cycle length is determined by the `--epoch` option.
-p, \--partition                                        Ground truth data partition ratio between train/validation set
-u, \--normalization                                    Ground truth Unicode normalization. One of NFC, NFKC, NFD, NFKD.
-c, \--codec                                            Load a codec JSON definition (invalid if loading existing model)
\--resize                                               Codec/output layer resizing option. If set
                                                        to `add` code points will be added, `both`
                                                        will set the layer to match exactly the
                                                        training data, `fail` will abort if training
                                                        data and model codec do not match. Only valid when refining an existing model.
-n, \--reorder / \--no-reorder                          Reordering of code points to display order.
-t, \--training-files                                   File(s) with additional paths to training data. Used to
                                                        enforce an explicit train/validation set split and deal with
                                                        training sets with more lines than the command line can process. Can be used more than once.
-e, \--evaluation-files                                 File(s) with paths to evaluation data. Overrides the `-p` parameter.
-f, \--format-type                                      Sets the training and evaluation data format.
                                                        Valid choices are 'path', 'xml' (default), 'alto', 'page', or binary.
                                                        In `alto`, `page`, and xml mode all data is extracted from XML files
                                                        containing both baselines and a link to source images.
                                                        In `path` mode arguments are image files sharing a prefix up to the last
                                                        extension with JSON `.path` files containing the baseline information.
                                                        In `binary` mode arguments are precompiled binary dataset files.
\--augment / \--no-augment                              Enables/disables data augmentation.
\--workers                                              Number of OpenMP threads and workers used to perform neural network passes and load samples from the dataset.
======================================================= ======

From Scratch
~~~~~~~~~~~~

The absolute minimal example to train a new recognition model from a number of
ALTO or PAGE XML documents is similar to the segmentation training:

.. code-block:: console

        $ ketos train -f xml training_data/*.xml

Training will continue until the error does not improve anymore and the best
model (among intermediate results) will be saved in the current directory; this
approach is called early stopping.

In some cases, such as color inputs, changing the network architecture might be
useful:

.. code-block:: console

        $ ketos train -f page -s '[1,0,0,3 Cr3,3,16 Mp3,3 Lfys64 Lbx128 Lbx256 Do]' syr/*.xml

Complete documentation for the network description language can be found on the
:ref:`VGSL <vgsl>` page.

Sometimes the early stopping default parameters might produce suboptimal
results such as stopping training too soon. Adjusting the minimum delta an/or
lag can be useful:

.. code-block:: console

        $ ketos train --lag 10 --min-delta 0.001 syr/*.png

To switch optimizers from Adam to SGD or RMSprop just set the option:

.. code-block:: console

        $ ketos train --optimizer SGD syr/*.png

It is possible to resume training from a previously saved model:

.. code-block:: console

        $ ketos train -i model_25.mlmodel syr/*.png

A good configuration for a small precompiled print dataset and GPU acceleration
would be:

.. code-block:: console

        $ ketos train -d cuda -f binary dataset.arrow

A better configuration for large and complicated datasets such as handwritten texts:

.. code-block:: console 

        $ ketos train --augment --workers 4 -d cuda -f binary --min-epochs 20 -w 0 -s '[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do.1,2 Lbx200 Do]' -r 0.0001 dataset_large.arrow

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

        $ ketos train -i model_5.mlmodel kamil/*.png
        Building training set  [####################################]  100%
        Building validation set  [####################################]  100%
        [0.8616] alphabet mismatch {'~', '»', '8', '9', 'ـ'} 
        Network codec not compatible with training set
        [0.8620] Training data and model codec alphabets mismatch: {'ٓ', '؟', '!', 'ص', '،', 'ذ', 'ة', 'ي', 'و', 'ب', 'ز', 'ح', 'غ', '~', 'ف', ')', 'د', 'خ', 'م', '»', 'ع', 'ى', 'ق', 'ش', 'ا', 'ه', 'ك', 'ج', 'ث', '(', 'ت', 'ظ', 'ض', 'ل', 'ط', '؛', 'ر', 'س', 'ن', 'ء', 'ٔ', '«', 'ـ', 'ٕ'} 
        
There are two modes dealing with mismatching alphabets, ``add`` and ``both``.
``add`` resizes the output layer and codec of the loaded model to include all
characters in the new training set without removing any characters. ``both``
will make the resulting model an exact match with the new training set by both
removing unused characters from the model and adding new ones.

.. code-block:: console

        $ ketos -v train --resize add -i model_5.mlmodel syr/*.png
        ...
        [0.7943] Training set 788 lines, validation set 88 lines, alphabet 50 symbols
        ...
        [0.8337] Resizing codec to include 3 new code points
        [0.8374] Resizing last layer in network to 52 outputs
        ...

In this example 3 characters were added for a network that is able to
recognize 52 different characters after sufficient additional training.

.. code-block:: console

        $ ketos -v train --resize both -i model_5.mlmodel syr/*.png
        ...
        [0.7593] Training set 788 lines, validation set 88 lines, alphabet 49 symbols
        ...
        [0.7857] Resizing network or given codec to 49 code sequences
        [0.8344] Deleting 2 output classes from network (46 retained)
        ...

In ``both`` mode 2 of the original characters were removed and 3 new ones were added.

Slicing
~~~~~~~

Refining on mismatched alphabets has its limits. If the alphabets are highly
different the modification of the final linear layer to add/remove character
will destroy the inference capabilities of the network. In those cases it is
faster to slice off the last few layers of the network and only train those
instead of a complete network from scratch.

Taking the default network definition as printed in the debug log we can see
the layer indices of the model:

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

        $ ketos train -i model_1.mlmodel --append 7 -s '[Lbx256 Do]' syr/*.png 
        Building training set  [####################################]  100%
        Building validation set  [####################################]  100%
        [0.8014] alphabet mismatch {'8', '3', '9', '7', '܇', '݀', '݂', '4', ':', '0'} 
        Slicing and dicing model ✓

The new model will behave exactly like a new one, except potentially training a
lot faster.

Text Normalization and Unicode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
whitespace characters into a simple space. It is highly recommended to leave
this function enabled as the variation of space width, resulting either from
text justification or the irregularity of handwriting, is difficult for a
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

While the system is functional it is not well-tested in practice and it is
unclear which approach works best for which kinds of inputs.

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

Text recognition models can be pretrained in an unsupervised fashion from text
line images, both in bounding box and baseline format. The pretraining is
performed through a contrastive surrogate task aiming to distinguish in-painted
parts of the input image features from randomly sampled distractor slices.

All data sources accepted by the supervised trainer are valid for pretraining
but for performance reasons it is recommended to use pre-compiled binary
datasets. One thing to keep in mind is that compilation filters out empty
(non-transcribed) text lines per default which is undesirable for pretraining.
With the `--keep-empty-lines` option all valid lines will be written to the
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

It is necessary to use learning rate warmup (`warmup`) for at least a couple of
epochs in addition to freezing the backbone (all but the last fully connected
layer performing the classification) to have the model converge during
fine-tuning. Fine-tuning models from pre-trained weights is quite a bit less
stable than training from scratch or fine-tuning an existing model. As such it
can be necessary to run a couple of trials with different hyperparameters
(principally learning rate) to find workable ones. It is entirely possible that
pretrained models do not converge at all even with reasonable hyperparameter
configurations.

Segmentation training
---------------------

.. _segtrain:

Training a segmentation model is very similar to training models for text
recognition. The basic invocation is:

.. code-block:: console

        $ ketos segtrain -f xml training_data/*.xml
        Training line types:
          default 2	53980
          foo     8     134
        Training region types:
          graphic	3	135
          text	4	1128
          separator	5	5431
          paragraph	6	10218
          table	7	16
        val check  [------------------------------------]  0/0

This takes all text lines and regions encoded in the XML files and trains a
model to recognize them.

Most other options available in transcription training are also available in
segmentation training. CUDA acceleration:

.. code-block:: console

        $ ketos segtrain -d cuda -f xml training_data/*.xml

Defining custom architectures:

.. code-block:: console

        $ ketos segtrain -d cuda -s '[1,1200,0,3 Cr7,7,64,2,2 Gn32 Cr3,3,128,2,2 Gn32 Cr3,3,128 Gn32 Cr3,3,256 Gn32]' -f xml training_data/*.xml

Fine tuning/transfer learning with last layer adaptation and slicing:

.. code-block:: console

        $ ketos segtrain --resize both -i segmodel_best.mlmodel training_data/*.xml
        $ ketos segtrain -i segmodel_best.mlmodel --append 7 -s '[Cr3,3,64 Do0.1]' training_data/*.xml

In addition there are a couple of specific options that allow filtering of
baseline and region types. Datasets are often annotated to a level that is too
detailed or contains undesirable types, e.g. when combining segmentation data
from different sources. The most basic option is the suppression of *all* of
either baseline or region data contained in the dataset:

.. code-block:: console

        $ ketos segtrain --suppress-baselines -f xml training_data/*.xml
        Training line types:
        Training region types:
          graphic	3	135
          text	4	1128
          separator	5	5431
          paragraph	6	10218
          table	7	16
        ...
        $ ketos segtrain --suppress-regions -f xml training-data/*.xml
        Training line types:
          default 2	53980
          foo     8     134
        ...

It is also possible to filter out baselines/regions selectively:

.. code-block:: console

        $ ketos segtrain -f xml --valid-baselines default training_data/*.xml
        Training line types:
          default 2	53980
        Training region types:
          graphic	3	135
          text	4	1128
          separator	5	5431
          paragraph	6	10218
          table	7	16
        $ ketos segtrain -f xml --valid-regions graphic --valid-regions paragraph training_data/*.xml
        Training line types:
          default 2	53980
         Training region types:
          graphic	3	135
          paragraph	6	10218

Finally, we can merge baselines and regions into each other:

.. code-block:: console 

        $ ketos segtrain -f xml --merge-baselines default:foo training_data/*.xml
        Training line types:
          default 2	54114
        ...
        $ ketos segtrain -f xml --merge-regions text:paragraph --merge-regions graphic:table training_data/*.xml
        ...
        Training region types:
          graphic	3	151
          text	4	11346
          separator	5	5431
        ...

These options are combinable to massage the dataset into any typology you want.

Then there are some options that set metadata fields controlling the
postprocessing. When computing the bounding polygons the recognized baselines
are offset slightly to ensure overlap with the line corpus. This offset is per
default upwards for baselines but as it is possible to annotate toplines (for
scripts like Hebrew) and centerlines (for baseline-free scripts like Chinese)
the appropriate offset can be selected with an option:

.. code-block:: console

        $ ketos segtrain --topline -f xml hebrew_training_data/*.xml
        $ ketos segtrain --centerline -f xml chinese_training_data/*.xml
        $ ketos segtrain --baseline -f xml latin_training_data/*.xml

Lastly, there are some regions that are absolute boundaries for text line
content. When these regions are marked as such the polygonization can sometimes
be improved:

.. code-block:: console

        $ ketos segtrain --bounding-regions paragraph -f xml training_data/*.xml
        ...

Recognition Testing
-------------------

Picking a particular model from a pool or getting a more detailed look on the
recognition accuracy can be done with the `test` command. It uses transcribed
lines, the test set, in the same format as the `train` command, recognizes the
line images with one or more models, and creates a detailed report of the
differences from the ground truth for each of them.

======================================================= ======
option                                                  action
======================================================= ======
-f, --format-type                                       Sets the test set data format.
                                                        Valid choices are 'path', 'xml' (default), 'alto', 'page', or binary.
                                                        In `alto`, `page`, and xml mode all data is extracted from XML files
                                                        containing both baselines and a link to source images.
                                                        In `path` mode arguments are image files sharing a prefix up to the last
                                                        extension with JSON `.path` files containing the baseline information.
                                                        In `binary` mode arguments are precompiled binary dataset files.
-m, --model                                             Model(s) to evaluate.
-e, --evaluation-files                                  File(s) with paths to evaluation data.
-d, --device                                            Select device to use.
--pad                                                   Left and right padding around lines.
======================================================= ======

Transcriptions are handed to the command in the same way as for the `train`
command, either through a manifest with `-e/--evaluation-files` or by just
adding a number of image files as the final argument:

.. code-block:: console

   $ ketos test -m $model -e test.txt test/*.png
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


