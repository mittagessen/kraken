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
of custom low level format, and the XML-based formats that are commony used for
archival of annotation and transcription data. It is recommended to use the XML
formats as they are interchangeable with other tools, do not incur
transformation losses, and allow training all components of kraken from the
same datasets easily.

ALTO
~~~~

Kraken parses and produces files according to the upcoming version of the ALTO
standard: 4.2. It validates against version 4.1 with the exception of the
`redefinition <https://github.com/altoxml/schema/issues/32>`_ of the `BASELINE`
attribute to accomodate polygonal chain baselines. An example showing the
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

Recognition training
--------------------

The training utility allows training of :ref:`VGSL <vgsl>` specified models
both from scratch and from existing models. Here are its command line options:

======================================================= ======
option                                                  action
======================================================= ======
-p, --pad                                               Left and right padding around lines
-o, --output                                            Output model file prefix. Defaults to model.
-s, --spec                                              VGSL spec of the network to train. CTC layer
                                                        will be added automatically. default:
                                                        [1,48,0,1 Cr3,3,32 Do0.1,2 Mp2,2 Cr3,3,64
                                                        Do0.1,2 Mp2,2 S1(1x12)1,3 Lbx100 Do]
-a, --append                                            Removes layers before argument and then
                                                        appends spec. Only works when loading an
                                                        existing model
-i, --load                                              Load existing file to continue training
-F, --savefreq                                          Model save frequency in epochs during
                                                        training
-R, --report                                            Report creation frequency in epochs
-q, --quit                                              Stop condition for training. Set to `early`
                                                        for early stopping (default) or `dumb` for fixed
                                                        number of epochs.
-N, --epochs                                            Number of epochs to train for. Set to -1 for indefinite training.
--lag                                                   Number of epochs to wait before stopping
                                                        training without improvement. Only used when using early stopping.
--min-delta                                             Minimum improvement between epochs to reset
                                                        early stopping. Defaults to 0.005.
-d, --device                                            Select device to use (cpu, cuda:0, cuda:1,...). GPU acceleration requires CUDA.
--optimizer                                             Select optimizer (Adam, SGD, RMSprop).
-r, --lrate                                             Learning rate  [default: 0.001]
-m, --momentum                                          Momentum used with SGD optimizer. Ignored otherwise.
-w, --weight-decay                                      Weight decay.
--schedule                                              Sets the learning rate scheduler. May be either constant or 1cycle. For 1cycle 
                                                        the cycle length is determined by the `--epoch` option.
-p, --partition                                         Ground truth data partition ratio between train/validation set
-u, --normalization                                     Ground truth Unicode normalization. One of NFC, NFKC, NFD, NFKD.
-c, --codec                                             Load a codec JSON definition (invalid if loading existing model)
--resize                                                Codec/output layer resizing option. If set
                                                        to `add` code points will be added, `both`
                                                        will set the layer to match exactly the
                                                        training data, `fail` will abort if training
                                                        data and model codec do not match. Only valid when refining an existing model.
-n, --reorder / --no-reorder                            Reordering of code points to display order.
-t, --training-files                                    File(s) with additional paths to training data. Used to 
                                                        enforce an explicit train/validation set split and deal with 
                                                        training sets with more lines than the command line can process. Can be used more than once.
-e, --evaluation-files                                  File(s) with paths to evaluation data. Overrides the `-p` parameter.
--preload / --no-preload                                Hard enable/disable for training data preloading. Preloading 
                                                        training data into memory is enabled per default for sets with less than 2500 lines.
--threads                                               Number of OpenMP threads when running on CPU. Defaults to min(4, #cores).
======================================================= ======

From Scratch
~~~~~~~~~~~~

The absolute minimal example to train a new recognition model from a number of
PAGE XML documents is similar to the segmentation training:

.. code-block:: console

        $ ketos train training_data/*.png

Training will continue until the error does not improve anymore and the best
model (among intermediate results) will be saved in the current directory.

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

Fine Tuning
~~~~~~~~~~~

Fine tuning an existing model for another typeface or new characters is also
possible with the same syntax as resuming regular training:

.. code-block:: console

        $ ketos train -f page -i model_best.mlmodel syr/*.xml

The caveat is that the alphabet of the base model and training data have to be
an exact match. Otherwise an error will be raised:

.. code-block:: console

        $ ketos train -i model_5.mlmodel --no-preload kamil/*.png
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

Segmentation training
---------------------

Training a segmentation model is very similar to training one for 


Testing
-------

Picking a particular model from a pool or getting a more detailled look on the
recognition accuracy can be done with the `test` command. It uses transcribed
lines, the test set, in the same format as the `train` command, recognizes the
line images with one or more models, and creates a detailled report of the
differences from the ground truth for each of them.

======================================================= ======
option                                                  action
======================================================= ======
-m, --model                                             Model(s) to evaluate.
-e, --evaluation-files                                  File(s) with paths to evaluation data.
-d, --device                                            Select device to use.
-p, --pad                                               Left and right padding around lines.


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

The report(s) contains character accuracy measured per script and a detailled
list of confusions. When evaluating multiple models the last line of the output
will the average accuracy and the standard deviation across all of them.
