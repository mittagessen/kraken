.. _training:

Training kraken
===============

kraken is an optical character recognition package that can be trained fairly
easily for a large number of scripts. In contrast to other system requiring
segmentation down to glyph level before classification, it is uniquely suited
for the recognition of connected scripts, because the neural network is trained
to assign correct character to unsegmented training data.

Both segmentation, the process finding lines and regions on a page image, and
recognition, the conversion of line images into text, can be trained in kraken.
To train models for either we require training data, i.e. examples of page
segmentations and transcriptions that are similar to what we want to be able to
recognize. For segmentation the examples are  the location of baselines, i.e.
the imaginary lines the text is written on, and polygons of regions. For
recognition these are the text contained in a line. There are multiple ways to
supply training data but the easiest is through PageXML or ALTO files.

Installing kraken
-----------------

The easiest way to install and use kraken is through `conda
<https://www.anaconda.com/download/>`_. kraken works both on Linux and Mac OS
X. After installing conda, download the environment file and create the
environment for kraken:

.. code-block:: console

   $ wget https://raw.githubusercontent.com/mittagessen/kraken/master/environment.yml
   $ conda env create -f environment.yml

Each time you want to use the kraken environment in a shell is has to be
activated first:

.. code-block:: console

   $ conda activate kraken

Image acquisition and preprocessing
-----------------------------------

First a number of high quality scans, preferably color or grayscale and at
least 300dpi are required. Scans should be in a lossless image format such as
TIFF or PNG, images in PDF files have to be extracted beforehand using a tool
such as ``pdftocairo`` or ``pdfimages``. While each of these requirements can
be relaxed to a degree, the final accuracy will suffer to some extent. For
example, only slightly compressed JPEG scans are generally suitable for
training and recognition.

Depending on the source of the scans some preprocessing such as splitting scans
into pages, correcting skew and warp, and removing speckles can be advisable
although it isn't strictly necessary as the segmenter can be trained to treat
noisy material with a high accuracy. A fairly user-friendly software for
semi-automatic batch processing of image scans is `Scantailor
<http://scantailor.org>`_ albeit most work can be done using a standard image
editor.

The total number of scans required depends on the kind of model to train
(segmentation or recognition), the complexity of the layout or the nature of
the script to recognize. Only features that are found in the training data can
later be recognized, so it is important that the coverage of typographic
features is exhaustive. Training a small segmentation model for a particular
kind of material might require less than a few hundred samples while a general
model can well go into the thousands of pages. Likewise a specific recognition
model for printed script with a small grapheme inventory such as Arabic or
Hebrew requires around 800 lines, with manuscripts, complex scripts (such as
polytonic Greek), and general models for multiple typefaces and hands needing
more training data for the same accuracy.

There is no hard rule for the amount of training data and it may be required to
retrain a model after the initial training data proves insufficient. Most
``western`` texts contain between 25 and 40 lines per page, therefore upward of
30 pages have to be preprocessed and later transcribed.

Annotation and transcription
----------------------------

kraken does not provide internal tools for the annotation and transcription of
baselines, regions, and text. There are a number of tools available that can
create ALTO and PageXML files containing the requisite information for either
segmentation or recognition training: `escriptorium
<https://escripta.hypotheses.org>`_ integrates kraken tightly including
training and inference, `Aletheia
<https://www.primaresearch.org/tools/Aletheia>`_ is a powerful desktop
application that can create fine grained annotations.

Dataset Compilation
-------------------

.. _compilation:

Training
--------

.. _training_step:

The training data, e.g. a collection of PAGE XML documents, obtained through
annotation and transcription may now be used to train segmentation and/or
transcription models.

The training data in ``output_dir`` may now be used to train a new model by
invoking the ``ketos train`` command. Just hand a list of images to the command
such as:

.. code-block:: console

        $ ketos train output_dir/*.png

to start training.

A number of lines will be split off into a separate held-out set that is used
to estimate the actual recognition accuracy achieved in the real world. These
are never shown to the network during training but will be recognized
periodically to evaluate the accuracy of the model. Per default the validation
set will comprise of 10% of the training data.

Basic model training is mostly automatic albeit there are multiple parameters
that can be adjusted:

--output
        Sets the prefix for models generated during training. They will best as
        ``prefix_epochs.mlmodel``.
--report
        How often evaluation passes are run on the validation set. It is an
        integer equal or larger than 1 with 1 meaning a report is created each
        time the complete training set has been seen by the network.
--savefreq
        How often intermediate models are saved to disk. It is an integer with
        the same semantics as ``--report``.
--load
        Continuing training is possible by loading an existing model file with
        ``--load``. To continue training from a base model with another
        training set refer to the full :ref:`ketos <ketos>` documentation.
--preload
        Enables/disables preloading of the training set into memory for
        accelerated training. The default setting preloads data sets with less
        than 2500 lines, explicitly adding ``--preload`` will preload arbitrary
        sized sets. ``--no-preload`` disables preloading in all circumstances.

Training a network will take some time on a modern computer, even with the
default parameters. While the exact time required is unpredictable as training
is a somewhat random process a rough guide is that accuracy seldom improves
after 50 epochs reached between 8 and 24 hours of training. 

When to stop training is a matter of experience; the default setting employs a
fairly reliable approach known as `early stopping
<https://en.wikipedia.org/wiki/Early_stopping>`_ that stops training as soon as
the error rate on the validation set doesn't improve anymore.  This will
prevent `overfitting <https://en.wikipedia.org/wiki/Overfitting>`_, i.e.
fitting the model to recognize only the training data properly instead of the
general patterns contained therein. 

.. code-block:: console
        
        $ ketos train output_dir/*.png
        Building training set  [####################################]  100%
        Building validation set  [####################################]  100%
        [270.2364] alphabet mismatch {'9', '8', '݂', '3', '݀', '4', '1', '7', '5', '\xa0'}
        Initializing model ✓
        Accuracy report (0) -1.5951 3680 9550
        epoch 0/-1  [####################################]  788/788
        Accuracy report (1) 0.0245 3504 3418
        epoch 1/-1  [####################################]  788/788
        Accuracy report (2) 0.8445 3504 545
        epoch 2/-1  [####################################]  788/788             
        Accuracy report (3) 0.9541 3504 161
        epoch 3/-1  [------------------------------------]  13/788  0d 00:22:09
        ...

By now there should be a couple of models model_name-1.mlmodel,
model_name-2.mlmodel, ... in the directory the script was executed in. Lets
take a look at each part of the output.

.. code-block:: console

        Building training set  [####################################]  100%
        Building validation set  [####################################]  100%

shows the progress of loading the training and validation set into memory. This
might take a while as preprocessing the whole set and putting it into memory is
computationally intensive. Loading can be made faster without preloading at the
cost of performing preprocessing repeatedly during the training process.

.. code-block:: console

        [270.2364] alphabet mismatch {'9', '8', '݂', '3', '݀', '4', '1', '7', '5', '\xa0'}

is a warning about missing characters in either the validation or training set,
i.e.  that the alphabets of the sets are not equal. Increasing the size of the
validation set will often remedy this warning.

.. code-block:: console

        Accuracy report (2) 0.8445 3504 545

this line shows the results of the validation set evaluation. The error after 2
epochs is 545 incorrect characters out of 3504 characters in the validation set
for a character accuracy of 84.4%. It should decrease fairly rapidly.  If
accuracy remains around 0.30 something is amiss, e.g. non-reordered
right-to-left or wildly incorrect transcriptions. Abort training, correct the
error(s) and start again.

After training is finished the best model is saved as
``model_name_best.mlmodel``. It is highly recommended to also archive the
training log and data for later reference.

``ketos`` can also produce more verbose output with training set and network
information by appending one or more ``-v`` to the command:

.. code-block:: console

        $ ketos -vv train syr/*.png
        [0.7272] Building ground truth set from 876 line images 
        [0.7281] Taking 88 lines from training for evaluation 
        ...
        [0.8479] Training set 788 lines, validation set 88 lines, alphabet 48 symbols
        [0.8481] alphabet mismatch {'\xa0', '0', ':', '݀', '܇', '݂', '5'}
        [0.8482] grapheme	count
        [0.8484] SPACE	5258
        [0.8484] 	ܐ	3519
        [0.8485] 	ܘ	2334
        [0.8486] 	ܝ	2096
        [0.8487] 	ܠ	1754
        [0.8487] 	ܢ	1724
        [0.8488] 	ܕ	1697
        [0.8489] 	ܗ	1681
        [0.8489] 	ܡ	1623
        [0.8490] 	ܪ	1359
        [0.8491] 	ܬ	1339
        [0.8491] 	ܒ	1184
        [0.8492] 	ܥ	824
        [0.8492] 	.	811
        [0.8493] COMBINING DOT BELOW	646
        [0.8493] 	ܟ	599
        [0.8494] 	ܫ	577
        [0.8495] COMBINING DIAERESIS	488
        [0.8495] 	ܚ	431
        [0.8496] 	ܦ	428
        [0.8496] 	ܩ	307
        [0.8497] COMBINING DOT ABOVE	259
        [0.8497] 	ܣ	256
        [0.8498] 	ܛ	204
        [0.8498] 	ܓ	176
        [0.8499] 	܀	132
        [0.8499] 	ܙ	81
        [0.8500] 	*	66
        [0.8501] 	ܨ	59
        [0.8501] 	܆	40
        [0.8502] 	[	40
        [0.8503] 	]	40
        [0.8503] 	1	18
        [0.8504] 	2	11
        [0.8504] 	܇	9
        [0.8505] 	3	8
        [0.8505] 		6
        [0.8506] 	5	5
        [0.8506] NO-BREAK SPACE	4
        [0.8507] 	0	4
        [0.8507] 	6	4
        [0.8508] 	:	4
        [0.8508] 	8	4
        [0.8509] 	9	3
        [0.8510] 	7	3
        [0.8510] 	4	3
        [0.8511] SYRIAC FEMININE DOT	1
        [0.8511] SYRIAC RUKKAKHA	1
        [0.8512] Encoding training set
        [0.9315] Creating new model [1,1,0,48 Lbx100 Do] with 49 outputs
        [0.9318] layer		type	params
        [0.9350] 0		rnn	direction b transposed False summarize False out 100 legacy None
        [0.9361] 1		dropout	probability 0.5 dims 1
        [0.9381] 2		linear	augmented False out 49
        [0.9918] Constructing RMSprop optimizer (lr: 0.001, momentum: 0.9)
        [0.9920] Set OpenMP threads to 4
        [0.9920] Moving model to device cpu
        [0.9924] Starting evaluation run


indicates that the training is running on 788 transcribed lines and a
validation set of 88 lines. 49 different classes, i.e. Unicode code points,
where found in these 788 lines. These affect the output size of the network;
obviously only these 49 different classes/code points can later be output by
the network.  Importantly, we can see that certain characters occur markedly
less often than others. Characters like the Syriac feminine dot and numerals
that occur less than 10 times will most likely not be recognized well by the
trained net.


Evaluation and Validation
-------------------------

While output during training is detailed enough to know when to stop training
one usually wants to know the specific kinds of errors to expect. Doing more
in-depth error analysis also allows to pinpoint weaknesses in the training
data, e.g. above average error rates for numerals indicate either a lack of
representation of numerals in the training data or erroneous transcription in
the first place.

First the trained model has to be applied to some line transcriptions with the
`ketos test` command:

.. code-block:: console

      $ ketos test -m syriac_best.mlmodel lines/*.png
      Loading model syriac_best.mlmodel	✓
      Evaluating syriac_best.mlmodel
      Evaluating  [#-----------------------------------]    3%  00:04:56
      ...

After all lines have been processed a evaluation report will be printed:

.. code-block:: console

      === report  ===
      
      35619	Characters
      336	Errors
      99.06%	Accuracy
      
      157	Insertions
      81	Deletions
      98	Substitutions
      
      Count	Missed	%Right
      27046	143	99.47%	Syriac
      7015	52	99.26%	Common
      1558	60	96.15%	Inherited
      
      Errors	Correct-Generated
      25	{  } - { COMBINING DOT BELOW }
      25	{ COMBINING DOT BELOW } - {  }
      15	{ . } - {  }
      15	{ COMBINING DIAERESIS } - {  }
      12	{ ܢ } - {  }
      10	{  } - { . }
      8	{ COMBINING DOT ABOVE } - {  }
      8	{ ܝ } - {  }
      7	{ ZERO WIDTH NO-BREAK SPACE } - {  }
      7	{ ܆ } - {  }
      7	{ SPACE } - {  }
      7	{ ܣ } - {  }
      6	{  } - { ܝ }
      6	{ COMBINING DOT ABOVE } - { COMBINING DIAERESIS }
      5	{ ܙ } - {  }
      5	{ ܬ } - {  }
      5	{  } - { ܢ }
      4	{ NO-BREAK SPACE } - {  }
      4	{ COMBINING DIAERESIS } - { COMBINING DOT ABOVE }
      4	{  } - { ܒ }
      4	{  } - { COMBINING DIAERESIS }
      4	{ ܗ } - {  }
      4	{  } - { ܬ }
      4	{  } - { ܘ }
      4	{ ܕ } - { ܢ }
      3	{  } - { ܕ }
      3	{ ܐ } - {  }
      3	{ ܗ } - { ܐ }
      3	{ ܝ } - { ܢ }
      3	{ ܀ } - { . }
      3	{  } - { ܗ }

	.....

The first section of the report consists of a simple accounting of the number
of characters in the ground truth, the errors in the recognition output and the
resulting accuracy in per cent.

The next table lists the number of insertions (characters occurring in the
ground truth but not in the recognition output), substitutions (misrecognized
characters), and deletions (superfluous characters recognized by the model).

Next is a grouping of errors (insertions and substitutions) by Unicode script.

The final part of the report are errors sorted by frequency and a per
character accuracy report. Importantly most errors are incorrect recognition of
combining marks such as dots and diaereses. These may have several sources:
different dot placement in training and validation set, incorrect transcription
such as non-systematic transcription, or unclean speckled scans. Depending on
the error source, correction most often involves adding more training data and
fixing transcriptions. Sometimes it may even be advisable to remove
unrepresentative data from the training set.

Recognition
-----------

The ``kraken`` utility is employed for all non-training related tasks. Optical
character recognition is a multi-step process consisting of binarization
(conversion of input images to black and white), page segmentation (extracting
lines from the image), and recognition (converting line image to character
sequences). All of these may be run in a single call like this:

.. code-block:: console

        $ kraken -i INPUT_IMAGE OUTPUT_FILE binarize segment ocr -m MODEL_FILE

producing a text file from the input image. There are also `hocr
<http://hocr.info>`_ and `ALTO <https://www.loc.gov/standards/alto/>`_ output
formats available through the appropriate switches:

.. code-block:: console

        $ kraken -i ... ocr -h
        $ kraken -i ... ocr -a

For debugging purposes it is sometimes helpful to run each step manually and
inspect intermediate results:

.. code-block:: console

        $ kraken -i INPUT_IMAGE BW_IMAGE binarize
        $ kraken -i BW_IMAGE LINES segment
        $ kraken -i BW_IMAGE OUTPUT_FILE ocr -l LINES ...

It is also possible to recognize more than one file at a time by just chaining
``-i ... ...`` clauses like this:

.. code-block:: console

        $ kraken -i input_1 output_1 -i input_2 output_2 ...

Finally, there is a central repository containing freely available models.
Getting a list of all available models:

.. code-block:: console

        $ kraken list

Retrieving model metadata for a particular model:

.. code-block:: console

	$ kraken show arabic-alam-al-kutub
	name: arabic-alam-al-kutub.mlmodel
	
	An experimental model for Classical Arabic texts.
	
	Network trained on 889 lines of [0] as a test case for a general Classical
	Arabic model. Ground truth was prepared by Sarah Savant
	<sarah.savant@aku.edu> and Maxim Romanov <maxim.romanov@uni-leipzig.de>.
	
	Vocalization was omitted in the ground truth. Training was stopped at ~35000
	iterations with an accuracy of 97%.
	
	[0] Ibn al-Faqīh (d. 365 AH). Kitāb al-buldān. Edited by Yūsuf al-Hādī, 1st
	edition. Bayrūt: ʿĀlam al-kutub, 1416 AH/1996 CE.
	alphabet:  !()-.0123456789:[] «»،؟ءابةتثجحخدذرزسشصضطظعغفقكلمنهوىي ARABIC
	MADDAH ABOVE, ARABIC HAMZA ABOVE, ARABIC HAMZA BELOW

and actually fetching the model:

.. code-block:: console

	$ kraken get arabic-alam-al-kutub

The downloaded model can then be used for recognition by the name shown in its metadata, e.g.:

.. code-block:: console

        $ kraken -i INPUT_IMAGE OUTPUT_FILE binarize segment ocr -m arabic-alam-al-kutub.mlmodel

For more documentation see the kraken `website <http://kraken.re>`_.
