.. _training:

Training a kraken model
=======================

kraken is an optical character recognition package that can be trained fairly
easily for a large number of scripts. In contrast to other system requiring
segmentation down to glyph level before classification, it is uniquely suited
for the recognition of connected scripts, because the neural network is trained
to assign correct character to unsegmented training data.

Training a new model for kraken requires a variable amount of training data
manually generated from page images which have to be typographically similar to
the target prints that are to be recognized. As the system works on unsegmented
inputs for both training and recognition and its base unit is a text line,
training data are just transcriptions aligned to line images. 

Installing kraken
-----------------

The easiest way to install and use kraken is through a `vagrant
<https://vagrantup.com>`_ virtual machine. After downloading and installing
vagrant, the box can be provisioned.

.. code-block:: console

        $ vagrant init openphilology/kraken
        $ vagrant up

After running the above commands, the box should be up and running. The
directory these commands are executed in is mapped into the virtual machine and
can be used to exchange data with the host system. The virtual machine can be
accessed through running:

.. code-block:: console

        $ vagrant ssh

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
into pages, correcting skew and warp, and removing speckles is usually
required. For complex layouts such as newspapers it is advisable to split the
page manually into columns as the line extraction algorithm run to create
transcription environments does not deal well with non-codex page layouts. A
fairly user-friendly software for semi-automatic batch processing of image
scans is `Scantailor <http://scantailor.org>`_ albeit most work can be done
using a standard image editor.

The total number of scans required depends on the nature of the script to be
recognized. Only features that are found on the page images and training data
derived from it can later be recognized, so it is important that the coverage
of typographic features is exhaustive. Training a single script model for a
fairly small script such as Arabic or Hebrew requires at least 800 lines, while
multi-script models, e.g. combined polytonic Greek and Latin, will require
significantly more transcriptions. 

There is no hard rule for the amount of training data and it may be required to
retrain a model after the initial training data proves insufficient. Most
``western`` texts contain between 25 and 40 lines per page, therefore upward of
30 pages have to be preprocessed and later transcribed.

Transcription
-------------

Transcription is done through local browser based HTML transcription
environments. These are created by the ``ketos transcribe`` command line util
that is part of kraken. Its basic input is just a number of image files and an
output path to write the HTML file to:

.. code-block:: console
        
        $ ketos transcribe -o output.html image_1.png image_2.png ...

While it is possible to put multiple images into a single transcription
environment splitting into one-image-per-HTML will ease parallel transcription
by multiple people.

The above command reads in the image files, converts them to black and white if
necessary, tries to split them into line images, and puts an editable text
field next to the image in the HTML.

Transcription has to be diplomatic, i.e. contain the exact character sequence
in the line image, including original orthography. Some deviations, such as
consistently omitting vocalization in Arabic texts, is possible as long as they
are systematic and relatively minor.

.. note::

        The page segmentation algorithm extracting lines from images is
        optimized for ``western`` page layouts and may recognize lines
        erroneously, lumping multiple lines together or cutting them in half.
        The most efficient way to deal with these errors is just skipping the
        affected lines by leaving the text box empty.

.. tip::

        Copy-paste transcription can significantly speed up the whole process.
        Either transcribe scans of a work where a digital edition already
        exists (but does not for typographically similar prints) or find a
        sufficiently similar edition as a base.

After transcribing a number of lines the results have to be saved, either using
the ``Download`` button on the lower left or through the regular ``Save Page
As`` (CTRL+S) function of the browser. All the work done is contained directly
in the saved files and it is possible to save partially transcribed files and
continue work later.

Next the contents of the filled transcription environments have to be
extracted through the ``ketos extract`` command:

.. code-block:: console 

        $ ketos extract --output output_directory --normalization NFD *.html

with

--output
        The output directory where all line image-text pairs (training data)
        are written, defaulting to ``training/``
--normalization
        Unicode has code points to encode most glyphs encountered in the wild.
        A lesser known feature is that there usually are multiple ways to
        encode a glyph.  `Unicode normalization
        <http://www.unicode.org/reports/tr15/>`_ ensures that equal glyphs are
        encoded in the same way, i.e. that the encoded representation across
        the training data set is consistent and there is only one way the
        network can recognize a particular feature on the page. Usually it is
        sufficient to set the normalization to Normalization Form
        Decomposed (NFD), as it reduces the the size of the overall script to
        be recognized slightly.

The result will be a directory filled with line image text pairs ``NNNNNN.png``
and ``NNNNNN.gt.txt`` and a ``manifest.txt`` containing a list of all extracted
lines.

.. note::

        At this point it is recommended to review the content of the training
        data directory before proceeding. 

Training
--------

The training data in ``output_dir`` may now be used to train a new model by
invoking the ``ketos train`` command. Just hand a list of images to the command
such as:

.. code-block:: console

        $ ketos train output_dir/*.png

to start training.

A number of lines will be split off into a separate held-out set that is used
to estimate the actual recognition accuracy achieved in the real world. These
are never shown to the network during training but will be recognized
periodically to evaluate the accuracy of the model. Per default the test set
will comprise of 10% of the training data.

Basic model training is mostly automatic albeit there are multiple parameters
that can be adjusted:

--output
        Sets the prefix for models generated during training. They will best as
        ``prefix_iterations.mlmodel``.
--report
        How often evaluation passes are run on the test set. It is a number
        between 0 and 1 with 1 meaning a report is created each time the
        complete training set has been seen by the network.
--savefreq
        How often intermediate models are saved to disk. It is a number between
        0 and 1 with the same semantics as ``--report``.
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
is a somewhat random process a rough guide is that accuracy seldomly improves
after 50 epochs reached between 8 and 24 hours of training. 

When to stop training is a matter of experience; the default setting employs a
fairly reliable approach known as `early stopping
<https://en.wikipedia.org/wiki/Early_stopping>`_ that stops training as soon as
the error rate on the test set doesn't improve anymore.  This will prevent
`overfitting <https://en.wikipedia.org/wiki/Overfitting>`_, i.e. fitting the
model to recognize only the training data properly instead of the general
patterns contained therein. 

.. code-block:: console
        
        $ ketos train output_dir/*.png
        Building training set  [####################################]  100%
        Building test set  [####################################]  100%
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
        Building test set  [####################################]  100%

shows the progress of loading the training and test set into memory. This might
take a while as preprocessing the whole set and putting it into memory is
computationally intensive. Loading can be made faster without preloading at the
cost of performing preprocessing repeatedlyduring the training process. 

.. code-block:: console

        [270.2364] alphabet mismatch {'9', '8', '݂', '3', '݀', '4', '1', '7', '5', '\xa0'}

is a warning about missing characters in either the test or training set, i.e.
that the alphabets of the sets are not equal. Increasing the size of the test
set will often remedy this warning.

.. code-block:: console

        Accuracy report (2) 0.8445 3504 545

this line shows the results of the test set evaluation. The error after 2
epochs is 545 incorrect characters out of 3504 characters in the test set for a
character accuracy of 84.4%. It should decrease fairly rapidly.  If accuracy
remains around 0.30 something is amiss, e.g. non-reordered right-to-left or
wildly incorrect transcriptions. Abort training, correct the error(s) and start
again.

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
        [0.8479] Training set 788 lines, test set 88 lines, alphabet 48 symbols
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


indicates that the training is running on 788 transcribed lines and a test set
of 88 lines. 49 different classes, i.e. Unicode code points, where found in
these 788 lines. These affect the output size of the network; obviously only
these 49 different classes/code points can later be output by the network.
Importantly, we can see that certain characters occur markedly less often than
others. Characters like the Syriac feminine dot and numerals that occur less
than 10 times will most likely not be recognized well by the trained net.


Evaluation and Validation
-------------------------

While output during training is detailed enough to know when to stop training
one usually wants to know the specific kinds of errors to expect. Doing more
in-depth error analysis also allows to pinpoint weaknesses in the training
data, e.g. above average error rates for numerals indicate either a lack of
representation of numerals in the training data or erroneous transcription in
the first place.

First the trained model has to be applied to the line images by invoking
``eval.py`` with the model and a directory containing line images:

.. code-block:: console

        $ ./eval.py output_dir model_file

The recognition output is written into ``rec.txt``, the ground truth is
concatenated into a file called ``gt.txt``. There will also be a file
``report.txt`` containing the detailed accuracy report:

.. code-block:: console

	UNLV-ISRI OCR Accuracy Report Version 5.1
	-----------------------------------------
	   35632   Characters
	    1477   Errors
	   95.85%  Accuracy
	
	       0   Reject Characters
	       0   Suspect Markers
	       0   False Marks
	    0.00%  Characters Marked
	   95.85%  Accuracy After Correction
	
	     Ins    Subst      Del   Errors
	       0        0        0        0   Marked
	     151      271     1055     1477   Unmarked
	     151      271     1055     1477   Total
	
	   Count   Missed   %Right
	   27046      155    99.43   Unassigned
	    5843       13    99.78   ASCII Spacing Characters
	    1089      108    90.08   ASCII Special Symbols
	      77       53    31.17   ASCII Digits
	      15       15     0.00   ASCII Uppercase Letters
	       4        4     0.00   Latin1 Spacing Characters
	    1558       74    95.25   Combining Diacritical Marks
	   35632      422    98.82   Total
	
	  Errors   Marked   Correct-Generated
	     815        0   {}-{ }
	      29        0   {}-{̈}
	      29        0   {}-{̣}
	      20        0   {[}-{ ]}
	      18        0   {̈}-{}
	      18        0   {̣}-{}
	      15        0   {̇}-{}
	      13        0   {}-{.}
	      12        0   {}-{. }
	      12        0   {}-{ܝ}
	       9        0   {}-{ܠ}
	       9        0   {}-{ܢ}
	       8        0   { }-{}
	       8        0   {ܨ}-{ܢ}
	       8        0   {[SECTIO}-{ ] ܐܘܘ...}
	
	.....

	Count   Missed   %Right
	 5843       13    99.78   { }
	   72        0   100.00   {*}
	  909       13    98.57   {.}
	    4        4     0.00   {0}
	   22        6    72.73   {1}
	   15       12    20.00   {2}
	    9        7    22.22   {3}
	    4        4     0.00   {4}
	    5        3    40.00   {5}
	    5        5     0.00   {6}
	    4        4     0.00   {7}
	    5        4    20.00   {8}
	    4        4     0.00   {9}
	    4        4     0.00   {:}
	    2        2     0.00   {C}
	    2        2     0.00   {E}
	    5        5     0.00   {I}
	    2        2     0.00   {O}
	    2        2     0.00   {S}
	    2        2     0.00   {T}
	   52       45    13.46   {[}
	   52       46    11.54   {]}
	    4        4     0.00   { }
	  297       22    92.59   {̇}
	  538       26    95.17   {̈}
	  723       26    96.40   {̣}
	  149        6    95.97   {܀}
	   46       12    73.91   {܆}
	    9        8    11.11   {܇}
	 3891       16    99.59   {ܐ}
	 1309        6    99.54   {ܒ}
	  190        1    99.47   {ܓ}
	 1868        9    99.52   {ܕ}
	 1862        7    99.62   {ܗ}
	 2588       10    99.61   {ܘ}
	   87        2    97.70   {ܙ}
	  484        2    99.59   {ܚ}
	  225        0   100.00   {ܛ}

	.....

The first section of the report consists of a simple accounting of the number
of characters in the ground truth, the errors in the recognition output and the
resulting accuracy in per cent.

The next section can be ignored.

The next table lists the number of insertions (characters occuring in the
ground truth but not in the recognition output), substitutions (misrecognized
characters), and deletions (superfluous characters recognized by the model).

Next is a grouping of errors (insertions and substitutions) by Unicode
character class. As the report tool does not have proper Unicode support,
Syriac characters are classified as ``Unassigned``. Nevertheless it is apparent
that numerals are recognized markedly worse than every other class, presumably
because they are severely underrepresented (77) in the training set. Further
all Latin text is misrecognized, as the training set did not contain any and
there is a small inconsistency in the test set caused by Latin-1 spacing
characters. 

The final two parts of the report are errors sorted by frequency and a per
character accuracy report. Importantly, over half the overall errors are caused
by incorrect whitespace produced by the model. These may have several sources:
different spacing in training and test set, incorrect transcription such as
leading/trailing whitespace, or. Depending on the error source, correction most
often involves adding more training data and fixing transcriptions. Sometimes
it may even be advisable to remove unrepresentative data from the training set.

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

Finally, there is an central repository containing freely available models.
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
