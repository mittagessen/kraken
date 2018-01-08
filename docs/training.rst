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

        $ ketos extract --reorder --output output_directory --normalization NFD *.html

with

--reorder
        Tells ketos to reorder the code point for each line into left-to-right
        order. Unicode code points are always in reading order, e.g. the first
        code point in an Arabic line will be the rightmost character. This
        option reorders them into ``display order``, i.e. the first code point
        is the leftmost, the second one the next from the left and so on. As
        the neural network does not know beforehand if part of an image
        contains left-to-right or right-to-left text, all glyphs are assumed to
        be left-to-right and later reordered for correct display.
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
invoking the ``train.sh`` script in the vagrant box's home directory. 100 lines
will be split off from the actual training set into a separate test set for
validation purposes. These are never shown to the network for training purposes
but will be periodically recognized to evaluate the accuracy of the model.

Model training is mostly automatic albeit there are parameters in the training
script that may be adjusted if necessary:

save_every
        Sets the frequence the model is saved during training. Per default the
        network is serialized after 1000 epochs of training. An epoch is the
        number of training steps after which each randomly drawn line has been
        seens once by the network (on average).
test_every
        How often the trained network is evaluated on the test set.
hidden
        The number of nodes in the internal ``hidden`` layer of the network.
        Larger networks are capable of capturing more complex patterns but will
        slow down training and recognition significantly.
lrate
        Learning rate of the network. Lower values cause training to take
        longer while higher values may cause the network not to converge at
        all, i.e. not learn a set of weights producing high recognition
        accuracy.

Training a network will take some time on a modern computer, even with the
default parameters. While the exact time required is unpredictable as training
is a somewhat random process a rough guide is that accuracy seldomly improves
after 40000 epochs reached between 8 and 24 hours of training. 

When to stop is a matter of experience (and personal preference); a fairly
reliable approach known as `early stopping
<https://en.wikipedia.org/wiki/Early_stopping>`_ is stopping training
immediately as soon as the error rate on the test set worsens. This will
prevent `overfitting <https://en.wikipedia.org/wiki/Overfitting>`_, i.e. fitting
the model to recognize only the training data properly instead of the general
patterns contained therein. 

Running the script will look like this after a couple of minutes:

.. code-block:: console
        
        $ ./train.sh output_dir model_name
        *** charsep 
        got 778 files, 100 tests
        got 55 classes
        .stacked: 0.0001 0.9 in 0 48 out 0 55
        .stacked.parallel: 0.0001 0.9 in 0 48 out 0 200
        .stacked.parallel.lstm: 0.0001 0.9 in 0 48 out 0 100
        .stacked.parallel.reversed: 0.0001 0.9 in 0 48 out 0 100
        .stacked.parallel.reversed.lstm: 0.0001 0.9 in 0 48 out 0 100
        .stacked.softmax: 0.0001 0.9 in 0 200 out 0 55
        0
        TRU ܇ܢܝܬܝܡ ܐܠܕ ܐ̈ܝܚ .̣ܢܘܗܠ ̣ܒܗܝ ܐܬܘܒܝܛܒ ܡܕܡ ܠܘܟ
        ALN 
        OUT 
        ERROR 1000 0.60224     2474 4108
        1000
        TRU ܀ܐ̣ܘܗ ܛܝܠܫ ܐܝܬ̈ܚܬ ܠܥܘ ܐܝܠ̈ܥ ܠܥܕ
        ALN ܀܀ܐ̣ܘܗ ܛܛܝܠܫ ܐܝܬܚܬ ܠܥܘ ܐܝܠ ܠܠܠܥܕ
        OUT ܘ ܐܘܘ ܘ ܝܫ ܐܢܘܢ ܢܘ ܐܝ  
        ERROR 2000 0.204236     839 4108
        2000
        TRU ܐܘܚ .ܗܠ ܐ̣ܘܗ ܚܫܚ ܗܬܘܟܐܕ ܐܣܢܓ ܪܒ ܐܠܐ .̣ܐܢ̈ܪܕܥܡ
        ALN ܐܘܚ .ܗܠ ܐ̣ܘܗ ܚܫܚ ܗܬܘܟܐ ܐܣܢܓ ܪܒ ܐܠܐ .ܐܢ̈ܪܕܥܡ
        OUT ܐܘܬܚ .ܗ ܐܘܗ ܚܫܚ ܗܬܘܟܐ ܐܘܢܓ ܪ ܐܠܐ .ܐܢܪܕܥܡa
        ERROR 3000 0.0946933     389 4108
        3000
        TRU ܢܘܗܝ̈ܒܘܥ ܘܓ ܢܡ ܢܕܥ ܠܟܒܕ ܐ̈ܝܡܠܘ ܐܥܪܐܠ ̇ܗܠ ܕܩܦܬܐ
        ALN ܢܘܗܝ̈ܒܘܥ ܘܓ ܢܡ  ܢܕܥ ܠܟܒܕ ܐ̈ܝܡܠܘ ܐܥܪܐܠ ̇ܗܠ ܕܩܦܬܐ
        OUT ܢܘܗܝܒܘܥ ܘܓ ܢܒܢܕܥ ܠܟܒܕܐܝܡܠܘ ܐܥܪܐܠ ܗܠ ܕܩܦܬܐ
        ...

By now there should be a couple of models model_name-1000.clstm,
model_name-2000.clstm, ... in the directory the script was executed in. Lets
take a look at each part of the output.

.. code-block:: console

        got 778 files, 100 tests
        got 55 classes

indicates that the training is running on 778 transcribed lines and a test set
of 100 lines. 55 different classes, i.e. Unicode code points, where found in
these 778 lines. These affect the output size of the network; obviously only
these 55 different classes/code points can later be output by the network.

.. code-block:: console

        ERROR 1000 0.60224     2474 4108

this line shows the results of the test set evaluation. The error after the
``1000`` epochs is ``2474`` incorrect code points out of ``4108`` in the set or
``0.60224``/60%. It should decrease fairly rapidly, in the example it drops to
0.20 after 2000 epochs.

If it remains around 0.70 something is amiss, e.g. non-reordered right-to-left
or wildly incorrect transcriptions. Abort training, correct the error(s) and
start again.

The next part is just the network output for a random line where ``TRU`` is the
transcription and ``OUT`` is the recognized text. ``ALN`` is a derivative
output used internally for adjusting the network's weights. It should not be
used for any kind of evaluation and is purely for informational purposes.

.. code-block:: console

        TRU ܐܘܚ .ܗܠ ܐ̣ܘܗ ܚܫܚ ܗܬܘܟܐܕ ܐܣܢܓ ܪܒ ܐܠܐ .̣ܐܢ̈ܪܕܥܡ
        ALN ܐܘܚ .ܗܠ ܐ̣ܘܗ ܚܫܚ ܗܬܘܟܐ ܐܣܢܓ ܪܒ ܐܠܐ .ܐܢ̈ܪܕܥܡ
        OUT ܐܘܬܚ .ܗ ܐܘܗ ܚܫܚ ܗܬܘܟܐ ܐܘܢܓ ܪ ܐܠܐ .ܐܢܪܕܥܡ

After stoppig training, pick your chosen model and copy it somehwere safe. It
is highly recommended to also archive the training log and data for later
reference.

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
	name: arabic-alam-al-kutub.clstm
	
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

        $ kraken -i INPUT_IMAGE OUTPUT_FILE binarize segment ocr -m arabic-alam-al-kutub.clstm

For more documentation see the kraken `website <http://kraken.re>`_.
