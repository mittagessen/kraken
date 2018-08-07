.. _ketos:

Training
========

This page describes the training utilities available through the ``ketos``
command line utility in depth. For a gentle introduction on model training
please refer to the :ref:`tutorial <training>`. 

Thanks to the magic of `Connectionist Temporal Classification
<ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf>`_ prerequisites for creating a
new recognition model are quite modest. The basic requirement is a number of
text lines (``ground truth``) that correspond to line images and some time for
training.

Transcription
-------------

Transcription is done through local browser based HTML transcription
environments. These are created by the ``ketos transcribe`` command line util.
Its basic input is just a number of image files and an output path to write the
HTML file to:

.. code-block:: console

        $ ketos transcribe -o output.html image_1.png image_2.png ...


While it is possible to put multiple images into a single transcription
environment splitting into one-image-per-HTML will ease parallel transcription
by multiple people.

The above command reads in the image files, converts them to black and white if
desired, tries to split them into line images, and puts an editable text
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
the ``Download`` button on the lower right or through the regular ``Save Page
As`` function of the browser. All the work done is contained directly in the
saved files and it is possible to save partially transcribed files and continue
work later.

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
        are written.
--normalization
        Unicode has code points to encode most glyphs encountered in the wild.
        A lesser known feature is that there usually are multiple ways to
        encode a string of printed characters, i.e. a series of what a human
        would consider a single character (in Unicode jargon ``grapheme
        clusters``) in multiple ways. Crucially, accented characters marks may
        either be a single code point (``precomposed``) or two or more separate
        symbols (``decomposed``). Many texts contain a mixture of both.
        `Unicode normalization <http://www.unicode.org/reports/tr15/>`_ ensures
        that equal grapheme cluster are encoded in the same way, i.e. that the
        encoded representation across the training data set is consistent and
        there is only one way the network can recognize a particular feature on
        the page. Usually it is sufficient to set the normalization to
        Normalization Form Decomposed (NFD), as it reduces the the size of the
        overall script to be recognized slightly.

The result will be a directory filled with line image text pairs ``NNNNNN.png``
and ``NNNNNN.gt.txt`` and a ``manifest.txt`` containing a list of all extracted
lines.

Training
--------

Currently kraken does not contain a training interface. Use the
``clstmocrtrain`` command contained in the CLSTM distribution.

Validation
----------

TODO

Artificial Training Data
------------------------

It is possible to rely on artificially created training data, instead of
laborously creating ground truth by manual means. A proper typeface and some
text in the target language will be needed. 

For many popular historical fonts there are free reproductions which quite
closely match printed editions. Most are available in your distribution's
repositories and often shipped with TeX Live.

Some good places to start for non-Latin scripts are:

- `Amiri <http://www.amirifont.org/>`_, a classical Arabic typeface by Khaled
  Hosny
- The `Greek Font Society <http://www.greekfontsociety.gr/>`_ offers freely
  licensed (historical) typefaces for polytonic Greek.
- The friendly religious fanatics from `SIL <http://scripts.sil.org/>`_
  assemble a wide variety of fonts for non-Latin scripts.

Next we need some text to generate artificial line images from. It should be a
typical example of the type of printed works you want to recognize and at least
500-1000 lines in length. 

A minimal invocation to the line generation tool will look like this:

.. code-block:: console

        $ ketos linegen -f Amiri da1.txt da2.txt
        Reading texts   ✓
        Read 3692 unique lines
        Σ (len: 99)
        Symbols:  !(),-./0123456789:ABEFGHILMNPRS[]_acdefghiklmnoprstuvyz«»،؟ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىيپ
        Writing images  ✓

The output will be written to a directory called ``training_data``, although
this may be changed using the ``-o`` option. Each text line is rendered using
the Amiri typeface.

Alphabet and Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's take a look at important information in the preamble:

.. code-block:: console

        Read 3692 unique lines
        Σ (len: 99)
        Symbols:  !(),-./0123456789:ABEFGHILMNPRS[]_acdefghiklmnoprstuvyz«»،؟ﺀﺁﺃﺅﺈﺋﺎﺑﺔﺘﺜﺠﺤﺧﺩﺫﺭﺰﺴﺸﺼﻀﻄﻈﻌﻐـﻔﻘﻜﻠﻤﻨﻫﻭﻰﻳپ

ketos tells us that it found 3692 unique lines which contained 99 different
``symbols`` or ``code points``.  We can see the training data contains all of
the Arabic script including accented precomposed characters, but only a subset
of Latin characters, numerals, and punctuation. A trained model will be able to
recognize only these exact symbols, e.g. a ``C`` or ``j`` on the page will
never be recognized. Either accept this limitation or add additional text lines
to the training corpus until the alphabet matches your needs.

We can also force a normalization form using the ``-u`` option; per default
none is applied. For example:

.. code-block:: console

        $ ketos linegen -u NFD -f "GFS Philostratos" grc.txt
        Reading texts   ✓
        Read 2860 unique lines
        Σ (len: 132)
        Symbols:  #&'()*,-./0123456789:;ABCDEGHILMNOPQRSTVWXZ]abcdefghiklmnopqrstuvxy §·ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρςστυφχψω—‘’“
        Combining Characters: COMBINING GRAVE ACCENT, COMBINING ACUTE ACCENT, COMBINING DIAERESIS, COMBINING COMMA ABOVE, COMBINING REVERSED COMMA ABOVE, COMBINING DOT BELOW, COMBINING GREEK PERISPOMENI, COMBINING GREEK YPOGEGRAMMENI


        $ ketos linegen -u NFC -f "GFS Philostratos" grc.txt
        Reading texts   ✓
        Read 2860 unique lines
        Σ (len: 231)
        Symbols:  #&'()*,-./0123456789:;ABCDEGHILMNOPQRSTVWXZ]abcdefghiklmnopqrstuvxy §·ΐΑΒΓΔΕΖΘΙΚΛΜΝΞΟΠΡΣΤΦΧΨΩάέήίαβγδεζηθικλμνξοπρςστυφχψωϊϋόύώἀἁἂἃἄἅἈἌἎἐἑἓἔἕἘἙἜἝἠἡἢἣἤἥἦἧἩἭἮἰἱἳἴἵἶἷἸἹἼὀὁὂὃὄὅὈὉὌὐὑὓὔὕὖὗὙὝὠὡὢὤὥὦὧὨὩὰὲὴὶὸὺὼᾄᾐᾑᾔᾗᾠᾤᾧᾳᾶᾷῃῄῆῇῒῖῥῦῬῳῴῶῷ—‘’“
        Combining Characters: COMBINING ACUTE ACCENT, COMBINING DOT BELOW

While there hasn't been any study on the effect of different normalizations on
recognition accuracy there are some benefits to NFD, namely decreased model
size and easier validation of the alphabet.

Other Parameters
~~~~~~~~~~~~~~~~

Sometimes it is desirable to draw a certain number of lines randomly from one
or more large texts. The ``-n`` option does just that:

.. code-block:: console
        
        $ ketos linegen -u NFD -n 100 -f Amiri da1.txt da2.txt da3.txt da4.txt
        Reading texts   ✓
        Read 114265 unique lines
        Sampling 100 lines      ✓
        Σ (len: 64)
        Symbols:  !(),-./0123456789:[]{}«»،؛؟ءابةتثجحخدذرزسشصضطظعغـفقكلمنهوىي–
        Combining Characters: ARABIC MADDAH ABOVE, ARABIC HAMZA ABOVE, ARABIC HAMZA BELOW
        Writing images ⢿

It is also possible to adjust to amount of degradation/distortion of line
images by using the ``-s/-r/-d/-ds`` switches:

.. code-block:: console

        $ ketos linegen -m 0.2 -s 0.002 -r 0.001 -d 3 Downloads/D/A/da1.txt
        Reading texts   ✓
        Read 859 unique lines
        Σ (len: 46)
        Symbols:  !"-.:،؛؟ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي﻿
        Writing images  ⣽


Sometimes the shaping engine misbehaves using some fonts (notably ``GFS
Philostratos``) by rendering texts in certain normalizations incorrectly if the
font does not contain glyphs for decomposed characters. One sign are misplaced
diacritics and glyphs in different fonts. A workaround is renormalizing the
text for rendering purposes (here to NFC):

.. code-block:: console

        $ ketos linegen -ur NFC -u NFD -f "GFS Philostratos" grc.txt


