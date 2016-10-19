Training
========

Thanks to the magic of `Connectionist Temporal Classification
<ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf>`_ prerequisites for creating a
new recognition model are quite modest. The basic requirement is a number of
text lines (``ground truth``) that correspond to line images and some time for
training.

Training tasks are covered by subcommands attached to the ``ketos`` command.

Ground Truth Editing
--------------------

Training
--------
 
Validation
----------

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
``symbols`` or ``code points``. The important part is to remember that one can
encode a string of printed characters, i.e. a series of what a human would
consider a single character (in Unicode jargon ``grapheme clusters``) in
multiple ways. Crucially, accented characters marks may either be a single code
point (``precomposed``) or two or more separate symbols (``decomposed``). Many
texts contain a mixture of both. A text that is maximally
decomposed/precomposed as said to be in normalization form (NF) D/C.  There is
a `Wikipedia <https://en.wikipedia.org/wiki/Unicode_equivalence>`_ lemma on
this phenomenon.

With that in mind we can see the training data contains all of the Arabic
script including accented precomposed characters, but only a subset of Latin
characters, numerals, and punctuation. A trained model will be able to
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


