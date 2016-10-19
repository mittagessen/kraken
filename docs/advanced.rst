Advanced usage
==============

Optical character recognition is the serial execution of multiple steps, in the
case of kraken binarization (converting color and grayscale images into bitonal
ones), layout analysis/page segmentation (extracting topological text lines
from an image), recognition (feeding text lines images into an classifiers),
and finally serialization of results into an appropriate format such as hOCR or
ALTO.

Input specification
-------------------

All kraken subcommands operating on input-output pairs, i.e. producing one
output document for one input document follow the basic syntax:

.. code-block:: console

        $ kraken -i input_1 output_1 -i input_2 output_2 ... subcommand_1 subcommand_2 ... subcommand_n

In particular subcommands may be chained.

Binarization
------------

The binarization subcommand accepts almost the same parameters as
``ocropus-nlbin``. Only options not related to binarization, e.g. skew
detection are missing. In addition, error checking (image sizes, inversion
detection, grayscale enforcement) is always disabled and kraken will happily
binarize any image that is thrown at it.

Available parameters are:

  --threshold FLOAT
  --zoom FLOAT
  --escale FLOAT
  --border FLOAT
  --perc INTEGER RANGE
  --range INTEGER
  --low INTEGER RANGE
  --high INTEGER RANGE


Page segmentation
-----------------

Page segmentation is mostly parameterless, although a switch to change the
color of column separators has been retained. The segmentation is written as a
plain text CSV file. Each record corresponds to a single line bounding box in
the format (x0, y0, x1, y1). Lines are printed in reading order:

.. code-block:: console

        $ kraken -i 14.tif lines.txt segment
        $ cat lines.txt
        422,188,2158,327
        421,328,2158,430
        415,464,2153,599
        412,604,2153,745
        406,744,2152,882
        405,877,2144,1020
        403,1020,2139,1150
        399,1160,2136,1297
        394,1292,2137,1435
        391,1431,2131,1572
        385,1566,2128,1709
        379,1710,2128,1830
        383,1854,2126,1986
        370,1985,2125,2127
        369,2123,2118,2268
        369,2268,2111,2407
        366,2401,2116,2514
        363,2541,2107,2682
        364,2677,2109,2819
        358,2815,2106,2956
        354,2955,2098,3092
        355,3092,2094,3230
        1859,3233,2084,3354

Model repository
----------------

There is a semi-curated `repository
<https://github.com/mittagessen/kraken-models>`_ of freely licensed recognition
models that can be accessed from the command line using a few subcommands. For
evaluating a series of models it is also possible to just clone the repository
using the normal git client. 

The ``list`` subcommand retrieves a list of all models available and prints
them including some additional information (identifier, type, and a short
description):

.. code-block:: console

        $ kraken list
        Retrieving model list   ✓
        default (pyrnn) - A converted version of en-default.pyrnn.gz
        toy (clstm) - A toy model trained on 400 lines of the UW3 data set.


To access more detailed information the ``show`` subcommand may be used:

.. code-block:: console

        $ kraken show toy
        name: toy.clstm

        A toy model trained on 400 lines of the UW3 data set.

        author: Benjamin Kiessling (mittagessen@l.unchti.me)
        http://kraken.re

If a suitable model has been decided upon it can be retrieved using the ``get``
subcommand:

.. code-block:: console

        $ kraken get toy
        Retrieving model        ✓

Models will be placed in $XDG_BASE_DIR and can be accessed using their name as
shown by the ``show`` command, e.g.:

.. code-block:: console

        $ kraken -i ... ... ocr -m toy

Additions and updates to existing models are always welcome! Just open a pull
request or write an email.

Recognition
-----------

Recognition requires a grey-scale or binarized image, a page segmentation for
that image, and a pyrnn or protobuf model. In particular there is no
requirement to use the page segmentation algorithm contained in the ``segment``
subcommand or the binarization provided by kraken. 

The ``ocr`` subcommand is able to serialize the recognitino results either as
plain text (default), as `hOCR
<https://docs.google.com/document/d/1QQnIQtvdAC_8n92-LhwPcjtAUFwBlzE8EWnKAxlgVf0/preview>`_,
or into `ALTO <http://www.loc.gov/standards/alto/>`_ containing additional
metadata such as bounding boxes and confidences:

.. code-block:: console

        $ kraken -i ... ... ocr -t # text output
        $ kraken -i ... ... ocr -h # hOCR output
        $ kraken -i ... ... ocr -a # ALTO output

hOCR output is slightly different from hOCR files produced by ocropus. Each
``ocr_line`` span contains not only the bounding box of the line but also
character cuts (``cuts`` attribute) indicating the coordinates of each
character. In each line sequences of alphanumeric (in the unicode sense)
character are put into ``ocrx_word`` spans. Non-alphanumeric sequences are
described by ``ocrx_word`` spans. Both have bounding boxes as attributes and
the recognition confidence for each character in the ``x_conf`` attribute.

Paragraph detection has been removed as it was deemed to be unduly dependent on
certain typographic features which may not be valid for your input.
