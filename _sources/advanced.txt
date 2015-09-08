Advanced usage
==============

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
the format (x0, y0, x1, y1). Lines are printed in reading order::

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

Recognition
-----------

Recognition requires a grey-scale or binarized image, a page segmentation for
that image, and a pyrnn or HDF5 model. In particular there is no requirement to
use the page segmentation algorithm contained in the ``segment`` subcommand or
the binarization provided by kraken. 

The ``ocr`` subcommand is able to print the recognition results either as plain
text (default) or as `hOCR
<https://docs.google.com/document/d/1QQnIQtvdAC_8n92-LhwPcjtAUFwBlzE8EWnKAxlgVf0/preview>`_
containing additional information about the results.

hOCR output is slightly different from hOCR files produced by ocropus. Each
``ocr_line`` span contains not only the bounding box of the line but also
character cuts (``cuts`` attribute) indicating the coordinates of each
character. In each line sequences of alphanumeric (in the unicode sense)
character are put into ``ocrx_word`` spans. Non-alphanumeric sequences are
described by ``ocrx_word`` spans. Both have bounding boxes as attributes and
the recognition confidence for each character in the ``x_conf`` attribute.

Paragraph detection has been removed as it was deemed to be unduly dependent on
certain typographic features which may not be valid for your input.

Parallelization
---------------

Per default execution is parallelized to some extend. Binarization and
segmentation are atomic operations, while recognition is parallelized with each
process operating on a line separately. The default number of processes in the
worker pool is equal to the number of CPU cores, although it may be necessary
to reduce this as kraken is mostly memory-bound on modern system.
