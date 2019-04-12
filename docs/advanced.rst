.. _advanced:

Advanced Usage
==============

Optical character recognition is the serial execution of multiple steps, in the
case of kraken binarization (converting color and grayscale images into bitonal
ones), layout analysis/page segmentation (extracting topological text lines
from an image), recognition (feeding text lines images into an classifiers),
and finally serialization of results into an appropriate format such as hOCR or
ALTO.

Input Specification
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

===========     ====
option          type
===========     ==== 
--threshold     FLOAT
--zoom          FLOAT
--escale        FLOAT
--border        FLOAT
--perc          INTEGER RANGE
--range         INTEGER
--low           INTEGER RANGE
--high          INTEGER RANGE
===========     ====

Page Segmentation and Script Detection
--------------------------------------

The `segment` subcommand access two operations page segmentation into lines and
script detection of those lines.

Page segmentation is mostly parameterless, although a switch to change the
color of column separators has been retained. The segmentation is written as a
`JSON <http://json.org/>`_ file containing bounding boxes in reading order and
the general text direction (horizontal, i.e. LTR or RTL text in top-to-bottom
reading order or vertical-ltr/rtl for vertical lines read from left-to-right or
right-to-left).

The script detection splits extracted lines from the segmenter into strip
sharing a particular script that can then be recognized by supplying
appropriate models for each detected script to the `ocr` subcommand.

Combined output from both consists of lists in the `boxes` field corresponding
to a topographical line and containing one or more bounding boxes of a
particular script. Identifiers are `ISO 15924
<http://www.unicode.org/iso15924/iso15924-codes.html>`_ 4 character codes.

.. code-block:: console

        $ kraken -i 14.tif lines.txt segment
        $ cat lines.json
	{
	   "boxes" : [
            [
                ["Grek", [561, 216, 1626,309]]
            ],
            [
                ["Latn", [2172, 197, 2424, 244]]
            ],
            [
                ["Grek", [1678, 221, 2236, 320]],
                ["Arab", [2241, 221, 2302, 320]]
            ],
            
                ["Grek", [412, 318, 2215, 416]],
                ["Latn", [2208, 318, 2424, 416]]
            ],
            ...
   	   ],
           "script_detection": true,
           "text_direction" : "horizontal-tb"
	}

Script detection is automatically enabled; by explicitly disabling script
detection the `boxes` field will contain only a list of line bounding boxes:

.. code-block:: console

	      [546, 216, 1626, 309],
	      [2169, 197, 2423, 244],
	      [1676, 221, 2293, 320],
              ...
	      [503, 2641, 848, 2681]

Available page segmentation parameters are:

=============================================== ======
option                                          action
=============================================== ======
-d, --text-direction                            Sets principal text direction. Valid values are `horizontal-lr`, `horizontal-rl`, `vertical-lr`, and `vertical-rl`.
--scale FLOAT                                   Estimate of the average line height on the page
-m, --maxcolseps                                Maximum number of columns in the input document. Set to `0` for uni-column layouts.
-b, --black-colseps / -w, --white-colseps       Switch to black column separators.
-r, --remove-hlines / -l, --hlines              Disables prefiltering of small horizontal lines. Improves segmenter output on some Arabic texts.
=============================================== ======

The parameters specific to the script identification are:

=============================================== ======
option                                          action
=============================================== ======
-s/-n                                           Enables/disables script detection
-a, --allowed-script                            Whitelists specific scripts for detection output. Other detected script runs are merged with their adjacent scripts, after a heuristic pre-merging step.
=============================================== ======

Model Repository
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
        ...

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
that image, and a model file. In particular there is no requirement to use the
page segmentation algorithm contained in the ``segment`` subcommand or the
binarization provided by kraken. 

Multi-script recognition is possible by supplying a script-annotated
segmentation and a mapping between scripts and models:

.. code-block:: console

        $ kraken -i ... ... ocr -m Grek:porson.clstm -m Latn:antiqua.clstm

All polytonic Greek text portions will be recognized using the `porson.clstm`
model while Latin text will be fed into the `antiqua.clstm` model. It is
possible to define a fallback model that other text will be fed to:

.. code-block:: console

        $ kraken -i ... ... ocr -m ... -m ... -m default:porson.clstm

It is also possible to disable recognition on a particular script by mapping to
the special model keyword `ignore`. Ignored lines will still be serialized but
will not contain any recognition results.

The ``ocr`` subcommand is able to serialize the recognition results either as
plain text (default), as `hOCR <http://hocr.info>`_, into `ALTO
<http://www.loc.gov/standards/alto/>`_, or abbyyXML containing additional
metadata such as bounding boxes and confidences:

.. code-block:: console

        $ kraken -i ... ... ocr -t # text output
        $ kraken -i ... ... ocr -h # hOCR output
        $ kraken -i ... ... ocr -a # ALTO output
        $ kraken -i ... ... ocr -y # abbyyXML output

hOCR output is slightly different from hOCR files produced by ocropus. Each
``ocr_line`` span contains not only the bounding box of the line but also
character boxes (``x_bboxes`` attribute) indicating the coordinates of each
character. In each line alternating sequences of alphanumeric and
non-alphanumeric (in the unicode sense) characters are put into ``ocrx_word``
spans. Both have bounding boxes as attributes and the recognition confidence
for each character in the ``x_conf`` attribute.

Paragraph detection has been removed as it was deemed to be unduly dependent on
certain typographic features which may not be valid for your input.
