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

There are other ways to define inputs and outputs as the syntax shown above can
become rather cumbersome for large amounts of files.

As such there are a couple of ways to deal with multiple files in a compact
way. The first is batch processing:

.. code-block:: console

        $ kraken -I '*.png' -o ocr.txt segment ...

which expands the `glob expression
<https://en.wikipedia.org/wiki/Glob_(programming)>`_ in kraken internally and
appends the suffix defined with `-o` to each output file. An input file
`xyz.png` will therefore produce an output file `xyz.png.ocr.txt`. A second way
is to input multi-image files directly. These can be either in PDF, TIFF, or
JPEG2000 format and are specified like:

.. code-block:: console

        $ kraken -I some.pdf -o ocr.txt -f pdf segment ...

This will internally extract all page images from the input PDF file and write
one output file with an index (can be changed using the `-p` option) and the
suffix defined with `-o`.

The `-f` option can not only be used to extract data from PDF/TIFF/JPEG2000
files but also various XML formats. In these cases the appropriate data is
automatically selected from the inputs, image data for segmentation or line and
region segmentation for recognition:

.. code-block:: console

        $ kraken -i alto.xml alto.ocr.txt -i page.xml page.ocr.txt -f xml ocr ...

The code is able to automatically determine if a file is in PageXML or ALTO format.

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

Model Repository
----------------

.. _repo:

There is a semi-curated `repository
<https://zenodo.org/communities/ocr_models>`_ of freely licensed recognition
models that can be interacted with from the command line using a few
subcommands. 

Querying and Model Retrieval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``list`` subcommand retrieves a list of all models available and prints
them including some additional information (identifier, type, and a short
description):

.. code-block:: console

        $ kraken list
        Retrieving model list ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 8/8 0:00:00 0:00:07
        10.5281/zenodo.6542744 (pytorch) - LECTAUREP Contemporary French Model (Administration)
        10.5281/zenodo.5617783 (pytorch) - Cremma-Medieval Old French Model (Litterature)
        10.5281/zenodo.5468665 (pytorch) - Medieval Hebrew manuscripts in Sephardi bookhand version 1.0
        ...

To access more detailed information the ``show`` subcommand may be used:

.. code-block:: console

        $ kraken show 10.5281/zenodo.5617783
        name: 10.5281/zenodo.5617783

        Cremma-Medieval Old French Model (Litterature)

        ....
        scripts: Latn
        alphabet: &'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVXabcdefghijklmnopqrstuvwxyz¶ãíñõ÷ħĩłũƺᵉẽ’•⁊⁹ꝑꝓꝯꝰ SPACE, COMBINING ACUTE ACCENT, COMBINING TILDE, COMBINING MACRON, COMBINING ZIGZAG ABOVE, COMBINING LATIN SMALL LETTER A, COMBINING LATIN SMALL LETTER E, COMBINING LATIN SMALL LETTER I, COMBINING LATIN SMALL LETTER O, COMBINING LATIN SMALL LETTER U, COMBINING LATIN SMALL LETTER C, COMBINING LATIN SMALL LETTER R, COMBINING LATIN SMALL LETTER T, COMBINING UR ABOVE, COMBINING US ABOVE, COMBINING LATIN SMALL LETTER S, 0xe8e5, 0xf038, 0xf128
        accuracy: 95.49%
        license: CC-BY-SA-2.0
        author(s): Pinche, Ariane
        date: 2021-10-29

If a suitable model has been decided upon it can be retrieved using the ``get``
subcommand:

.. code-block:: console

        $ kraken get 10.5281/zenodo.5617783
        Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 16.1/16.1 MB 0:00:00 0:00:10
        Model name: cremma_medieval_bicerin.mlmodel

Models will be placed in ``$XDG_BASE_DIR`` and can be accessed using their name as
printed in the last line of the ``kraken get`` output.

.. code-block:: console

        $ kraken -i ... ... ocr -m cremma_medieval_bicerin.mlmodel

Publishing
^^^^^^^^^^

When one would like to share a model with the wider world (for fame and glory!)
it is possible (and recommended) to upload them to repository. The process
consists of 2 stages: the creation of the deposit on the Zenodo platform
followed by approval of the model in the community making it discoverable for
other kraken users.

For uploading model a Zenodo account and a personal access token is required.
After account creation tokens can be created under the account settings:

.. image:: _static/pat.png
  :width: 800
  :alt: Zenodo token creation dialogue

With the token models can then be uploaded:

.. code-block:: console

   $ ketos publish -a $ACCESS_TOKEN aaebv2-2.mlmodel
   DOI: 10.5281/zenodo.5617783

A number of important metadata will be asked for such as a short description of
the model, long form description, recognized scripts, and authorship.
Afterwards the model is deposited at Zenodo. This deposit is persistent, i.e.
can't be changed or deleted so it is important to make sure that all the
information is correct. Each deposit also has a unique persistent identifier, a
DOI, that can be used to refer to it, e.g. in publications or when pointing
someone to a particular model.

Once the deposit has been created a request (requiring manual approval) for
inclusion in the repository will automatically be created which will make it
discoverable by other users.

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
