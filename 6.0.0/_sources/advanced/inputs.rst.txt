Input and Outputs
-----------------

Kraken inputs and their outputs can be defined in multiple ways. The most
simple are input-output pairs, i.e. producing one output document for one input
document follow the basic syntax:

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
`xyz.png` will therefore produce an output file `xyz.png.ocr.txt`. `-I` batch
inputs can also be specified multiple times:

.. code-block:: console

        $ kraken -I '*.png' -I '*.jpg' -I '*.tif' -o ocr.txt segment ...

A second way is to input multi-image files directly. These can be either in
PDF, TIFF, or JPEG2000 format and are specified like:

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

Output formats
^^^^^^^^^^^^^^

All commands have a default output format such as raw text for `ocr`, a plain
image for `binarize`, or a JSON definition of the the segmentation for
`segment`. These are specific to kraken and generally not suitable for further
processing by other software but a number of standardized data exchange formats
can be selected. Per default `ALTO <http://www.loc.gov/standards/alto/>`_,
`PageXML <https://en.wikipedia.org/wiki/PAGE_(XML)>`_, `hOCR
<http://hocr.info>`_, and abbyyXML containing additional metadata such as
bounding boxes and confidences are implemented. In addition, custom `jinja
<https://jinja.palletsprojects.com>`_ templates can be loaded to create
individualised output such as TEI.

Output formats are selected on the main `kraken` command and apply to the last
subcommand defined in the subcommand chain. For example:

.. code-block:: console

        $ kraken --alto -i ... segment -bl

will serialize a plain segmentation in ALTO into the specified output file.

The currently available format switches are:

.. code-block:: console

        $ kraken -n -i ... ... # native output
        $ kraken -a -i ... ... # ALTO output
        $ kraken -x -i ... ... # PageXML output
        $ kraken -h -i ... ... # hOCR output
        $ kraken -y -i ... ... # abbyyXML output

Custom templates can be loaded with the ``--template`` option:

.. code-block:: console

        $ kraken --template /my/awesome/template.tmpl -i ... ...

The data objects used by the templates are considered internal to kraken and
can change from time to time. The best way to get some orientation when writing
a new template from scratch is to have a look at the existing templates `here
<https://github.com/mittagessen/kraken/tree/main/kraken/templates>`_.
