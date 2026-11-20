.. _ketos_synth:

Synthetic training data
=======================

Synthetic training data is intended to decrease the amount of manually
annotated data needed for training recognition models. Kraken does not include
a synthetic training data generation util anymore but `pangoline
<https://github.com/mittagessen/pangoline>`_ can be used with only minor
adaptation necessary. The data produced through this way is most useful for
training models intended to recognize machine-printed text due to it using
digital fonts to create type approximating real world data which is usually
insufficiently close to handwritten text to achieve acceptable error rates.

Installing pangoline
--------------------

The most up-to-date instructions on pangoline installation can be found `here
<https://github.com/mittagessen/pangoline>`_ but the basic procedure is fairly
simple:

.. code-block:: console

  $ pipx install pangoline-tool

Note that pangoline requires the pygobject, pango, and cairo libraries to be
installed on the system in order to function properly. It is usually necessary
to install them through the package manager of your operating system. An
alternative is to use `anaconda <https://anaconda.org>`_ which has binary
packages for all dependencies:

.. code-block:: console

   $ conda create --name pangoline-py3.11 -c conda-forge python=3.11
   $ conda activate pangoline-py3.11
   $ conda install -c conda-forge pygobject pango Cairo click jinja2 rich pypdfium2 lxml pillow
   $ pip install pangoline-tool

Rendering text
--------------

Pangoline produces output data in two steps. First text is rendered into vector
PDFs with blank canvases and ALTO files containing physical dimensions are
written. In a second pass the vectorized data is rasterized with the desired
resolution, optionally placed on a background image, and the ALTO's positional
information converted into the pixel domain.

The most basic invocation of pangoline is:

.. code-block:: console

   $ pangoline render doc.txt
   Rendering ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

which will produce output `doc.0.pdf, doc.0.xml, doc.1.pdf, doc.1.xml`...

Various options to direct rendering such as page size, margins, language, and
base direction can be manually set, for example:

.. code-block:: console

  $ pangoline render -p 216 279 -l en-us -f "Noto Sans 24" doc.txt
   Rendering ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

which renders the text with the Noto Sans font in size 24 onto US Letter size
pages (216x279mm) and sets the language to American English.

Text can also be styled with `Pango Markup
<https://docs.gtk.org/Pango/pango_markup.html>`_. Parsing is disabled per
default but can be enabled with a switch:

.. code-block:: console

  $ pangoline render --markup doc.txt
  Rendering ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

A more useful approach in most cases is to apply random styles (bolding,
italicization, ...) to some words in the source text: 

.. code-block:: console

  $ pangoline render --random-markup-probability 0.01 doc.txt
  Rendering ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

The above command will apply one or more random styles to roughly 1% of words.
The set of styles to apply can also be configured:

.. code-block:: console

  $ pangoline render --random-markup-probability 0.01 --random-markup style_italic --random-markup variant_smallcaps doc.txt
  Rendering ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

A list of available styles as well as the set of default styles is available on
the help screen `pangoline render --help`. The semantics of each value can be
found in the `pango
documentation <https://docs.gtk.org/Pango/pango_markup.html>`_.

Rasterization
-------------

In a second step those vector files can be rasterized into PNGs and the
coordinates in the ALTO files scaled to the selected resolution (per default
300dpi):

.. code-block:: console

  $ pangoline rasterize doc.0.xml doc.1.xml ...
  Rasterizing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

To obtain slightly more realistic input images it is possible to overlay the
rasterized text into images of writing surfaces.

.. code-block:: console

  $ pangoline rasterize -w ~/background_1.jpg doc.0.xml doc.1.xml ...
  Rasterizing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

Rasterization can be invoked with multiple background images in which case they
will be sampled randomly for each output page. A tarball with 70 empty paper
backgrounds of different origins, digitization qualities, and states of
preservation can be found `here <http://l.unchti.me/paper.tar>`_. A more
ergonomic way to specify multiple background images exists with the `-W`
option that takes a manifest file containing a list of image file paths:

.. code-block:: console

  $ cat backgrounds.lst
  background_0.jpg
  background_1.jpg
  ...
  background_N.jpg
  $ pangoline rasterize -W backgrounds.lst doc.0.xml doc.1.xml ...
  Rasterizing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

Repolygonization
----------------

The ALTO files produced by pangoline contain baselines and bounding polygons
for each text line. Nevertheless, they are not immediately usable for training
purposes as the bounding polygons provided through the Pango layout engine are
just rectangular boxes that are quite different from the bounding polygons
produced by kraken's native polygonizer. Hence, it is necessary to recompute
the bounding polygons around the line content.

The kraken source repository contains a `script
<https://github.com/mittagessen/kraken/blob/main/kraken/contrib/repolygonize.py>`_
that aids in repolygonization by wrapping the API in an easy to use command
line interface. After downloading it you can repolygonize XML files as such:

.. code-block:: console

   $ python repolygonize.py -f alto *.xml

The repolygonized files with have a `_rewrite.xml` suffix attached to them. You
can use these files as is or :ref:`compile <binary_datasets>` them into binary
datasets.
