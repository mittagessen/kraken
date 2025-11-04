Recognition
-----------

Recognition requires an input image, a page segmentation for that image (both
can be supplied as through a single XML files), and a text recognition model
file. In particular there is no requirement to use the page segmentation
algorithm contained in the ``segment`` subcommand or the binarization provided
by kraken.

It is possible to apply different recognition models to different parts of a
segmentation, for example if the lines in the segmentation are annotated by
language, script, or hand:

.. code-block:: console

        $ kraken -i ... ... ocr -m Grek:porson.mlmodel -m Latn:antiqua.mlmodel

All lines having the line type `Grek` (polytonic Greek) will be recognized
using the `porson.mlmodel` model while lines with `Latn` type will be fed into
the `antiqua.mlmodel` model. Obviously this requires the segmentation model to
output lines with these line types. It is possible to define a fallback model
that other text will be fed to:

.. code-block:: console

        $ kraken -i ... ... ocr -m ... -m ... -m default:porson.mlmodel

It is also possible to disable recognition on a particular script by mapping to
the special model keyword `ignore`. Ignored lines will still be serialized but
will not contain any recognition results.


