Recognition
-----------

Recognition requires an input image, a page segmentation for that image (both
can be supplied as through a single XML files), and a text recognition model
file. In particular there is no requirement to use the page segmentation
algorithm contained in the ``segment`` subcommand or the binarization provided
by kraken.

Multi-script recognition is possible by supplying a script-annotated
segmentation and a mapping between scripts and models:

.. code-block:: console

        $ kraken -i ... ... ocr -m Grek:porson.mlmodel -m Latn:antiqua.mlmodel

All polytonic Greek text portions will be recognized using the `porson.mlmodel`
model while Latin text will be fed into the `antiqua.mlmodel` model. It is
possible to define a fallback model that other text will be fed to:

.. code-block:: console

        $ kraken -i ... ... ocr -m ... -m ... -m default:porson.mlmodel

It is also possible to disable recognition on a particular script by mapping to
the special model keyword `ignore`. Ignored lines will still be serialized but
will not contain any recognition results.
