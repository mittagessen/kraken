.. _segtrain:

Segmentation Training
=====================

Segmentation training allows kraken to perform layout analysis, finding text
lines and regions, on material that is insufficiently well recognized by the
default model. It is not necessary to train a new segmentation model for each
new type of document, the default segmentation model works well on quite a
variety of handwritten and printed documents. On the other hand, if a specific
typology of lines and regions is needed or the output is subpar it can often be
a good choice to train a new segmentation model.

Best practices
--------------

* The segmenter is fairly robust when it comes to hyperparameter choice.
* Start by finetuning from the default model for a fixed number of epochs (50 for reasonably sized datasets) with a cosine schedule.
* Segmentation models' performance is difficult to evaluate. Pixel accuracy doesn't mean much because there are many more pixels that aren't part of a line or region than just background. Frequency-weighted IoU is good for overall performance, while mean IoU overrepresents rare classes. The best way to evaluate segmentation models is to look at the output on unlabelled data.
* If you don't have rare classes you can use a fairly small validation set to make sure everything is converging and just visually validate on unlabelled data.
* Upload your models to the :ref:`model repository <repo>`.

Training data formats
---------------------

Segmentation training supports the XML formats described in the :ref:`data
format <ketos_format>` section.

Training
--------

Training a segmentation model is very similar to training models for :ref:`text
recognition <rectrain>` in terms of overall process and available options. The
basic invocation is:

.. code-block:: console

        $ ketos segtrain -f xml training_data/*.xml

This takes all text lines and regions encoded in the XML files and trains a
model to recognize them.

Most other options available in transcription training are also available in
segmentation training. CUDA acceleration:

.. code-block:: console

        $ ketos -d cuda segtrain -f xml training_data/*.xml

Defining custom architectures:

.. code-block:: console

        $ ketos -d cuda segtrain -s '[1,1200,0,3 Cr7,7,64,2,2 Gn32 Cr3,3,128,2,2 Gn32 Cr3,3,128 Gn32 Cr3,3,256 Gn32]' -f xml training_data/*.xml

Fine tuning/transfer learning with last layer adaptation and slicing:

.. code-block:: console

        $ ketos segtrain --resize new -i segmodel_best.mlmodel training_data/*.xml
        $ ketos segtrain -i segmodel_best.mlmodel --append 7 -s '[Cr3,3,64 Do0.1]' training_data/*.xml

In addition there are a couple of specific options that allow filtering of
baseline and region types. Datasets are often annotated to a level that is too
detailed or contains undesirable types, e.g. when combining segmentation data
from different sources. The most basic option is the suppression of *all* of
either baseline or region data contained in the dataset:

.. code-block:: console

        $ ketos segtrain --suppress-baselines -f xml training_data/*.xml
        Training line types:
        Training region types:
          graphic	3	135
          text	4	1128
          separator	5	5431
          paragraph	6	10218
          table	7	16
        ...
        $ ketos segtrain --suppress-regions -f xml training-data/*.xml
        Training line types:
          default 2	53980
          foo     8     134
        ...

It is also possible to filter out baselines/regions selectively:

.. code-block:: console

        $ ketos segtrain -f xml --valid-baselines default training_data/*.xml
        Training line types:
          default 2	53980
        Training region types:
          graphic	3	135
          text	4	1128
          separator	5	5431
          paragraph	6	10218
          table	7	16
        $ ketos segtrain -f xml --valid-regions graphic --valid-regions paragraph training_data/*.xml
        Training line types:
          default 2	53980
          foo     3     134
         Training region types:
          graphic	4	135
          paragraph	5	10218

We can merge baselines and regions into each other:

.. code-block:: console

        $ ketos segtrain -f xml --merge-baselines default:foo training_data/*.xml
        Training line types:
          default 2	54114
        ...
        $ ketos segtrain -f xml --merge-regions text:paragraph --merge-regions graphic:table training_data/*.xml
        ...
        Training region types:
          graphic	3	151
          text	4	11346
          separator	5	5431
        ...

Finally, it is possible to merge all baseline or region types into a single type with the ``--merge-all-*`` options:

.. code-block:: console

        $ ketos segtrain -f xml --merge-all-baselines DefaultLine training_data/*.xml
        Training line types:
          DefaultLine 2	54114
        ...
        $ ketos segtrain -f xml --merge-all-regions DefaultRegion training_data/*.xml
        ...
        Training region types:
          DefaultRegion 3	16928
        ...

These options are combinable to massage the dataset into any typology you want.
Tags containing the separator character `:` can be specified by escaping them
with backslash.

Then there are some options that set metadata fields controlling the
postprocessing. When computing the bounding polygons the recognized baselines
are offset slightly to ensure overlap with the line corpus. This offset is per
default upwards for baselines but as it is possible to annotate toplines (for
scripts like Hebrew) and centerlines (for baseline-free scripts like Chinese)
the appropriate offset can be selected with an option:

.. code-block:: console

        $ ketos segtrain --topline -f xml hebrew_training_data/*.xml
        $ ketos segtrain --centerline -f xml chinese_training_data/*.xml
        $ ketos segtrain --baseline -f xml latin_training_data/*.xml

Evaluation
----------

Numerical evaluation options are of limited value for the segmenter and it is
generally recommended to visually confirm the quality of an existing or newly
created model.

Nevertheless, the ``segtest`` tool exists which is able 

