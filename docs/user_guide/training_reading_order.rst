.. _training_reading_order:

Reading Order Model Training
============================

This guide covers the training of reading order models. Reading order models
predict the correct sequence in which text lines or regions on a page should be
read. They are trained separately and can then be combined with a segmentation
model for end-to-end layout analysis. Kraken supports baseline-level ordering,
region-level ordering, and a hybrid mode where both models are attached to a
segmentation model to produce a hierarchical order (regions first, then lines
within each region).

Data Preparation
----------------

Reading order models are trained on ALTO or PageXML files that contain reading
order annotations, i.e. an explicit ordering of the baselines or regions on
each page. The annotations must be present in the XML reading order elements
(``<ReadingOrder>`` in PageXML, ``<ReadingOrder>`` in ALTO).

The recommended way to provide training and evaluation data to kraken is
through manifest files. A manifest is a simple text file containing a list of
paths to image files, one per line. These manifests are passed to the ``ketos``
training commands using the ``-t`` and ``-e`` options.

For annotating reading order, we recommend the `eScriptorium`_ platform which
supports reading order annotation and is tightly integrated with kraken.

.. _`eScriptorium`: https://www.escriptorium.fr/

Training
--------

To train a reading order model, use the ``ketos rotrain`` command. You need to
provide the training data and, optionally, a validation set:

.. code-block:: console

    $ ketos rotrain -t training_manifest.txt -e validation_manifest.txt

To train a region-level model, set ``--level regions``:

.. code-block:: console

    $ ketos rotrain --level regions -t training_manifest.txt -e validation_manifest.txt

If no explicit validation set is given, the training data is automatically
split according to the ``--partition`` ratio (default: 0.9).

**Ordering Level**

The ``--level`` option selects whether the model learns to order baselines or
regions:

- ``baselines`` (default): Train on baseline-level reading order.
- ``regions``: Train on region-level reading order.

**Hybrid Mode (Region + Baseline)**

You can attach both a region-ordering model and a baseline-ordering model to
the same segmentation model. At inference time this produces a hierarchical
order: regions are ordered first, then lines are ordered within each region.
This is useful for complex page layouts where region order alone is
insufficient.

Combine models using ``ketos convert`` (recommended). ``ketos roadd`` still
works but is deprecated:

.. code-block:: console

    $ ketos convert -i seg.mlmodel \
                    --add-ro line_ro.safetensors \
                    --add-ro region_ro.safetensors \
                    -o seg_with_line_and_region_ro.mlmodel

    # deprecated
    $ ketos roadd -o seg_with_line_ro.mlmodel line_ro.safetensors -i seg.mlmodel
    $ ketos roadd -o seg_with_line_and_region_ro.mlmodel region_ro.safetensors \
                  -i seg_with_line_ro.mlmodel

**Reading Order Selection**

Documents may contain multiple named reading orders. The ``--reading-order``
option selects which one to train on. By default it uses ``line_implicit`` for
baseline-level and ``region_implicit`` for region-level ordering. If your data
has named reading orders, specify the name here.

**Fine-tuning**

To fine-tune an existing reading order model:

.. code-block:: console

    $ ketos rotrain --load existing_ro_model.safetensors -t training_manifest.txt -e validation_manifest.txt

As with other training commands, ``--load`` resets the learning rate schedule
while ``--resume`` continues from where a previous training run left off. The
two options are mutually exclusive.

**Resuming Training**

If a training run is interrupted (e.g. by a crash, timeout, or manual
cancellation), it can be continued from the last saved checkpoint using the
``--resume`` option:

.. code-block:: console

    $ ketos rotrain --resume checkpoint_0005.ckpt

Unlike ``--load``, which only restores the model weights and starts a fresh
training run, ``--resume`` restores the full training state: model weights,
optimizer state, learning rate scheduler position, and the current epoch count.
Training continues exactly where it left off. No training data or
hyperparameter options need to be provided on the command line as these are
restored from the checkpoint as well.

Kraken also saves an emergency checkpoint (``checkpoint_abort.ckpt``) when
training is interrupted by an unhandled exception.

Class Mappings
--------------

Reading order models use a class mapping to assign integer labels to each
baseline or region type encountered in the training data. By default, indices
are auto-assigned starting at 1 (0 is reserved for unknown/default types). This
works well for standalone training but can cause mismatches when combining a
reading order model with a segmentation model via ``roadd`` or ``ketos
convert``.

**From a segmentation checkpoint**

The simplest way to ensure matching class mappings is to extract the mapping
from an existing segmentation checkpoint:

.. code-block:: console

    $ ketos rotrain --class-mapping-from-ckpt segmentation_model.ckpt \
                    --level baselines -t training_manifest.txt

The ``--level`` option determines which mapping is extracted: ``baselines``
reads the ``line_class_mapping``, ``regions`` reads the ``region_class_mapping``
from the checkpoint. This ensures that the reading order model's mapping is
exactly the same as the segmentation model's, which is required when combining
models. Only ``.ckpt`` checkpoint files are supported (not ``.safetensors`` or
``.mlmodel`` weights files).

.. note::

    Loading may fail if the checkpoint's serialized config references training
    data paths that have moved or been deleted. In that case, specify the
    mapping explicitly via an experiment file instead.

**Explicit mapping via YAML**

A class mapping can also be specified directly in a YAML experiment file using
the ``class_mapping`` key under ``rotrain:``:

.. code-block:: yaml

    rotrain:
      class_mapping:
        - [default, 1]
        - [heading, 2]
        - [commentary, 3]
        - ['*', 0]

The syntax is the same as for ``line_class_mapping`` in segmentation training:
a list of ``[class_name, label]`` pairs. The ``*`` wildcard maps all unmatched
types to the given label. Without a wildcard, unknown types are silently
skipped.

``--class-mapping-from-ckpt`` and the YAML ``class_mapping`` key are mutually
exclusive.

**YAML example**

A minimal ``rotrain`` configuration in YAML:

.. code-block:: yaml

    rotrain:
      # training data manifests
      training_data:
        - train.lst
      evaluation_data:
        - val.lst
      # reading order configuration
      level: baselines
      reading_order: line_implicit
      # optional: extract class mapping from segmentation checkpoint
      # class_mapping_from_ckpt: segtrain_checkpoint.ckpt
      # training hyperparameters
      batch_size: 15000
      epochs: 3000
      min_epochs: 500
      lag: 300
      lrate: 1e-3

Experiment Files
----------------

Instead of supplying all options through the command line, it is possible to
put options into a YAML file and pass it to the ``ketos`` command using the
``--config`` option:

.. code-block:: console

    $ ketos --config experiment.yml rotrain

Global options (``precision``, ``device``, ``workers``, ``threads``, ``seed``,
``deterministic``) are placed at the top level of the YAML file. Subcommand
options are nested under the subcommand name:

.. code-block:: yaml

    precision: 32-true
    device: auto
    num_workers: 32
    num_threads: 1
    rotrain:
      # training data manifests
      training_data:
        - train.lst
      evaluation_data:
        - val.lst
      # format of the training data
      format_type: xml
      # directory to save checkpoints in
      checkpoint_path: checkpoints
      # change to `coreml` to save best model with kraken < 7 compatibility
      weights_format: safetensors
      # reading order configuration
      level: baselines
      reading_order: line_implicit
      batch_size: 15000
      # optional: explicit class mapping (list of [class, label] pairs)
      # class_mapping:
      #   - [default, 1]
      #   - [heading, 2]
      #   - ['*', 0]
      # base configuration of training epochs and LR schedule
      quit: early
      epochs: 3000
      min_epochs: 500
      lag: 300
      lrate: 1e-3
      weight_decay: 0.01
      schedule: cosine
      cos_t_max: 100
      cos_min_lr: 1e-3

.. note::

    The YAML keys correspond to the Python parameter names of the click
    options, not the CLI flag names. For instance, the ``--output`` flag maps
    to the ``checkpoint_path`` key, ``--sched-patience`` maps to
    ``rop_patience``, and ``--cos-max`` maps to ``cos_t_max``.

Command Line Options
--------------------

**rotrain**

.. code-block::

    -B, --batch-size INTEGER        batch sample size
    --weights-format TEXT           Output weights format.
    -o, --output PATH               Output model file
    -i, --load PATH                 Load existing checkpoint or weights file
                                    to train from.
    --resume PATH                   Load a checkpoint to continue training
    -F, --freq FLOAT                Model saving and report generation
                                    frequency in epochs during training. If
                                    frequency is >1 it must be an integer,
                                    i.e. running validation every n-th epoch.
    -q, --quit [early, fixed, aneal]
                                    Stop condition for training.
    -N, --epochs INTEGER            Number of epochs to train for
    --min-epochs INTEGER            Minimal number of epochs to train for when
                                    using early stopping.
    --lag INTEGER                   Number of evaluations (--report frequency)
                                    to wait before stopping training without
                                    improvement
    --min-delta FLOAT               Minimum improvement between epochs to reset
                                    early stopping. By default it scales the
                                    delta by the best loss
    --optimizer [Adam, SGD, RMSprop, Lamb, AdamW]
                                    Select optimizer
    -r, --lrate FLOAT               Learning rate
    -m, --momentum FLOAT            Momentum
    -w, --weight-decay FLOAT        Weight decay
    --gradient-clip-val FLOAT       Gradient clip value
    --accumulate-grad-batches INTEGER
                                    Number of batches to accumulate gradient
                                    across.
    --warmup INTEGER                Number of samples to ramp up to `lrate`
                                    initial learning rate.
    --schedule [constant, 1cycle, exponential, step, reduceonplateau, cosine, cosine_warm_restarts]
                                    Set learning rate scheduler. For 1cycle,
                                    cycle length is determined by the
                                    `--step-size` option.
    -g, --gamma FLOAT               Decay factor for exponential, step, and
                                    reduceonplateau learning rate schedules
    -ss, --step-size INTEGER        Number of validation runs between learning
                                    rate decay for exponential and step LR
                                    schedules
    --sched-patience INTEGER        Minimal number of validation runs between
                                    LR reduction for reduceonplateau LR
                                    schedule.
    --cos-max INTEGER               Epoch of minimal learning rate for cosine
                                    LR scheduler.
    --cos-min-lr FLOAT              Minimal final learning rate for cosine LR
                                    scheduler.
    -p, --partition FLOAT           Ground truth data partition ratio between
                                    train/validation set
    -t, --training-data FILENAME   File(s) with additional paths to training
                                    data
    -e, --evaluation-data FILENAME
                                    File(s) with paths to evaluation data.
                                    Overrides the `-p` parameter
    -f, --format-type [xml, alto, page]
                                    Sets the training data format. In ALTO and
                                    PageXML mode all data is extracted from
                                    xml files containing both baselines and a
                                    link to source images.
    --logger [tensorboard]          Logger used by PyTorch Lightning to track
                                    metrics such as loss and accuracy.
    --log-dir PATH                  Path to directory where the logger will
                                    store the logs. If not set, a directory
                                    will be created in the current working
                                    directory.
    --level [baselines, regions]    Selects level to train reading order model
                                    on.
    --reading-order TEXT            Select reading order to train. Defaults to
                                    `line_implicit`/`region_implicit`
    --class-mapping-from-ckpt PATH  Extract class mapping from a segmentation
                                    checkpoint (.ckpt). Uses --level to select
                                    baseline or region mapping.

Combining with Other Models
---------------------------

A reading order model is trained independently and must be combined with a
segmentation model before it can be used during inference. The ``ketos
convert`` command can merge multiple related models into a single weights file:

.. code-block:: console

    $ ketos convert -o combined_model.safetensors segmentation_model.ckpt reading_order_model.ckpt

The command accepts both checkpoints (``.ckpt``) and weights files
(``.safetensors``, ``.mlmodel``) interchangeably. It can also be used to
convert between serialization formats:

.. code-block:: console

    $ ketos convert -o combined_model.mlmodel --weights-format coreml segmentation_model.safetensors reading_order_model.safetensors

The two models must have compatible baseline class mappings, i.e. the line
types known to the reading order model must match those in the segmentation
model.

.. deprecated::

    The ``roadd`` command previously served this purpose but has been
    superseded by ``ketos convert`` which handles combining arbitrary model
    types.
