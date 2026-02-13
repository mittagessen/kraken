.. _training_segmentation:

Segmentation Model Training
===========================

This guide covers the training and evaluation of segmentation models. kraken's
segmentation models detect baselines and regions on document images, producing
the layout information needed before text recognition can be run.

Changes Since 6.0
-------------------

.. important::
   For a complete migration checklist, see :doc:`migration_6_0`.

- There are new line segmentation metrics that are actually meaningful now.
  Higher baseline accuracy correspond to better line segmentation.
- Manifest options were renamed to ``--training-data`` and
  ``--evaluation-data``.
- Class filtering/merging is now expressed through
  ``line_class_mapping``/``region_class_mapping`` instead of the older
  ``--valid-*``/``--merge-*`` flag family.

Data Preparation
----------------

Segmentation models are trained on ALTO or PageXML files containing baseline
and region annotations.

For annotating training data, we recommend the `eScriptorium`_ platform which
is tightly integrated with kraken.

The recommended way to provide training and evaluation data to kraken is
through manifest files. A manifest is a simple text file containing a list of
paths to image files, one per line. These manifests are passed to the ``ketos``
training commands using the ``-t`` and ``-e`` options.

Each training sample is a document image together with its XML annotation file.
The annotation files must contain baseline polylines for text lines and,
optionally, region polygons.

.. _`eScriptorium`: https://www.escriptorium.fr/

Use Cases
---------

Training from Scratch
~~~~~~~~~~~~~~~~~~~~~

To train a new model from scratch, you need to provide at least a training
set. It is highly recommended to also provide a validation set to evaluate the
model's performance during training.

.. code-block:: console

    $ ketos segtrain -t training_manifest.txt -e validation_manifest.txt

This will train a model using the data specified in the manifest files.
Checkpoints will be saved periodically, and the best performing model on the
validation set will be saved as a weights file at the end.

If no explicit validation set is given, the training data is automatically
split according to the ``--partition`` ratio (default: 0.9, meaning 90%
training, 10% validation).

Fine-tuning
~~~~~~~~~~~

To fine-tune a pre-existing model, use the ``--load`` option to load its
weights. This is useful when you want to adapt a general-purpose model to a
specific type of material.

.. code-block:: console

    $ ketos segtrain --load existing_model.safetensors -t training_manifest.txt -e validation_manifest.txt

This will initialize the model with the weights from
``existing_model.safetensors`` and then start training on your new data. The
learning rate schedule is reset.

When fine-tuning, the ``--resize`` option controls how the output layer is
adapted when the training data contains classes not present in the original
model:

- ``fail``: Abort if training data and model classes do not match (default).
- ``union``: Add new classes to the existing model.
- ``new``: Replace the output layer to match exactly the training data classes.

Resuming Training
~~~~~~~~~~~~~~~~~

If a training run is interrupted (e.g. by a crash, timeout, or manual
cancellation), it can be continued from the last saved checkpoint using the
``--resume`` option:

.. code-block:: console

    $ ketos segtrain --resume checkpoint_0005.ckpt

Unlike ``--load``, which only restores the model weights and starts a fresh
training run, ``--resume`` restores the full training state: model weights,
optimizer state, learning rate scheduler position, and the current epoch count.
Training continues exactly where it left off. No training data or
hyperparameter options need to be provided on the command line as these are
restored from the checkpoint as well.

kraken also saves an emergency checkpoint (``checkpoint_abort.ckpt``) when
training is interrupted by an unhandled exception.

Class Mappings
--------------

By default, kraken assigns an automatic label to each unique line and region
type found in the training data. If you want to control how classes are mapped
to output labels, or merge multiple types into a single class, use the
``line_class_mapping`` and ``region_class_mapping`` options in an experiment
file.

Line and region class mappings share a label space that starts from ``3`` as
``0-2`` is reserved for internal use. Duplicate label values will result in
merging of classes. The special value ``*`` matches all class names and can be
used to merge everything into a single class:

.. code-block:: yaml

    segtrain:
      line_class_mapping:
        - ['*', 3]              # merge all line types into label 3
        - ['DefaultLine', 3]    # assign a name to the merged class
        - ['Marginal_Note', 4]  # keep this type separate
      region_class_mapping:
        - ['*', 5]
        - ['Text_Region', 5]
        - ['Foot_Notes', 6]

When using ``*``, make sure to also define a specific entry for the label to
control which class name is used in the output (otherwise a random one of the
merged classes will be picked).

To not train on certain regions and baselines in the source files it is
possible to suppress classes by not having a catch-all ``*`` class and simply
omitting the identifier from the mapping. It is also possible to suppress a
category entirely by setting a class mapping to an empty dictionary. For
example, to train a model that only detects baselines without any region
output, set ``region_class_mapping`` to ``{}``.

Baseline Position
-----------------

The ``--topline``/``--centerline``/``--baseline`` flags control the expected
position of the baseline annotation in the training data:

- ``--baseline`` (default): The annotation marks the baseline of the script,
  as is standard for Latin and most left-to-right scripts.
- ``--topline``: The annotation marks a hanging baseline, as is common with
  Hebrew, Bengali, Devanagari, and similar scripts.
- ``--centerline``: The annotation marks a central line.

The corresponding experiment file key is ``topline``: ``false`` for baseline,
``true`` for topline, ``null`` for centerline.

Experiment Files
----------------

Instead of supplying all options through the command line, it is possible to
put options into a YAML file and pass it to the ``ketos`` command using the
``--config`` option:

.. code-block:: console

    $ ketos --config experiment.yml segtrain

Global options (``precision``, ``device``, ``workers``, ``threads``, ``seed``,
``deterministic``) are placed at the top level of the YAML file. Subcommand
options are nested under the subcommand name:

.. code-block:: yaml

    precision: 32-true
    device: auto
    num_workers: 32
    num_threads: 1
    segtrain:
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
      # dataset metadata and transformations
      topline: false
      augment: true
      # class mappings (set to null to generate a mapping automatically)
      line_class_mapping:
        - ['*', 3]
        - ['DefaultLine', 3]
        - ['Marginal_Note', 4]
      region_class_mapping:
        - ['*', 5]
        - ['Text_Region', 5]
        - ['Foot_Notes', 6]
      # base configuration of training epochs and LR schedule
      quit: fixed
      epochs: 50
      lrate: 2e-4
      weight_decay: 1e-5
      schedule: cosine
      cos_t_max: 50
      cos_min_lr: 2e-5
      warmup: 200

.. note::

    The YAML keys correspond to the Python parameter names of the click
    options, not the CLI flag names. For instance, the ``--output`` flag maps
    to the ``checkpoint_path`` key, ``--sched-patience`` maps to
    ``rop_patience``, and ``--cos-max`` maps to ``cos_t_max``.

Command Line Options
--------------------

segtrain
~~~~~~~~

.. code-block::

    -o, --output PATH               Output checkpoint path
    --weights-format TEXT           Output weights format.
    -s, --spec TEXT                 VGSL spec of the baseline labeling network
    --line-width INTEGER            The height of each baseline in the target
                                    after scaling
    --pad INTEGER...                Padding (left/right, top/bottom) around
                                    the page image
    -i, --load PATH                 Load existing file to continue training
    --resume PATH                   Load a checkpoint to continue training
    -F, --freq FLOAT                Model saving and report generation
                                    frequency in epochs during training. If
                                    frequency is >1 it must be an integer,
                                    i.e. running validation every n-th epoch.
    -q, --quit [early, fixed, aneal]
                                    Stop condition for training. Set to `early`
                                    for early stopping or `fixed` for fixed
                                    number of epochs
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
    --warmup INTEGER                Number of steps to ramp up to `lrate`
                                    initial learning rate.
    --schedule [constant, 1cycle, exponential, step, reduceonplateau, cosine, cosine_warm_restarts]
                                    Set learning rate scheduler. For 1cycle,
                                    cycle length is determined by the
                                    `--step-size` option.
    -g, --gamma FLOAT               Decay factor for exponential, step, and
                                    reduceonplateau learning rate schedules
    -ss, --step-size FLOAT          Number of validation runs between learning
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
    --augment / --no-augment        Enable image augmentation
    --resize [add, union, both, new, fail]
                                    Output layer resizing option. If set to
                                    `union` new classes will be added, `new`
                                    will set the layer to match exactly the
                                    training data classes, `fail` will abort
                                    if training data and model classes do not
                                    match.
    -tl, --topline                  Switch for the baseline location in the
                                    scripts. Set to topline if the data is
                                    annotated with a hanging baseline, as is
                                    common with Hebrew, Bengali, Devanagari,
                                    etc. Set to  centerline for scripts
                                    annotated with a central line.
    -cl, --centerline
    -bl, --baseline
    --logger [tensorboard]          Logger used by PyTorch Lightning to track
                                    metrics such as loss and accuracy.
    --log-dir PATH                  Path to directory where the logger will
                                    store the logs. If not set, a directory
                                    will be created in the current working
                                    directory.

Evaluation
----------

To evaluate a trained segmentation model, use the ``ketos segtest`` command.
You need to provide the model and the test data.

.. code-block:: console

    $ ketos segtest -m model_best.safetensors -e test_manifest.txt

This computes:

- Per-class pixel accuracy and intersection-over-union (IoU)
- Baseline detection precision/recall/F1 (overall and per baseline class)

The baseline metrics have been improved significantly and increased scores
should now correspond to better real-world line segmentation accuracy.
segtest
~~~~~~~

.. code-block::

    -m, --model PATH                Model(s) to evaluate
    -e, --test-data FILENAME       File(s) with paths to evaluation data.
    -f, --format-type [xml, alto, page]
                                    Sets the training data format. In ALTO and
                                    PageXML mode all data is extracted from
                                    xml files containing both baselines and a
                                    link to source images.
    --bl-tol FLOAT                  Tolerance in pixels for baseline detection
                                    metrics.
    --test-class-mapping-mode [full, canonical, custom]
                                    Controls how the test-set class mapping is
                                    resolved. `full` uses the many-to-one
                                    mapping from a training checkpoint (with
                                    canonical fallback for plain weights),
                                    `canonical` uses the model's one-to-one
                                    mapping, and `custom` uses user-provided
                                    mappings.

Mode guidance:

- ``full``: Best when your training setup merged aliases/classes and your test
  set uses the same class taxonomy as the training set source files.
- ``canonical``: Best when your test set uses the same class taxonomy as the
  output classes of the model.
- ``custom``: Best when you need explicit remapping/filtering at test time
  using ``line_class_mapping``/``region_class_mapping`` from your config. This
  is usually necessary when the test set has its own classes that are
  completely different from those of the original training dataset which makes
  using ``full`` mode not possible. Instead you will have to provide a manual
  mapping of the class taxonomy in the source files to the output taxonomy of
  the segmentation model.
