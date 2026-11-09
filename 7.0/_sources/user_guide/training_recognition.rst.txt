.. _training_recognition:

Recognition Model Training
==========================

This guide covers the training, fine-tuning, and evaluation of recognition
models.

Changes Since 6.0
-------------------

.. important::
   For a consolidated migration checklist, see :doc:`migration_6_0`.

- Training now creates checkpoint files that need to be converted into final
  weights format.
- Manifest options were renamed to ``--training-data`` and
  ``--evaluation-data``.
- ``--resume`` continues a run from a checkpoint with optimizer/scheduler
  state, while ``--load`` starts a fresh run from existing weights.

Data Preparation
----------------

Before training a recognition model, you need to prepare your training data.
kraken can use ALTO and PageXML files as a source of training data.
Alternatively, you can use a simple line-based format, where each line in a
text file corresponds to a line image with the same name.

For annotating training data, we recommend the `eScriptorium`_ platform which
is tightly integrated with kraken.

The recommended way to provide training and evaluation data to kraken is
through manifest files. A manifest is a simple text file containing a list of
paths to image files, one per line. These manifests are passed to the ``ketos``
training commands using the ``-t`` and ``-e`` options.

To speed up training, it is recommended to compile the training data into a
binary format using the ``ketos compile`` command.

.. code-block:: console

    $ ketos compile -f xml -o training_data.arrow *.xml

This will create a file named ``training_data.arrow`` that can be used for
training. The manifest would then contain the path to this ``.arrow`` file.
When using compiled datasets, set ``format_type`` to ``binary``.

.. _`eScriptorium`: https://www.escriptorium.fr/

Use Cases
---------

Training from Scratch
~~~~~~~~~~~~~~~~~~~~~

To train a new model from scratch, you need to provide at least a training
set. It is highly recommended to also provide a validation set to evaluate the
model's performance during training.

.. code-block:: console

    $ ketos train -t training_manifest.txt -e validation_manifest.txt

This will train a model using the data specified in the manifest files.
Checkpoints will be saved periodically, and the best performing model on the
validation set will be saved as a weights file at the end.

If no explicit validation set is given, the training data is automatically
split according to the ``--partition`` ratio (default: 0.9, meaning 90%
training, 10% validation). This automatic split is entirely random so will
change between training runs which in turn makes the reported validation
metrics not comparable.

Fine-tuning
~~~~~~~~~~~

To fine-tune a pre-existing model, use the ``--load`` option to load its
weights. This is useful when you want to adapt a general-purpose model to a
specific type of material.

.. code-block:: console

    $ ketos train --load existing_model.safetensors -t training_manifest.txt -e validation_manifest.txt

This will initialize the model with the weights from
``existing_model.safetensors`` and then start training on your new data. The
learning rate schedule is reset.

When fine-tuning, the ``--resize`` option controls how the output layer is
adapted when the training data contains characters not seen in the original
model:

- ``fail``: Abort if training data and model codec do not match (default).
- ``union``: Add new code points to the existing codec.
- ``new``: Replace the output layer to match exactly the training data codec.

Resuming Training
~~~~~~~~~~~~~~~~~

If a training run is interrupted (e.g. by a crash, timeout, or manual
cancellation), it can be continued from the last saved checkpoint using the
``--resume`` option:

.. code-block:: console

    $ ketos train --resume checkpoint_0005.ckpt

Unlike ``--load``, which only restores the model weights and starts a fresh
training run, ``--resume`` restores the full training state: model weights,
optimizer state, learning rate scheduler position, and the current epoch count.
Training continues exactly where it left off. No training data or
hyperparameter options need to be provided on the command line as these are
restored from the checkpoint as well.

kraken also saves an emergency checkpoint (``checkpoint_abort.ckpt``) when
training is interrupted by an unhandled exception.

Freezing the Backbone
~~~~~~~~~~~~~~~~~~~~~

When fine-tuning on a small dataset, it can be useful to freeze the backbone
(all layers except the output layer) for an initial number of samples to avoid
catastrophic forgetting. The ``--freeze-backbone`` option sets the number of
training samples to keep the backbone frozen for:

.. code-block:: console

    $ ketos train --load existing_model.safetensors --freeze-backbone 5000 --resize union -t training_manifest.txt

Early Stopping
--------------

By default, the ``train`` command uses early stopping (``--quit early``). This
means training will stop when the validation metric (character accuracy) does
not improve for ``--lag`` consecutive evaluation runs. The ``--min-epochs``
option ensures a minimum number of epochs are trained regardless of early
stopping. Set ``--quit fixed`` combined with ``--epochs`` to train for an exact
number of epochs.

Learning Rate Schedules
-----------------------

The ``--schedule`` option selects a learning rate schedule:

- ``constant``: Fixed learning rate throughout training.
- ``1cycle``: One-cycle policy; cycle length is determined by ``--epochs``.
- ``exponential``: Exponential decay by ``--gamma`` every ``--step-size``
  validation runs.
- ``step``: Step-wise decay by ``--gamma`` every ``--step-size`` validation
  runs.
- ``reduceonplateau``: Reduce by ``--gamma`` when validation loss stagnates
  for ``--sched-patience`` evaluation runs.
- ``cosine``: Cosine annealing from ``--lrate`` to ``--cos-min-lr`` over
  ``--cos-max`` epochs.

All schedules support ``--warmup`` which linearly ramps the learning rate from
zero to ``--lrate`` over the specified number of steps.

Experiment Files
----------------

Instead of supplying all options through the command line, it is possible to
put options into a YAML file and pass it to the ``ketos`` command using the
``--config`` option:

.. code-block:: console

    $ ketos --config experiment.yml train

Global options (``precision``, ``device``, ``workers``, ``threads``, ``seed``,
``deterministic``) are placed at the top level of the YAML file. Subcommand
options are nested under the subcommand name:

.. code-block:: yaml

    precision: 32-true
    device: auto
    num_workers: 32
    num_threads: 1
    train:
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
      # text transforms
      normalization: NFD
      normalize_whitespace: true
      # base configuration of training epochs and LR schedule
      quit: early
      epochs: 24
      lag: 10
      lrate: 1e-3
      schedule: constant
      warmup: 200
      augment: true
      # effective batch size params
      batch_size: 32
      accumulate_grad_batches: 1
      # codec definition
      # codec: null  # creates a codec automatically
      codec:
        'a': [1]
        'b': [22]
        'c': [23, 24]

.. note::

    The YAML keys correspond to the Python parameter names of the click
    options, not the CLI flag names. For instance, the ``--output`` flag maps
    to the ``checkpoint_path`` key, and ``--sched-patience`` maps to
    ``rop_patience``.

Model Conversion
----------------

At the end of a successful training run, kraken will automatically convert the
best performing checkpoint into a weights file in the format specified by
``--weights-format`` (default: ``safetensors``).

If you need to convert a checkpoint manually, you can use the ``ketos convert``
command.

Command Line Options
--------------------

train
~~~~~

.. code-block::

    -B, --batch-size INTEGER        batch sample size
    --pad INTEGER                   Left and right padding around lines
    -o, --output PATH               Directory to save checkpoints into.
    --weights-format TEXT           Output weights format.
    -s, --spec TEXT                 VGSL spec of the network to train. CTC
                                    layer will be added automatically.
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
                                    early stopping.
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
    --freeze-backbone INTEGER       Number of samples to keep the backbone
                                    (everything but last layer) frozen.
    --schedule [constant, 1cycle, exponential, step, reduceonplateau, cosine, cosine_warm_restarts]
                                    Set learning rate scheduler. For 1cycle,
                                    cycle length is determined by the `--epoch`
                                    option.
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
    -u, --normalization [NFD, NFKD, NFC, NFKC]
                                    Ground truth normalization
    -n, --normalize-whitespace / --no-normalize-whitespace
                                    Normalizes unicode whitespace
    -c, --codec UNPROCESSED         Load a codec JSON definition (invalid if
                                    loading existing model)
    --resize [add, union, both, new, fail]
                                    Codec/output layer resizing option. If set
                                    to `union` code points will be added, `new`
                                    will set the layer to match exactly the
                                    training data, `fail` will abort if
                                    training data and model codec do not
                                    match.
    --reorder / --no-reorder        Reordering of code points to display order
    --base-dir [L, R, auto]         Set base text direction.  This should be
                                    set to the direction used during the
                                    creation of the training data. If set to
                                    `auto` it will be overridden by any
                                    explicit value given in the input files.
    -t, --training-data FILENAME   File(s) with paths to training data
    -e, --evaluation-data FILENAME
                                    File(s) with paths to evaluation data.
                                    Overrides the `-p` parameter
    -f, --format-type [path, xml, alto, page, binary]
                                    Sets the training data format. In ALTO and
                                    PageXML mode all data is extracted from
                                    xml files containing both line definitions
                                    and a link to source images. In `path`
                                    mode arguments are image files sharing a
                                    prefix up to the last extension with
                                    `.gt.txt` text files containing the
                                    transcription. In binary mode files are
                                    datasets files containing pre-extracted
                                    text lines.
    --augment / --no-augment        Enable image augmentation
    --logger [tensorboard]          Logger used by PyTorch Lightning to track
                                    metrics such as loss and accuracy.
    --log-dir PATH                  Path to directory where the logger will
                                    store the logs. If not set, a directory
                                    will be created in the current working
                                    directory.
    --legacy-polygons               Use the legacy polygon extractor.

Evaluation
----------

To evaluate a trained model, use the ``ketos test`` command. You need to
provide the model (either a checkpoint or a weights file) and the test data.

.. code-block:: console

    $ ketos test -m model_best.safetensors -e test_manifest.txt

This will output a report with the character error rate, word error rate , and
per-script accuracy metrics, as well as character confusion statistics.

test
~~~~

.. code-block::

    -B, --batch-size INTEGER        Batch sample size
    -m, --model PATH                Model to evaluate
    -e, --test-data FILENAME       File(s) with paths to evaluation data.
    -f, --format-type [path, xml, alto, page, binary]
                                    Sets the training data format. In ALTO and
                                    PageXML mode all data is extracted from
                                    xml files containing both line definitions
                                    and a link to source images. In `path`
                                    mode arguments are image files sharing a
                                    prefix up to the last extension with
                                    `.gt.txt` text files containing the
                                    transcription. In binary mode files are
                                    datasets files containing pre-extracted
                                    text lines.
    --pad INTEGER                   Left and right padding around lines
    --reorder / --no-reorder        Reordering of code points to display order
    --base-dir [L, R, auto]         Set base text direction.  This should be
                                    set to the direction used during the
                                    creation of the training data. If set to
                                    `auto` it will be overridden by any
                                    explicit value given in the input files.
    -u, --normalization [NFD, NFKD, NFC, NFKC]
                                    Ground truth normalization
    -n, --normalize-whitespace / --no-normalize-whitespace
                                    Normalizes unicode whitespace
    --no-legacy-polygons            Force disable the legacy polygon extractor.

Pretraining
-----------

kraken supports unsupervised pretraining of recognition models using a
wav2vec2-style contrastive learning approach. Given a set of text line images
(without transcriptions), the model learns useful representations by predicting
masked portions of the input from context. This can improve performance when
labeled data is scarce.

To pretrain a model:

.. code-block:: console

    $ ketos pretrain -t line_images_manifest.txt

The pretrained model is saved as a checkpoint which can then be used as a
starting point for supervised fine-tuning with ``ketos train --load``.

The key pretraining-specific parameters control the masking strategy:

- ``--mask-width``: Width of masks at the scale of the subsampled tensor
  (default: 4). With 4x subsampling in the convolutional layers, a mask width
  of 3 results in an effective mask width of 12 pixels.
- ``--mask-probability``: Probability of each position being the start of a
  mask (default: 0.5).
- ``--num-negatives``: Number of negative samples for the contrastive loss
  (default: 100).
- ``--logit-temp``: Temperature for the contrastive loss logits (default: 0.1).

Pretraining experiment file example:

.. code-block:: yaml

    precision: 32-true
    device: auto
    num_workers: 32
    num_threads: 1
    pretrain:
      training_data:
        - pretrain_lines.lst
      evaluation_data:
        - pretrain_val.lst
      format_type: binary
      checkpoint_path: pretrain_checkpoints
      batch_size: 64
      epochs: 100
      lrate: 1e-6
      schedule: cosine
      cos_t_max: 100
      cos_min_lr: 1e-7
      warmup: 32000
      weight_decay: 0.01
      augment: true
      mask_width: 4
      mask_prob: 0.5
      num_negatives: 100
      logit_temp: 0.1

pretrain
~~~~~~~~

.. code-block::

    -B, --batch-size INTEGER        batch sample size
    --pad INTEGER                   Left and right padding around lines
    -o, --output PATH               Output checkpoint path
    -s, --spec TEXT                 VGSL spec of the network to train.
    -i, --load PATH                 Load existing file to continue training
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
                                    early stopping. Default is scales the
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
    --warmup FLOAT                  Number of samples to ramp up to `lrate`
                                    initial learning rate.
    --schedule [constant, 1cycle, exponential, step, reduceonplateau, cosine, cosine_warm_restarts]
                                    Set learning rate scheduler. For 1cycle,
                                    cycle length is determined by the `--epoch`
                                    option.
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
    -f, --format-type [path, xml, alto, page, binary]
                                    Sets the training data format. In ALTO and
                                    PageXML mode all data is extracted from
                                    xml files containing both line definitions
                                    and a link to source images. In `path`
                                    mode arguments are image files sharing a
                                    prefix up to the last extension with
                                    `.gt.txt` text files containing the
                                    transcription. In binary mode files are
                                    datasets files containing pre-extracted
                                    text lines.
    --augment / --no-augment        Enable image augmentation
    -mw, --mask-width INTEGER       Width of sampled masks at scale of the
                                    sampled tensor, e.g. 4X subsampling in
                                    convolutional layers with mask width 3
                                    results in an effective mask width of 12.
    -mp, --mask-probability FLOAT   Probability of a particular position being
                                    the start position of a mask.
    -nn, --num-negatives INTEGER    Number of negative samples for the
                                    contrastive loss.
    -lt, --logit-temp FLOAT         Multiplicative factor for the logits used
                                    in contrastive loss.
