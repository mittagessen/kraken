.. _migration_6_0:

Upgrading from 6.0
====================

This page summarizes user-facing behavior changes between kraken ``6.0`` and
the ``7.x`` releases.

For detailed release notes describing all changes in depth, see the
:doc:`/changelog`.

Manifest Option Renames
-----------------------

Training and evaluation manifest options were renamed across ``ketos``
commands:

- ``--training-files`` -> ``--training-data``
- ``--evaluation-files`` -> ``--evaluation-data``

This affects ``ketos train``, ``ketos pretrain``, ``ketos segtrain``, ``ketos
segtest``, ``ketos rotrain``, and related contrib scripts.

If you use YAML experiment files, use ``training_data``/``evaluation_data`` as
the key name.

Experiment Files Migration
---------------------------------

Experiment files in YAML format are now the recommended way to express
non-trivial training configuration, especially class mappings and evaluation
behavior.

Pass an experiment file with ``--config`` before the command name:

.. code-block:: console

   $ ketos --config experiments.yml segtrain

Key migration points:

- Use python parameter names (e.g. ``checkpoint_path``, ``evaluation_data``),
  not legacy CLI flag spellings.
- Put global options (e.g. ``precision``, ``device``, ``num_workers``,
  ``num_threads``) at top-level.
- Put command-specific options under their command key
  (``train``, ``segtrain``, ``rotrain``, ``pretrain``, etc.).
- Replace legacy ``training_files``/``evaluation_files`` keys with
  ``training_data``/``evaluation_data``.
- Replace old segmentation class filtering/merging flags with
  ``line_class_mapping``/``region_class_mapping`` entries under ``segtrain``.

A minimal example for a segmentation training experiment file:

.. code-block:: yaml

   precision: 32-true
   device: auto
   num_workers: 32
   num_threads: 1
   segtrain:
     training_data:
       - train.lst
     evaluation_data:
       - val.lst
     checkpoint_path: checkpoints
     weights_format: safetensors
     line_class_mapping:
       - ['*', 3]
       - ['DefaultLine', 3]

You can keep configurations for multiple commands in a single file. kraken
will pick the parameter block matching the command you run:

.. code-block:: yaml

   precision: 32-true
   device: auto
   num_workers: 16
   num_threads: 1

   train:
     training_data:
       - rec_train.lst
     evaluation_data:
       - rec_val.lst
     format_type: xml
     checkpoint_path: rec_checkpoints
     weights_format: safetensors

   segtrain:
     training_data:
       - seg_train.lst
     evaluation_data:
       - seg_val.lst
     format_type: xml
     checkpoint_path: seg_checkpoints
     weights_format: safetensors
     line_class_mapping:
       - ['*', 3]
       - ['DefaultLine', 3]

Example usage with the same file:

.. code-block:: console

   $ ketos --config experiments.yml train
   $ ketos --config experiments.yml segtrain

Training Outputs, Checkpoints, and Weights
------------------------------------------

For ``ketos train``, ``ketos segtrain``, and ``ketos rotrain``, training now
produces Lightning checkpoints (``.ckpt``) as the primary artifact instead of
writing a CoreML weights file directly during training.

Checkpoint files include the full training state (model weights, optimizer
state, scheduler state, epoch/step counters, and serialized training config),
which enables exact continuation of interrupted runs.

There are now two distinct continuation modes:

- ``--resume`` restores and continues from the checkpoint's exact previous
  training state. The checkpoint state is authoritative, even if command-line
  flags or config files specify different values.
- ``--load`` keeps the previous "fine-tune/start new run" behavior. It loads
  weights only and starts a fresh training run using current CLI/config
  hyperparameters.

Because checkpoints include much more than deployable model weights (and may
execute arbitrary python code), distribute converted weights files, not raw
checkpoints. Conversion strips training-only state and produces a
weights artifact suitable for distribution.

At the end of training, kraken automatically converts the best checkpoint into
a weights file. You can also convert manually with ``ketos convert``.

The default weights format is now ``safetensors``. Compared to legacy
``coreml`` weights, ``safetensors`` is a more modern format that supports
serialization of arbitrary model types, while ``coreml`` is limited to the core
model methods implemented in kraken.

Use ``--weights-format coreml`` only when you explicitly need legacy
compatibility.

Segmentation Class Mapping Workflow
-----------------------------------

Segmentation class filtering/merging moved from various CLI flags to explicit
class mappings ``line_class_mapping`` / ``region_class_mapping`` which are no
longer accessible from the command line but exclusively from the new YAML
experiment configs

1. Equivalent of ``--merge-all-*`` (merge all lines/regions into one class)

   .. code-block:: yaml

      segtrain:
        line_class_mapping:
          - ['*', 3]
          - ['DefaultLine', 3]
        region_class_mapping:
          - ['*', 5]
          - ['Text_Region', 5]

   Legacy equivalent:

   - ``--merge-all-baselines DefaultLine``
   - ``--merge-all-regions Text_Region``

2. Explicit class merging (selected classes merged by name)

   This maps two specific baseline classes to the same label:

   .. code-block:: yaml

      segtrain:
        line_class_mapping:
          - ['DefaultLine', 3]
          - ['Running_Title', 3]
          - ['Marginal_Note', 4]

   Legacy equivalent:

   - Like ``--merge-baselines DefaultLine:Running_Title`` while keeping
     ``Marginal_Note`` separate.

3. Filtering (equivalent of ``--valid-baselines``)

   To keep only selected baseline classes, define only those classes and do
   not use ``'*'`` in ``line_class_mapping``:

   .. code-block:: yaml

      segtrain:
        line_class_mapping:
          - ['DefaultLine', 3]
          - ['Marginal_Note', 4]

   With this configuration, other baseline types are ignored during training.

``ketos segtest`` now has a ``--test-class-mapping-mode`` with ``full``,
``canonical``, and ``custom``.

Mode semantics:

- ``full``: Evaluate with the full many-to-one mapping used during training
  (aliases/merged classes preserved). This is the closest equivalent to legacy
  merged-class evaluation behavior. If that full mapping is unavailable (e.g.
  pre-7.0 CoreML weights files), kraken falls back to ``canonical``.
- ``canonical``: Evaluate with the model's one-to-one class mapping (one
  canonical class name per label index), i.e., the classes assigned to the
  final output of the model.
- ``custom``: Evaluate with explicitly provided
  ``line_class_mapping``/``region_class_mapping`` for the test dataset. Use
  this mode to enforce a compatibility mapping, e.g., when the class taxonomy
  of the test set is radically different from the segmentation model.

Reading Order Integration
-------------------------

- ``ketos roadd`` is deprecated in favor of ``ketos convert``.
- ``ketos rotrain`` can import class mappings from a segmentation checkpoint
  via ``--class-mapping-from-ckpt``.

Inference Changes
-----------------

- Global ``kraken --precision`` now controls inference precision.
- The old segment ``--autocast`` workflow was replaced by precision settings.
- Recognition is now explicitly batchable and parallelizable:

  - ``ocr -B/--batch-size`` controls how many line images are sent per forward
    pass.
  - ``ocr --num-line-workers`` controls parallel line extraction/preprocessing
    on CPU (``0`` keeps extraction in-process).

  This significantly improves throughput and allows effective GPU utilization
  during recognition.

- The older tag-based multi-model recognition workflow is deprecated, is
  incompatible with this batched pipeline, and is scheduled for removal in
  kraken 8.
