.. _rotrain:

Reading Order Training
======================

Reading order models work slightly differently from segmentation and reading
order models. They are closely linked to the typology used in the dataset they
were trained on as they use type information on lines and regions to make
ordering decisions. As the same typology was probably used to train a specific
segmentation model, reading order models are trained separately but bundled
with their segmentation model in a subsequent step. The general sequence is
therefore:

.. code-block:: console

        $ ketos segtrain -o fr_manu_seg.mlmodel -f xml french/*.xml
        ...
        $ ketos rotrain -o fr_manu_ro.mlmodel -f xml french/*.xml
        ...
        $ ketos roadd -o fr_manu_seg_with_ro.mlmodel -i fr_manu_seg_best.mlmodel  -r fr_manu_ro_best.mlmodel

Only the `fr_manu_seg_with_ro.mlmodel` file will contain the trained reading
order model.  Segmentation models can exist with or without reading order
models. If one is added, the neural reading order will be computed *in
addition* to the one produced by the default heuristic during segmentation and
serialized in the final XML output (in ALTO/PAGE XML).

.. note::

        Reading order models work purely on the typology and geometric features
        of the lines and regions. They construct an approximate ordering matrix
        by feeding feature vectors of two lines (or regions) into the network
        to decide which of those two lines precedes the other.

        These feature vectors are quite simple; just the lines' types, and
        their start, center, and end points. Therefore they can *not* reliably
        learn any ordering relying on graphical features of the input page such
        as: line color, typeface, or writing system.

Reading order models are extremely simple and do not require a lot of memory or
computational power to train. In fact, the default parameters are extremely
conservative and it is recommended to increase the batch size for improved
training speed. Large batch size above 128k are easily possible with
sufficiently large training datasets:

.. code-block:: console

        $ ketos rotrain -o fr_manu_ro.mlmodel -B 128000 -f french/*.xml
        Training RO on following baselines types:
          DefaultLine	1
          DropCapitalLine	2
          HeadingLine	3
          InterlinearLine	4
        GPU available: False, used: False
        TPU available: False, using: 0 TPU cores
        IPU available: False, using: 0 IPUs
        HPU available: False, using: 0 HPUs
        ┏━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
        ┃   ┃ Name        ┃ Type              ┃ Params ┃
        ┡━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
        │ 0 │ criterion   │ BCEWithLogitsLoss │      0 │
        │ 1 │ ro_net      │ MLP               │  1.1 K │
        │ 2 │ ro_net.fc1  │ Linear            │  1.0 K │
        │ 3 │ ro_net.relu │ ReLU              │      0 │
        │ 4 │ ro_net.fc2  │ Linear            │     45 │
        └───┴─────────────┴───────────────────┴────────┘
        Trainable params: 1.1 K
        Non-trainable params: 0
        Total params: 1.1 K
        Total estimated model params size (MB): 0
        stage 0/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0/35 0:00:00 • -:--:-- 0.00it/s val_spearman: 0.912 val_loss: 0.701 early_stopping: 0/300 inf

During validation a metric called Spearman's footrule is computed. To calculate
Spearman's footrule, the ranks of the lines of text in the ground truth reading
order and the predicted reading order are compared. The footrule is then
calculated as the sum of the absolute differences between the ranks of pairs of
lines. The score increases by 1 for each line between the correct and predicted
positions of a line.

A lower footrule score indicates a better alignment between the two orders. A
score of 0 implies perfect alignment of line ranks.
