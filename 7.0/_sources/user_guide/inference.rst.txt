.. _inference:

Inference with kraken
=====================

.. important::
   For migration details across all training and inference commands, see
   :doc:`migration_6_0`.

The ``kraken`` command-line interface (CLI) is the primary entry point for all
inference tasks. It employs a **chainable subcommand architecture**, allowing
you to define a complete processing pipeline (e.g., binarization -> segmentation
-> recognition) in a single invocation.

Synopsis
--------

.. code-block:: bash

   $ kraken [global-options] subcommand [subcommand-options] ...

A typical pipeline establishes a global I/O context and passes the data through
segmentation and recognition stages.

.. code-block:: bash

   # Complete pipeline: Input Image -> Segmentation -> Recognition -> Output Text
   $ kraken -i input.jpg output.txt segment -bl ocr -m model.safetensors

.. note::
   **Order Matters:** Arguments placed *before* a subcommand (like ``segment``
   or ``ocr``) are global options (e.g., input files, hardware selection).
   Arguments placed *after* a subcommand apply only to that specific step.

Models
------

Inference requires trained models for both segmentation and recognition.

* Stock models: kraken includes a default model for baseline segmentation (enabled via ``segment -bl``).
* Custom models: For text recognition (and specialized segmentation), you must provide a trained model file.

To learn how to search for and download models from the official repository, please refer to the :doc:`models` documentation.

.. tip::
   You can specify models using absolute paths, relative paths, or just the
   filename if the model is installed in the default global directory
   (``~/.local/share/kraken``).

Input and Output
----------------

kraken handles input files through either explicit pairings (processing specific files) or batch globbing (processing folders).

Global I/O Options
~~~~~~~~~~~~~~~~~~

These options must be specified before any subcommands.

``-i, --input <src> <dest>``
    Defines an explicit input/output pair. This option can be repeated multiple times in a single command.

    .. code-block:: bash

       $ kraken -i page1.tif page1.xml -i page2.tif page2.xml segment -bl

``-I, --batch-input <glob>``
    Accepts a glob expansion for batch processing. This **requires** the ``-o`` option to define how output filenames are generated.

``-o, --suffix <suffix>``
    Used with ``-I``. Defines the suffix appended to the input filename to create the output filename.

    .. code-block:: bash

       # Processes all pngs, saving results as 'filename.png.txt'
       $ kraken -I "data/*.png" -o .txt ocr -m model.mlmodel

``-f, --format-type <type>``
    Forces a specific input handler.
    
    * ``image``: Standard raster images (default).
    * ``pdf``: Extracts images from PDF files. Output filenames are generated using a format string provided with the ``-p, --pdf-format`` option.
    * ``xml``, ``alto``, ``page``: Parses existing segmentation from XML files. Useful for modular workflows (e.g., running recognition on already segmented ALTO files).

Output Serialization
~~~~~~~~~~~~~~~~~~~~

The format of the final output is controlled by global flags.

* ``-a``: **ALTO XML** (the preferred standard)
* ``-x``: **PageXML**
* ``-h``: **hOCR**
* ``-n``: **Native** (JSON for segmentation, plain text for recognition)
* ``-t <file>``: custom **Jinja2** template

The Processing Pipeline
-----------------------

1. Binarization (``binarize``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts input images to 1-bit monochrome.

.. tip::
   Modern recognition models usually accept grayscale or color inputs directly. This step is typically optional unless you are using the legacy bounding box segmenter or a model specifically trained on binary data.

* ``--threshold <float>``: Sets the binarization threshold.

Example
^^^^^^^

.. code-block:: bash

   # Binarize a page and write a 1bpp output image
   $ kraken -i page.tif page_bin.png binarize --threshold 0.5

2. Segmentation (``segment``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzes the layout of the page to extract text lines.

* ``-bl, --baseline``: Uses the neural baseline segmenter (**default/recommended**). It handles complex layouts and curved lines effectively.
* ``-i, --model``: Specify a custom segmentation model. If omitted, the internal stock model is used.
* ``-d, --text-direction``: Hints for the reading order heuristics. Options: ``horizontal-lr``, ``horizontal-rl``, ``vertical-lr``, ``vertical-rl``.
* ``-x, --boxes``: Uses the legacy bounding box segmenter. **requires binary input.**

Examples
^^^^^^^^

.. code-block:: bash

   # Neural baseline segmentation to ALTO XML output
   $ kraken -a -i page.tif page.seg.json -n segment -bl

.. code-block:: bash

   # Legacy bbox segmentation (requires binary input)
   $ kraken -i page_bin.png page_bbox.json -n segment -x

3. Recognition (``ocr``)
~~~~~~~~~~~~~~~~~~~~~~~~

Transcribes text from the segmented lines.

* ``-m, --model <file>``: Path to the recognition model. Supports both CoreML (``.mlmodel``) and safetensors (``.safetensors``) formats.
* ``--no-segmentation``: Skips segmentation and treats the input image(s) as single text lines. Useful for processing folders of pre-cropped line images.
* ``--reorder / --no-reorder``: Applies the Unicode BiDi algorithm to the output (enabled by default).
* ``--base-dir <dir>``: Forces a specific initial text direction when running the BiDi algorithm (e.g., ``L``, ``R``).
* ``--temperature <float>``: Adjusts the softmax temperature during decoding.
    * *Values < 1.0*: Sharpen the probability distribution.
    * *Values > 1.0*: Smoothes out the distribution.
* ``--no-legacy-polygons``: Disables the legacy fast-path polygon extractor.
* ``-B, --batch-size <int>``: Number of lines processed per recognition forward pass.
* ``--num-line-workers <int>``: Number of CPU workers for parallel line extraction/pre-processing.

.. note::
   The older tag-based multi-model recognition workflow is deprecated and
   scheduled for removal in kraken 8.

Examples
^^^^^^^^

.. code-block:: bash

   # Complete pipeline: segment + recognize into ALTO
   $ kraken -i page.tif page.xml -a segment -bl ocr -m model.safetensors

.. code-block:: bash

   # Recognize pre-segmented XML files
   $ kraken -f xml -I "data/*.xml" -o _ocr.xml ocr -m specialized.safetensors

.. code-block:: bash

   # Batched + parallelized recognition (GPU)
   $ kraken -i page.tif page.txt --device cuda:0 --precision bf16-mixed \
     segment -bl ocr -m model.safetensors -B 64 --num-line-workers 8

Performance Tuning
------------------

You can tune hardware utilization and floating-point precision via global options.

``--device <str>``
    Selects the compute device.
   
    * ``cpu``: Default.
    * ``cuda:N``: NVIDIA GPU (e.g., ``cuda:0``).
    * ``mps``: Apple Silicon (Metal Performance Shaders).

``--precision <str>``
    Sets the floating-point precision for inference.
    
    * ``32``: Standard FP32.
    * ``16``, ``16-mixed``: FP16 (half-precision).
    * ``bf16``, ``bf16-mixed``: BFloat16 (Recommended for NVIDIA Ampere+ GPUs to prevent numerical instability).

On the `ocr` subcommand only:

``-B, --batch-size <int>``
    Number of lines processed in parallel on the GPU. Higher values increase throughput but require more VRAM.

``--num-line-workers <int>``
    Number of CPU processes used for pre-processing image batches before they
    are sent to the GPU. ``0`` runs extraction in the main process.
