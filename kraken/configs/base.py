"""
kraken.configs.base
~~~~~~~~~~~~~~~~~~~

Base classes for configurations
"""
import logging

from collections import defaultdict

__all__ = ['Config',
           'RecognitionInferenceConfig',
           'SegmentationInferenceConfig',
           'TrainingConfig',
           'TrainingDataConfig',
           'RecognitionTrainingDataConfig',
           'SegmentationTrainingDataConfig']

logger = logging.getLogger(__name__)


class Config:
    """
    Generic configuration for all tasks supported by kraken.

    Arg:
        > General parameters

        precision (one of kraken.registry.PRECISIONS, defaults to 32-true):
            Sets the precision to run the model in.
        accelerator (str, defaults to 'auto'):
        device (str, default to 'auto'):
        batch_size (int, defaults to 1):
            Sets the batch size for inference.
        compile_config (dict[str, Any], optional, defaults to None):
            Decides how kraken will compile the forward pass of the model. If
            not given compilation will be disabled. To enable with default
            parameters set an empty dictionary.

        > Error handling parameters

        raise_on_error (bool, defaults to False):
            Causes an exception to be raised instead of internal handling when
            functional blocks that can fail for misshapen input crash.

        > Parallelism parameters

        num_threads (int, defaults to 1):
            Number of threads to use for intra-op parallelisation.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.precision = kwargs.pop('precision', '32-true')
        self.accelerator = kwargs.pop('accelerator', 'auto')
        self.device = kwargs.pop('device', 'auto')
        self.batch_size = kwargs.pop('batch_size', 1)
        self.compile_config = kwargs.pop('compile', None)
        self.raise_on_error = kwargs.pop('raise_on_error', False)
        self.num_threads = kwargs.pop('num_threads', 1)

class TrainingDataConfig:
    """
    Generic configuration for datasets for all tasks.

    Arg:
        > Universal parameters

        training_data (list of paths):
            A list of training data files.
        evaluation_data (list of paths, optional):
            A list of evaluation data files.
        test_data (list of paths, optional):
            A list of evaluation data files.
        partition (float, defaults to 0.9):
            Automatic partition of training data files if no evaluation data is
            defined.
        num_workers (int, defaults to 1):
            Number of dataloader workers.
        augment (bool, defaults to False):
            Switch to enable augmentation.
        batch_size (int, defaults to 1):
            Number of items to pack into a single sample.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.training_data = kwargs.pop('training_data', None)
        self.evaluation_data = kwargs.pop('evaluation_data', None)
        self.test_data = kwargs.pop('test_data', None)
        self.partition = kwargs.pop('partition', 0.9)
        self.num_workers = kwargs.pop('num_workers', 1)
        self.augment = kwargs.pop('augment', False)
        self.batch_size = kwargs.pop('batch_size', 1)


class SegmentationTrainingDataConfig(TrainingDataConfig):
    """
    Configuration for segmentation training data.

    Arg:
        > Universal parameters

        format_type (Literal['alto', 'page', 'xml'] defaults to 'xml'):
            Format of the training data.
        line_class_mapping (dict[str, int], defaults to defaultdict):
            Mapping between line class identifiers and integer labels.
        region_class_mapping (dict[str, int], defaults to None):
            Mapping between region class identifiers and integer labels. Same
            semantics as `line_class_mapping`.
        topline (Union[bool, None], defaults to False):
            Indicator of baseline position in dataset. False = baseline, True =
            topline, None = centerline.

    Notes:
        line_class_mapping and region_class_mapping share a label space, i.e.,
        duplicate label values will result in merging of classes. This feature
        can be used to merge multiple line or region types into a single class:

        ```
        line_class_mapping = OrderedDict([('line_type_1', 2), ('line_type_2', 2)])
        ```

        will merge 'line_type_1' and 'line_type_2' in the detector. The
        OrderedDict is necessary to make the reverse lookup of labels to
        identifiers deterministic (the first occurence of the type will be used
        as the identifier in the output).

        Regions or baselines can be suppressed to train only on one or the
        other. To do so, set line_class_mapping or region_class_mapping to an
        empty dictionary.

        The values for both mappings default to a modified defaultdict that
        assigns a unique label to each line/region class in the training data.
    """
    def __init__(self, **kwargs):
        counter = {'i': 2}

        def idx_factory():
            val = counter['i']
            counter['i'] += 1
            return val

        self.format_type = kwargs.pop('format_type', 'xml')
        self.line_class_mapping = kwargs.pop('line_class_mapping', defaultdict(idx_factory))
        self.region_class_mapping = kwargs.pop('region_class_mapping', defaultdict(idx_factory))
        self.topline = kwargs.pop('topline', False)
        super().__init__(**kwargs)


class RecognitionTrainingDataConfig(TrainingDataConfig):
    """
    Configuration for recognition training data.

    Arg:
        > Text recognition parameters

        binary_dataset_split (bool, defaults to False):
            Flag to retrieve fixed splits from binary datasets.
        format_type (Literal['alto', 'page', 'xml', 'binary'], defaults to 'xml'):
            Format of the training data.
        codec: (Union[dict[str, Sequence[int]], Sequence[str], str], defaults to None):
            Codec mapping one or more Unicode code points to one or more
            integers.
    """
    def __init__(self, **kwargs):
        self.binary_dataset_split = kwargs.pop('binary_dataset_split', False)
        self.format_type = kwargs.pop('format_type', 'xml')
        self.codec = kwargs.pop('codec', None)
        super().__init__(**kwargs)


class RecognitionInferenceConfig(Config):
    """
    Configuration for recognition inference.

    Arg:
        > Universal parameters

        temperature (float, defaults to 1.0):
            The value used to smoothen the softmax distribution of the output cofidences.
        return_line_image (bool defaults to False):
            Switch to add the extracted line image to the output records.
        return_logits (bool, default to False):
            Switch to add raw logits to output records.

        > Line recognizer parameters

        padding (int, defaults to 16):
            Extra blank padding to the left and right of text line
        num_line_workers (int, defaults to 2):
            Number of worker processes to extract lines from images.
        no_legacy_polygons (bool, defaults to False):
            disables the fast path for polygonal line extraction
        text_direction (Literal['horizontal-tb', 'vertical-lr', 'vertical-rl'], defaults to 'horizontal-tb'):
            Sets the orientation of bounding box segmentation data

        > CTC model parameters

        decoder (Callable, defaults to `kraken.lib.ctc_decoder.greedy_decoder`):
            CTC decoder
        bidi_reordering (bool, defaults to True):
            Reorder classes in the ocr_record according to the Unicode
            bidirectional algorithm for correct display. Set to L|R to override
            default text direction.
    """
    def __init__(self, **kwargs):
        import kraken.lib.ctc_decoder
        self.temperature = kwargs.pop('temperature', 1.0)
        self.return_logits = kwargs.pop('return_logits', False)
        self.return_line_image = kwargs.pop('return_line_image', False)
        self.padding = kwargs.pop('padding', 16)
        self.num_line_workers = kwargs.pop('num_line_workers', 2)
        self.no_legacy_polygons = kwargs.pop('no_legacy_polygons', False)
        self.decoder = kwargs.pop('decoder', kraken.lib.ctc_decoder.greedy_decoder)
        self.bidi_reordering = kwargs.pop('bidi_reordering', True)
        self.text_direction = kwargs.pop('text_direction', 'horizontal-tb')
        super().__init__(**kwargs)


class SegmentationInferenceConfig(Config):
    """
    Configuration for recognition inference.

    Arg:
        > Universal parameters

        text_direction (Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'], defaults to 'horizontal-lr'):
            Sets the principal text direction and line orientation of the input page.
        input_padding (int or tuple[int, int] or tuple[int, int, int, int], defaults to 0):
            Padding to add around the input image.

        > Parameters for the legacy bounding box segmenter

        legacy_scale: Optional[float] = None,
        legacy_maxcolseps: float = 2,
        legacy_black_colseps: bool = False,
        legacy_no_hlines: bool = True,

        > Parameters for bounding box segmenters (currently only the legacy segmenter)

        bbox_line_padding (Union[int, tuple[int, int]], defaults to 0):
            Padding to be added around left/right side of bounding boxes in bbox *line* segmenter.

        > Parameters for reading order determination

        bbox_ro_fn (Callable, defaults t kraken.lib.segmentation.reading_order):
            Function to compute the basic reading order of a set of lines in
            bbox format.
        baseline_ro_fn (Callable, defaults to kraken.lib.segmentation.polygonal_reading_order):
            Function to compute the basic reading order of a set of lines in
            baseline format.
    """
    def __init__(self, **kwargs):
        self.text_direction = kwargs.pop('text_direction', 'horizontal-lr')
        self.legacy_scale = kwargs.pop('legacy_scale', None)
        self.legacy_maxcolseps = kwargs.pop('legacy_maxcolseps', 2)
        self.legacy_black_colseps = kwargs.pop('legacy_black_colseps', False)
        self.legacy_no_hlines = kwargs.pop('legacy_no_hlines', True)
        self.bbox_line_padding = kwargs.pop('bbox_line_padding', 0)
        self.input_padding = kwargs.pop('input_padding', 0)
        from kraken.lib.segmentation import reading_order, polygonal_reading_order
        self.bbox_ro_fn = kwargs.pop('bbox_ro_fn', reading_order)
        self.baseline_ro_fn = kwargs.pop('baseline_ro_fn', polygonal_reading_order)
        super().__init__(**kwargs)


class TrainingConfig(Config):
    """
    Generic configuration for model training of all tasks supported by kraken.

    Arg:
        > General parameters

        epochs (int, optional with early stopping):
            Number of epochs to train for.
        completed_epochs (int):
            How many epochs of the schedule have already been completed.
        freq (float, defaults to 1.0):
            Evaluation and checkpoint saving frequency
        checkpoint_path (PathLike, defaults to `model`):
            Path prefix to save checkpoints during training.
        weights_format (Literal[safetensors, coreml], defaults to 'safetensors'):
            Weight format to convert checkpoint at end of training to.

        > Optimizer configuration

        optimizer (str, defaults to `AdamW`):
            Optimizer to use.
        lrate (float, defaults to 1e-5):
            Learning rate
        momentum (float, defaults to 0.9):
            Momentum parameter. Ignored if optimizer doesn't use it.
        weight_decay (float, defaults to 0.0):
            Weight decay. Ignored if optimizer doesn't support it.
        gradient_clip_val (float, defaults to 1.0):
            Threshold for gradient clipping.
        accumulate_grad_batches (int, defaults to 1):
            Number of batches to aggregate before backpropagation.

        > Learning rate scheduling parameters

        schedule (str, defaults to `constant`):
            Type of learning rate schedule.
        warmup (int, defaults to 0):
            Number of iterations to warmup learning rate.
        step_size (int, defaults to 10):
            Learning rate decay in stepped schedule.
        gamma (float, defaults to 0.1):
            Learning rate decay in exponential schedule.
        rop_factor (float, defaults to 0.1):
            Learning rate decay in reduce on plateau schedule.
        rop_patience (int, defaults to 5):
            Number of epochs to wait before reducing learning rate.
        cos_t_max (int, defaults to 10):
            Epoch at which cosine schedule reaches final learning rate.
        cos_min_lr (float, defaults to 1e-6):
            Final learning rate with cosine schedule.

        > Early stopping parameters

        quit (str, `early` or `fixed`, defaults to `fixed`):
        min_epochs (int, defaults to 0):
             Minimum number of epochs to train without considering validation
             scores.
        lag (int, defaults to 10):
            Number of epochs to wait for improvement in validation scores
            before aborting.
        min_delta (float, defaults to 0.0):
            Minimum delta of validation scores.
    """
    def __init__(self, **kwargs):
        self.epochs = kwargs.pop('epochs', -1)
        self.completed_epochs = kwargs.pop('completed_epochs', 0)
        self.freq = kwargs.pop('freq', 1.0)
        self.checkpoint_path = kwargs.pop('checkpoint_path', 'model')
        self.weights_format = kwargs.pop('weights_format', 'safetensors')
        self.optimizer = kwargs.pop('optimizer', 'AdamW')
        self.lrate = kwargs.pop('lrate', 1e-5)
        self.momentum = kwargs.pop('momentum', 0.9)
        self.weight_decay = kwargs.pop('weight_decay', 0.0)
        self.gradient_clip_val = kwargs.pop('gradient_clip_val', 1.0)
        self.accumulate_grad_batches = kwargs.pop('accumulate_grad_batches', 1)
        self.schedule = kwargs.pop('schedule', 'constant')
        self.warmup = kwargs.pop('warmup', 0)
        self.step_size = kwargs.pop('step_size', 10)
        self.gamma = kwargs.pop('gamma', 0.1)
        self.rop_factor = kwargs.pop('rop_factor', 0.1)
        self.rop_patience = kwargs.pop('rop_patience', 5)
        self.cos_t_max = kwargs.pop('cos_t_max', 10)
        self.cos_min_lr = kwargs.pop('cos_min_lr', 1e-6)
        self.quit = kwargs.pop('quit', 'fixed')
        self.min_epochs = kwargs.pop('min_epochs', 0)
        self.lag = kwargs.pop('lag', 10)
        self.min_delta = kwargs.pop('min_delta', 0.0)
        super().__init__(**kwargs)
