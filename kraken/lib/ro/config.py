from kraken.models import TrainingConfig


class ROTrainingConfig(TrainingConfig):
    """
    Base configuration for training a reading order model.
    """
    def __init__(self, **kwargs),
        kwargs.setdefault('lrate', 0.001)
        kwargs.setdefault('batch_size', 15000)
        kwargs.setdefault('min_epochs', 500)
        kwargs.setdefault('epochs', 3000)
        kwargs.setdefault('lag', 300)
        kwargs.setdefault('quit', 'early')
        kwargs.setdefault('weight_decay', 0.01)
        kwargs.setdefault('schedule', 'cosine')
        kwargs.setdefault('cos_t_max', 100)
        kwargs.setdefault('cos_min_lr', 0.001)
        super().__init__(**kwargs)
