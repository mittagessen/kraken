import tempfile
import unittest
from pathlib import Path

from lightning.pytorch.callbacks import OnExceptionCheckpoint

from kraken.train.utils import KrakenOnExceptionCheckpoint


class _DummyTrainer:
    def __init__(self):
        self.saved_path = None

    def save_checkpoint(self, path):
        self.saved_path = path


class TestTrainUtils(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_kraken_on_exception_checkpoint_saves_file(self):
        cb = KrakenOnExceptionCheckpoint(self.tmp_path, filename='checkpoint_abort')
        trainer = _DummyTrainer()

        cb.on_exception(trainer)

        self.assertEqual(trainer.saved_path, str(self.tmp_path / 'checkpoint_abort.ckpt'))

    def test_kraken_on_exception_checkpoint_not_lightning_fault_tolerance_callback(self):
        cb = KrakenOnExceptionCheckpoint(self.tmp_path)
        self.assertNotIsInstance(cb, OnExceptionCheckpoint)
