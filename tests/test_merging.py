from pathlib import Path
from unicodedata import normalize
from unittest import TestCase

import lightning as L

from kraken.train import VGSLRecognitionModel, VGSLRecognitionDataModule
from kraken.configs import VGSLRecognitionTrainingConfig, VGSLRecognitionTrainingDataConfig

_here = Path(__file__).parent
base_model = _here / "resources" / "merge_tests" / "merge_codec_nfd.mlmodel"
training_data = [str(_here / "resources" / "merge_tests" / "merger.arrow")]
xml_data = [str(_here / "resources" / "merge_tests" / "0014.xml")]


class TestMerging(TestCase):
    """
    Testing merging and fine-tuning models with previous codecs


    The base model is trained with 0006.gt.txt and 0007.gt.txt (base.arrow)
    The merger.arrow is composed of 0008.gt.txt and 0021.gt.txt.

    This expands on test_train to test unicode normalization on top of `resize`
    """
    def _get_model_and_data(self, resize='fail', normalization=None,
                            format_type='binary', data=None):
        data_config = VGSLRecognitionTrainingDataConfig(
            training_data=data if data is not None else training_data,
            format_type=format_type,
            num_workers=1,
            normalization=normalization,
        )
        config = VGSLRecognitionTrainingConfig(
            resize=resize,
            quit='fixed',
            epochs=1,
        )
        model = VGSLRecognitionModel.load_from_weights(base_model, config)
        data_module = VGSLRecognitionDataModule(data_config)
        return model, data_module

    def _run_setup(self, model, data_module):
        """Run a minimal trainer.fit() to trigger model and data module setup."""
        trainer = L.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator='cpu',
        )
        trainer.fit(model, data_module)

    def test_no_resize_fails(self):
        """ Asserts that not setting up resize fails to fit """
        model, data_module = self._get_model_and_data(resize='fail')
        with self.assertRaises(ValueError):
            self._run_setup(model, data_module)

    def test_merging(self):
        """
        Asserts codec merging semantics of the `new` and `union` resize modes,
        with and without NFD normalization, over binary and XML datasets. Each
        case fine-tunes a fresh copy of the base model and probes the merged
        codec.
        """
        cases = [
            ('new', dict(resize='new'),
             [("1", 0, "1 is unknown to the original model and the second dataset, produces nothing"),
              ("9", 1, "9 is known to the new dataset and should be encoded through `new`"),
              ("x", 0, "x is known to the loaded model and shouldn't be encoded through `new`")]),
            ('union', dict(resize='union'),
             [("1", 0, "1 is unknown to the original model and the second dataset, produces nothing"),
              ("9", 1, "9 is known to the new dataset and should be encoded through `union`"),
              ("x", 1, "x is known to the loaded model and should be encoded through `union`")]),
            ('union_nfd', dict(resize='union', normalization='NFD'),
             [("1", 0, "1 is unknown to the original model and the second dataset, produces nothing"),
              ("9", 1, "9 is known to the new dataset and should be encoded through `union`"),
              ("x", 1, "x is known to the loaded model and should be encoded through `union`"),
              ("ẽ", 0, "ẽ (unnormalized) should not work in `union` mode because it should be split in two"),
              (normalize("NFD", "ẽ"), 2, "ẽ should work in `union` mode because it should be split in two and is in the training data"),
              (normalize("NFD", "Ũ"), 2, "Ũ should work in `union` mode because it should be split in two and is in the training data and the original model")]),
            ('new_nfd', dict(resize='new', normalization='NFD'),
             [("1", 0, "1 is unknown to the original model and the second dataset, produces nothing"),
              ("9", 1, "9 is known to the new dataset and should be encoded through `new`"),
              ("x", 0, "x is only known to the loaded model and shouldn't be encoded through `new`"),
              ("ẽ", 0, "ẽ (unnormalized) should not work in `new` mode because it should be split in two"),
              (normalize("NFD", "ẽ"), 2, "ẽ should work in `new` mode because it should be split in two and is in the training data"),
              (normalize("NFD", "Ũ"), 1, "Ũ should not work in `new` mode because it should be split in two and U is only in the original model")]),
            ('new_nfd_xml', dict(resize='new', normalization='NFD', format_type='xml', data=xml_data),
             [("1", 0, "1 is unknown to the original model and the second dataset, produces nothing"),
              ("9", 1, "9 is known to the new dataset and should be encoded through `new`"),
              ("x", 0, "x is known to the loaded model and shouldn't be encoded through `new`"),
              ("ẽ", 0, "ẽ (unnormalized) should not work in `new` mode because it should be split in two"),
              (normalize("NFD", "ẽ"), 2, "ẽ should work in `new` mode because it should be split in two and is in the training data"),
              (normalize("NFD", "Ũ"), 1, "Ũ should not work in `new` mode because it should be split in two and U is only in the original model"),
              (normalize("NFD", "ã"), 2, "ã should work in `new` mode because it should be split in two")]),
            ('union_nfd_xml', dict(resize='union', normalization='NFD', format_type='xml', data=xml_data),
             [("1", 0, "1 is unknown to the original model and the second dataset, produces nothing"),
              ("9", 1, "9 is known to the new dataset and should be encoded through `union`"),
              ("x", 1, "x is known to the loaded model and should be encoded through `union`"),
              ("ẽ", 0, "ẽ (unnormalized) should not work in `union`+NFD mode because it should be split in two"),
              (normalize("NFD", "ẽ"), 2, "ẽ should work in `union` mode because it should be split in two and is in the training data"),
              (normalize("NFD", "Ũ"), 2, "Ũ should work in `union` mode because it should be split in two and U is in the original model"),
              (normalize("NFD", "ã"), 2, "ã should work in `union` mode because it should be split in two")]),
        ]
        for name, kwargs, probes in cases:
            with self.subTest(name):
                model, data_module = self._get_model_and_data(**kwargs)
                self._run_setup(model, data_module)
                for probe, expected_len, msg in probes:
                    self.assertEqual(model.net.codec.encode(probe).shape,
                                     (expected_len, ), msg)
