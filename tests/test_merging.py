#from kraken.ketos.util import _expand_gt
from pathlib import Path
from unicodedata import normalize
from unittest import TestCase

from kraken.lib.default_specs import RECOGNITION_HYPER_PARAMS
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.train import RecognitionModel

_here = Path(__file__).parent
base_model = _here.joinpath(Path("./resources/merge_tests/merge_codec_nfd.mlmodel"))
training_data = [str(_here.joinpath(Path("./resources/merge_tests/merger.arrow")))]
xml_data = [str(_here.joinpath(Path("./resources/merge_tests/0014.xml")))]


class TestMerging(TestCase):
    """
    Testing merging and fine-tuning models with previous codecs


    The base model is trained with 0006.gt.txt and 0007.gt.txt (base.arrow)
    The merger.arrow is composed of 0008.gt.txt and 0021.gt.txt.

    This expands on test_train to test unicode normalization on top of `resize`
    """
    def _get_model(self, new_hyp_params=None, **generic_params):
        hyp_params = RECOGNITION_HYPER_PARAMS.copy()
        if new_hyp_params:
            hyp_params.update(new_hyp_params)

        params = dict(
            hyper_params=hyp_params,
            model=base_model,
            training_data=training_data,
            num_workers=1,
            format_type="binary",
            resize="fail"
        )
        if generic_params:
            params.update(**generic_params)
        return RecognitionModel(**params)

    def test_no_resize_fails(self):
        """ Asserts that not setting up resize fails to fit """
        model = self._get_model()
        with self.assertRaises(KrakenInputException) as E:
            model.setup("fit")

    def test_merging_new(self):
        """ Asserts that new, which only takes into account new data, works as intended """
        model = self._get_model(resize="new")
        model.setup("fit")
        self.assertEqual(
            model.nn.codec.encode("1").shape, (0, ),
            "1 is unknown to the original model and the second dataset, produces nothing"
        )
        self.assertEqual(
            model.nn.codec.encode("9").shape, (1, ),
            "9 is known to the new dataset and should be encoded through `new`"
        )
        self.assertEqual(
            model.nn.codec.encode("x").shape, (0, ),
            "x is known to the loaded model and shouldn't be encoded through `new`"
        )

    def test_merging_union(self):
        """ Asserts that union, which only takes into account new the original codec and the new data,
            works as intended
        """
        model = self._get_model(resize="union")
        model.setup("fit")
        self.assertEqual(
            model.nn.codec.encode("1").shape, (0, ),
            "1 is unknown to the original model and the second dataset, produces nothing"
        )
        self.assertEqual(
            model.nn.codec.encode("9").shape, (1, ),
            "9 is known to the new dataset and should be encoded through `new`"
        )
        self.assertEqual(
            model.nn.codec.encode("x").shape, (1, ),
            "x is known to the loaded model and should be encoded through `new`"
        )

    def test_merging_union_with_nfd(self):
        """ Asserts that union, which only takes into account new the original codec and the new data,
            works as intended
        """
        model = self._get_model(resize="union", new_hyp_params={"normalization": "NFD"})
        model.setup("fit")
        self.assertEqual(
            model.nn.codec.encode("1").shape, (0, ),
            "1 is unknown to the original model and the second dataset, produces nothing"
        )
        self.assertEqual(
            model.nn.codec.encode("9").shape, (1, ),
            "9 is known to the new dataset and should be encoded through `union`"
        )
        self.assertEqual(
            model.nn.codec.encode("x").shape, (1, ),
            "x is known to the loaded model and should be encoded through `union`"
        )
        self.assertEqual(
            model.nn.codec.encode("ẽ").shape, (0, ),
            "ẽ (unnormalized) should not work in `union` mode because it should be split in two"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "ẽ")).shape, (2, ),
            "ẽ should work in `union` mode because it should be split in two and is in the training data"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "Ũ")).shape, (2, ),
            "Ũ should work in `union` mode because it should be split in two and is in the training data and the "
            "original model"
        )

    def test_merging_new_with_NFD(self):
        """ Asserts that new, which only takes into account new data, works as intended """
        model = self._get_model(resize="new", new_hyp_params={"normalization": "NFD"})
        model.setup("fit")
        self.assertEqual(
            model.nn.codec.encode("1").shape, (0, ),
            "1 is unknown to the original model and the second dataset, produces nothing"
        )
        self.assertEqual(
            model.nn.codec.encode("9").shape, (1, ),
            "9 is known to the new dataset and should be encoded through `new`"
        )
        self.assertEqual(
            model.nn.codec.encode("x").shape, (0, ),
            "x is only known to the loaded model and shouldn't be encoded through `new`"
        )
        self.assertEqual(
            model.nn.codec.encode("ẽ").shape, (0, ),
            "ẽ (unnormalized) should not work in `new` mode because it should be split in two"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "ẽ")).shape, (2, ),
            "ẽ should work in `new` mode because it should be split in two and is in the training data"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "Ũ")).shape, (1, ),
            "Ũ should not work in `union` mode because it should be split in two and U is only in the original model"
        )

    def test_merging_new_with_NFD_two_different_kind_of_dataset(self):
        """ Asserts that new, which only takes into account new data, works as intended, including with XML Dataset """
        model = self._get_model(resize="new", format_type="xml",
                                training_data=xml_data, new_hyp_params={"normalization": "NFD"})
        model.setup("fit")
        self.assertEqual(
            model.nn.codec.encode("1").shape, (0, ),
            "1 is unknown to the original model and the second dataset, produces nothing"
        )
        self.assertEqual(
            model.nn.codec.encode("9").shape, (1, ),
            "9 is known to the new dataset and should be encoded through `new`"
        )
        self.assertEqual(
            model.nn.codec.encode("x").shape, (0, ),
            "x is known to the loaded model and shouldn't be encoded through `new`"
        )
        self.assertEqual(
            model.nn.codec.encode("ẽ").shape, (0, ),
            "ẽ (unnormalized) should not work in `new` mode because it should be split in two"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "ẽ")).shape, (2, ),
            "ẽ should work in `new` mode because it should be split in two and is in the training data"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "Ũ")).shape, (1, ),
            "Ũ should not work in `new` mode because it should be split in two and U is only in the original model"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "ã")).shape, (2, ),
            "ã should work in `new` mode because it should be split in two"
        )

    def test_merging_union_with_NFD_two_different_kind_of_dataset(self):
        """ Asserts that union works as intended, including with XML Dataset """
        model = self._get_model(resize="union", format_type="xml",
                                training_data=xml_data, new_hyp_params={"normalization": "NFD"})
        model.setup("fit")
        self.assertEqual(
            model.nn.codec.encode("1").shape, (0, ),
            "1 is unknown to the original model and the second dataset, produces nothing"
        )
        self.assertEqual(
            model.nn.codec.encode("9").shape, (1, ),
            "9 is known to the new dataset and should be encoded through `union`"
        )
        self.assertEqual(
            model.nn.codec.encode("x").shape, (1, ),
            "x is known to the loaded model and should be encoded through `union`"
        )
        self.assertEqual(
            model.nn.codec.encode("ẽ").shape, (0, ),
            "ẽ (unnormalized) should not work in `union`+NFD mode because it should be split in two"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "ẽ")).shape, (2, ),
            "ẽ should work in `union` mode because it should be split in two and is in the training data"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "Ũ")).shape, (2, ),
            "Ũ should work in `union` mode because it should be split in two and U is in the original model"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "ã")).shape, (2, ),
            "ã should work in `union` mode because it should be split in two"
        )

