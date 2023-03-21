from kraken.lib.train import RecognitionModel
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.default_specs import RECOGNITION_HYPER_PARAMS
#from kraken.ketos.util import _expand_gt
from pathlib import Path
from unittest import TestCase
from unicodedata import normalize


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

    def test_merging_both(self):
        """ Asserts that both, which only takes into account new data, works as intended """
        model = self._get_model(resize="both")
        model.setup("fit")
        self.assertEqual(
            model.nn.codec.encode("1").shape, (0, ),
            "1 is unknown to the original model and the second dataset, produces nothing"
        )
        self.assertEqual(
            model.nn.codec.encode("9").shape, (1, ),
            "9 is known to the new dataset and should be encoded through `both`"
        )
        self.assertEqual(
            model.nn.codec.encode("x").shape, (0, ),
            "x is known to the loaded model and shouldn't be encoded through `both`"
        )

    def test_merging_add(self):
        """ Asserts that add, which only takes into account both the original codec and the new data,
            works as intended
        """
        model = self._get_model(resize="add")
        model.setup("fit")
        self.assertEqual(
            model.nn.codec.encode("1").shape, (0, ),
            "1 is unknown to the original model and the second dataset, produces nothing"
        )
        self.assertEqual(
            model.nn.codec.encode("9").shape, (1, ),
            "9 is known to the new dataset and should be encoded through `both`"
        )
        self.assertEqual(
            model.nn.codec.encode("x").shape, (1, ),
            "x is known to the loaded model and should be encoded through `both`"
        )
        
    def test_merging_add_with_nfd(self):
        """ Asserts that add, which only takes into account both the original codec and the new data,
            works as intended
        """
        model = self._get_model(resize="add", new_hyp_params={"normalization": "NFD"})
        model.setup("fit")
        self.assertEqual(
            model.nn.codec.encode("1").shape, (0, ),
            "1 is unknown to the original model and the second dataset, produces nothing"
        )
        self.assertEqual(
            model.nn.codec.encode("9").shape, (1, ),
            "9 is known to the new dataset and should be encoded through `add`"
        )
        self.assertEqual(
            model.nn.codec.encode("x").shape, (1, ),
            "x is known to the loaded model and should be encoded through `add`"
        )
        self.assertEqual(
            model.nn.codec.encode("ẽ").shape, (0, ),
            "ẽ (unnormalized) should not work in `add` mode because it should be split in two"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "ẽ")).shape, (2, ),
            "ẽ should work in `add` mode because it should be split in two and is in the training data"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "Ũ")).shape, (2, ),
            "Ũ should work in `add` mode because it should be split in two and is in the training data and the "
            "original model"
        )

    def test_merging_both_with_NFD(self):
        """ Asserts that both, which only takes into account new data, works as intended """
        model = self._get_model(resize="both")
        model.setup("fit")
        self.assertEqual(
            model.nn.codec.encode("1").shape, (0, ),
            "1 is unknown to the original model and the second dataset, produces nothing"
        )
        self.assertEqual(
            model.nn.codec.encode("9").shape, (1, ),
            "9 is known to the new dataset and should be encoded through `both`"
        )
        self.assertEqual(
            model.nn.codec.encode("x").shape, (0, ),
            "x is known to the loaded model and shouldn't be encoded through `both`"
        )
        self.assertEqual(
            model.nn.codec.encode("ẽ").shape, (0, ),
            "ẽ (unnormalized) should not work in `both` mode because it should be split in two"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "ẽ")).shape, (2, ),
            "ẽ should work in `both` mode because it should be split in two and is in the training data"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "Ũ")).shape, (1, ),
            "Ũ should not work in `add` mode because it should be split in two and U is only in the original model"
        )

    def test_merging_both_with_NFD_two_different_kind_of_dataset(self):
        """ Asserts that both, which only takes into account new data, works as intended, including with XML Dataset """
        model = self._get_model(resize="both", format_type="xml", training_data=xml_data)
        model.setup("fit")
        self.assertEqual(
            model.nn.codec.encode("1").shape, (0, ),
            "1 is unknown to the original model and the second dataset, produces nothing"
        )
        self.assertEqual(
            model.nn.codec.encode("9").shape, (1, ),
            "9 is known to the new dataset and should be encoded through `both`"
        )
        self.assertEqual(
            model.nn.codec.encode("x").shape, (0, ),
            "x is known to the loaded model and shouldn't be encoded through `both`"
        )
        self.assertEqual(
            model.nn.codec.encode("ẽ").shape, (0, ),
            "ẽ (unnormalized) should not work in `both` mode because it should be split in two"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "ẽ")).shape, (2, ),
            "ẽ should work in `both` mode because it should be split in two and is in the training data"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "Ũ")).shape, (1, ),
            "Ũ should not work in `both` mode because it should be split in two and U is only in the original model"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "ã")).shape, (2, ),
            "ã should work in `both` mode because it should be split in two"
        )

    def test_merging_add_with_NFD_two_different_kind_of_dataset(self):
        """ Asserts that add works as intended, including with XML Dataset """
        model = self._get_model(resize="add", format_type="xml", training_data=xml_data)
        model.setup("fit")
        self.assertEqual(
            model.nn.codec.encode("1").shape, (0, ),
            "1 is unknown to the original model and the second dataset, produces nothing"
        )
        self.assertEqual(
            model.nn.codec.encode("9").shape, (1, ),
            "9 is known to the new dataset and should be encoded through `add`"
        )
        self.assertEqual(
            model.nn.codec.encode("x").shape, (1, ),
            "x is known to the loaded model and should be encoded through `add`"
        )
        self.assertEqual(
            model.nn.codec.encode("ẽ").shape, (0, ),
            "ẽ (unnormalized) should not work in `add`+NFD mode because it should be split in two"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "ẽ")).shape, (2, ),
            "ẽ should work in `add` mode because it should be split in two and is in the training data"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "Ũ")).shape, (2, ),
            "Ũ should work in `add` mode because it should be split in two and U is in the original model"
        )
        self.assertEqual(
            model.nn.codec.encode(normalize("NFD", "ã")).shape, (2, ),
            "ã should work in `add` mode because it should be split in two"
        )

