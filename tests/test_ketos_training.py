# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from kraken.ketos import cli as ketos_cli
from kraken.train.blla import _fit_class_mapping

RESOURCES = Path(__file__).resolve().parent / 'resources'
AUX = {'_start_separator': 0, '_end_separator': 1}


class TestKetosTraining(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.runner = CliRunner()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_manifest(self, entries: list[Path], name: str) -> Path:
        manifest = self.tmp_path / name
        manifest.write_text('\n'.join(str(p) for p in entries) + '\n')
        return manifest

    def _run_ketos_train(self, args, out_dir, tasks, model_class):
        """
        Runs a ketos training command and verifies the converted best model:
        exactly one weights file, loadable, containing a model of the
        expected class and task type.
        """
        from kraken.models import load_models

        result = self.runner.invoke(ketos_cli, args)
        self.assertEqual(result.exit_code, 0, result.output)
        weights = list(Path(out_dir).glob('best_*.safetensors'))
        self.assertEqual(len(weights), 1,
                         f'Expected exactly one best model in {out_dir}, found {weights}')
        models = load_models(weights[0], tasks=tasks)
        self.assertGreater(len(models), 0)
        self.assertIsInstance(models[0], model_class)

    def test_ketos_train_recognition_smoke(self):
        from kraken.lib.vgsl import TorchVGSLModel

        xml = RESOURCES / '170025120000003,0074-lite.xml'
        manifest = self._write_manifest([xml], 'rec_train.lst')
        out_dir = self.tmp_path / 'rec_out'
        out_dir.mkdir()

        args = [
            '-d', 'cpu', '--threads', '1', '--workers', '0',
            'train',
            '-o', str(out_dir),
            '-f', 'xml',
            '--spec', '[1,12,0,1 Cr3,3,8 S1(1x0)1,3]',
            '--quit', 'fixed',
            '-N', '1',
            '--min-epochs', '1',
            '-F', '1',
            '-B', '1',
            '-t', str(manifest),
            '-e', str(manifest),
        ]
        self._run_ketos_train(args, out_dir, ['recognition'], TorchVGSLModel)

    def test_ketos_train_segmentation_smoke(self):
        from kraken.lib.vgsl import TorchVGSLModel

        xml = RESOURCES / '170025120000003,0074-lite.xml'
        manifest = self._write_manifest([xml], 'seg_train.lst')
        out_dir = self.tmp_path / 'seg_out'
        out_dir.mkdir()

        args = [
            '-d', 'cpu', '--threads', '1', '--workers', '0',
            'segtrain',
            '-o', str(out_dir),
            '-f', 'xml',
            '--spec', '[1,240,0,1 Cr3,3,8]',
            '--quit', 'fixed',
            '-N', '1',
            '--min-epochs', '1',
            '-F', '1',
            '-t', str(manifest),
            '-e', str(manifest),
        ]
        self._run_ketos_train(args, out_dir, ['segmentation'], TorchVGSLModel)

    def test_ketos_train_reading_order_smoke(self):
        from kraken.lib.ro import ROMLP

        xml = RESOURCES / 'page' / 'explicit_ro.xml'
        manifest = self._write_manifest([xml], 'ro_train.lst')
        out_dir = self.tmp_path / 'ro_out'
        out_dir.mkdir()

        args = [
            '-d', 'cpu', '--threads', '1', '--workers', '0',
            'rotrain',
            '-o', str(out_dir),
            '--quit', 'fixed',
            '-N', '1',
            '--min-epochs', '1',
            '-F', '1',
            '-B', '4',
            '-t', str(manifest),
            '-e', str(manifest),
        ]
        self._run_ketos_train(args, out_dir, ['reading_order'], ROMLP)


class TestFitClassMapping(unittest.TestCase):
    """
    Tests fitting of dataset class mappings to pretrained segmentation models.
    """
    def setUp(self):
        # baseline and region indices are interleaved on purpose
        self.net_mapping = {'aux': AUX,
                            'baselines': {'default': 2, 'heading': 4},
                            'regions': {'text': 3, 'margin': 5}}

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            _fit_class_mapping(self.net_mapping, self.net_mapping, 'add')

    def test_fail_identical(self):
        data_mapping = {'aux': AUX,
                        'baselines': {'heading': 7, 'default': 8},
                        'regions': {'margin': 9, 'text': 10}}
        mapping, output_size, del_indices = _fit_class_mapping(self.net_mapping, data_mapping, 'fail')
        self.assertEqual(mapping, self.net_mapping)
        self.assertIsNone(output_size)
        self.assertEqual(del_indices, [])

    def test_fail_aliases(self):
        """
        Aliases of a matching canonical class do not trigger a failure and are
        kept in the fitted mapping.
        """
        data_mapping = {'aux': AUX,
                        'baselines': {'default': 2, 'DefaultLine': 2, 'heading': 3},
                        'regions': {'text': 4, 'MainZone': 4, 'margin': 5}}
        mapping, output_size, _ = _fit_class_mapping(self.net_mapping, data_mapping, 'fail')
        self.assertIsNone(output_size)
        self.assertEqual(mapping['baselines'], {'default': 2, 'DefaultLine': 2, 'heading': 4})
        self.assertEqual(mapping['regions'], {'text': 3, 'MainZone': 3, 'margin': 5})

    def test_fail_mismatch(self):
        data_mapping = {'aux': AUX,
                        'baselines': {'default': 2, 'heading': 3},
                        'regions': {'text': 4, 'stamp': 5}}
        with self.assertRaises(ValueError):
            _fit_class_mapping(self.net_mapping, data_mapping, 'fail')

    def test_union(self):
        """
        Matching classes keep their outputs, new classes are appended with
        merged classes sharing a single output, and model classes not in the
        dataset are retained.
        """
        data_mapping = {'aux': AUX,
                        'baselines': {'default': 2},
                        'regions': {'stamp': 3, 'text': 4, 'seal': 3}}
        mapping, output_size, del_indices = _fit_class_mapping(self.net_mapping, data_mapping, 'union')
        self.assertEqual(output_size, 7)
        self.assertEqual(del_indices, [])
        self.assertEqual(mapping['aux'], AUX)
        self.assertEqual(mapping['baselines'], {'default': 2, 'heading': 4})
        self.assertEqual(mapping['regions'], {'stamp': 6, 'seal': 6, 'text': 3, 'margin': 5})

    def test_union_subset(self):
        """
        A dataset with a subset of the model classes does not resize the model.
        """
        data_mapping = {'aux': AUX,
                        'baselines': {'heading': 2},
                        'regions': {}}
        mapping, output_size, del_indices = _fit_class_mapping(self.net_mapping, data_mapping, 'union')
        self.assertIsNone(output_size)
        self.assertEqual(mapping, self.net_mapping)

    def test_new(self):
        """
        Outputs of model classes not in the dataset are removed, the remaining
        outputs are renumbered in output layer order, and new classes are
        appended.
        """
        data_mapping = {'aux': AUX,
                        'baselines': {'heading': 2},
                        'regions': {'text': 3, 'stamp': 4, 'seal': 4}}
        mapping, output_size, del_indices = _fit_class_mapping(self.net_mapping, data_mapping, 'new')
        self.assertEqual(del_indices, [2, 5])
        self.assertEqual(output_size, 5)
        self.assertEqual(mapping['aux'], AUX)
        self.assertEqual(mapping['baselines'], {'heading': 3})
        self.assertEqual(mapping['regions'], {'text': 2, 'stamp': 4, 'seal': 4})

    def test_new_alias_canonical_name(self):
        """
        Dataset classes are matched through the canonical (first) name of
        their group only.
        """
        data_mapping = {'aux': AUX,
                        'baselines': {'default': 2, 'heading': 3},
                        'regions': {'MainZone': 4, 'text': 4, 'margin': 5}}
        mapping, output_size, del_indices = _fit_class_mapping(self.net_mapping, data_mapping, 'new')
        self.assertEqual(del_indices, [3])
        self.assertEqual(output_size, 6)
        self.assertEqual(mapping['regions'], {'MainZone': 5, 'text': 5, 'margin': 4})

    def test_new_resize_keeps_weights(self):
        """
        Fitted class indices point to the original output weights after
        resizing the network.
        """
        import torch
        from kraken.models import create_model

        net = create_model('TorchVGSLModel',
                           vgsl='[1,48,0,1 Cr3,3,4 O2l6]',
                           model_type=['segmentation'],
                           class_mapping=self.net_mapping)
        weight = net.nn[-1].co.weight.detach().clone()
        data_mapping = {'aux': AUX,
                        'baselines': {'heading': 2},
                        'regions': {'text': 3, 'stamp': 4}}
        mapping, output_size, del_indices = _fit_class_mapping(self.net_mapping, data_mapping, 'new')
        net.resize_output(output_size, del_indices)
        new_weight = net.nn[-1].co.weight.detach()
        self.assertEqual(new_weight.shape[0], output_size)
        for section in ('aux', 'baselines', 'regions'):
            for name, idx in mapping[section].items():
                if name in self.net_mapping[section]:
                    self.assertTrue(torch.equal(new_weight[idx], weight[self.net_mapping[section][name]]), name)
