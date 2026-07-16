# -*- coding: utf-8 -*-
"""
Tests for the multi-architecture dispatch of `ketos train`/`ketos test`:
parameter-source filtering, per-architecture option validation, and
architecture auto-detection from artifacts.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import click
import pytest
import torch
from click.testing import CliRunner

from kraken.ketos import cli as ketos_cli
from kraken.ketos.util import (_arch_names, _load_config, _resolve_arch,
                               _user_supplied_params)

RESOURCES = Path(__file__).resolve().parent / 'resources'


class TestUserSuppliedParams(unittest.TestCase):
    """
    Tests the parameter-source filter that keeps per-architecture config
    defaults intact: seeded default_map values must be excluded, command-line
    and user-YAML values included.
    """
    def setUp(self):
        self.runner = CliRunner()
        self.captured = {}
        captured = self.captured

        @click.group(context_settings=dict(default_map={'cmd': {'lrate': 0.001,
                                                                'schedule': 'constant'}}))
        @click.option('--config', type=click.File(mode='r', lazy=True),
                      callback=_load_config, is_eager=True, expose_value=False,
                      required=False)
        def grp():
            pass

        @grp.command('cmd')
        @click.pass_context
        @click.option('--lrate', type=float)
        @click.option('--schedule')
        @click.option('--variant', default='small')
        def cmd(ctx, **kwargs):
            captured.update(_user_supplied_params(ctx))

        self.grp = grp

    def test_seeded_defaults_are_filtered(self):
        result = self.runner.invoke(self.grp, ['cmd'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.captured, {})

    def test_cli_values_are_explicit(self):
        result = self.runner.invoke(self.grp, ['cmd', '--lrate', '0.0005'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.captured, {'lrate': 0.0005})

    def test_yaml_values_are_explicit(self):
        with self.runner.isolated_filesystem():
            with open('conf.yml', 'w') as fp:
                fp.write('cmd:\n  schedule: cosine\n')
            result = self.runner.invoke(self.grp, ['--config', 'conf.yml', 'cmd'])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(self.captured, {'schedule': 'cosine'})

    def test_cli_overrides_yaml(self):
        with self.runner.isolated_filesystem():
            with open('conf.yml', 'w') as fp:
                fp.write('cmd:\n  schedule: cosine\n  lrate: 0.1\n')
            result = self.runner.invoke(self.grp, ['--config', 'conf.yml', 'cmd',
                                                   '--schedule', 'exponential'])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(self.captured, {'schedule': 'exponential', 'lrate': 0.1})


class TestArchRegistry(unittest.TestCase):
    """
    Tests the per-task architecture registry helpers. Entry-point membership
    itself is asserted in test_plugins.py.
    """
    def test_resolve_arch(self):
        from kraken.train import PPOCRv6RecognitionModel, VGSLRecognitionModel
        self.assertIs(_resolve_arch('recognition', 'vgsl'), VGSLRecognitionModel)
        self.assertIs(_resolve_arch('recognition', 'ppocrv6'), PPOCRv6RecognitionModel)

    def test_resolve_arch_segmentation(self):
        """The dispatch is task-generic: segmentation families resolve too."""
        from kraken.train import BLLASegmentationModel
        self.assertIs(_resolve_arch('segmentation', 'blla'), BLLASegmentationModel)

    def test_arch_names_are_task_scoped(self):
        self.assertEqual(set(_arch_names('recognition')), {'vgsl', 'ppocrv6'})
        self.assertEqual(set(_arch_names('segmentation')), {'blla'})
        # unknown task -> empty registry, no error
        self.assertEqual(_arch_names('nonexistent'), [])

    def test_resolve_unknown_arch(self):
        with self.assertRaises(click.BadParameter):
            _resolve_arch('recognition', 'nonexistent')

    def test_resolve_arch_wrong_task(self):
        """A recognition family is not selectable under the segmentation task."""
        with self.assertRaises(click.BadParameter):
            _resolve_arch('segmentation', 'vgsl')

    def test_trainer_classvars(self):
        from kraken.train import (BLLASegmentationModel, PPOCRv6RecognitionModel,
                                  VGSLRecognitionModel)
        for cls, task in ((VGSLRecognitionModel, 'recognition'),
                          (PPOCRv6RecognitionModel, 'recognition'),
                          (BLLASegmentationModel, 'segmentation')):
            self.assertEqual(cls._task, task)
            self.assertIsNotNone(cls._arch)
            self.assertIsNotNone(cls._data_config_class)
            self.assertIsNotNone(cls._data_module_class)


class TestArchDetection(unittest.TestCase):
    """
    Tests architecture auto-detection from checkpoints and weights files.
    """
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _fake_ckpt(self, config):
        path = str(self.tmp_path / 'fake.ckpt')
        torch.save({'_module_config': config}, path)
        return path

    def test_checkpoint_detection_ppocr(self):
        from kraken.configs import PPOCRv6RecognitionTrainingConfig
        from kraken.models.convert import find_checkpoint_module
        from kraken.train import PPOCRv6RecognitionModel
        path = self._fake_ckpt(PPOCRv6RecognitionTrainingConfig())
        self.assertIs(find_checkpoint_module(path), PPOCRv6RecognitionModel)

    def test_checkpoint_detection_vgsl(self):
        from kraken.configs import VGSLRecognitionTrainingConfig
        from kraken.models.convert import find_checkpoint_module
        from kraken.train import VGSLRecognitionModel
        path = self._fake_ckpt(VGSLRecognitionTrainingConfig())
        self.assertIs(find_checkpoint_module(path), VGSLRecognitionModel)

    def test_checkpoint_detection_pretrain_decoy(self):
        """
        VGSLPreTrainingConfig subclasses VGSLRecognitionTrainingConfig; the
        exact-type match must route it to the pretrain module, not vgsl.
        """
        from kraken.configs import VGSLPreTrainingConfig
        from kraken.lib.pretrain import RecognitionPretrainModel
        from kraken.models.convert import find_checkpoint_module
        path = self._fake_ckpt(VGSLPreTrainingConfig())
        self.assertIs(find_checkpoint_module(path), RecognitionPretrainModel)

    def test_checkpoint_without_config(self):
        from kraken.models.convert import find_checkpoint_module
        path = str(self.tmp_path / 'noconf.ckpt')
        torch.save({'state_dict': {}}, path)
        with self.assertRaises(ValueError):
            find_checkpoint_module(path)

    def test_weights_detection_ppocr(self):
        from kraken.lib.ppocr import PPOCRv6Model
        from kraken.models import write_safetensors
        from kraken.models.convert import find_weights_archs
        model = PPOCRv6Model(variant='tiny', num_classes=4, codec={'a': [1]})
        path = str(self.tmp_path / 'model.safetensors')
        write_safetensors([model], path)
        self.assertEqual(find_weights_archs(path), {'ppocrv6'})

    def test_weights_detection_vgsl(self):
        from kraken.models.convert import find_weights_archs
        self.assertEqual(find_weights_archs(RESOURCES / 'model_small.safetensors',
                                            task='recognition'),
                         {'vgsl'})

    def test_weights_detection_shared_model_class(self):
        """
        Without a task the search spans every ``kraken.archs.*`` group, so a
        model class shared by families of different tasks (``TorchVGSLModel``
        backs both vgsl recognition and blla segmentation) matches both.
        """
        from kraken.models.convert import find_weights_archs
        self.assertEqual(find_weights_archs(RESOURCES / 'model_small.safetensors'),
                         {'vgsl', 'blla'})

    def test_weights_detection_task_scoped(self):
        """Scoping by task disambiguates a shared model class."""
        from kraken.models.convert import find_weights_archs
        path = RESOURCES / 'model_small.safetensors'
        self.assertEqual(find_weights_archs(path, task='recognition'), {'vgsl'})
        self.assertEqual(find_weights_archs(path, task='segmentation'), {'blla'})

    def test_weights_detection_legacy_fallback(self):
        from kraken.models.convert import find_weights_archs
        self.assertIsNone(find_weights_archs(RESOURCES / 'overfit.mlmodel'))


class TestKetosArchCLI(unittest.TestCase):
    """
    CLI-level tests of `--arch` validation and dispatch.
    """
    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _ppocr_weights(self):
        from kraken.lib.ppocr import PPOCRv6Model
        from kraken.models import write_safetensors
        model = PPOCRv6Model(variant='tiny', num_classes=4, codec={'a': [1]})
        path = str(self.tmp_path / 'ppocr.safetensors')
        write_safetensors([model], path)
        return path

    def test_foreign_option_vgsl_rejected_for_ppocr(self):
        result = self.runner.invoke(ketos_cli, ['-d', 'cpu', 'train',
                                                '--arch', 'ppocrv6',
                                                '--spec', '[1,48,0,1 Lbx100]',
                                                'foo.png'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--spec is not supported by architecture 'ppocrv6'", result.output)

    def test_foreign_option_ppocr_rejected_for_vgsl(self):
        result = self.runner.invoke(ketos_cli, ['-d', 'cpu', 'train',
                                                '--variant', 'tiny',
                                                'foo.png'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--variant is not supported by architecture 'vgsl'", result.output)

    def test_arch_conflict_with_weights(self):
        weights = self._ppocr_weights()
        result = self.runner.invoke(ketos_cli, ['-d', 'cpu', 'train',
                                                '--arch', 'vgsl',
                                                '-i', weights,
                                                'foo.png'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('conflicts with', result.output)

    def test_arch_conflict_with_checkpoint(self):
        from kraken.configs import PPOCRv6RecognitionTrainingConfig
        ckpt = str(self.tmp_path / 'fake.ckpt')
        torch.save({'_module_config': PPOCRv6RecognitionTrainingConfig()}, ckpt)
        result = self.runner.invoke(ketos_cli, ['-d', 'cpu', 'train',
                                                '--arch', 'vgsl',
                                                '-i', ckpt,
                                                'foo.png'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('conflicts with', result.output)

    def test_non_recognition_checkpoint_rejected(self):
        from kraken.configs import VGSLPreTrainingConfig
        ckpt = str(self.tmp_path / 'pretrain.ckpt')
        torch.save({'_module_config': VGSLPreTrainingConfig()}, ckpt)
        result = self.runner.invoke(ketos_cli, ['-d', 'cpu', 'train',
                                                '-i', ckpt,
                                                'foo.png'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('not a trainable recognition model', result.output)

    @pytest.mark.slow
    def test_ketos_train_ppocr_smoke(self):
        """
        Full `ketos train --arch ppocrv6` CLI smoke including conversion of the
        best checkpoint to a loadable safetensors file.
        """
        from helpers import build_path_dataset
        img_paths = build_path_dataset(self.tmp_path)
        out_dir = self.tmp_path / 'out'
        out_dir.mkdir()
        args = ['-d', 'cpu', '--threads', '1', '--workers', '0',
                'train',
                '--arch', 'ppocrv6',
                '--variant', 'tiny',
                '--height', '48',
                '-o', str(out_dir),
                '-f', 'path',
                '--quit', 'fixed',
                '-N', '1',
                '--min-epochs', '1',
                '-F', '1',
                '-B', '4',
                '--warmup', '2'] + img_paths
        # keep the smoke eager and fast by stubbing out torch.compile
        with mock.patch('torch.compile', side_effect=lambda fn, **kw: fn):
            result = self.runner.invoke(ketos_cli, args)
        self.assertEqual(result.exit_code, 0, result.output)
        weights = list(out_dir.glob('best_*.safetensors'))
        self.assertEqual(len(weights), 1)

        from kraken.lib.ppocr import PPOCRv6Model
        from kraken.models import load_models
        net = load_models(weights[0], tasks=['recognition'])[0]
        self.assertIsInstance(net, PPOCRv6Model)
        self.assertEqual(net.variant, 'tiny')

        # per-architecture defaults must have reached the saved config
        from kraken.configs import PPOCRv6RecognitionTrainingConfig
        from kraken.models.convert import _register_safe_globals
        _register_safe_globals()
        ckpt = torch.load(sorted(out_dir.glob('checkpoint_*.ckpt'))[0],
                          weights_only=True, map_location='cpu')
        config = ckpt['_module_config']
        self.assertIsInstance(config, PPOCRv6RecognitionTrainingConfig)
        self.assertEqual(config.optimizer, 'AdamW+Muon')
        self.assertEqual(config.schedule, 'cosine')
