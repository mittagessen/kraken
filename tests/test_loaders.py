# -*- coding: utf-8 -*-
"""
Tests for model loaders/writers/creators.
"""
import os
import torch
import pytest
import tempfile
import unittest

from pathlib import Path
from pytest import raises
from safetensors import safe_open

from kraken.models.utils import create_model
from kraken.models import load_models, write_safetensors

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


class TestCreateModel(unittest.TestCase):
    """
    Tests for the `create_model` factory function that uses the model
    registry to instantiate model objects.
    """

    def test_create_model_invalid_name(self):
        """
        Tests that create_model raises ValueError for unknown model names.
        """
        with raises(ValueError, match='not in model registry'):
            create_model('NonExistentModel')

    def test_create_model_torch_vgsl(self):
        """
        Tests that create_model can instantiate a TorchVGSLModel from the
        registry.
        """
        from kraken.lib.vgsl import TorchVGSLModel
        model = create_model('TorchVGSLModel',
                             vgsl='[1,48,0,1 Cr3,3,32 Mp2,2 Cr3,3,64 Mp2,2 S1(1x0)1,3 Lbx200 Do O1c10]')
        self.assertIsInstance(model, TorchVGSLModel)


class TestLoadModels(unittest.TestCase):
    """
    Tests for the `load_models` factory function that uses registered loaders
    to deserialize model files.
    """

    def test_load_models_coreml(self):
        """
        Tests loading a CoreML model file through the loader registry.
        """
        models = load_models(resources / 'overfit.mlmodel')
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        self.assertIn('recognition', models[0].model_type)

    def test_load_models_invalid_file(self):
        """
        Tests that load_models raises ValueError for invalid model files.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as fp:
            fp.write(b'invalid data')
            fp.flush()
            try:
                with raises(ValueError):
                    load_models(fp.name)
            finally:
                os.unlink(fp.name)

    def test_load_models_nonexistent_file(self):
        """
        Tests that load_models raises ValueError for non-existent files.
        """
        with raises(ValueError, match='not a regular file'):
            load_models('/nonexistent/path/model.mlmodel')

    def test_load_models_task_filter(self):
        """
        Tests that the task filter parameter works correctly.
        """
        models = load_models(resources / 'overfit.mlmodel', tasks=['recognition'])
        self.assertGreater(len(models), 0)

    def test_load_models_task_filter_mismatch(self):
        """
        Tests that task filtering returns empty list for mismatched tasks.
        """
        models = load_models(resources / 'overfit.mlmodel', tasks=['segmentation'])
        self.assertEqual(len(models), 0)

    def test_load_fp32(self):
        """Load a standard fp32 model and verify all parameters are float32."""
        models = load_models(resources / 'model_small.safetensors')
        self.assertEqual(len(models), 1)
        for name, param in models[0].named_parameters():
            self.assertEqual(param.dtype, torch.float32, f'{name} is not float32')

    def test_load_fp16(self):
        """Load a model saved with all-fp16 weights without errors.

        The model is constructed with float32 parameters and weights are
        loaded via copy_(), so the resulting parameters remain float32.
        This test verifies that loading fp16 weights into a float32 model
        succeeds (the strict=False fix).
        """
        models = load_models(resources / 'model_small_fp16.safetensors')
        self.assertEqual(len(models), 1)
        for name, param in models[0].named_parameters():
            self.assertEqual(param.dtype, torch.float32, f'{name} is not float32')

    def test_load_mixed(self):
        """Load a model saved with mixed fp16/fp32 weights without errors.

        Conv layer weights are stored as fp16, linear layer weights as fp32.
        After loading, all parameters are float32 due to copy_() semantics.
        """
        models = load_models(resources / 'model_small_mixed.safetensors')
        self.assertEqual(len(models), 1)
        for name, param in models[0].named_parameters():
            self.assertEqual(param.dtype, torch.float32, f'{name} is not float32')


class TestWriteModels(unittest.TestCase):
    """
    Tests for the `write_safetensors`/`write`
    """
    def test_write_read_roundtrip_safetensors(self):
        """
        Tests that models can be written and read back in safetensors format.
        """
        models = load_models(resources / 'model_small.safetensors')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_model.safetensors'
            opath = write_safetensors(models, path)
            self.assertTrue(opath.exists())
            loaded = load_models(opath)
            self.assertEqual(len(loaded), len(models))
            self.assertEqual(loaded[0].model_type, models[0].model_type)

    def test_roundtrip_fp16(self):
        """Cast model to fp16, write, and verify stored tensors are fp16."""
        models = load_models(resources / 'model_small.safetensors')
        models[0].to(torch.float16)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'fp16.safetensors'
            opath = write_safetensors(models, path)

            with safe_open(opath, framework='pt') as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    self.assertEqual(tensor.dtype, torch.float16,
                                     f'{key} is not float16')

    def test_roundtrip_mixed(self):
        """Cast only conv layer to fp16, write, and verify stored dtypes."""
        models = load_models(resources / 'model_small.safetensors')
        models[0].nn.C_0.to(torch.float16)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'mixed.safetensors'
            opath = write_safetensors(models, path)

            with safe_open(opath, framework='pt') as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if '.C_0.' in key:
                        self.assertEqual(tensor.dtype, torch.float16,
                                         f'{key} should be float16')
                    elif '.O_1.' in key:
                        self.assertEqual(tensor.dtype, torch.float32,
                                         f'{key} should be float32')
