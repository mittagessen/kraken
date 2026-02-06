# -*- coding: utf-8 -*-
"""
Tests for safetensors writer dtype preservation.

Verifies that `write_models` correctly preserves non-float32 dtypes
(uniform float16 and mixed-precision) in the safetensors file.
"""
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

from kraken.models import load_models, write_models

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


class TestSafetensorsWriterDtype(unittest.TestCase):

    def test_roundtrip_fp16(self):
        """Cast model to fp16, write, and verify stored tensors are fp16."""
        models = load_models(resources / 'model_small.safetensors')
        models[0].to(torch.float16)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'fp16.safetensors'
            write_models(models, path)

            with safe_open(path, framework='pt') as f:
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
            write_models(models, path)

            with safe_open(path, framework='pt') as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if '.C_0.' in key:
                        self.assertEqual(tensor.dtype, torch.float16,
                                         f'{key} should be float16')
                    elif '.O_1.' in key:
                        self.assertEqual(tensor.dtype, torch.float32,
                                         f'{key} should be float32')
