# -*- coding: utf-8 -*-
"""
Tests for safetensors loader dtype handling.

Verifies that models saved with non-float32 dtypes (uniform float16 and
mixed-precision) can be loaded without errors through `load_models`.
"""
import unittest
from pathlib import Path

import torch

from kraken.models import load_models

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


class TestSafetensorsLoaderDtype(unittest.TestCase):

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
