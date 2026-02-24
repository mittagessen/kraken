# -*- coding: utf-8 -*-
"""
Tests for model loaders/writers/creators.
"""
import json
import os
import torch
import tempfile
import unittest
from contextlib import contextmanager

from pathlib import Path
from pytest import raises
from safetensors import safe_open
from safetensors.torch import save_file, load_file

from kraken.models.utils import create_model
from kraken.models import load_models, write_safetensors
from kraken.models.loaders import load_coreml, load_safetensors

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


@contextmanager
def _patched_safetensors_model_type(path: Path, model_type: list[str] = ['recognition']):
    """
    Creates a temporary copy of a safetensors file with explicit model_type.
    """
    tensors = load_file(path)
    with safe_open(path, framework='pt') as f:
        metadata = json.loads(f.metadata()['kraken_meta'])
    for rec in metadata.values():
        rec['model_type'] = model_type
    with tempfile.NamedTemporaryFile(delete=False, suffix='.safetensors') as fp:
        tmp_path = Path(fp.name)
    try:
        save_file(tensors, tmp_path, metadata={'kraken_meta': json.dumps(metadata)})
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


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
        with _patched_safetensors_model_type(resources / 'model_small.safetensors') as model_path:
            models = load_models(model_path)
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
        with _patched_safetensors_model_type(resources / 'model_small_fp16.safetensors') as model_path:
            models = load_models(model_path)
        self.assertEqual(len(models), 1)
        for name, param in models[0].named_parameters():
            self.assertEqual(param.dtype, torch.float32, f'{name} is not float32')

    def test_load_mixed(self):
        """Load a model saved with mixed fp16/fp32 weights without errors.

        Conv layer weights are stored as fp16, linear layer weights as fp32.
        After loading, all parameters are float32 due to copy_() semantics.
        """
        with _patched_safetensors_model_type(resources / 'model_small_mixed.safetensors') as model_path:
            models = load_models(model_path)
        self.assertEqual(len(models), 1)
        for name, param in models[0].named_parameters():
            self.assertEqual(param.dtype, torch.float32, f'{name} is not float32')

    def test_load_safetensors_missing_min_version_metadata(self):
        """
        Missing _kraken_min_version in safetensors metadata raises ValueError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'bad_min_version.safetensors'
            tensors = load_file(resources / 'model_small.safetensors')
            with safe_open(resources / 'model_small.safetensors', framework='pt') as f:
                metadata = json.loads(f.metadata()['kraken_meta'])
            for rec in metadata.values():
                rec['model_type'] = 'recognition'
                rec.pop('_kraken_min_version', None)
            save_file(tensors, path, metadata={'kraken_meta': json.dumps(metadata)})
            with raises(ValueError, match='_kraken_min_version'):
                load_safetensors(path)

    def test_load_safetensors_invalid_tasks_metadata(self):
        """
        Invalid _tasks type in safetensors metadata raises ValueError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'bad_tasks.safetensors'
            tensors = load_file(resources / 'model_small.safetensors')
            with safe_open(resources / 'model_small.safetensors', framework='pt') as f:
                metadata = json.loads(f.metadata()['kraken_meta'])
            for rec in metadata.values():
                rec['model_type'] = 'recognition'
                rec['_tasks'] = {'recognition': True}
            save_file(tensors, path, metadata={'kraken_meta': json.dumps(metadata)})
            with raises(ValueError, match='_tasks'):
                load_safetensors(path)

    def test_load_safetensors_missing_model_type_metadata(self):
        """
        Missing model_type in safetensors metadata raises ValueError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'bad_model_type.safetensors'
            tensors = load_file(resources / 'model_small.safetensors')
            with safe_open(resources / 'model_small.safetensors', framework='pt') as f:
                metadata = json.loads(f.metadata()['kraken_meta'])
            for rec in metadata.values():
                rec.pop('model_type', None)
            save_file(tensors, path, metadata={'kraken_meta': json.dumps(metadata)})
            with raises(ValueError, match='model_type'):
                load_safetensors(path)

    def test_load_coreml_missing_model_type_metadata(self):
        """
        Missing model_type in CoreML metadata raises ValueError.
        """
        from coremltools.models import MLModel
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'missing_model_type.mlmodel'
            model = MLModel((resources / 'overfit.mlmodel').as_posix())
            metadata = json.loads(model.user_defined_metadata.get('kraken_meta', '{}'))
            metadata.pop('model_type', None)
            model.user_defined_metadata['kraken_meta'] = json.dumps(metadata)
            model.save(path.as_posix())
            with raises(ValueError, match='model_type'):
                load_coreml(path)


class TestVersionCompatibility(unittest.TestCase):
    """
    Tests for the version compatibility check in load_safetensors.
    """

    def _make_versioned_model_file(self, version_overrides, path):
        """
        Creates a safetensors file containing two copies of the small test
        model, then patches _kraken_min_version in the metadata for selected
        models.

        Args:
            version_overrides: dict mapping prefix index (0-based) to a
                               _kraken_min_version string to set.
            path: Output path for the safetensors file.

        Returns:
            List of prefix strings in the file (in metadata key order).
        """
        import copy
        with _patched_safetensors_model_type(resources / 'model_small.safetensors') as model_path:
            models = load_models(model_path)
        write_safetensors([models[0], copy.deepcopy(models[0])], path)

        tensors = load_file(path)
        with safe_open(path, framework='pt') as f:
            meta = json.loads(f.metadata()['kraken_meta'])

        prefixes = list(meta.keys())
        for idx, version in version_overrides.items():
            meta[prefixes[idx]]['_kraken_min_version'] = version

        save_file(tensors, path, metadata={'kraken_meta': json.dumps(meta)})
        return prefixes

    def test_compatible_and_incompatible(self):
        """
        In a file with one compatible and one incompatible model, only the
        compatible model is loaded.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.safetensors'
            self._make_versioned_model_file({0: '999.0.0'}, path)
            models = load_models(path)
            self.assertEqual(len(models), 1)

    def test_all_incompatible(self):
        """
        When all models in a file are incompatible, an empty list is returned.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.safetensors'
            self._make_versioned_model_file({0: '999.0.0', 1: '999.0.0'}, path)
            models = load_models(path)
            self.assertEqual(len(models), 0)

    def test_all_compatible(self):
        """
        When all models in a file are compatible, all are loaded.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.safetensors'
            self._make_versioned_model_file({}, path)
            models = load_models(path)
            self.assertEqual(len(models), 2)

    def test_incompatible_model_warns(self):
        """
        Skipping an incompatible model emits a warning containing the
        required version.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.safetensors'
            self._make_versioned_model_file({0: '999.0.0'}, path)
            with self.assertLogs('kraken.models.loaders', level='WARNING') as cm:
                load_models(path)
            self.assertTrue(any('999.0.0' in msg for msg in cm.output))

    def test_compatible_model_has_weights(self):
        """
        The compatible model loaded from a mixed file has actual (non-zero)
        weights.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.safetensors'
            self._make_versioned_model_file({0: '999.0.0'}, path)
            models = load_models(path)
            has_nonzero = any(p.abs().sum() > 0 for p in models[0].parameters())
            self.assertTrue(has_nonzero)


class TestWriteModels(unittest.TestCase):
    """
    Tests for the `write_safetensors`/`write`
    """
    def test_write_read_roundtrip_safetensors(self):
        """
        Tests that models can be written and read back in safetensors format.
        """
        with _patched_safetensors_model_type(resources / 'model_small.safetensors') as model_path:
            models = load_models(model_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_model.safetensors'
            opath = write_safetensors(models, path)
            self.assertTrue(opath.exists())
            loaded = load_models(opath)
            self.assertEqual(len(loaded), len(models))
            self.assertEqual(loaded[0].model_type, models[0].model_type)

    def test_roundtrip_fp16(self):
        """Cast model to fp16, write, and verify stored tensors are fp16."""
        with _patched_safetensors_model_type(resources / 'model_small.safetensors') as model_path:
            models = load_models(model_path)
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
        with _patched_safetensors_model_type(resources / 'model_small.safetensors') as model_path:
            models = load_models(model_path)
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
