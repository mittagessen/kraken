# -*- coding: utf-8 -*-
"""
Tests for the kraken plugin/entry point system.
"""
import importlib.metadata

import pytest

EXPECTED_GROUPS = {
    'kraken.models': {'TorchVGSLModel', 'Wav2Vec2Mask', 'ROMLP', 'PPOCRv6Model'},
    'kraken.loaders': {'safetensors', 'coreml'},
    'kraken.writers': {'safetensors', 'coreml'},
    'kraken.tasks': {'segmentation', 'recognition', 'alignment'},
    'kraken.cli': {'binarize', 'segment', 'ocr', 'show', 'list', 'get'},
    'ketos.cli': {'compile', 'pretrain', 'train', 'test', 'publish', 'rotrain',
                  'roadd', 'segtrain', 'segtest', 'convert'},
    'kraken.lightning_modules': {'blla', 'vgsl', 'ppocrv6', 'pretrain', 'ro'},
    'kraken.archs.recognition': {'vgsl', 'ppocrv6'},
    'kraken.archs.segmentation': {'blla'},
}


@pytest.mark.parametrize('group,expected', EXPECTED_GROUPS.items(), ids=EXPECTED_GROUPS.keys())
def test_entry_point_group(group, expected):
    """
    Verifies that the entry point group contains kraken's expected entries.
    """
    eps = {ep.name for ep in importlib.metadata.entry_points(group=group)}
    assert expected <= eps, f'missing entries in {group}: {expected - eps}'
