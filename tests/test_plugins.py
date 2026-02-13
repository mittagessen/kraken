# -*- coding: utf-8 -*-
"""
Tests for the kraken plugin/entry point system.

Tests the entry point-based plugin architecture used for model registration,
loader/writer discovery, task registration, and CLI subcommand loading.
"""
import importlib.metadata
import os
import tempfile
import unittest
from pathlib import Path

import pytest
from pytest import raises

from kraken.models import load_models, write_models
from kraken.models.utils import create_model

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


class TestEntryPointRegistration(unittest.TestCase):
    """
    Tests that all expected entry point groups are registered and resolvable.
    """

    def test_kraken_models_entry_points(self):
        """
        Verifies that the kraken.models entry point group contains the
        expected model classes.
        """
        eps = {ep.name for ep in importlib.metadata.entry_points(group='kraken.models')}
        self.assertIn('TorchVGSLModel', eps)

    def test_kraken_loaders_entry_points(self):
        """
        Verifies that the kraken.loaders entry point group contains the
        expected loaders.
        """
        eps = {ep.name for ep in importlib.metadata.entry_points(group='kraken.loaders')}
        self.assertIn('safetensors', eps)
        self.assertIn('coreml', eps)

    def test_kraken_writers_entry_points(self):
        """
        Verifies that the kraken.writers entry point group contains the
        expected writers.
        """
        eps = {ep.name for ep in importlib.metadata.entry_points(group='kraken.writers')}
        self.assertIn('safetensors', eps)
        self.assertIn('coreml', eps)

    def test_kraken_tasks_entry_points(self):
        """
        Verifies that the kraken.tasks entry point group contains the
        expected task models.
        """
        eps = {ep.name for ep in importlib.metadata.entry_points(group='kraken.tasks')}
        self.assertIn('segmentation', eps)
        self.assertIn('recognition', eps)
        self.assertIn('alignment', eps)

    def test_kraken_cli_entry_points(self):
        """
        Verifies that the kraken.cli entry point group contains the expected
        subcommands.
        """
        eps = {ep.name for ep in importlib.metadata.entry_points(group='kraken.cli')}
        for cmd in ('binarize', 'segment', 'ocr', 'show', 'list', 'get'):
            self.assertIn(cmd, eps)

    def test_ketos_cli_entry_points(self):
        """
        Verifies that the ketos.cli entry point group contains the expected
        training subcommands.
        """
        eps = {ep.name for ep in importlib.metadata.entry_points(group='ketos.cli')}
        for cmd in ('compile', 'pretrain', 'train', 'test', 'publish',
                    'rotrain', 'roadd', 'segtrain', 'segtest', 'convert'):
            self.assertIn(cmd, eps)

    def test_kraken_lightning_modules_entry_points(self):
        """
        Verifies that the kraken.lightning_modules entry point group contains
        the expected lightning modules.
        """
        eps = {ep.name for ep in importlib.metadata.entry_points(group='kraken.lightning_modules')}
        for mod in ('blla', 'vgsl', 'pretrain', 'ro'):
            self.assertIn(mod, eps)



