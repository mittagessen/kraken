# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from kraken.ketos import cli as ketos_cli

RESOURCES = Path(__file__).resolve().parent / 'resources'


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

    def _run_ketos(self, args):
        result = self.runner.invoke(ketos_cli, args)
        self.assertEqual(result.exit_code, 0, result.output)

    def test_ketos_train_recognition_smoke(self):
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
        self._run_ketos(args)

    def test_ketos_train_segmentation_smoke(self):
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
        self._run_ketos(args)

    def test_ketos_train_reading_order_smoke(self):
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
        self._run_ketos(args)
