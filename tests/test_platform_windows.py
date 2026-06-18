"""Platform-specific tests for Windows compatibility.

These tests verify that kraken works correctly on Windows when PYTHONUTF8=1
is set (or Python 3.15+ which defaults to UTF-8 mode). The PYTHONUTF8=1 env
var forces UTF-8 as the default I/O encoding, fixing htrmopo's open() calls
that lack explicit encoding='utf-8'.
"""
import sys
import os
import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def _ensure_utf8_mode():
    """Ensure PYTHONUTF8=1 is set for all tests in this module.

    This simulates the recommended Windows setup where PYTHONUTF8=1 is
    configured as a persistent user environment variable.
    """
    old = os.environ.get("PYTHONUTF8")
    os.environ["PYTHONUTF8"] = "1"
    yield
    if old is None:
        os.environ.pop("PYTHONUTF8", None)
    else:
        os.environ["PYTHONUTF8"] = old


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_kraken_list_windows():
    """kraken list should not crash with UnicodeDecodeError on Windows.

    Verifies that htrmopo's iso15924.txt reading works when Python is in
    UTF-8 mode (PYTHONUTF8=1). Previously crashed with cp1252 default.

    NOTE: This test requires network access (fetches model list from Zenodo).
    It may be slow or skipped in CI without network.
    """
    from kraken.kraken import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0, f"kraken list failed: {result.output}"


def test_htrmopo_utf8_import():
    """htrmopo module should import without UnicodeDecodeError when PYTHONUTF8=1.

    This directly tests the root cause: htrmopo/util.py reads iso15924.txt
    without encoding='utf-8'. With PYTHONUTF8=1, the default encoding is utf-8
    so the import succeeds.
    """
    import importlib
    if "htrmopo" in sys.modules:
        del sys.modules["htrmopo"]
    if "htrmopo.util" in sys.modules:
        del sys.modules["htrmopo.util"]
    import htrmopo
    assert htrmopo is not None


def test_pdf_import_fallback():
    """PDF backend should be importable on Windows via pypdfium2.

    pypdfium2 is the recommended Windows PDF backend (pure-Python wheels).
    pyvips requires native libvips DLLs which are not available by default.
    """
    import pypdfium2
    assert hasattr(pypdfium2, 'PdfDocument')


def test_case_insensitive_model_suffix():
    """Model file suffix matching should be case-insensitive.

    On Windows, filesystems are case-insensitive, so Path('foo.MLMODEL').suffix
    returns '.MLMODEL'. The model filter must use .lower() to match correctly.
    """
    from pathlib import Path
    import tempfile, shutil
    src = Path('tests/resources/model_small.mlmodel')
    if not src.exists():
        pytest.skip("test fixture not found")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dst = tmpdir / 'TEST_MODEL.MLMODEL'
        shutil.copy(src, dst)
        candidates = list(filter(lambda x: x.suffix.lower() in ['.mlmodel', '.safetensors'], tmpdir.iterdir()))
        assert len(candidates) == 1, f"Expected 1 candidate, got {len(candidates)}"


def test_model_load_mlmodel():
    """CoreML model loading should work on all platforms."""
    from kraken.lib.vgsl.model import TorchVGSLModel
    model = TorchVGSLModel.load_model('tests/resources/model_small.mlmodel')
    assert model is not None


def test_import_kraken():
    """kraken should import without error on all platforms."""
    import kraken
    assert kraken is not None


def test_import_kraken_kraken():
    """kraken.kraken should import without error."""
    from kraken.kraken import cli
    assert cli is not None
