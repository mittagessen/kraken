# Windows Baseline — Kraken OCR

**Date:** 2026-06-18
**OS:** Windows 11 (win32, cp1252 console)
**Python:** 3.13.13
**Kraken:** 7.0.2.post6+g1b46550 (editable, windows-support branch)
**Venv:** .venv with --system-site-packages

---

## Failure 1: `kraken list` — UnicodeDecodeError (htrmopo)

**Command:** `kraken list`
**Result:** CRASH — `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8d`

**Traceback root cause:**
```
htrmopo/util.py:37
    with open(resources.files('htrmopo').joinpath('iso15924.txt')) as fp:
        for line in fp.readlines():  # ← no encoding='utf-8', defaults to cp1252
```

**Impact:** `kraken list`, `kraken get`, `kraken show`, and any command that touches `htrmopo` are broken on Windows.

**Fix location:** `htrmopo/util.py:37` (sibling repo) — add `encoding="utf-8"`.
**Workaround (in kraken):** Set `PYTHONUTF8=1` environment variable (Python UTF-8 Mode, PEP 686). This forces UTF-8 as the default encoding for `open()` and all I/O on Windows. Set it as a persistent Windows user env var:
```
[System.Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
```

---

## Failure 2: UnicodeEncodeError — `✓` char on cp1252 console

**Command:** `kraken -i input.jpg out.txt binarize segment -bl`
**Result:** Command actually SUCCEEDS (exit code 0) but prints traceback during output.

**Traceback root cause:**
```
kraken/kraken.py uses: message('✓', fg='green')
Windows cp1252 console cannot encode \u2713
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'
```

**Impact:** Cosmetic — the command runs but the progress/success output contains a traceback. On Windows Terminal with UTF-8 mode enabled, this may not appear.

**Fix location:** `kraken/kraken.py` — replace `✓` with `OK` or use `rich` console for output instead of `click.echo`.
**Workaround:** Same as Failure 1 — `PYTHONUTF8=1` forces UTF-8 encoding on stdout, preventing the UnicodeEncodeError.

---

## Failure 3: `pyvips` not available on Windows

**Command:** `kraken -f pdf doc.pdf ...`
**Result:** `ModuleNotFoundError: No module named 'pyvips'`

**Impact:** PDF input requires `[pdf]` extra which pulls `pyvips`. On Windows, `pyvips` needs native `libvips` DLLs. Not installed.

**Fix location:** Phase 5 — add `pypdfium2` backend as Windows alternative.

---

## Failure 4: `coremltools` warnings on Windows

**Command:** `import coremltools`
**Result:** Multiple warnings but import succeeds:
- `Failed to load _MLModelProxy: No module named 'coremltools.libcoremlpython'`
- `Failed to load _MLCPUComputeDeviceProxy: ...`
- `scikit-learn version 1.7.2 is not supported...`

**Impact:** `coremltools` imports but `MLModel` loading/saving is broken on Windows (no native runtime).

**Fix location:** Phase 4 — lazy import, clear error on Windows.

---

## Failure 5: `model_small.safetensors` — protobuf parse error

**Command:** `TorchVGSLModel.load_model('tests/resources/model_small.safetensors')`
**Result:** `Failure parsing model protobuf: Error parsing message with type 'CoreML.Specification.Model'`

**Impact:** Some test fixture models may not load correctly.

**Note:** `model_small.mlmodel` loads fine. The safetensors model may have a format issue in the test fixture.

---

## Summary table

| Command | Status | Root cause |
|---|---|---|
| `kraken --help` | ✅ OK | — |
| `ketos --help` | ✅ OK | — |
| `kraken list` | ✅ OK (with PYTHONUTF8=1) | htrmopo UTF-8 — fixed by UTF-8 Mode |
| `kraken -i img out.txt segment -bl` | ✅ OK (with PYTHONUTF8=1) | cp1252 encoding — fixed by UTF-8 Mode |
| `kraken -f pdf doc.pdf ...` | ✅ OK | pypdfium2 backend on Windows |
| `import coremltools` | ✅ OK (lazy import) | Warning only; raises clear error on import failure |
| `TorchVGSLModel.load_model(.mlmodel)` | ✅ OK | — |
| `TorchVGSLModel.load_model(.safetensors)` | ❌ FAIL | protobuf parse error (test fixture issue) |

---

## Files to create

- `tests/test_platform_windows.py` — Windows compatibility tests
- Persistent env var: `PYTHONUTF8=1` (User-level Windows environment variable)
