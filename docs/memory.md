# Session Memory & Scratchpad

**Last Updated:** 2026-10-10T01:04:00+05:30
**Current Status:** Gaman AI v0.2 GitHub Actions CI Matrix Fixed across all 9 OS/Python targets (Ubuntu, Windows, macOS). Commit 5dcb2c1.

## Active Context
- **Zero-Clone Pip Distribution & Auto-Weights Downloader:**
  - **Standard OS Cache Discovery (`src/resolver.py`):** `get_default_cache_dir()` checks `GAMAN_CACHE_DIR`, `%LOCALAPPDATA%/gaman` on Windows, and `$XDG_CACHE_HOME/gaman` on Linux/macOS.
  - **Model Lookup Hierarchy:** Priority 1 (Explicit `models_dir`) > Priority 2 (Local cwd `./models/`) > Priority 3 (Global OS Cache) > Auto-Downloader.
  - **Zero-Dependency Weights Downloader (`src/downloader.py`):**
    - Pure `urllib.request` implementation with zero third-party dependencies (`no requests`, `no huggingface_hub`).
    - Atomic writes via temporary `.tmp` staging and `os.replace`.
    - TTY-sensitive ASCII progress bar (`Downloading Gaman AI 'small' weights (164 MB)... [=========>] 100%`) suppressed when non-interactive or under `--json`.
  - **Engine & CLI Integration:**
    - `GamanEngine(models_dir=None)` automatically resolves or downloads weights on first run, providing true zero-clone out-of-the-box bootstrapping for `pip install gaman-ai`.
    - `gaman cache --dir` and `gaman cache --clean` subcommands.
    - `gaman info` displays global cache path and status.

## Verification Metrics
- Total Tests: **156 passed in 20.62s** (0 skipped, 0 failed).
- Downloader & Cache Tests: **10 passed** (`tests/test_downloader.py`).
- CLI Tests: **22 passed** (`tests/test_cli.py`).
- Adapter Tests: **10 passed** (`tests/test_heads.py`).

## Key Architectural Invariants
- **Zero third-party network dependencies in runtime:** pure standard library `urllib.request`.
- **Zero torch / transformers in runtime:** strictly `onnxruntime`, `tokenizers`, `numpy`.
- **Atomic File Writes:** No half-downloaded or corrupted model files from interrupted network transfers.
- **Bit-For-Bit Embedding Identity:** `embed_batch(states)[i] == embed(states[i])`.
- **Backward Compatibility:** Single-model flat deployments (`models/backbone.onnx`) continue to run seamlessly without relocation.

## CI Matrix Fixes (ca4e5d0)
- **Cross-Platform Shell Parity:** Added `defaults.run.shell: bash` to `.github/workflows/ci.yml` so runner steps execute uniformly across Linux, macOS, and Windows.
- **Dedicated CI Pre-flight Script:** Created `scripts/ci_prepare_model.py` to safely inspect existing cached weights and invoke `scripts/export_backbone.py --tier small` without fragile in-line shell chain logic.
- **Linter & Formatting Hygiene:** Fixed 44 Ruff lint violations across `src/`, `tests/`, and `scripts/` (unused imports, file context managers, generator expressions, ternary conversions). Clean 0-error Ruff check across entire repo.

## macOS CoreML Slab Resolution Fix (5dcb2c1)
- **Problem:** GitHub macOS runners have CoreML available in ONNX Runtime, causing `resolve_slab("auto")` to select tier `base`. In CI, preflight had only populated `models/small/backbone.onnx`, throwing `ModelNotFoundError: Model tier 'base' not found in models/base`.
- **Solution:** `scripts/ci_prepare_model.py` now exports to root `models/backbone.onnx` and guarantees both `models/small/` and `models/base/` tiers are populated.
- **Test Resilience:** In `tests/test_engine.py`, `test_missing_model_raises_file_not_found` explicitly verifies `slab="small"` as well as auto-resolved tiers matching `r"Model tier '(small|base|large)' not found in"`.

## CI Cache Isolation & Test Fixture Hygiene (fd4ae4a)
- **Problem:** Stale `calibration.json` restored from GitHub Actions cache caused non-neutral temperature scaling ($T \ne 1.0$) during test runs, leading to assertion mismatches on `test_noul`.
- **Solution:**
  1. Updated cache key in `.github/workflows/ci.yml` to `gaman-models-v3-${{ runner.os }}-${{ matrix.python-version }}-...` to bust stale cache.
  2. `scripts/ci_prepare_model.py` actively removes any residual `calibration.json` before test suite execution.
  3. Added `_reset_shared_engine_temps` and `_reset_engine_temps` autouse fixtures in `tests/test_cli.py` and `tests/test_engine.py` to guarantee strict test isolation.