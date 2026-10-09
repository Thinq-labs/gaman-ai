# Session Memory & Scratchpad

**Last Updated:** 2026-10-09T23:41:00+05:30
**Current Status:** Gaman AI v0.2 Zero-Clone Pip Distribution & Auto-Weights Downloader Complete. 156/156 tests passing.

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