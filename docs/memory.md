# Session Memory & Scratchpad

**Last Updated:** 2026-10-09T23:11:00+05:30
**Current Status:** Gaman AI v0.2 Pillar 3 Lightweight Adapters & Multi-Tier Model Export Complete. 146/146 tests passing.

## Active Context
- **Pillar 3: Lightweight Adapters & Multi-Tier Export Complete:**
  - **Multi-Tier Export Tooling (`scripts/export_backbone.py`):** Added `--tier [small|base|large]` flag mapping to canonical HF checkpoints and output directories (`models/small/`, `models/base/`, `models/large/`). Cleaned non-ASCII characters to support Windows CP1252 consoles.
  - **ModelNotFoundError Guidance:** Implemented `ModelNotFoundError(FileNotFoundError)` in `src/resolver.py` and `src/engine.py`. Emits copy-paste remediation instructions: `Run: python scripts/export_backbone.py --tier {tier}`.
  - **Lightweight Linear Adapters (`src/heads.py`):**
    - `CustomLinearHead`: stores weight matrix $W \in \mathbb{R}^{K \times d}$, bias vector $b \in \mathbb{R}^K$, and class mappings. Pure NumPy Softmax inference with $< 10\,\mu\text{s}$ latency overhead.
    - `fit_adapter`: Closed-form analytical Ridge regression solver ($\tilde{W} = (\tilde{Z}^T \tilde{Z} + \lambda I')^{-1} \tilde{Z}^T Y$) in pure NumPy. Executes in $< 20\,\text{ms}$ on CPU for $N \le 2,000$ samples.
    - JSON serialization and deserialization at `models/heads/<name>.json`.
  - **Engine Runtime Integration (`src/engine.py`):**
    - `embed_batch`: vectorized batch embedding extraction with bit-for-bit identity with `embed`.
    - `predict(state, head_name)`: dynamic head resolution, in-memory caching, dimension validation, and inference reporting.
  - **CLI Commands (`src/cli.py`):**
    - `gaman fit`: fits linear adapter from CSV dataset and reports empirical metrics.
    - `gaman predict`: evaluates single state across trained adapter head with human-readable and `--json` outputs.

## Verification Metrics
- Total Tests: **146 passed in 22.62s** (0 skipped, 0 failed).
- Adapter Tests: **10 passed** (`tests/test_heads.py`).
- CLI Tests: **22 passed** (`tests/test_cli.py`).
- Analytical Fitting Benchmark: $N = 2,000$ in $\mathbb{R}^{768}$ fits in **16.8 ms** on CPU (< 1.0s requirement).
- Adapter Forward Pass Overhead: **< 5.0 µs** (< 10 µs requirement).

## Key Architectural Invariants
- **Zero torch in runtime:** strictly `onnxruntime`, `tokenizers`, `numpy`.
- **Zero scikit-learn in runtime:** closed-form normal equations solved in pure NumPy.
- **Bit-For-Bit Embedding Identity:** `embed_batch(states)[i] == embed(states[i])`.
- **Rank Preservation & Shift Invariance:** Universal primitives preserve mathematical invariants.
- **Backward Compatibility:** Single-model flat deployments (`models/backbone.onnx`) continue to run seamlessly without relocation.