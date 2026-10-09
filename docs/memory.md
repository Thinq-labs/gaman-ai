# Session Memory & Scratchpad

**Last Updated:** 2026-10-09T22:52:00+05:30
**Current Status:** Gaman AI v0.2 Pillar 2 Automated Spec Slab Resolver Complete. 133/133 tests passing.

## Active Context
- **Pillar 2: Automated Spec Slab Resolver Complete:**
  - **Hardware Detection (`src/resolver.py`):** Zero-dependency hardware probing (`detect_hardware()`, `HardwareProfile`). Discovers execution providers via `ort.get_available_providers()`, inspects system RAM via OS APIs (`GlobalMemoryStatusEx` on Windows, `/proc/meminfo` on Linux, `sysctl hw.memsize` on Darwin), counts logical cores, and probes SIMD (AVX512/NEON).
  - **Hierarchical Tier Dispatch:**
    - `large`: CUDA/ROCm execution provider with $\ge 8\,\text{GB}$ VRAM. Model: `cross-encoder/nli-deberta-v3-large` (1024 hidden dimension).
    - `base`: CoreML or CUDA/ROCm with $\ge 4\,\text{GB}$ VRAM, or host RAM $\ge 16\,\text{GB}$. Model: `cross-encoder/nli-deberta-v3-base` (768 hidden dimension).
    - `small`: Edge/CPU fallback when host RAM $< 16\,\text{GB}$ or on CPU-only edge systems. Model: `cross-encoder/nli-deberta-v3-small` (768 hidden dimension).
  - **Backward-Compatible Model Resolution:** `resolve_model_path()` checks `models/<tier>/backbone.onnx` first, falling back to flat `models/backbone.onnx` for `small` tier to maintain 100% compatibility with v0.1 model deployments.
  - **Dynamic Dimensioning in Engine (`src/engine.py`):** Engine dynamically initializes with `slab` tier and resolved `hidden_dim` (768 vs 1024), dynamically selecting execution providers. `embed(state)` emits representation vector of shape `(hidden_dim,)`.
  - **Decoupled Diagnostic Tooling (`src/cli.py`):** Added `gaman info` subcommand (instantaneous <50ms, no ONNX session overhead) with human-readable and `--json` outputs, and added global `--slab [auto|small|base|large]` flag.

## Verification Metrics
- Total Tests: **133 passed in 18.42s** (0 skipped, 0 failed).
- Resolver Tests: **12 passed** (`tests/test_resolver.py`).
- CLI Tests: **19 passed** (`tests/test_cli.py`).
- Real System Diagnostic: Host (Windows 11, 13.8 GB RAM, 8 logical cores, CPUExecutionProvider) cleanly auto-resolved to `small` tier.

## Key Architectural Invariants
- **Zero torch in runtime:** strictly `onnxruntime`, `tokenizers`, `numpy`.
- **Zero psutil:** hardware inspection via stdlib / native platform ctypes and procfs.
- **Shift Invariance:** All three primitives (`choice`, `noul`, `score`) are mathematically invariant to arbitrary constant logit shifts.
- **Backward Compatibility:** Single-model flat deployments (`models/backbone.onnx`) continue to run seamlessly without relocation.
- **Rank Preservation:** $\arg\max_k(z_k / T) = \arg\max_k(z_k)$ strictly preserved.