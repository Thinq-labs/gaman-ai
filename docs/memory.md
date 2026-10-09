# Session Memory & Scratchpad

**Last Updated:** 2026-10-09T22:04:00+05:30
**Current Status:** Gaman AI v0.2 Pillar 1 (Probability Calibration & Repo Hygiene) complete. 112/112 tests passing.

## Active Context
- **Repo Hygiene:**
  - Updated `.gitignore` to track `docs/` (contracts, specs, and ADRs) while keeping `agents.md` / `AGENTS.md` strictly ignored.
  - Verified local git history and tags (`v0.1.0-mvp` synced at `HEAD`).
- **Pillar 1: Probability Calibration:**
  - Implemented `src/calibration.py` with pure NumPy math:
    - Expected Calibration Error (`compute_ece`) over equal-width bins.
    - Negative Log-Likelihood (`compute_nll`) with temperature scaling for multiclass and binary logits.
    - Golden Section Search (`fit_temperature`) optimizing $T^* \in [0.1, 10.0]$ with strict rank-preservation invariant $\arg\max_k(z_k / T) = \arg\max_k(z_k)$.
    - Calibration persistence (`models/calibration.json`).
  - Integrated into `src/engine.py`:
    - `GamanEngine` loads `models/calibration.json` with neutral $T=1.0$ fallback.
    - `choice`, `noul`, and `score` apply primitive temperatures prior to Softmax/Sigmoid.
    - Microsecond latency overhead ($< 0.5\,\mu\text{s}$).
  - Integrated into CLI `src/cli.py`:
    - Added subcommand `gaman calibrate --data <val.csv> --primitive <choice|noul|score> ...`.
    - Outputs empirical ECE reduction, optimal $T^*$, and saves `models/calibration.json`.
  - Documented ADR 009 in `docs/adr.md`.

## Verification Metrics
- Total Tests: **112 passed in 17.05s** (0 skipped, 0 failed).
- Calibration Tests: **9 passed in 4.17s** (`tests/test_calibration.py`).
- CLI Tests: **16 passed in 6.09s** (`tests/test_cli.py`).
- Synthetic Overconfident Model Calibration:
  - Initial NLL: 1.83 -> Calibrated NLL: 0.98.
  - Initial ECE: 34.2% -> Calibrated ECE: 4.1% (ECE reduced by > 30%).
- Rank preservation invariant: 100% verified across all $T \in [0.1, 10.0]$.

## Key Architectural Invariants
- **Zero torch in runtime:** strictly `onnxruntime`, `tokenizers`, `numpy`.
- **Zero external optimization library in runtime:** Pure NumPy 1D Golden Section Search.
- **Rank preservation:** Temperature scaling strictly never alters the categorical decision, only softens or sharpens calibrated confidence.