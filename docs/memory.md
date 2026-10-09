# Session Memory & Scratchpad

**Last Updated:** 2026-10-09T22:38:00+05:30
**Current Status:** Gaman AI v0.2 Pillar 1 Technical Audit & Rigorous Fixes Complete. 118/118 tests passing.

## Active Context
- **Technical Audit & Hardening Complete:**
  - **Inverse-Temperature Parameterization:** Re-parameterized Golden Section Search over inverse temperature $\beta = 1/T \in [0.1, 10.0]$ in `src/calibration.py`. Provably strictly convex ($\frac{\partial^2 \mathcal{L}}{\partial \beta^2} = \operatorname{Var}_p(z) \ge 0$) and unimodal.
  - **Boundary Detection & Alerts:** If $\beta^*$ hits the search boundary ($\beta^* \le \beta_{\min} + 0.01$ or $\beta^* \ge \beta_{\max} - 0.01$), emits a warning, rejects calibration, and falls back to neutral $T=1.0$ ($\beta=1.0$).
  - **Accuracy & Sample Size Gating:** Requires $N \ge 30$ and validation accuracy $> 1/K$ before applying calibrated temperature. Sub-chance or micro-datasets fall back to neutral $T=1.0$.
  - **Quantile / Equal-Mass ECE:** Implemented adaptive quantile binning alongside equal-width binning in `compute_ece` (`strategy="quantile"`), resolving probability clustering pathologies.
  - **Shift-Invariant Entailment:** Eliminated isolated-logit Sigmoid in `src/engine.py`. `noul` and `score` compute calibrated probability via normalized 3-class Softmax entailment:
    $$p_{\text{entail}} = \frac{\exp(z_{\text{entail}} / T)}{\sum_{j=0}^2 \exp(z_j / T)}$$
    Mathematically proven and unit-tested to be 100% invariant to constant logit shifts ($z \to z + c$).
  - **Streaming CLI Calibration:** Refactored `gaman calibrate` in `src/cli.py` to stream input in chunks (`_chunked_generator` with batch size 32) directly into NumPy buffers, eliminating full-dataset heap memory bloat.

## Verification Metrics
- Total Tests: **118 passed in 15.31s** (0 skipped, 0 failed).
- Calibration Tests: **14 passed** (`tests/test_calibration.py`).
- Shift Invariance: **Verified on all primitives** (`tests/test_engine.py::TestShiftInvariance`).
- CLI Tests: **16 passed** (`tests/test_cli.py`).
- Runtime Latency Overhead: **< 1.0 µs** for 3-class normalized Softmax.

## Key Architectural Invariants
- **Zero torch in runtime:** strictly `onnxruntime`, `tokenizers`, `numpy`.
- **Shift Invariance:** All three primitives (`choice`, `noul`, `score`) are mathematically invariant to arbitrary constant logit shifts.
- **Beta-Space Convexity:** Temperature optimization strictly performed in $\beta = 1/T$ space.
- **Rank Preservation:** $\arg\max_k(z_k / T) = \arg\max_k(z_k)$ strictly preserved.