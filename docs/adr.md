# Architecture Decision Records (ADR)

---

## ADR 1: Zero Autoregressive Generation
- **Decision:** The engine will strictly output typed data structures (Logits, Scalars, Booleans). It will never generate text.
- **Reason:** Text generation requires a KV-cache and O(N) sequential time steps. We need O(1) single-pass execution for sub-20ms latency.
- **Status:** LOCKED.

---

## ADR 2: ONNX Runtime Inference
- **Decision:** The deployment runtime uses `onnxruntime` and `tokenizers`, not PyTorch.
- **Reason:** PyTorch is too heavy (2GB+ footprint) and slow to initialize for edge daemons and CLI tools. ONNX provides native hardware acceleration across CPU/GPU with a tiny footprint.
- **Status:** LOCKED.

---

## ADR 3: Explanations via Template Mapping
- **Decision:** When an explanation (`reason`) is required, it is generated via deterministic template interpolation based on the winning choice and threshold scores, not via an LLM.
- **Reason:** Prevents hallucinations and ensures auditability in production environments.
- **Status:** LOCKED.

---

## ADR 4: Primary Model & Export Strategy
- **Decision:** The v0.1 baseline model is `MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli` (a pre-trained NLI cross-encoder). The export utility (`scripts/export_backbone.py`) MUST accept a `--model_id` argument to enable swapping to `ModernBERT-small`, `DeBERTa-v3-base`, or `DeBERTa-v3-large` without code changes.
- **Weights:** Do NOT use random weights. The export script downloads pre-trained weights from Hugging Face Hub, exports to ONNX via `optimum`, applies dynamic INT8 quantization, and stores final artifacts (`backbone.onnx`, `tokenizer.json`, `config.json`, `manifest.json`) in `models/`.
- **Dev vs. Runtime Boundary:** `torch`, `transformers`, and `optimum` are permitted ONLY in `scripts/`. They are hard-banned from `src/` (the inference runtime).
- **Status:** LOCKED.

---

## ADR 5: Dual-Path Classification Head Architecture
- **Primary Zero-Shot Path (v0.1 Default):**
  - The NLI model is exported via `AutoModelForSequenceClassification`. The 3-class NLI head (`contradiction=0`, `neutral=1`, `entailment=2`) resides inside the exported ONNX graph.
  - Evaluation maps state to Premise and query (predicate/criterion/options) to Hypothesis. The calibrated `entailment` logit (index 2) is read directly from the single forward pass output.
- **Pluggable Adapter Heads Path (future fine-tuning):**
  - `src/engine.py` exposes a method to extract the raw pooled `[CLS]` embedding `Z ∈ ℝ^d` from the backbone.
  - Task-specific custom heads run as decoupled NumPy linear projections (`Z @ W.T + b`) in `src/heads.py`, adding <10µs overhead.
- **Status:** LOCKED.

---

## ADR 6: Dynamic K for `choice` via Batched NLI Cross-Encoder
- **Decision:** The `choice` primitive does NOT use dot-product similarity over raw embeddings (which suffers from representation collapse). Instead:
  1. For K options, construct K input pairs: `[Premise: <serialized state> | Hypothesis: "This state corresponds to: {option_i}"]`.
  2. Batch all K pairs into a single dynamic batch (`batch_size = K`) and execute **one parallel ONNX forward pass**.
  3. Extract the entailment logit for each option: `[e_1, e_2, ..., e_K]`.
  4. Compute `Softmax([e_1, ..., e_K])` in NumPy to get a calibrated probability distribution and the top selection.
- **Reason:** Maintains full cross-attention fidelity between state tokens and option tokens. Executes within the <20ms budget on modern CPUs.
- **Status:** LOCKED.

---

## ADR 7: Dynamic INT8 Quantization for MVP
- **Decision:** Use Dynamic INT8 Quantization via `onnxruntime.quantization.quantize_dynamic` (targeting `MatMul` and `Gemm` operators) for the Phase 1 MVP.
- **Reason:** Requires zero calibration data, avoids activation clipping errors, achieves ~95% of static INT8 speedups on CPU vector extensions (AVX-512/VNNI/NEON), and provides an immediate working artifact.
- **Roadmap:** Static quantization with synthetic calibration data is deferred to Phase 4 benchmarking.
- **Status:** LOCKED.

---

## ADR 8: `gaman batch` Schema & Pipeline
- **Input Parsing:**
  - **Single-Column Mode:** `--state-column <col>` extracts only that column's content (string or JSON-parseable string) as the state dict.
  - **Full-Row Mode (Default):** When `--state-column` is omitted, all columns are serialized into a dictionary and passed through `src/serializer.py`.
- **Output Preservation:** Never drop input metadata (IDs, keys, timestamps). Mirror input format via `--output` file extension (`.csv` or `.jsonl`).
  - **CSV append columns:** `gaman_selection`/`gaman_confidence` (choice), `gaman_passed`/`gaman_probability` (noul), `gaman_score` (score), plus `gaman_latency_ms` for all.
  - **JSONL:** Augment the original JSON dict with a top-level `"gaman"` payload object.
- **Status:** LOCKED.

---

## ADR 009: Post-Hoc Temperature Scaling for Probability Calibration
- **Decision:** Use post-hoc scalar Temperature Scaling ($T > 0$) to calibrate confidence outputs across all three primitives (`choice`, `noul`, `score`). Optimize $T^*$ via 1D Golden Section Search on validation Negative Log-Likelihood (NLL). Measure calibration reliability using Expected Calibration Error (ECE) over $M=10$ equal-width bins.
- **Mathematical Invariant:** Temperature scaling is strictly rank-preserving: $\arg\max_k (z_{i,k} / T) = \arg\max_k (z_{i,k})$. The winning class prediction never flips, only epistemic confidence is softened or sharpened.
- **Runtime Isolation:** Calibration parameters are stored in `models/calibration.json`. The runtime engine (`src/engine.py`) loads primitive temperatures ($T=1.0$ neutral default) and applies vectorized scalar division on logits prior to Softmax/Sigmoid with $< 5\,\mu\text{s}$ overhead.
- **Dependencies:** Pure NumPy implementation in `src/calibration.py` with zero runtime dependencies.
- **CLI Workflow:** `gaman calibrate --data <val.csv> --primitive <choice|noul|score> --target-column <col> --state-column <col>` optimizes $T^*$, reports empirical ECE reduction, and writes to `models/calibration.json`.
- **Status:** LOCKED.

---

## ADR 010: Automated Hardware Spec Slab Resolution and Dynamic Tiering
- **Decision:** Implement zero-dependency automated hardware probing (`src/resolver.py`) and dynamic tier dispatch across three canonical slabs (`small`, `base`, `large`):
  1. **Tier 1 (Edge / CPU Fallback - `small`):** `cross-encoder/nli-deberta-v3-small` (INT8 quantized, ~140MB, 6 layers, 768 hidden dimension, target <20ms on CPU). Dispatched when available RAM < 16GB or running on CPU-only edge hardware.
  2. **Tier 2 (Pro Workstation / Server - `base`):** `cross-encoder/nli-deberta-v3-base` (FP16 or INT8, ~400MB, 12 layers, 768 hidden dimension, target <10ms on GPU/Apple Silicon). Dispatched when CoreML is present, or CUDA/ROCm GPU has $\ge 4\,\text{GB}$ VRAM, or host RAM $\ge 16\,\text{GB}$.
  3. **Tier 3 (Datacenter / Dedicated Accelerator - `large`):** `cross-encoder/nli-deberta-v3-large` (FP16, ~800MB, 24 layers, 1024 hidden dimension, target <15ms batch GPU). Dispatched when dedicated CUDA/ROCm VRAM $\ge 8\,\text{GB}$.
- **Backward Compatibility:** Model resolution checks `models/<tier>/backbone.onnx` first, falling back to flat `models/backbone.onnx` for the `small` tier to maintain 100% backward compatibility with v0.1 model deployments.
- **Dynamic Hidden Dimensions:** The engine dynamically sets `self.hidden_dim` (768 for small/base, 1024 for large) and shapes `embed(state)` representation vectors accordingly.
- **Decoupled Diagnostic Tooling:** `gaman info` probes hardware topology directly without instantiating ONNX runtime sessions, guaranteeing instantaneous (<50ms) execution.
- **Status:** LOCKED.

---

## ADR 011: Analytical Linear Adapters for Custom Domain Taxonomies
- **Decision:** Implement pluggable lightweight classification heads (`src/heads.py`) operating on frozen backbone representations ($Z \in \mathbb{R}^d$, where $d=768$ for small/base and $d=1024$ for large).
- **Mathematical Formulation:**
  - Forward Pass: $z = Z W^T + b$, followed by standard numerically stable Softmax:
    $$p_k = \frac{\exp(z_k - \max(z))}{\sum_j \exp(z_j - \max(z))}$$
  - Fitting Formulation: Closed-form $L_2$-regularized Ridge regression in augmented space $\tilde{Z} = [Z, \mathbf{1}] \in \mathbb{R}^{N \times (d+1)}$:
    $$\tilde{W}^* = (\tilde{Z}^T \tilde{Z} + \lambda I')^{-1} \tilde{Z}^T Y$$
    where $Y \in \{0, 1\}^{N \times K}$ is the one-hot target matrix and $I'_{d+1, d+1} = 0$ (unpenalized intercept).
  - Solver: Pure NumPy via `np.linalg.solve` with `np.linalg.pinv` fallback. Zero Scikit-Learn or PyTorch dependencies at runtime.
- **Performance Characteristics:**
  - Fitting Latency: $< 1.0\text{s}$ on CPU for $N \le 2,000$ samples (empirically $< 20\,\text{ms}$ on modern CPUs).
  - Inference Latency: $< 10\,\mu\text{s}$ forward pass overhead on top of the backbone embedding.
- **Storage & Artifacts:** Serialized as clean JSON at `models/heads/<name>.json` containing `name`, `classes`, `hidden_dim`, `weights`, and `bias`.
- **CLI Workflow:**
  - Fit: `gaman fit --data <data.csv> --state-column <col> --target-column <col> --name <head_name> [--l2-reg 1.0]`
  - Predict: `gaman predict --state '<json>' --head <head_name> [--json]`
- **Status:** LOCKED.