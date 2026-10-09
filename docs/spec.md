# Technical Specification: Gaman AI (v0.1)

---

## 1. Core Architecture

Gaman AI is an encoder-only neural network inference engine. It does not predict next tokens. It computes probability distributions over input states via a single parallel forward pass through a bidirectional transformer.

---

## 2. The Machine Learning Pipeline

### 2.1 Backbone Model
- **v0.1 Default:** `MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli` (DeBERTa-v3-Small, NLI-fine-tuned)
- **Swappable via:** `--model_id` flag in `scripts/export_backbone.py`
- **Supported alternatives:** `microsoft/deberta-v3-base`, `microsoft/deberta-v3-large`, `answerdotai/ModernBERT-small`
- **Architecture:** Encoder-only transformer (bidirectional self-attention). No decoder. No KV-cache.

### 2.2 Execution Framework
- **Runtime:** `onnxruntime` exclusively.
- **HARD BAN:** `torch` and `tensorflow` are FORBIDDEN in `src/` (the inference runtime). They are permitted only in `scripts/` (export utilities).
- **Tokenization:** Hugging Face `tokenizers` library (Rust-backed, fast).

### 2.3 Precision
- **Deployed precision:** Dynamic INT8 quantization via `onnxruntime.quantization.quantize_dynamic`.
- **Operators quantized:** `MatMul`, `Gemm`.
- **Roadmap:** Static INT8 with calibration dataset deferred to Phase 4.

### 2.4 Input Serialization Format
Structured JSON states are flattened into a deterministic string by `src/serializer.py` before tokenization:

```
[STATE] key1: value1 | key2: value2 | nested.key: value3 [QUERY] <predicate / criterion / option text>
```

- Keys are sorted alphabetically for determinism.
- Nested dicts are flattened with dot-notation (`parent.child: value`).
- Lists are serialized as comma-separated values.
- The `[STATE]` and `[QUERY]` markers are literal strings acting as structural separators.

---

## 3. The Classification Heads (Dual-Path Architecture)

### 3.1 Primary Zero-Shot Path (v0.1 Default)
The NLI model is exported with its 3-class head baked into the ONNX graph.

**NLI Label Mapping (fixed for DeBERTa NLI checkpoint):**
| Index | Label | Meaning |
|-------|-------|---------|
| 0 | `contradiction` | The hypothesis contradicts the premise |
| 1 | `neutral` | The hypothesis is unrelated to the premise |
| 2 | `entailment` | The hypothesis follows from the premise |

All three primitives read the **entailment logit (index 2)** from the model output as the base confidence signal.

### 3.2 Primitive Execution Logic

**`choice` Head — Batched NLI Cross-Encoder:**
- For K options, build K premise-hypothesis pairs: `"[STATE] ... [QUERY] This state corresponds to: {option_i}"`
- Batch all K pairs (shape `[K, seq_len]`) into one ONNX forward pass.
- Extract entailment logits `[e_1, ..., e_K]`.
- Compute `Softmax([e_1, ..., e_K])` in NumPy.
- Output: `selection = options[argmax]`, `confidence = max(softmax_probs)`.

**`noul` Head — Single NLI Pass:**
- Build 1 premise-hypothesis pair: `"[STATE] ... [QUERY] {predicate}"`
- Single forward pass. Extract entailment logit → apply Sigmoid.
- Output: `passed = probability > 0.5`, `probability = sigmoid(entailment_logit)`.

**`score` Head — Single NLI Pass:**
- Build 1 premise-hypothesis pair: `"[STATE] ... [QUERY] {criterion}"`
- Single forward pass. Extract entailment logit → apply Sigmoid.
- Output: `value = sigmoid(entailment_logit)` ∈ `[0.0, 1.0]`.

### 3.3 Pluggable Adapter Heads Path (fine-tuning ready)
- `GamanEngine` exposes `embed(state: dict) -> np.ndarray` to extract raw `[CLS]` pooled embedding `Z ∈ ℝ^d`.
- Custom task heads live in `src/heads.py` as NumPy linear projections: `Z @ W.T + b`.
- Head weights stored as `.npy` files, loaded at engine initialization. Adds <10µs overhead.

---

## 4. Hardware Dispatch

The ONNX `InferenceSession` must dynamically probe available hardware in this priority order:

| Priority | Provider | Hardware |
|----------|----------|----------|
| 1 | `CUDAExecutionProvider` | Discrete NVIDIA GPU |
| 2 | `CoreMLExecutionProvider` | Apple Silicon (M-series) |
| 3 | `DirectMLExecutionProvider` | Windows / Integrated GPU |
| 4 | `CPUExecutionProvider` | Fallback (optimized threading, AVX-512/NEON) |

- Fallback to CPU must be **silent and automatic**. The caller must never see a crash due to missing hardware.
- CPU session must configure `intra_op_num_threads` to the physical core count for optimal SIMD utilization.

---

## 5. ONNX Artifact Layout (`models/`)

After running `scripts/export_backbone.py`, the `models/` directory contains:

```
models/
├── backbone.onnx          # INT8 quantized model (primary runtime artifact)
├── backbone_fp32.onnx     # FP32 model (only if --keep_fp32 flag used)
├── tokenizer.json         # HF fast tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── config.json            # Model config (num_labels, label2id, etc.)
├── spm.model              # SentencePiece vocab (DeBERTa)
└── manifest.json          # Export provenance (model_id, artifact list)
```

---

## 6. Output Schema (API Contract)

All primitives include `latency_ms` measured wall-clock from the start of `engine.choice/noul/score()` call to return.

### `choice`
```json
{"primitive": "choice", "selection": "<string>", "confidence": 0.941, "latency_ms": 14.1}
```

### `noul`
```json
{"primitive": "noul", "passed": true, "probability": 0.982, "latency_ms": 12.4}
```

### `score`
```json
{"primitive": "score", "value": 0.895, "latency_ms": 11.2}
```