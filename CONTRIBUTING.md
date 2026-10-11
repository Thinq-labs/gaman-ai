# Contributing to Gaman AI

Thank you for your interest in contributing to **Gaman AI**! We are building a local-first, non-autoregressive "System One" decision engine optimized for sub-20ms edge inference.

We are a performance-obsessed, minimalist engineering community. We welcome contributions from systems engineers, ML practitioners, compiler enthusiasts, and open-source developers.

---

## The Core Philosophy (Karpathy Standard)

Before contributing code, please align with our core design principles:

1. **Simplicity First:** Write the minimum code that solves the problem. No speculative abstractions. No over-engineered classes for single-use functions. We are building a lightweight edge runtime, not a bloated enterprise framework.
2. **Zero Inference Bloat:** The inference runtime (`src/`) has zero dependencies on massive ML frameworks like `torch`, `transformers`, or `tensorflow`. Inference must rely strictly on `onnxruntime`, `tokenizers`, and `numpy`.
3. **Surgical Changes:** Touch only what you must. Do not rewrite adjacent code, reformat files, or "clean up" unrelated lines. If your changes create orphaned code, delete the orphans.
4. **Goal-Driven Execution:** Define your success criteria before you start typing. Write the test, make it pass, and verify it locally. Never submit a PR saying "I made the changes, let me know if it works."

---

## Quickstart: Local Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/thinq-labs/gaman-ai.git
cd gaman-ai
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install in Editable Mode with Dev Dependencies
```bash
pip install --upgrade pip
pip install -e .
pip install -r requirements-dev.txt
```

### 4. Obtain the Model Weights
To run tests locally, you need the lightweight `small` model weights in `models/small/`. You can export them locally or use the automated downloader:

```bash
# Export local INT8 model (requires torch/optimum in build environment):
python scripts/export_backbone.py --tier small
```

---

## The 3-Command Verification Gate

Before submitting any Pull Request, ensure that your changes pass all three checks cleanly:

```bash
# 1. Run full test suite (146+ tests across serialization, engine, calibration, CLI)
python -m pytest tests/

# 2. Strict static type checking
mypy src/

# 3. Code formatting and linting
ruff check src/
```

If all three commands exit with code `0`, your code meets our baseline quality standard!

---

## Where We Need Help (High-Impact Areas)

Whether you are an expert in low-level systems or applied ML, here are the frontiers where your contributions will have the highest impact:

### 1. Edge & Systems Runtimes
* **WebAssembly Target:** Exporting and running Gaman AI in the browser via `onnxruntime-web` or Rust Wasm.
* **Native Rust & C++ SDKs:** Zero-Python standalone binaries and bindings for native embedded software loops.
* **Apple Silicon & CoreML:** Profiling and optimizing graph fusion for Apple Neural Engine (`CoreMLExecutionProvider`).
* **DirectML / GPU Hardware Fusion:** Benchmarking and tuning execution providers across AMD ROCm and Intel Arc GPUs.

### 2. Agent Framework Connectors
* **LangChain / LangGraph:** Fast reflex tool selection and loop guardrails without LLM roundtrips.
* **LlamaIndex:** Sub-20ms semantic node routing and query triage.
* **CrewAI / AutoGen:** Low-latency inter-agent intent handoff and destructive action gates.

### 3. Mathematical Rigor & Calibration
* **Conformal Prediction Sets:** Implementing adaptive prediction sets for guaranteed coverage at a chosen significance level $\alpha$.
* **Vectorized Multi-Class Temperature Scaling:** Matrix-scaling optimization for larger decision spaces ($K > 50$).
* **Out-of-Distribution (OOD) Rejection:** Information-theoretic entropy thresholds to trigger System 2 fallbacks.

### 4. Synthetic Datasets & Domain Benchmarks
* **Stress Suites:** Benchmarks in edge robotics, financial fraud, database query routing, and real-time gaming control.
* **Adversarial Hardening:** Expanding resilience against token obfuscation, indirect prompt injection, and delimiter attacks.

---

## Repository Structure

```
gaman-ai/
├── src/                  # The inference runtime (STRICTLY NO torch / transformers)
│   ├── engine.py         # Core GamanEngine (choice, noul, score primitives)
│   ├── tokenizer.py      # Rust-backed offline fast BPE tokenizer
│   ├── serializer.py     # Deterministic JSON / dictionary flattening
│   ├── resolver.py       # Automated hardware probe & spec slab dispatcher
│   ├── calibration.py    # Temperature scaling & ECE minimization
│   ├── heads.py          # Pure NumPy analytical linear adapters
│   ├── downloader.py     # Atomic weight caching & validation
│   └── cli.py            # Universal CLI binary entrypoint (`gaman`)
├── scripts/              # Build, export, and benchmark utilities
│   ├── export_backbone.py# Hugging Face NLI -> ONNX INT8 exporter
│   ├── optimize_model.py # Graph fusions & constant folding
│   └── benchmark.py      # Microsecond latency & ECE measurement suite
├── tests/                # 146+ comprehensive unit & property tests
├── models/               # Local ONNX weights and tokenizer artifacts
└── docs/                 # Architecture Decision Records (ADRs) and specs
```

---

## Pull Request Guidelines

1. **Open an Issue First:** For major features or architectural shifts, open an issue or GitHub Discussion to discuss design trade-offs before writing code.
2. **Atomic Commits:** Keep commits focused, well-titled, and self-contained.
3. **Include Tests:** Every bug fix must include a regression test. Every new feature must include comprehensive unit tests in `tests/`.
4. **Preserve Architectural Boundaries:** Never touch `src/` dependencies to import heavy frameworks. If a feature requires PyTorch, it belongs in `scripts/` or an offline training tool.
5. **Update Documentation:** If your PR changes CLI arguments, configuration options, or APIs, update the corresponding docs and docstrings.

---

## Community & Code of Conduct

We are dedicated to providing a welcoming, inclusive, and harassment-free environment for everyone. Please be respectful, constructive in code reviews, and supportive of fellow contributors.

* **GitHub Discussions:** Ask questions, share ideas, and showcase projects.
* **Issues:** Report bugs, request features, or claim a "Good First Issue".

Thank you for helping build the future of fast, deterministic edge intelligence! 🚀
