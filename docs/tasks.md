# Gaman AI - Task Backlog

## v0.1 MVP - Baseline System One Engine
### Phase 1: Environment & Scaffolding
- [x] Set up Python project structure (`src/`, `tests/`, `models/`).
- [x] Create `requirements.txt` with `onnxruntime`, `tokenizers`, and `numpy`.
- [x] Write script `scripts/export_backbone.py` to export and quantize backbone to `models/`.

### Phase 2: Serialization & Tokenization
- [x] Implement `src/serializer.py`: Universal deterministic flattener for arbitrary nested JSON dictionaries.
- [x] Implement `src/tokenizer.py`: Local offline fast tokenizer with dynamic batch-max padding.
- [x] Write Pytest tests verifying exact serialization formats across deeply nested JSON.

### Phase 3: The Engine Runtime
- [x] Implement `src/engine.py`: Create the `GamanEngine` class.
- [x] Implement dynamic ONNX hardware dispatch (CPU/GPU fallback).
- [x] Implement `choice` primitive: Takes generic `state` (dict) and `options` (list of strings).
- [x] Implement `noul` primitive: Takes generic `state` (dict) and `predicate` (string).
- [x] Implement `score` primitive: Takes generic `state` (dict) and `criterion` (string).

### Phase 4: The Universal CLI
- [x] Implement `src/cli.py`: Expose the engine via `argparse` for single-shot commands (`gaman choice`, `gaman noul`, `gaman score`).
- [x] Implement `gaman batch`: Generic streaming processor for CSV/JSONL datasets with metadata preservation.

### Phase 5: Verification
- [x] Write integration tests verifying the engine across three distinct domains (security audit logs, customer chat support, source code).

---

## v0.2 - Pillar 1: Probability Calibration & Repo Hygiene
- [x] Modify `.gitignore`: Track `docs/` (contracts and ADRs); keep `agents.md`/`AGENTS.md` ignored.
- [x] Verify local git history and tags (`v0.1.0-mvp`).
- [x] Implement `src/calibration.py`:
  - [x] `compute_ece`: Expected Calibration Error over $M=10$ equal-width bins.
  - [x] `compute_nll`: Negative Log-Likelihood loss with temperature scaling for multiclass and binary logits.
  - [x] `fit_temperature`: 1D Golden Section Search optimization for $T^* \in [0.1, 10.0]$ with rank preservation invariant.
  - [x] `load_calibration` / `save_calibration`: Persistence in `models/calibration.json`.
- [x] Integrate temperature scaling into `src/engine.py` (`self.temp_choice`, `self.temp_noul`, `self.temp_score`).
- [x] Add CLI subcommand `gaman calibrate` in `src/cli.py` for dataset calibration and diagnostic reporting.
- [x] Write comprehensive unit & integration tests in `tests/test_calibration.py` and `tests/test_cli.py`.
- [x] Record ADR 009 in `docs/adr.md`.

---

## v0.2 - Pillar 2: Automated Spec Slab Resolver
- [x] Implement `src/resolver.py`:
  - [x] Hardware detection: `detect_hardware()`, `HardwareProfile` (providers, VRAM, RAM, CPU cores, AVX512/NEON).
  - [x] Slab definitions: `SlabConfig` for `small`, `base`, `large` tiers.
  - [x] Hierarchical tier resolution: `resolve_slab(target_tier="auto")` with strict fallback.
  - [x] Model path resolution: `resolve_model_path(tier, models_dir)` with backward compatibility for flat `models/` layout.
- [x] Parameterize `GamanEngine(slab="auto", models_dir="models")` in `src/engine.py`:
  - [x] Dynamic hardware provider dispatch based on resolved slab profile.
  - [x] Dynamic hidden dimension (`768` vs `1024`).
  - [x] Update `embed(state)` method to emit representation vector of shape `(hidden_dim,)`.
- [x] Implement CLI integration in `src/cli.py`:
  - [x] Global `--slab [auto|small|base|large]` flag across all commands.
  - [x] `gaman info` diagnostic subcommand with human-readable and `--json` outputs.
- [x] Write comprehensive unit & integration tests in `tests/test_resolver.py` and `tests/test_cli.py`.
- [x] Record ADR 010 in `docs/adr.md`.

---

## v0.2 - Pillar 3: Lightweight Adapters & Multi-Tier Model Export
- [x] Enhance `scripts/export_backbone.py` with `--tier [small|base|large]` multi-tier export tooling.
- [x] Implement friendly `ModelNotFoundError(FileNotFoundError)` guidance on missing model assets.
- [x] Implement `src/heads.py`:
  - [x] `CustomLinearHead` with pure NumPy forward pass and Softmax inference (< 10µs latency).
  - [x] `fit_adapter`: Closed-form analytical Ridge regression solver for $N \le 2,000$ samples in < 1.0s.
  - [x] JSON head serialization and deserialization at `models/heads/<name>.json`.
- [x] Integrate adapters into `src/engine.py`:
  - [x] Add `embed_batch` method with bit-for-bit consistency with `embed`.
  - [x] Add `predict(state, head_name)` method with automatic head loading and caching.
- [x] Integrate adapter workflows into `src/cli.py`:
  - [x] `gaman fit --data <csv> --state-column <col> --target-column <col> --name <name> [--l2-reg 1.0]`.
  - [x] `gaman predict --state '<json>' --head <name> [--json]`.
- [x] Write comprehensive unit and integration tests in `tests/test_heads.py` and `tests/test_cli.py`.
- [x] Record ADR 011 in `docs/adr.md`.

---

## v0.2 - Zero-Clone Pip Distribution & Auto-Weights Downloader
- [x] Implement standard OS cache discovery in `src/resolver.py`:
  - [x] `get_default_cache_dir()` checking `GAMAN_CACHE_DIR`, `%LOCALAPPDATA%/gaman`, and `$XDG_CACHE_HOME/gaman`.
  - [x] Model lookup hierarchy in `resolve_model_path`: Explicit > Local cwd (`./models`) > Global OS Cache.
- [x] Implement zero-dependency weights downloader in `src/downloader.py`:
  - [x] Pure `urllib.request` implementation with zero third-party packages.
  - [x] Atomic write via temporary files (`.tmp`) and `os.replace`.
  - [x] TTY-sensitive ASCII progress bar suppressed when non-interactive or under `--json`.
  - [x] `ensure_model_tier(tier, cache_dir, silent)` on-demand bootstrap.
- [x] Engine & CLI integration:
  - [x] `GamanEngine(models_dir=None)` automatically resolves or downloads weights on first run.
  - [x] `gaman cache --dir` and `gaman cache --clean` subcommands in `src/cli.py`.
  - [x] `gaman info` displays global cache path and status.
- [x] Write comprehensive tests in `tests/test_downloader.py` (10/10 passing).
- [x] Record ADR 012 in `docs/adr.md`.

---

## v0.2 - Pillar 4: Adversarial Hardening & Intent Expansion
- [x] Input Encapsulation & Delimiter Sandboxing:
  - [x] Wrap conversational & free-form natural language state in `[CONTEXT]: Document classification task... <payload> {sanitized_state} </payload>`.
  - [x] Neutralize imperative prompts, prompt injection, and jailbreak formatting.
- [x] Semantic Intent Expansion:
  - [x] Support dictionary options mapping choice keys to natural language intent definitions (`dict[str, str]`).
  - [x] Naturalize SCREAMING_SNAKE_CASE option labels and format hypothesis templates as authentic communicative intent statements.
- [x] Lexical Echo Dampening:
  - [x] Detect verbatim token matches in input state and compute divergence against neutral baseline to penalize superficial token matching.
- [x] Entropy-Based OOD Gating:
  - [x] Calculate normalized Shannon entropy $H(p) / \log K$. Flag `low_confidence = True` when normalized entropy exceeds 0.85 or when entailment logits are negative.
- [x] Comprehensive TDD Verification:
  - [x] Created `tests/test_adversarial.py` covering prompt hijack, roleplay jailbreak, token injection bait, gibberish token, and out-of-distribution queries (5/5 passing).
  - [x] All 161 tests passing across entire test suite with 0 regressions. Clean Ruff lint checks.