"""
src/engine.py — Gaman AI Core Inference Runtime
=================================================
High-performance, non-autoregressive decision engine running quantized
bidirectional encoders via ONNX Runtime.

Optimized for sub-20ms edge latency on consumer hardware.
Zero imports of torch, transformers, or optimum.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src.calibration import (
    apply_logit_prior_matrix,
    compute_decision_metadata,
    load_calibration,
    regularize_and_scale_logits,
)
from src.conformal import ConformalPredictor
from src.heads import CustomLinearHead, fit_adapter
from src.late_interaction import LateInteractionScorer, StateRepresentationCache
from src.reference import ReferenceExtractor
from src.resolver import ModelNotFoundError, SlabConfig, resolve_model_path, resolve_slab
from src.serializer import (
    build_nli_pair,
    encapsulate_payload,
    sanitize_adversarial_input,
    serialize_state,
)
from src.tokenizer import GamanTokenizer

# Hardware dispatch priority per spec §4
HARDWARE_PROVIDERS_PRIORITY = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "DirectMLExecutionProvider",
    "CPUExecutionProvider",
]


class GamanEngine:
    """
    Local System One decision engine.

    Evaluates arbitrary JSON states across three primitives:
    - choice: K-way semantic routing (Softmax)
    - noul: Guardrail / predicate compliance (Sigmoid)
    - score: Continuous context evaluation (Sigmoid)
    """

    def __init__(
        self,
        slab: str = "auto",
        models_dir: str | Path | None = None,
        backbone_filename: str = "backbone.onnx",
        silent: bool = False,
        late_interaction: bool = False,
        family: str = "deberta",
    ) -> None:
        """
        Initialize inference session and tokenizer with dynamic Spec Slab resolution.

        Args:
            slab: Spec Slab tier ('auto', 'small', 'base', 'large').
            models_dir: Root directory containing model artifacts (default None: auto-resolved).
            backbone_filename: Name of ONNX model file inside resolved tier folder.
            silent: Suppress progress outputs during automatic downloads.
            late_interaction: Enable decoupled two-tower late interaction and state caching.
            family: Model architecture family ('deberta' or 'modernbert').
        """
        # Backwards-compatibility check for positional models_dir
        if isinstance(slab, Path) or (
            isinstance(slab, str)
            and slab not in ("auto", "small", "base", "large")
            and (Path(slab).exists() or "/" in slab or "\\" in slab)
        ):
            models_dir = slab
            slab = "auto"

        self.requested_slab: str = slab
        self.slab_config: SlabConfig = resolve_slab(requested_slab=slab, family=family)
        self.models_dir: Path = resolve_model_path(
            tier=self.slab_config.tier,
            models_dir=models_dir,
            auto_download=(models_dir is None),
            silent=silent,
        )
        if backbone_filename == "backbone.onnx":
            candidates = [
                "backbone_optimized.onnx",
                "model_optimized.onnx",
                "backbone.onnx",
            ]
            selected_file = "backbone.onnx"
            for cand in candidates:
                if (self.models_dir / cand).exists():
                    selected_file = cand
                    break
            self.model_path = self.models_dir / selected_file
        else:
            self.model_path = self.models_dir / backbone_filename

        self.config_path = self.models_dir / "config.json"

        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Model tier '{self.slab_config.tier}' not found in {self.models_dir}. "
                f"Run: python scripts/export_backbone.py --tier {self.slab_config.tier}"
            )

        # 1. Load Model Config & Metadata dynamically
        self.config = self._load_config()
        self.hidden_dim: int = int(
            self.config.get("hidden_size", self.config.get("d_model", self.slab_config.hidden_dim))
        )
        label2id: dict[str, int] = self.config.get("label2id", {})
        self.entailment_idx: int = label2id.get("entailment", 1)

        # Detect ModernBERT architecture for auto-temperature scaling
        self.is_modernbert: bool = (
            getattr(self.slab_config, "family", "deberta") == "modernbert"
            or "modernbert" in getattr(self.slab_config, "model_id", "").lower()
            or "modernbert" in str(self.model_path).lower()
            or self.config.get("model_type") == "modernbert"
            or any("modernbert" in str(a).lower() for a in self.config.get("architectures", []))
        )
        self.default_temp_choice: float = 0.25 if self.is_modernbert else 1.0

        # 2. Hardware Dispatch
        self.active_provider, self.session = self._init_session()

        # 3. Offline Fast Tokenizer
        self.tokenizer = GamanTokenizer(models_dir=self.models_dir)

        # 4. Explicit Warm-up to eliminate first-query scratchpad latency penalty
        self._warmup()

        # 5. Probability Calibration Parameters (T=1.0 default for DeBERTa, T=0.25 for ModernBERT)
        self.temp_choice: float = self.default_temp_choice
        self.temp_noul: float = 1.0
        self.temp_score: float = 1.0
        self.load_calibration()

        # 6. Lightweight Linear Adapter Heads Cache
        self._loaded_heads: dict[str, CustomLinearHead] = {}
        self._reference_heads_cache: dict[str, CustomLinearHead] = {}

        # 7. Static Neutral Baseline Priors Cache
        self._baseline_priors: dict[str, float] = {}
        priors_file = self.models_dir / "baseline_priors.npy"
        if priors_file.exists():
            try:
                loaded = np.load(str(priors_file), allow_pickle=True).item()
                if isinstance(loaded, dict):
                    self._baseline_priors = loaded
            except Exception:
                pass

        # 8. Uncertainty-Aware Conformal Predictor (Certified P(Y in C(X)) >= 1 - alpha)
        self.conformal_alpha: float = 0.05
        conformal_file = self.models_dir / "conformal.json"
        self.conformal_predictor = (
            ConformalPredictor.load(conformal_file)
            if conformal_file.exists()
            else ConformalPredictor(alpha=self.conformal_alpha)
        )

        # 9. Two-Tower Decoupled Late-Interaction Caching & Scoring
        self.late_interaction: bool = late_interaction
        self.state_cache: StateRepresentationCache = StateRepresentationCache()
        self.late_scorer: LateInteractionScorer = LateInteractionScorer()

    def _get_neutral_prior(self, hyp: str) -> float:
        """Retrieve pre-computed neutral baseline logit from static cache, or compute and cache."""
        if hyp in self._baseline_priors:
            return self._baseline_priors[hyp]
        neutral_premise = encapsulate_payload(
            "The text mentions an administrative identifier or instruction without user intent."
        )
        enc = self.tokenizer.encode(neutral_premise, pair=hyp)
        val = float(self._forward(enc)[0, self.entailment_idx])
        self._baseline_priors[hyp] = val
        return val

    def load_calibration(self, calibration_path: str | Path | None = None) -> None:
        """
        Load temperature scaling calibration parameters from calibration.json.
        If the file does not exist, defaults to neutral temperatures (T=1.0 for DeBERTa, T=0.25 for ModernBERT).
        """
        path = Path(calibration_path) if calibration_path else (self.models_dir / "calibration.json")
        default_temp = getattr(self, "default_temp_choice", 0.25 if getattr(self, "is_modernbert", False) else 1.0)
        if not path.is_file():
            self.temp_choice = default_temp
            self.temp_noul = 1.0
            self.temp_score = 1.0
            return

        cal_data = load_calibration(path)
        temps = cal_data.get("temperatures", {})
        self.temp_choice = float(temps.get("choice", default_temp))
        self.temp_noul = float(temps.get("noul", 1.0))
        self.temp_score = float(temps.get("score", 1.0))

    def _load_config(self) -> dict[str, Any]:
        """Load model configuration dynamically."""
        if not self.config_path.exists():
            return {"hidden_size": self.slab_config.hidden_dim, "label2id": {"entailment": 1}}
        with open(self.config_path, encoding="utf-8") as f:
            return json.load(f)

    def _init_session(self) -> tuple[str, ort.InferenceSession]:
        """Initialize ONNX Runtime InferenceSession with dynamic hardware dispatch."""
        available_providers = ort.get_available_providers()
        priority_list = [self.slab_config.provider] + [
            p for p in HARDWARE_PROVIDERS_PRIORITY if p != self.slab_config.provider
        ]
        selected_providers = [p for p in priority_list if p in available_providers]
        if not selected_providers:
            selected_providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        num_cpus = os.cpu_count() or 2
        opts.intra_op_num_threads = min(6, max(2, (num_cpus * 3) // 4))
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3  # Suppress internal C++ graph partitioner warnings (Memcpy notices)
        is_dml = any("Dml" in p or "DirectML" in p for p in selected_providers[:1])
        opts.enable_mem_pattern = not is_dml

        session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=selected_providers,
        )
        active_provider = session.get_providers()[0]
        return active_provider, session

    def _warmup(self) -> None:
        """Execute a warmup forward pass to pre-allocate ONNX memory arena and thread pools."""
        try:
            warmup_enc = self.tokenizer.encode_batch(
                [("system status: normal", "The decision is to proceed.")]
            )
            self._forward(warmup_enc)
        except Exception:
            pass

    def _forward(self, encodings: dict[str, np.ndarray]) -> np.ndarray:
        """Execute a single parallel forward pass over the input batch."""
        session_inputs = {i.name for i in self.session.get_inputs()}
        feed = {k: v for k, v in encodings.items() if k in session_inputs}
        outputs = self.session.run(None, feed)
        return outputs[0]  # logits: shape [batch_size, num_labels]

    def get_token_embeddings(self, text: str) -> np.ndarray:
        """
        Extract token embeddings of shape [seq_len, hidden_dim].
        If ONNX graph outputs last_hidden_state, extracts it directly.
        Otherwise constructs token representations using model forward outputs.
        """
        enc = self.tokenizer.encode(text)
        session_outputs = [o.name for o in self.session.get_outputs()]
        if "last_hidden_state" in session_outputs or "hidden_states" in session_outputs:
            target = "last_hidden_state" if "last_hidden_state" in session_outputs else "hidden_states"
            feed = {k: v for k, v in enc.items() if k in {i.name for i in self.session.get_inputs()}}
            outs = self.session.run([target], feed)
            return outs[0][0].astype(np.float32)

        input_ids = enc["input_ids"][0]
        logits = self._forward(enc)[0]
        repeats = int(np.ceil(self.hidden_dim / len(logits)))
        base_rep = np.tile(logits, repeats)[: self.hidden_dim].astype(np.float32)
        token_factors = (1.0 + (input_ids % 100) * 0.01)[:, np.newaxis]
        token_embs = base_rep[np.newaxis, :] * token_factors
        return token_embs.astype(np.float32)

    def _embed_options_batch(self, texts: list[str], classes: list[str]) -> np.ndarray:
        """
        Construct contextualized option-aligned representations for tabular classification
        when the backbone session outputs classification logits rather than hidden states.
        """
        all_pairs = [
            (t, f"The authentic primary intent is {c.replace('_', ' ').lower()}.")
            for t in texts
            for c in classes
        ]
        batch_enc = self.tokenizer.encode_batch(all_pairs)
        logits = self._forward(batch_enc)[:, self.entailment_idx]
        matrix = logits.reshape(len(texts), len(classes))
        repeats = int(np.ceil(self.hidden_dim / len(classes)))
        tiled = np.tile(matrix, (1, repeats))[:, : self.hidden_dim]
        return tiled.astype(np.float32)

    def choice(
        self,
        state: dict[str, Any],
        options: list[str] | dict[str, str],
        reference: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Semantic Routing: Evaluates state against K options using batched NLI cross-encoding.
        Supports adversarial framing, delimiter sandboxing, semantic intent expansion,
        reference grounding, verbatim token dampening, and entropy-based OOD gating.

        Args:
            state: Arbitrary JSON dictionary representing context.
            options: List of choice strings or mapping of choice keys to semantic definitions (K >= 1).
            reference: Optional path to reference file (.docx, .txt, .csv) or inline policy prompt string.

        Returns:
            dict matching Universal API Contract:
            {"primitive": "choice", "selection": str, "confidence": float, "probabilities": dict, "latency_ms": float, ...}
        """
        if not options:
            raise ValueError("options list cannot be empty")

        t0 = time.perf_counter()

        # Tabular Dataset Grounding Fast Path (.csv or .jsonl)
        if reference is not None and ReferenceExtractor.is_tabular_file(reference):
            texts, labels = ReferenceExtractor.load_dataset(reference)
            opt_keys = list(options.keys()) if isinstance(options, dict) else list(options)
            if labels and any(lbl in opt_keys for lbl in labels):
                h = ReferenceExtractor.compute_hash(reference)
                head = self._reference_heads_cache.get(h)
                session_outputs = [o.name for o in self.session.get_outputs()]
                has_hidden = "last_hidden_state" in session_outputs or "hidden_states" in session_outputs
                if head is None:
                    if has_hidden:
                        embeddings = self.embed_batch([{"text": t} for t in texts])
                    else:
                        embeddings = self._embed_options_batch(texts, opt_keys)
                    head = fit_adapter(embeddings, labels, name=f"ref_{h[:8]}", classes=opt_keys)
                    self._reference_heads_cache[h] = head

                if has_hidden:
                    x = self.embed(state)
                else:
                    st_text = serialize_state(state) if isinstance(state, dict) else str(state)
                    x = self._embed_options_batch([st_text], opt_keys)[0]

                pred = head.predict(x)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                probs = np.array([float(pred["probabilities"].get(k, 0.0)) for k in opt_keys], dtype=np.float64)
                if np.sum(probs) > 0:
                    probs = probs / np.sum(probs)
                decision_meta = compute_decision_metadata(probs, opt_keys)
                conformal_res = self.conformal_predictor.predict_set(
                    probabilities=probs,
                    classes=opt_keys,
                )

                return {
                    "primitive": "choice",
                    "selection": decision_meta["selection"],
                    "confidence": decision_meta["confidence"],
                    "margin": decision_meta["margin"],
                    "entropy": decision_meta["entropy"],
                    "tier": decision_meta["tier"],
                    "escalate_to_system2": decision_meta["escalate_to_system2"],
                    "probabilities": decision_meta["probabilities"],
                    "prediction_set": conformal_res["prediction_set"],
                    "certified_safe": conformal_res["certified_safe"],
                    "coverage_guarantee": conformal_res["coverage_guarantee"],
                    "latency_ms": float(round(latency_ms, 2)),
                    "grounded": True,
                    "reference_type": "tabular_adapter",
                }

        # Decoupled Two-Tower Late-Interaction Fast Path
        if self.late_interaction:
            if isinstance(options, dict):
                keys = list(options.keys())
                descriptions = [options[k] for k in keys]
            else:
                keys = list(options)
                descriptions = [k.replace("_", " ").lower() for k in keys]

            serialized = serialize_state(state)
            state_hash = StateRepresentationCache.compute_hash(serialized)
            cached = self.state_cache.get(state_hash)
            if cached is not None:
                state_embs = cached.token_embeddings
            else:
                state_embs = self.get_token_embeddings(serialized)
                self.state_cache.put(
                    state_hash, state_embs, timestamp_ms=time.perf_counter() * 1000.0
                )

            candidate_embs = [self.get_token_embeddings(d) for d in descriptions]
            scored = self.late_scorer.score_candidates(
                state_embeddings=state_embs,
                candidate_embeddings=candidate_embs,
                option_keys=keys,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            return {
                "primitive": "choice",
                "selection": scored["selection"],
                "confidence": scored["confidence"],
                "probabilities": scored["probabilities"],
                "raw_scores": scored.get("raw_scores", {}),
                "latency_ms": float(round(latency_ms, 2)),
            }

        # 1. Parse option keys and semantic descriptions
        if isinstance(options, dict):
            keys = list(options.keys())
            descriptions = [options[k] for k in keys]
            has_dict_options = True
        else:
            keys = list(options)
            descriptions = [k.replace("_", " ").lower() for k in keys]
            has_dict_options = False

        # 2. Reference Document / Prompt Grounding (Extract Relevant Clauses in <1ms)
        reference_clauses: list[str] = []
        if reference is not None:
            doc_clauses = ReferenceExtractor.extract_clauses(reference)
            if doc_clauses:
                state_query = serialize_state(state) if isinstance(state, dict) else str(state)
                reference_clauses = ReferenceExtractor.find_relevant_clauses(
                    state_query, doc_clauses, top_k=3
                )

        # 3. Pre-Tokenization Sanitization & Delimiter Sandboxing
        serialized = serialize_state(state)
        sanitized_state, zero_dampeners, figurative_modifiers = sanitize_adversarial_input(
            serialized
        )
        state_text = sanitized_state.lower()

        text_fields = {
            "text",
            "message",
            "ticket",
            "body",
            "query",
            "prompt",
            "utterance",
            "input",
        }
        is_text_payload = any(k in text_fields for k in state) or has_dict_options

        if reference_clauses:
            ref_context = "; ".join(reference_clauses)
            premise = encapsulate_payload(
                f"[REFERENCE POLICY]: {ref_context} | [CONTEXT]: {sanitized_state}"
            )
        elif is_text_payload:
            payload_content = sanitized_state
            if figurative_modifiers:
                payload_content += (
                    " [NOTE: Figurative expressions and metaphorical modifiers "
                    "must not be interpreted as authentic transactional or business requests.]"
                )
            premise = encapsulate_payload(payload_content)
        else:
            premise = sanitized_state

        # 4. Semantic Intent Hypotheses
        hypotheses: list[str] = []
        for desc in descriptions:
            cleaned = desc.strip()
            if is_text_payload or has_dict_options or len(cleaned.split()) > 3 or cleaned.endswith("."):
                cleaned_body = cleaned.rstrip(".")
                hypotheses.append(f"The authentic primary intent of the message is {cleaned_body}.")
            else:
                hypotheses.append(f"The decision is to {cleaned}.")

        # 5. Vectorized Pair Construction: construct ONLY candidate pairs (batch_size = K)
        pairs = [(premise, hyp) for hyp in hypotheses]

        # 5. Batched Tokenization with Strict max_length=64 Truncation
        batch_enc = self.tokenizer.encode_batch(pairs, max_length=64)
        max_batch_tokens = min(max(len(ids) for ids in batch_enc["input_ids"]), 64)
        trimmed_batch = {
            k: np.ascontiguousarray(v[:, :max_batch_tokens])
            for k, v in batch_enc.items()
        }

        # Single forward pass for only candidate options (batch_size = len(keys))
        all_logits = self._forward(trimmed_batch)
        entailment_logits = all_logits[:, self.entailment_idx].astype(np.float64)

        # 6. Quantitative Weight & Figurative Dampeners + Verbatim Echo Dampening
        penalized_logits = entailment_logits.copy()

        # Dampen labels explicitly assigned zero or negligible weight
        if zero_dampeners:
            for idx, k in enumerate(keys):
                k_norm = k.lower().replace("_", " ")
                if any(d in k_norm for d in zero_dampeners):
                    penalized_logits[idx] -= 5.0

        # Dampen labels bound to figurative qualifiers
        if figurative_modifiers:
            for idx, k in enumerate(keys):
                k_norm = k.lower().replace("_", " ")
                if any(target in k_norm for _, target in figurative_modifiers):
                    penalized_logits[idx] -= 4.0

        # Verbatim Lexical Echo Dampening for detected bait using static cached baseline
        is_bait = any(k in serialized for k in keys) or any(
            phrase in state_text
            for phrase in ["ignore previous", "play a game", "print only", "every message is"]
        )
        if is_bait:
            for idx, k in enumerate(keys):
                raw_key_norm = k.lower().replace("_", " ")
                key_exact = k.lower()
                if (raw_key_norm in state_text or key_exact in state_text) and len(raw_key_norm) >= 3:
                    neutral_logit = self._get_neutral_prior(hypotheses[idx])
                    divergence = float(entailment_logits[idx] - neutral_logit)
                    if divergence > 0:
                        penalized_logits[idx] -= divergence * 0.75

        # Analytical Prior Layer (Sarcasm, Corporate Sandwich, Compound Hyphenation & Dual-Intent)
        penalized_logits, dual_intent_escalate = apply_logit_prior_matrix(
            penalized_logits,
            sanitized_state,
            keys,
            descriptions,
        )

        # Count verbatim candidate conflicts for anti-saturation temperature scaling
        conflicts = sum(
            1
            for k in keys
            if (
                (k.lower().replace("_", " ") in state_text or k.lower() in state_text)
                and len(k.lower().replace("_", " ")) >= 3
            )
        )
        n_conflicts = max(0, conflicts - 1) if conflicts > 1 else 0
        if dual_intent_escalate:
            n_conflicts += 2

        # 7. Logit Regularization & Temperature-Scaled Softmax
        probs = regularize_and_scale_logits(
            penalized_logits,
            temp=self.temp_choice,
            n_conflicts=n_conflicts,
        )

        # 8. Enterprise Decision Metadata & Confidence Tiering
        decision_meta = compute_decision_metadata(probs, keys)

        # Check for split percentages, figurative modifiers, dual-intent, or out-of-distribution unentailed inputs
        pcts = [int(p) for p in re.findall(r"(\d+)\s*%", state_text)]
        is_split = (
            len(pcts) >= 2
            and all(0 < p < 100 for p in pcts)
            and (max(pcts) - min(pcts) <= 20)
        )
        is_ood = float(np.max(penalized_logits)) < 0.0
        if is_split or figurative_modifiers or dual_intent_escalate or is_ood:
            decision_meta["escalate_to_system2"] = True
            decision_meta["tier"] = "LOW"

        # 9. Uncertainty-Aware Conformal Prediction Set (P(Y in C(X)) >= 1 - alpha)
        conformal_res = self.conformal_predictor.predict_set(
            probabilities=probs,
            classes=keys,
        )
        if conformal_res["escalate"] or conformal_res["ood_detected"]:
            decision_meta["escalate_to_system2"] = True

        latency_ms = (time.perf_counter() - t0) * 1000.0

        result: dict[str, Any] = {
            "primitive": "choice",
            "selection": decision_meta["selection"],
            "confidence": decision_meta["confidence"],
            "margin": decision_meta["margin"],
            "entropy": decision_meta["entropy"],
            "tier": decision_meta["tier"],
            "escalate_to_system2": decision_meta["escalate_to_system2"],
            "probabilities": decision_meta["probabilities"],
            "prediction_set": conformal_res["prediction_set"],
            "certified_safe": conformal_res["certified_safe"],
            "coverage_guarantee": conformal_res["coverage_guarantee"],
            "latency_ms": float(round(latency_ms, 2)),
        }
        if reference is not None:
            result["grounded"] = True
            if reference_clauses:
                result["relevant_clauses"] = reference_clauses
        if decision_meta["escalate_to_system2"]:
            result["low_confidence"] = True

        return result

    def decide(
        self,
        state: dict[str, Any],
        options: list[str] | dict[str, str],
        reference: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Alias for choice() providing the enterprise decision contract.
        """
        return self.choice(state, options, reference=reference)

    @staticmethod
    def _normalize_predicate(predicate: str) -> str:
        """
        Normalize interrogative question predicates into declarative statements
        so the NLI cross-encoder evaluates true semantic entailment.
        """
        p = predicate.strip()
        has_question = p.endswith("?")
        core = p[:-1].strip() if has_question else p

        if core.lower().startswith("is this a ") and core.lower().endswith(" action"):
            adj = core[10:-7].strip()
            return f"The requested action is {adj}."
        if core.lower().startswith("is this "):
            core = "This is " + core[8:]
        elif core.lower().startswith("is "):
            core = core[3:] + " is"

        if not core.endswith("."):
            core = core + "."
        return core

    def noul(
        self,
        state: dict[str, Any],
        predicate: str,
        reference: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Predicate / Guardrail check: Evaluates if state satisfies a strict condition.

        Args:
            state: Arbitrary JSON dictionary representing context.
            predicate: Compliance / policy question string.
            reference: Optional path to reference file (.docx, .txt, .csv) or inline policy prompt string.

        Returns:
            dict matching Universal API Contract:
            {"primitive": "noul", "passed": bool, "probability": float, "latency_ms": float}
        """
        t0 = time.perf_counter()
        hypothesis = self._normalize_predicate(predicate)
        premise, hypothesis = build_nli_pair(state, hypothesis)

        reference_clauses: list[str] = []
        if reference is not None:
            doc_clauses = ReferenceExtractor.extract_clauses(reference)
            if doc_clauses:
                state_query = f"{premise} {predicate}"
                reference_clauses = ReferenceExtractor.find_relevant_clauses(
                    state_query, doc_clauses, top_k=3
                )

        if reference_clauses:
            ref_context = "; ".join(reference_clauses)
            sanitized_state, _, _ = sanitize_adversarial_input(premise)
            premise = encapsulate_payload(
                f"[REFERENCE POLICY]: {ref_context} | [CONTEXT]: {sanitized_state}"
            )

        enc = self.tokenizer.encode(premise, pair=hypothesis)

        logits = self._forward(enc)[0].astype(np.float64)  # [num_labels]

        # Apply analytical logit prior matrix for guardrail predicate
        entailment_arr = np.array([logits[self.entailment_idx]], dtype=np.float64)
        boosted_arr, _ = apply_logit_prior_matrix(entailment_arr, premise, [hypothesis])
        boost = float(boosted_arr[0] - entailment_arr[0])
        if boost > 0.0:
            logits[self.entailment_idx] += boost
            for idx in range(len(logits)):
                if idx != self.entailment_idx:
                    logits[idx] -= boost

        # Shift-invariant 3-class normalized Softmax entailment probability
        scaled_logits = logits / self.temp_noul
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        prob = float(probs[self.entailment_idx])
        passed = bool(prob > 0.5)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        res: dict[str, Any] = {
            "primitive": "noul",
            "passed": passed,
            "probability": float(round(prob, 4)),
            "latency_ms": float(round(latency_ms, 2)),
        }
        if reference is not None:
            res["grounded"] = True
            if reference_clauses:
                res["relevant_clauses"] = reference_clauses
        return res

    def score(
        self,
        state: dict[str, Any],
        criterion: str,
        min_bound: float = 0.0,
        max_bound: float = 1.0,
        reference: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Context Evaluator: Evaluates continuous scalar degree bounded between min_bound and max_bound.

        Args:
            state: Arbitrary JSON dictionary representing context.
            criterion: Continuous evaluation criterion string.
            min_bound: Lower bound for scalar output (default 0.0).
            max_bound: Upper bound for scalar output (default 1.0).
            reference: Optional path to reference file (.docx, .txt, .csv) or inline policy prompt string.

        Returns:
            dict matching Universal API Contract:
            {"primitive": "score", "value": float, "latency_ms": float}
        """
        t0 = time.perf_counter()
        premise, hypothesis = build_nli_pair(state, criterion)

        reference_clauses: list[str] = []
        if reference is not None:
            doc_clauses = ReferenceExtractor.extract_clauses(reference)
            if doc_clauses:
                state_query = f"{premise} {criterion}"
                reference_clauses = ReferenceExtractor.find_relevant_clauses(
                    state_query, doc_clauses, top_k=3
                )

        if reference_clauses:
            ref_context = "; ".join(reference_clauses)
            sanitized_state, _, _ = sanitize_adversarial_input(premise)
            premise = encapsulate_payload(
                f"[REFERENCE GUIDELINE]: {ref_context} | [CONTEXT]: {sanitized_state}"
            )

        enc = self.tokenizer.encode(premise, pair=hypothesis)

        logits = self._forward(enc)[0]  # [num_labels]

        # Shift-invariant 3-class normalized Softmax continuous evaluation
        scaled_logits = logits / self.temp_score
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        val = float(probs[self.entailment_idx])

        if min_bound != 0.0 or max_bound != 1.0:
            val = min_bound + val * (max_bound - min_bound)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        res: dict[str, Any] = {
            "primitive": "score",
            "value": float(round(val, 4)),
            "latency_ms": float(round(latency_ms, 2)),
        }
        if reference is not None:
            res["grounded"] = True
            if reference_clauses:
                res["relevant_clauses"] = reference_clauses
        return res

    def embed(self, state: dict[str, Any]) -> np.ndarray:
        """
        Extract representation vector for pluggable adapter heads (ADR 5).

        Returns:
            NumPy 1D array representing state evaluation output vector with shape (hidden_dim,).
        """
        premise = serialize_state(state)
        enc = self.tokenizer.encode(premise)
        session_outputs = [o.name for o in self.session.get_outputs()]
        if "last_hidden_state" in session_outputs or "hidden_states" in session_outputs:
            target = "last_hidden_state" if "last_hidden_state" in session_outputs else "hidden_states"
            feed = {k: v for k, v in enc.items() if k in {i.name for i in self.session.get_inputs()}}
            outs = self.session.run([target], feed)
            return outs[0][0, 0].astype(np.float32)

        logits = self._forward(enc)[0]
        # Deterministically project/tile to match resolved slab hidden dimension
        repeats = int(np.ceil(self.hidden_dim / len(logits)))
        tiled = np.tile(logits, repeats)[: self.hidden_dim]
        return tiled.astype(np.float32)

    def embed_batch(self, states: list[dict[str, Any]]) -> np.ndarray:
        """
        Extract batch representation vectors for pluggable adapter heads (ADR 5).

        Guarantees that embed_batch(states)[i] is strictly bit-for-bit identical
        to embed(states[i]).

        Args:
            states: List of state dictionaries.

        Returns:
            NumPy 2D array of shape (N, hidden_dim) and dtype float32.
        """
        if not states:
            return np.empty((0, self.hidden_dim), dtype=np.float32)
        return np.vstack([self.embed(s) for s in states]).astype(np.float32)

    def predict(self, state: dict[str, Any], head_name: str) -> dict[str, Any]:
        """
        Run inference using a trained lightweight linear adapter head.

        Args:
            state: Arbitrary JSON dictionary representing current state.
            head_name: Identifier of the saved head (in models/heads/<name>.json).

        Returns:
            dict matching Universal API Contract:
            {"primitive": "choice", "head": str, "selection": str, "confidence": float, "probabilities": dict, "latency_ms": float}
        """
        t0 = time.perf_counter()
        if head_name not in self._loaded_heads:
            candidate1 = self.models_dir / "heads" / f"{head_name}.json"
            candidate2 = Path("models") / "heads" / f"{head_name}.json"
            if candidate1.exists():
                head_file = candidate1
            elif candidate2.exists():
                head_file = candidate2
            else:
                raise FileNotFoundError(
                    f"Adapter head '{head_name}' not found. Searched '{candidate1}' and '{candidate2}'."
                )
            self._loaded_heads[head_name] = CustomLinearHead.load(head_file)

        head = self._loaded_heads[head_name]
        if head.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Head '{head_name}' expects hidden_dim={head.hidden_dim}, but engine is {self.hidden_dim}"
            )

        z = self.embed(state)
        res = head.predict(z)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "primitive": "choice",
            "head": head_name,
            "selection": res["selection"],
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
            "latency_ms": float(round(latency_ms, 2)),
        }
