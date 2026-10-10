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
    compute_decision_metadata,
    load_calibration,
    regularize_and_scale_logits,
)
from src.heads import CustomLinearHead
from src.resolver import ModelNotFoundError, SlabConfig, resolve_model_path, resolve_slab
from src.serializer import (
    build_nli_pair,
    sanitize_adversarial_input,
    serialize_state,
)
from src.tokenizer import GamanTokenizer

# Hardware dispatch priority per spec §4
HARDWARE_PROVIDERS_PRIORITY = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",
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
    ) -> None:
        """
        Initialize inference session and tokenizer with dynamic Spec Slab resolution.

        Args:
            slab: Spec Slab tier ('auto', 'small', 'base', 'large').
            models_dir: Root directory containing model artifacts (default None: auto-resolved).
            backbone_filename: Name of ONNX model file inside resolved tier folder.
            silent: Suppress progress outputs during automatic downloads.
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
        self.slab_config: SlabConfig = resolve_slab(requested_slab=slab)
        self.models_dir: Path = resolve_model_path(
            tier=self.slab_config.tier,
            models_dir=models_dir,
            auto_download=(models_dir is None),
            silent=silent,
        )
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

        # 2. Hardware Dispatch
        self.active_provider, self.session = self._init_session()

        # 3. Offline Fast Tokenizer
        self.tokenizer = GamanTokenizer(models_dir=self.models_dir)

        # 4. Explicit Warm-up to eliminate first-query scratchpad latency penalty
        self._warmup()

        # 5. Probability Calibration Parameters (T=1.0 default if uncalibrated)
        self.temp_choice: float = 1.0
        self.temp_noul: float = 1.0
        self.temp_score: float = 1.0
        self.load_calibration()

        # 6. Lightweight Linear Adapter Heads Cache
        self._loaded_heads: dict[str, CustomLinearHead] = {}

    def load_calibration(self, calibration_path: str | Path | None = None) -> None:
        """
        Load temperature scaling calibration parameters from calibration.json.
        If the file does not exist, defaults to neutral temperatures (T=1.0).
        """
        path = Path(calibration_path) if calibration_path else (self.models_dir / "calibration.json")
        cal_data = load_calibration(path)
        temps = cal_data.get("temperatures", {})
        self.temp_choice = float(temps.get("choice", 1.0))
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

        # Optimize CPU threading for SIMD / vector extensions.
        # Scale intra-op threads up to available physical/logical cores (cap at 8).
        num_threads = min(os.cpu_count() or 4, 8)
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1

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

    # ── Primitives ────────────────────────────────────────────────────────────

    def choice(
        self, state: dict[str, Any], options: list[str] | dict[str, str]
    ) -> dict[str, Any]:
        """
        Semantic Routing: Evaluates state against K options using batched NLI cross-encoding.
        Supports adversarial framing, delimiter sandboxing, semantic intent expansion,
        verbatim token dampening, and entropy-based OOD gating.

        Args:
            state: Arbitrary JSON dictionary representing context.
            options: List of choice strings or mapping of choice keys to semantic definitions (K >= 1).

        Returns:
            dict matching Universal API Contract:
            {"primitive": "choice", "selection": str, "confidence": float, "probabilities": dict, "latency_ms": float, ...}
        """
        if not options:
            raise ValueError("options list cannot be empty")

        t0 = time.perf_counter()

        # 1. Parse option keys and semantic descriptions
        if isinstance(options, dict):
            keys = list(options.keys())
            descriptions = [options[k] for k in keys]
            has_dict_options = True
        else:
            keys = list(options)
            descriptions = [k.replace("_", " ").lower() for k in keys]
            has_dict_options = False

        # 2. Pre-Tokenization Sanitization & Delimiter Sandboxing
        serialized = serialize_state(state)
        sanitized_state, zero_dampeners, figurative_modifiers = sanitize_adversarial_input(
            serialized
        )

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

        if is_text_payload:
            payload_content = sanitized_state
            if figurative_modifiers:
                payload_content += (
                    "\n[SEMANTIC CLARIFIER]: Figurative expressions and metaphorical modifiers "
                    "must not be interpreted as authentic transactional or business requests."
                )
            premise = (
                "[CONTEXT]: Intent classification.\n"
                "<payload>\n"
                f"{payload_content}\n"
                "</payload>"
            )
        else:
            premise = sanitized_state

        # 3. Semantic Intent Hypotheses
        hypotheses: list[str] = []
        for desc in descriptions:
            cleaned = desc.strip()
            if is_text_payload or has_dict_options or len(cleaned.split()) > 3 or cleaned.endswith("."):
                cleaned_body = cleaned.rstrip(".")
                hypotheses.append(f"The authentic primary intent of the message is {cleaned_body}.")
            else:
                hypotheses.append(f"The decision is to {cleaned}.")

        # 4. Vectorized Pair Construction: construct all K pairs in memory before tokenization
        pairs = [(premise, hyp) for hyp in hypotheses]

        # 5. Batched Tokenization & Single-Pass Inference
        batch_enc = self.tokenizer.encode_batch(pairs)
        logits = self._forward(batch_enc)  # [K, num_labels]
        entailment_logits = logits[:, self.entailment_idx].astype(np.float64)

        # 6. Quantitative Weight & Figurative Dampeners + Verbatim Echo Dampening
        penalized_logits = entailment_logits.copy()
        state_text = sanitized_state.lower()

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

        # Verbatim Lexical Echo Dampening
        for idx, k in enumerate(keys):
            raw_key_norm = k.lower().replace("_", " ")
            key_exact = k.lower()
            if (raw_key_norm in state_text or key_exact in state_text) and len(raw_key_norm) >= 3:
                neutral_premise = (
                    "[CONTEXT]: Document classification task. Analyze the authentic communicative intent of the enclosed message.\n"
                    "<payload>\n"
                    "The text mentions an administrative identifier or instruction without user intent.\n"
                    "</payload>"
                )
                neutral_enc = self.tokenizer.encode(neutral_premise, pair=hypotheses[idx])
                neutral_logit = float(self._forward(neutral_enc)[0, self.entailment_idx])

                divergence = float(entailment_logits[idx] - neutral_logit)
                if divergence > 0:
                    penalized_logits[idx] -= divergence * 0.75

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

        # 7. Logit Regularization & Temperature-Scaled Softmax
        probs = regularize_and_scale_logits(
            penalized_logits,
            temp=self.temp_choice,
            n_conflicts=n_conflicts,
        )

        # 8. Enterprise Decision Metadata & Confidence Tiering
        decision_meta = compute_decision_metadata(probs, keys)

        # Check for split percentages (e.g., 50% refund, 50% inquiry) or figurative modifiers
        pcts = [int(p) for p in re.findall(r"(\d+)\s*%", state_text)]
        is_split = (
            len(pcts) >= 2
            and all(0 < p < 100 for p in pcts)
            and (max(pcts) - min(pcts) <= 20)
        )
        if is_split or figurative_modifiers:
            decision_meta["escalate_to_system2"] = True
            decision_meta["tier"] = "LOW"

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
            "latency_ms": float(round(latency_ms, 2)),
        }
        if decision_meta["escalate_to_system2"]:
            result["low_confidence"] = True

        return result

    def decide(
        self, state: dict[str, Any], options: list[str] | dict[str, str]
    ) -> dict[str, Any]:
        """
        Alias for choice() providing the enterprise decision contract.
        """
        return self.choice(state, options)

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

    def noul(self, state: dict[str, Any], predicate: str) -> dict[str, Any]:
        """
        Predicate / Guardrail check: Evaluates if state satisfies a strict condition.

        Args:
            state: Arbitrary JSON dictionary representing context.
            predicate: Compliance / policy question string.

        Returns:
            dict matching Universal API Contract:
            {"primitive": "noul", "passed": bool, "probability": float, "latency_ms": float}
        """
        t0 = time.perf_counter()
        hypothesis = self._normalize_predicate(predicate)
        premise, hypothesis = build_nli_pair(state, hypothesis)
        enc = self.tokenizer.encode(premise, pair=hypothesis)

        logits = self._forward(enc)[0]  # [num_labels]

        # Shift-invariant 3-class normalized Softmax entailment probability
        scaled_logits = logits / self.temp_noul
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        prob = float(probs[self.entailment_idx])
        passed = bool(prob > 0.5)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "primitive": "noul",
            "passed": passed,
            "probability": float(round(prob, 4)),
            "latency_ms": float(round(latency_ms, 2)),
        }

    def score(self, state: dict[str, Any], criterion: str) -> dict[str, Any]:
        """
        Context Evaluator: Evaluates continuous scalar degree bounded between 0.0 and 1.0.

        Args:
            state: Arbitrary JSON dictionary representing context.
            criterion: Continuous evaluation criterion string.

        Returns:
            dict matching Universal API Contract:
            {"primitive": "score", "value": float, "latency_ms": float}
        """
        t0 = time.perf_counter()
        premise, hypothesis = build_nli_pair(state, criterion)
        enc = self.tokenizer.encode(premise, pair=hypothesis)

        logits = self._forward(enc)[0]  # [num_labels]

        # Shift-invariant 3-class normalized Softmax continuous evaluation
        scaled_logits = logits / self.temp_score
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        val = float(probs[self.entailment_idx])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "primitive": "score",
            "value": float(round(val, 4)),
            "latency_ms": float(round(latency_ms, 2)),
        }

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
