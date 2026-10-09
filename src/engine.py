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
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src.serializer import build_nli_pair, serialize_state
from src.tokenizer import GamanTokenizer

# Hardware dispatch priority per spec §4
HARDWARE_PROVIDERS_PRIORITY = [
    "CUDAExecutionProvider",
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
        models_dir: str | Path = "models",
        backbone_filename: str = "backbone.onnx",
    ) -> None:
        """
        Initialize the inference session and tokenizer from local models directory.

        Args:
            models_dir: Directory containing backbone.onnx, tokenizer.json, and config.json.
            backbone_filename: Name of the ONNX model file inside models_dir.
        """
        self.models_dir = Path(models_dir)
        self.model_path = self.models_dir / backbone_filename
        self.config_path = self.models_dir / "config.json"

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'.\n"
                "Run `python scripts/export_backbone.py` to export and quantize the model."
            )

        # 1. Load Model Config & Metadata dynamically
        self.config = self._load_config()
        self.hidden_dim: int = int(
            self.config.get("hidden_size", self.config.get("d_model", 768))
        )
        label2id: dict[str, int] = self.config.get("label2id", {})
        self.entailment_idx: int = label2id.get("entailment", 1)

        # 2. Hardware Dispatch
        self.active_provider, self.session = self._init_session()

        # 3. Offline Fast Tokenizer
        self.tokenizer = GamanTokenizer(models_dir=self.models_dir)

    def _load_config(self) -> dict[str, Any]:
        """Load model configuration dynamically."""
        if not self.config_path.exists():
            return {"hidden_size": 768, "label2id": {"entailment": 1}}
        with open(self.config_path, encoding="utf-8") as f:
            return json.load(f)

    def _init_session(self) -> tuple[str, ort.InferenceSession]:
        """Initialize ONNX Runtime InferenceSession with dynamic hardware dispatch."""
        available_providers = ort.get_available_providers()
        selected_providers = [
            p for p in HARDWARE_PROVIDERS_PRIORITY if p in available_providers
        ]
        if not selected_providers:
            selected_providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Optimize CPU threading for SIMD / vector extensions
        num_threads = os.cpu_count() or 4
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1

        session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=selected_providers,
        )
        active_provider = session.get_providers()[0]
        return active_provider, session

    def _forward(self, encodings: dict[str, np.ndarray]) -> np.ndarray:
        """Execute a single parallel forward pass over the input batch."""
        session_inputs = {i.name for i in self.session.get_inputs()}
        feed = {k: v for k, v in encodings.items() if k in session_inputs}
        outputs = self.session.run(None, feed)
        return outputs[0]  # logits: shape [batch_size, num_labels]

    # ── Primitives ────────────────────────────────────────────────────────────

    def choice(self, state: dict[str, Any], options: list[str]) -> dict[str, Any]:
        """
        Semantic Routing: Evaluates state against K options using batched NLI cross-encoding.

        Args:
            state: Arbitrary JSON dictionary representing context.
            options: List of discrete choice strings (K >= 1).

        Returns:
            dict matching Universal API Contract:
            {"primitive": "choice", "selection": str, "confidence": float, "latency_ms": float}
        """
        if not options:
            raise ValueError("options list cannot be empty")

        t0 = time.perf_counter()
        premise = serialize_state(state)
        pairs = [(premise, f"This state corresponds to: {opt}") for opt in options]

        # Batched tokenization with dynamic batch padding
        batch_enc = self.tokenizer.encode_batch(pairs)

        # Single parallel forward pass (batch_size = K)
        logits = self._forward(batch_enc)  # [K, num_labels]
        entailment_logits = logits[:, self.entailment_idx]

        # Softmax over options
        exp_logits = np.exp(entailment_logits - np.max(entailment_logits))
        probs = exp_logits / np.sum(exp_logits)

        best_idx = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "primitive": "choice",
            "selection": options[best_idx],
            "confidence": float(round(float(probs[best_idx]), 4)),
            "latency_ms": float(round(latency_ms, 2)),
        }

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
        premise, hypothesis = build_nli_pair(state, predicate)
        enc = self.tokenizer.encode(premise, pair=hypothesis)

        logits = self._forward(enc)  # [1, num_labels]
        logit = float(logits[0, self.entailment_idx])

        # Strict Sigmoid probability
        prob = 1.0 / (1.0 + float(np.exp(-logit)))
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

        logits = self._forward(enc)  # [1, num_labels]
        logit = float(logits[0, self.entailment_idx])

        # Bounded scalar [0.0, 1.0] via Sigmoid
        val = 1.0 / (1.0 + float(np.exp(-logit)))
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
            NumPy 1D array representing state evaluation output vector.
        """
        premise = serialize_state(state)
        enc = self.tokenizer.encode(premise)
        logits = self._forward(enc)
        return logits[0]
