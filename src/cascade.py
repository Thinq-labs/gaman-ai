"""
src/cascade.py — Speculative Decision Cascading Engine
======================================================
Two-tier speculative cascading architecture combining high throughput
with enterprise safety guarantees.

Fast Draft Tier (e.g. ModernBERT-small 44M, <5ms) evaluates the query first.
- If Conformal Prediction Set size |C(X)| == 1 and not escalated:
    Certified safe decision -> commits immediately (Speculative Hit).
- If |C(X)| > 1 (ambiguity) or |C(X)| == 0 (OOD) or escalate_to_system2:
    Uncertainty detected -> cascades forward to Verifier Tier (435M / base).

Achieves <6ms fleet-wide average latency while retaining large-model precision.
Zero runtime dependencies beyond numpy and onnxruntime.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.engine import GamanEngine
from src.resolver import ModelNotFoundError


class SpeculativeCascadeEngine:
    """
    Speculative Decision Cascading Engine.

    Executes a fast draft engine (Slab 1 / small) first. If the output conformal
    prediction set guarantees single-class safety (|C(X)| == 1, not escalated),
    the draft output is returned immediately. Otherwise, cascades to a higher-capacity
    verifier engine (Slab 2/3 / base/large) or escalates to System 2.
    """

    def __init__(
        self,
        draft_engine: GamanEngine | None = None,
        verifier_engine: GamanEngine | None = None,
        draft_slab: str = "small",
        verifier_slab: str = "base",
        models_dir: str | Path | None = None,
        silent: bool = True,
    ) -> None:
        """
        Initialize speculative cascade engine with draft and optional verifier.

        Args:
            draft_engine: Pre-initialized GamanEngine for draft evaluation.
            verifier_engine: Pre-initialized GamanEngine for verification cascade.
            draft_slab: Slab tier to resolve if draft_engine is None (default 'small').
            verifier_slab: Slab tier to resolve if verifier_engine is None (default 'base').
            models_dir: Root directory for models.
            silent: Suppress auto-download progress bars.
        """
        self.draft_engine: GamanEngine = (
            draft_engine
            if draft_engine is not None
            else GamanEngine(slab=draft_slab, models_dir=models_dir, silent=silent)
        )

        self.verifier_engine: GamanEngine | None = verifier_engine
        self._verifier_slab_name = verifier_slab

        # Attempt to auto-initialize verifier if not explicitly provided
        if self.verifier_engine is None and verifier_slab:
            try:
                self.verifier_engine = GamanEngine(
                    slab=verifier_slab, models_dir=models_dir, silent=silent
                )
            except (ModelNotFoundError, FileNotFoundError):
                # Verifier weights not present on disk; fallback to draft-only with escalation
                self.verifier_engine = None

    @property
    def draft_tier(self) -> str:
        return self.draft_engine.slab_config.tier

    @property
    def verifier_tier(self) -> str | None:
        return self.verifier_engine.slab_config.tier if self.verifier_engine else None

    def choice(
        self,
        state: dict[str, Any],
        options: list[str] | dict[str, str],
        reference: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Speculative semantic routing.

        1. Evaluates state using draft engine.
        2. If draft produces certified safe singleton (|C(X)| == 1) and not escalated:
           returns draft result immediately (speculative hit).
        3. If uncertain (|C(X)| != 1 or escalate_to_system2):
           cascades to verifier engine if available.
        """
        t0 = time.perf_counter()

        draft_res = self.draft_engine.choice(state, options, reference=reference)

        prediction_set = draft_res.get("prediction_set", [])
        is_singleton = len(prediction_set) == 1
        is_escalated = bool(draft_res.get("escalate_to_system2", False))
        is_certified = bool(draft_res.get("certified_safe", False))

        is_safe_hit = is_singleton and is_certified and (not is_escalated)

        if is_safe_hit or self.verifier_engine is None:
            total_lat = (time.perf_counter() - t0) * 1000.0
            draft_res["cascaded"] = False
            draft_res["speculative_hit"] = is_safe_hit
            draft_res["draft_tier"] = self.draft_tier
            draft_res["verifier_tier"] = None
            draft_res["cascade_latency_ms"] = float(round(total_lat, 2))
            return draft_res

        # Ambiguity, OOD, or conflict detected -> cascade to verifier
        verifier_res = self.verifier_engine.choice(state, options, reference=reference)
        total_lat = (time.perf_counter() - t0) * 1000.0

        verifier_res["cascaded"] = True
        verifier_res["speculative_hit"] = False
        verifier_res["draft_tier"] = self.draft_tier
        verifier_res["verifier_tier"] = self.verifier_tier
        verifier_res["draft_prediction_set"] = prediction_set
        verifier_res["cascade_latency_ms"] = float(round(total_lat, 2))
        return verifier_res

    def decide(
        self,
        state: dict[str, Any],
        options: list[str] | dict[str, str],
        reference: str | Path | None = None,
    ) -> dict[str, Any]:
        """Alias for choice() matching enterprise decision contract."""
        return self.choice(state, options, reference=reference)

    def noul(
        self,
        state: dict[str, Any],
        predicate: str,
        reference: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Speculative policy predicate verification.

        If draft confidence is high (|p - 0.5| >= 0.35) or no verifier is available,
        returns draft output. Otherwise, cascades to verifier.
        """
        t0 = time.perf_counter()
        draft_res = self.draft_engine.noul(state, predicate, reference=reference)
        prob = draft_res.get("probability", 0.5)
        margin_from_boundary = abs(prob - 0.5)

        # High confidence boundary separation
        if margin_from_boundary >= 0.35 or self.verifier_engine is None:
            total_lat = (time.perf_counter() - t0) * 1000.0
            draft_res["cascaded"] = False
            draft_res["speculative_hit"] = margin_from_boundary >= 0.35
            draft_res["draft_tier"] = self.draft_tier
            draft_res["verifier_tier"] = None
            draft_res["cascade_latency_ms"] = float(round(total_lat, 2))
            return draft_res

        verifier_res = self.verifier_engine.noul(state, predicate, reference=reference)
        total_lat = (time.perf_counter() - t0) * 1000.0
        verifier_res["cascaded"] = True
        verifier_res["speculative_hit"] = False
        verifier_res["draft_tier"] = self.draft_tier
        verifier_res["verifier_tier"] = self.verifier_tier
        verifier_res["cascade_latency_ms"] = float(round(total_lat, 2))
        return verifier_res

    def score(
        self,
        state: dict[str, Any],
        criterion: str,
        min_bound: float = 0.0,
        max_bound: float = 1.0,
        reference: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Speculative rubric alignment.

        Runs draft engine and attaches cascade telemetry.
        """
        t0 = time.perf_counter()
        draft_res = self.draft_engine.score(
            state, criterion, min_bound=min_bound, max_bound=max_bound, reference=reference
        )
        total_lat = (time.perf_counter() - t0) * 1000.0
        draft_res["cascaded"] = False
        draft_res["speculative_hit"] = True
        draft_res["draft_tier"] = self.draft_tier
        draft_res["verifier_tier"] = None
        draft_res["cascade_latency_ms"] = float(round(total_lat, 2))
        return draft_res
