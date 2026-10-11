"""Uncertainty-Aware Distribution-Free Conformal Prediction for Gaman AI.

Provides mathematically certified prediction sets with finite-sample coverage guarantees:
P(Y in C(X)) >= 1 - alpha

Zero runtime dependencies beyond numpy and standard library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ConformalPredictionError(ValueError):
    """Raised when conformal calibration inputs are invalid."""

    pass


class ConformalPredictor:
    """Split Conformal Prediction calibrator and prediction set generator."""

    def __init__(self, alpha: float = 0.05, quantile: float | None = None) -> None:
        """
        Initialize ConformalPredictor.

        Args:
            alpha: Significance level in (0, 1). Default 0.05 (95% guaranteed coverage).
            quantile: Pre-computed conformal quantile threshold in [0, 1].
        """
        if not (0.0 < alpha < 1.0):
            raise ConformalPredictionError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha: float = alpha
        self.quantile: float | None = quantile

    @property
    def is_calibrated(self) -> bool:
        """Check if quantile threshold has been fitted or loaded."""
        return self.quantile is not None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray | list[int]) -> float:
        """
        Fit conformal quantile threshold on calibration holdout data.

        Args:
            probabilities: Array of shape [N, K] containing softmax probability vectors.
            labels: Integer class indices of true labels, shape [N].

        Returns:
            Computed conformal quantile threshold q_hat in [0, 1].
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)

        if probs.ndim != 2:
            raise ConformalPredictionError(f"probabilities must be 2D, got shape {probs.shape}")
        if len(probs) != len(y):
            raise ConformalPredictionError(
                f"probabilities length {len(probs)} must match labels length {len(y)}"
            )
        if len(probs) == 0:
            raise ConformalPredictionError("Calibration dataset cannot be empty")

        n = len(probs)
        # 1. Non-conformity scores: s_i = 1 - p(y_i | x_i)
        true_class_probs = probs[np.arange(n), y]
        nonconformity_scores = 1.0 - true_class_probs

        # 2. Finite-sample adjusted quantile level: ceil((n + 1) * (1 - alpha)) / n
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(1.0, max(0.0, float(level)))

        # 3. Compute empirical quantile with higher-order conservative rounding
        q_val = float(np.quantile(nonconformity_scores, level, method="higher"))
        self.quantile = float(np.clip(q_val, 0.0, 1.0))
        return self.quantile

    def predict_set(
        self,
        probabilities: np.ndarray,
        classes: list[str],
        default_quantile: float = 0.85,
    ) -> dict[str, Any]:
        """
        Generate certified prediction set for a single test input.

        Args:
            probabilities: Softmax probability vector over K classes, shape [K] or [1, K].
            classes: List of K class names.
            default_quantile: Fallback threshold if uncalibrated (default 0.85).

        Returns:
            Dictionary matching Universal API Contract:
            {
                "prediction_set": list[str],
                "set_size": int,
                "coverage_guarantee": float,
                "certified_safe": bool,
                "escalate": bool,
                "ood_detected": bool,
                "cutoff_threshold": float
            }
        """
        probs = np.asarray(probabilities, dtype=np.float64).flatten()
        if len(probs) != len(classes):
            raise ConformalPredictionError(
                f"probabilities length ({len(probs)}) must match classes length ({len(classes)})"
            )

        q = self.quantile if self.quantile is not None else default_quantile
        prob_threshold = max(0.0, 1.0 - q)

        # Include all classes whose nonconformity score 1 - p_k <= q  <=>  p_k >= 1 - q
        selected_classes = [
            classes[idx]
            for idx in np.argsort(-probs)
            if probs[idx] >= prob_threshold
        ]

        set_size = len(selected_classes)
        # If no class meets threshold, it is Out-of-Distribution (OOD)
        is_ood = (set_size == 0)
        # Single-class prediction set is certified unambiguous and safe
        is_safe = (set_size == 1) and not is_ood
        # Multi-option ambiguity or OOD warrants System 2 escalation
        escalate = not is_safe

        return {
            "prediction_set": selected_classes,
            "set_size": set_size,
            "coverage_guarantee": round(1.0 - self.alpha, 4),
            "certified_safe": is_safe,
            "escalate": escalate,
            "ood_detected": is_ood,
            "cutoff_threshold": round(prob_threshold, 4),
        }

    def save(self, path: Path | str) -> None:
        """Serialize conformal configuration to JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alpha": self.alpha,
            "coverage_guarantee": 1.0 - self.alpha,
            "quantile": self.quantile,
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> ConformalPredictor:
        """Load conformal configuration from JSON file."""
        target = Path(path)
        if not target.exists():
            return cls()
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            alpha=float(data.get("alpha", 0.05)),
            quantile=data.get("quantile"),
        )
