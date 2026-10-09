"""
src/heads.py — Lightweight Linear Adapters for Custom Domain Taxonomies
========================================================================
Analytical closed-form Ridge classification heads on top of frozen backbone
representations.

Zero runtime dependencies beyond numpy. Fitting executes in < 1.0s on CPU.
Inference overhead is < 10 microseconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CustomLinearHead:
    """
    Lightweight classification head on top of frozen representations (Z in R^d).

    Forward pass: z = Z @ W.T + b, followed by Softmax.
    """

    name: str
    classes: list[str]
    weights: np.ndarray  # Shape: (K, d)
    bias: np.ndarray  # Shape: (K,)
    hidden_dim: int

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float32)
        self.bias = np.asarray(self.bias, dtype=np.float32)
        k = len(self.classes)
        if self.weights.shape != (k, self.hidden_dim):
            raise ValueError(
                f"Weights shape {self.weights.shape} does not match (K={k}, d={self.hidden_dim})"
            )
        if self.bias.shape != (k,):
            raise ValueError(f"Bias shape {self.bias.shape} does not match (K={k},)")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Softmax probabilities for representation vectors.

        Args:
            X: Input vector with shape (d,) or 2D batch with shape (N, d).

        Returns:
            Probability array of shape (K,) for 1D input or (N, K) for 2D input.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        is_1d = X_arr.ndim == 1

        if is_1d:
            if X_arr.shape[0] != self.hidden_dim:
                raise ValueError(
                    f"Expected vector of shape ({self.hidden_dim},), got {X_arr.shape}"
                )
            logits = np.dot(self.weights, X_arr) + self.bias  # shape (K,)
            logits_max = np.max(logits)
            exp_logits = np.exp(logits - logits_max)
            return (exp_logits / np.sum(exp_logits)).astype(np.float32)

        if X_arr.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Expected matrix of shape (N, {self.hidden_dim}), got {X_arr.shape}"
            )
        logits = np.matmul(X_arr, self.weights.T) + self.bias  # shape (N, K)
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs.astype(np.float32)

    def predict(self, x: np.ndarray) -> dict[str, Any]:
        """
        Predict top class and full probability distribution for a single representation vector.

        Args:
            x: Input vector of shape (d,).

        Returns:
            dict containing selection, confidence, and probabilities mapping.
        """
        probs = self.predict_proba(x)
        top_idx = int(np.argmax(probs))
        selection = self.classes[top_idx]
        confidence = float(probs[top_idx])

        return {
            "selection": selection,
            "confidence": float(round(confidence, 4)),
            "probabilities": {
                cls: float(round(float(p), 4)) for cls, p in zip(self.classes, probs, strict=False)
            },
        }

    def save(self, filepath: str | Path) -> None:
        """Serialize adapter head to JSON on disk."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "classes": self.classes,
            "hidden_dim": self.hidden_dim,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> CustomLinearHead:
        """Load adapter head from JSON on disk."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Adapter head file not found at: {p}")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            name=data["name"],
            classes=data["classes"],
            weights=np.array(data["weights"], dtype=np.float32),
            bias=np.array(data["bias"], dtype=np.float32),
            hidden_dim=int(data["hidden_dim"]),
        )


def fit_adapter(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray,
    name: str = "custom_head",
    l2_reg: float = 1.0,
    classes: list[str] | None = None,
) -> CustomLinearHead:
    """
    Fit an L2-regularized linear classification head via closed-form Ridge regression.

    Solves:
        min_W,b || Z W^T + 1 b^T - Y ||_F^2 + lambda ||W||_F^2

    In closed-form using normal equations in pure NumPy.

    Args:
        embeddings: Representation matrix Z of shape (N, d).
        labels: Ground-truth target strings of length N.
        name: Name identifier for the head.
        l2_reg: L2 regularization strength lambda >= 0.
        classes: Optional fixed list of classes. If None, derived from unique labels.

    Returns:
        Fitted CustomLinearHead ready for inference.
    """
    Z = np.asarray(embeddings, dtype=np.float32)
    if Z.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array of shape (N, d), got shape {Z.shape}")

    N, d = Z.shape
    if len(labels) != N:
        raise ValueError(f"Number of labels ({len(labels)}) must match samples N={N}")

    target_classes = sorted(list(set(labels))) if classes is None else list(classes)

    k = len(target_classes)
    if k < 2:
        raise ValueError(f"Need at least 2 distinct classes to fit adapter, got {k}")

    cls_to_idx = {c: i for i, c in enumerate(target_classes)}

    # Build one-hot target matrix Y in R^(N x K)
    Y = np.zeros((N, k), dtype=np.float32)
    for i, lbl in enumerate(labels):
        if lbl not in cls_to_idx:
            raise ValueError(f"Label '{lbl}' not found in target classes {target_classes}")
        Y[i, cls_to_idx[lbl]] = 1.0

    # Augment Z with bias column: Z_tilde in R^(N x (d+1))
    ones_col = np.ones((N, 1), dtype=np.float32)
    Z_tilde = np.hstack([Z, ones_col])  # shape (N, d+1)

    # Regularization matrix: penalize weights, but not bias intercept
    reg = l2_reg * np.eye(d + 1, dtype=np.float32)
    reg[-1, -1] = 0.0

    # Normal equations: (Z_tilde^T Z_tilde + reg) W_tilde = Z_tilde^T Y
    A = np.matmul(Z_tilde.T, Z_tilde) + reg  # shape (d+1, d+1)
    B = np.matmul(Z_tilde.T, Y)  # shape (d+1, K)

    try:
        W_tilde = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W_tilde = np.matmul(np.linalg.pinv(A), B)

    # Decompose into weights W (K, d) and bias b (K,)
    W = W_tilde[:d, :].T.astype(np.float32)
    b = W_tilde[d, :].astype(np.float32)

    return CustomLinearHead(
        name=name,
        classes=target_classes,
        weights=W,
        bias=b,
        hidden_dim=d,
    )
