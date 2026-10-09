"""Probability calibration and reliability diagnostics for Gaman AI.

Provides post-hoc temperature scaling, Expected Calibration Error (ECE),
and Negative Log-Likelihood (NLL) optimization for non-autoregressive decision models.
Zero runtime dependencies beyond numpy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CALIBRATION: dict[str, Any] = {
    "version": "0.2.0",
    "temperatures": {
        "choice": 1.0,
        "noul": 1.0,
        "score": 1.0,
    },
    "ece_before": None,
    "ece_after": None,
}


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) partitioned into equal-width confidence bins.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        confidences: 1D array of predicted class confidences in [0.0, 1.0].
        predictions: 1D array of predicted class labels.
        targets: 1D array of ground truth labels.
        n_bins: Number of equal-width bins (default: 10).

    Returns:
        float: Expected Calibration Error in [0.0, 1.0].
    """
    confidences = np.asarray(confidences, dtype=float)
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    n_samples = len(confidences)
    if n_samples == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for m in range(n_bins):
        bin_lower = bin_edges[m]
        bin_upper = bin_edges[m + 1]

        if m == 0:
            mask = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            mask = (confidences > bin_lower) & (confidences <= bin_upper)

        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(predictions[mask] == targets[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(round(ece, 6))


def compute_nll(
    logits: np.ndarray,
    targets: np.ndarray,
    temperature: float = 1.0,
) -> float:
    """
    Compute Negative Log-Likelihood (NLL) with temperature scaling.

    Args:
        logits: Unnormalized logits of shape [N, K] or [N] for binary classification.
        targets: Ground truth class indices in {0, ..., K-1}.
        temperature: Positive scalar temperature parameter (T > 0).

    Returns:
        float: Mean negative log-likelihood loss.
    """
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)

    if len(targets) == 0:
        return 0.0

    # Handle 1D binary logits by mapping to 2-class representation [0, z]
    if logits.ndim == 1:
        logits_2d = np.stack([np.zeros_like(logits), logits], axis=-1)
    else:
        logits_2d = logits

    temperature = max(float(temperature), 1e-6)
    scaled_logits = logits_2d / temperature

    # Numerically stable log-softmax using LogSumExp trick
    max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(scaled_logits - max_logits)
    log_sum_exp = max_logits + np.log(np.sum(exp_logits, axis=-1, keepdims=True))
    log_probs = scaled_logits - log_sum_exp

    n_samples = len(targets)
    target_log_probs = log_probs[np.arange(n_samples), targets]
    return float(-np.mean(target_log_probs))


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    bounds: tuple[float, float] = (0.1, 10.0),
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """
    Optimize scalar temperature T* via 1D Golden Section Search to minimize NLL loss.

    Strictly preserves prediction rank: argmax_k(z_k / T) == argmax_k(z_k).

    Args:
        logits: Unnormalized validation logits (shape [N, K] or [N]).
        targets: Ground truth validation labels (shape [N]).
        bounds: Optimization search interval [min_T, max_T] (default: (0.1, 10.0)).
        tol: Convergence tolerance for interval width.
        max_iter: Maximum search iterations.

    Returns:
        float: Optimized temperature T*.
    """
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)

    a, b = float(bounds[0]), float(bounds[1])
    # Golden ratio constant: (sqrt(5) - 1) / 2
    phi = (np.sqrt(5.0) - 1.0) / 2.0

    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = compute_nll(logits, targets, temperature=c)
    fd = compute_nll(logits, targets, temperature=d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = compute_nll(logits, targets, temperature=c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = compute_nll(logits, targets, temperature=d)

    opt_t = float((a + b) / 2.0)
    opt_t = float(np.clip(opt_t, bounds[0], bounds[1]))

    # Strict rank-preservation invariant assertion
    if logits.ndim > 1 and logits.shape[1] > 1:
        orig_preds = np.argmax(logits, axis=-1)
        scaled_preds = np.argmax(logits / opt_t, axis=-1)
        if not np.array_equal(orig_preds, scaled_preds):
            raise RuntimeError("Temperature scaling violated argmax rank preservation.")

    return float(round(opt_t, 4))


def load_calibration(path: str | Path) -> dict[str, Any]:
    """
    Load calibration configuration from JSON file.

    Falls back to default neutral calibration (T=1.0) if file does not exist.

    Args:
        path: Path to calibration.json file.

    Returns:
        dict containing version, temperatures, and ECE diagnostics.
    """
    p = Path(path)
    if not p.is_file():
        return {
            "version": DEFAULT_CALIBRATION["version"],
            "temperatures": dict(DEFAULT_CALIBRATION["temperatures"]),
            "ece_before": None,
            "ece_after": None,
        }

    with open(p, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    # Validate and ensure all default keys exist
    temps = dict(DEFAULT_CALIBRATION["temperatures"])
    if "temperatures" in data and isinstance(data["temperatures"], dict):
        for k, v in data["temperatures"].items():
            if isinstance(v, (int, float)):
                temps[k] = float(v)

    return {
        "version": str(data.get("version", DEFAULT_CALIBRATION["version"])),
        "temperatures": temps,
        "ece_before": data.get("ece_before"),
        "ece_after": data.get("ece_after"),
    }


def save_calibration(calibration_data: dict[str, Any], path: str | Path) -> None:
    """
    Save calibration parameters to JSON file.

    Args:
        calibration_data: Calibration dictionary matching schema.
        path: Destination file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2)
