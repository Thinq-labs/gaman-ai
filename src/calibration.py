"""Probability calibration and reliability diagnostics for Gaman AI.

Provides post-hoc temperature scaling, Expected Calibration Error (ECE),
and Negative Log-Likelihood (NLL) optimization for non-autoregressive decision models.
Zero runtime dependencies beyond numpy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("gaman.calibration")

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
    strategy: str = "equal_width",
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Supports:
    - 'equal_width': M equal-width confidence intervals over [0.0, 1.0].
    - 'quantile' / 'equal_mass': M adaptive quantile intervals containing ~N/M samples.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        confidences: 1D array of predicted class confidences in [0.0, 1.0].
        predictions: 1D array of predicted class labels.
        targets: 1D array of ground truth labels.
        n_bins: Number of bins (default: 10).
        strategy: 'equal_width' or 'quantile' / 'equal_mass'.

    Returns:
        float: Expected Calibration Error in [0.0, 1.0].
    """
    confidences = np.asarray(confidences, dtype=float)
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    n_samples = len(confidences)
    if n_samples == 0:
        return 0.0

    if strategy in ("quantile", "equal_mass"):
        quantiles = np.linspace(0.0, 100.0, n_bins + 1)
        bin_edges = np.percentile(confidences, quantiles)
        bin_edges = np.unique(bin_edges)
        actual_bins = len(bin_edges) - 1
        if actual_bins <= 0:
            # All confidences identical
            acc = float(np.mean(predictions == targets))
            conf = float(np.mean(confidences))
            return float(round(abs(acc - conf), 6))
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        actual_bins = n_bins

    ece = 0.0

    for m in range(actual_bins):
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
    logits_2d = np.stack([np.zeros_like(logits), logits], axis=-1) if logits.ndim == 1 else logits

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
    enforce_gating: bool = True,
) -> float:
    """
    Optimize scalar temperature T* via 1D Golden Section Search in inverse-temperature
    beta-space (beta = 1/T) to minimize NLL loss.

    Convexity & Unimodality Invariant:
    d^2 L_NLL / d beta^2 = Var_p(z) >= 0 guarantees strict unimodality on (0, infinity).

    Production Safeguards:
    1. Sample size gating: N >= 30 required (if enforce_gating=True).
    2. Accuracy gating: Validation accuracy must exceed random chance (1/K).
    3. Boundary detection: Boundary-hit alerts reject calibration and fallback to T=1.0.
    4. Rank preservation: argmax_k(z_k / T) == argmax_k(z_k).

    Args:
        logits: Unnormalized validation logits (shape [N, K] or [N]).
        targets: Ground truth validation labels (shape [N]).
        bounds: Optimization search interval for T [min_T, max_T] (default: (0.1, 10.0)).
        tol: Convergence tolerance for interval width.
        max_iter: Maximum search iterations.
        enforce_gating: Enforce sample size and accuracy validation gates (default: True).

    Returns:
        float: Optimized temperature T*.
    """
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)
    n_samples = len(targets)

    if n_samples == 0:
        return 1.0

    num_classes = logits.shape[1] if (logits.ndim > 1 and logits.shape[1] > 1) else 2

    # 1. Sample size gating
    if enforce_gating and n_samples < 30:
        logger.warning(
            f"Validation sample size N={n_samples} < 30 is too small for reliable calibration. "
            "Falling back to neutral temperature T=1.0."
        )
        return 1.0

    # 2. Accuracy gating: Model must beat random chance prior to calibration
    if logits.ndim > 1 and logits.shape[1] > 1:
        base_preds = np.argmax(logits, axis=-1)
    else:
        base_preds = (logits > 0.0).astype(int)

    base_accuracy = float(np.mean(base_preds == targets))
    chance_threshold = 1.0 / num_classes

    if enforce_gating and base_accuracy <= chance_threshold:
        logger.warning(
            f"Validation accuracy ({base_accuracy:.4f}) does not exceed random chance ({chance_threshold:.4f}). "
            "Rejecting calibration; falling back to neutral temperature T=1.0."
        )
        return 1.0

    # 3. Inverse temperature beta-space optimization (beta = 1/T)
    # T in [T_min, T_max] corresponds to beta in [1/T_max, 1/T_min]
    beta_min = 1.0 / float(bounds[1])
    beta_max = 1.0 / float(bounds[0])

    a, b = beta_min, beta_max
    phi = (np.sqrt(5.0) - 1.0) / 2.0  # Golden ratio constant (~0.6180339887)

    c = b - phi * (b - a)
    d = a + phi * (b - a)

    fc = compute_nll(logits, targets, temperature=1.0 / c)
    fd = compute_nll(logits, targets, temperature=1.0 / d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = compute_nll(logits, targets, temperature=1.0 / c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = compute_nll(logits, targets, temperature=1.0 / d)

    opt_beta = float((a + b) / 2.0)

    # 4. Boundary detection
    epsilon = 1e-2
    if opt_beta <= beta_min + epsilon or opt_beta >= beta_max - epsilon:
        logger.warning(
            f"Inverse temperature optimization hit boundary (beta*={opt_beta:.4f}). "
            "Model predictions diverge toward extreme entropy. Falling back to neutral T=1.0."
        )
        return 1.0

    opt_t = float(np.clip(1.0 / opt_beta, bounds[0], bounds[1]))

    # 5. Strict rank-preservation invariant assertion
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

    with open(p, encoding="utf-8") as f:
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


def compute_decision_metadata(
    probs: list[float] | np.ndarray,
    options: list[str],
) -> dict[str, Any]:
    """
    Compute enterprise decision metadata from a calibrated probability distribution:
    - Shannon Entropy: H(P) = -sum p_i ln(p_i + 1e-12)
    - Normalized Margin: M = p_(1) - p_(2) (top-1 minus runner-up)
    - Tier: "HIGH" (p_max >= 0.70 and margin >= 0.35) | "MEDIUM" (p_max >= 0.50) | "LOW"
    - Escalation: True if p_max < 0.50 or margin < 0.20 or entropy > 0.85 * ln(K)

    Args:
        probs: 1D probability distribution across options.
        options: List of choice option labels.

    Returns:
        Dictionary containing enterprise decision contract fields.
    """
    p_arr = np.asarray(probs, dtype=float)
    if len(p_arr) != len(options):
        raise ValueError("probs length must match options length")

    best_idx = int(np.argmax(p_arr))
    winner = options[best_idx]
    p_max = float(p_arr[best_idx])

    # Runner-up probability & margin
    sorted_p = np.sort(p_arr)[::-1]
    p_runner_up = float(sorted_p[1]) if len(sorted_p) > 1 else 0.0
    margin = float(p_max - p_runner_up)

    # Shannon Entropy
    entropy = float(-np.sum(p_arr * np.log(p_arr + 1e-12)))

    K = len(options)
    tier = (
        "HIGH"
        if p_max >= 0.70 and margin >= 0.35
        else ("MEDIUM" if p_max >= 0.50 else "LOW")
    )
    escalate = bool(
        p_max < 0.50 or margin < 0.20 or (K > 1 and entropy > 0.85 * np.log(K))
    )

    return {
        "selection": winner,
        "confidence": float(round(p_max, 4)),
        "margin": float(round(margin, 4)),
        "entropy": float(round(entropy, 4)),
        "tier": tier,
        "escalate_to_system2": escalate,
        "probabilities": {k: float(round(float(v), 4)) for k, v in zip(options, p_arr, strict=True)},
    }


def apply_logit_regularization(
    logits: np.ndarray,
    temp: float = 1.0,
    n_conflicts: int = 0,
    clip_range: tuple[float, float] = (-8.0, 8.0),
) -> tuple[np.ndarray, float]:
    """
    Apply logit clipping and conflict-aware dynamic temperature scaling.

    Mitigates Softmax over-saturation and resolves lexical ambiguity by:
    1. Clamping unnormalized logits to clip_range (default: [-8.0, 8.0]).
    2. Scaling temperature dynamically based on conflict count:
       T_eff = T * (1.0 + 0.25 * n_conflicts).

    Args:
        logits: Unnormalized logits array.
        temp: Base temperature (T > 0).
        n_conflicts: Count of conflicting intent markers or candidate options.
        clip_range: Interval (min_logit, max_logit) for clamping.

    Returns:
        tuple[np.ndarray, float]: (clamped_logits, effective_temperature)
    """
    arr = np.asarray(logits, dtype=float)
    clamped = np.clip(arr, clip_range[0], clip_range[1])
    n_conf = max(0, int(n_conflicts))
    t_eff = float(temp * (1.0 + 0.25 * n_conf))
    return clamped, t_eff


def regularize_and_scale_logits(
    logits: np.ndarray,
    temp: float = 1.0,
    n_conflicts: int = 0,
    clip_range: tuple[float, float] = (-8.0, 8.0),
) -> np.ndarray:
    """
    Apply logit regularization followed by temperature-scaled Softmax.
    Zero-centers logits along the last axis to preserve shift invariance across
    arbitrary constant offsets prior to clamping and scaling.

    Args:
        logits: Unnormalized logits array.
        temp: Base temperature (default: 1.0).
        n_conflicts: Count of conflicting candidate options.
        clip_range: Bounds for logit clipping (default: (-8.0, 8.0)).

    Returns:
        np.ndarray: Calibrated probability distribution.
    """
    arr = np.asarray(logits, dtype=float)
    centered = arr - np.mean(arr, axis=-1, keepdims=True)
    clamped, t_eff = apply_logit_regularization(
        centered, temp=temp, n_conflicts=n_conflicts, clip_range=clip_range
    )
    t_eff = max(t_eff, 1e-6)
    scaled = clamped / t_eff
    max_logit = np.max(scaled, axis=-1, keepdims=True)
    exp_logits = np.exp(scaled - max_logit)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    probs = exp_logits / sum_exp
    return probs

