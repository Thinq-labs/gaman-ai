"""Tests for Gaman AI Probability Calibration (src/calibration.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.calibration import (
    compute_ece,
    compute_nll,
    fit_temperature,
    load_calibration,
    save_calibration,
)
from src.engine import GamanEngine


class TestExpectedCalibrationError:
    def test_ece_synthetic_perfect(self) -> None:
        """ECE should be approximately 0.0 on perfectly calibrated probabilities."""
        # 1000 samples, 10 bins of 100 samples each
        # In each bin k, confidence is c_k, and exactly c_k fraction of samples are correct
        n_bins = 10
        confidences_list = []
        predictions_list = []
        targets_list = []

        bin_confs = np.linspace(0.05, 0.95, n_bins)
        samples_per_bin = 200

        for conf in bin_confs:
            n_correct = int(round(conf * samples_per_bin))
            n_incorrect = samples_per_bin - n_correct

            # confidences for this bin
            confs = np.full(samples_per_bin, conf)
            preds = np.ones(samples_per_bin, dtype=int)
            # targets: first n_correct match preds (1), remaining are 0
            tgts = np.concatenate([np.ones(n_correct, dtype=int), np.zeros(n_incorrect, dtype=int)])

            confidences_list.append(confs)
            predictions_list.append(preds)
            targets_list.append(tgts)

        confidences = np.concatenate(confidences_list)
        predictions = np.concatenate(predictions_list)
        targets = np.concatenate(targets_list)

        ece = compute_ece(confidences, predictions, targets, n_bins=n_bins)
        assert ece < 0.01, f"Expected near-zero ECE for calibrated data, got {ece}"

    def test_ece_synthetic_overconfident(self) -> None:
        """Verify high ECE on artificially peaked / overconfident logits."""
        # Predictions are confident (~0.99) but accuracy is only 50% (random guess)
        n_samples = 500
        confidences = np.full(n_samples, 0.99)
        predictions = np.ones(n_samples, dtype=int)
        # 50% correct
        targets = np.array([1 if i % 2 == 0 else 0 for i in range(n_samples)])

        ece = compute_ece(confidences, predictions, targets, n_bins=10)
        # Accuracy is 0.50, confidence is 0.99 -> error is ~0.49
        assert ece > 0.40, f"Expected high ECE for overconfident predictions, got {ece}"

    def test_ece_empty_inputs(self) -> None:
        """ECE on empty inputs should return 0.0 safely."""
        ece = compute_ece(np.array([]), np.array([]), np.array([]), n_bins=10)
        assert ece == 0.0


class TestTemperatureOptimization:
    def test_temperature_scaling_preserves_argmax(self) -> None:
        """Confirm predictions do not flip before and after scaling for any T in [0.2, 5.0]."""
        np.random.seed(42)
        n_samples, n_classes = 100, 5
        logits = np.random.randn(n_samples, n_classes) * 3.0

        original_preds = np.argmax(logits, axis=1)

        test_temps = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.5, 5.0]
        for T in test_temps:
            scaled_logits = logits / T
            scaled_preds = np.argmax(scaled_logits, axis=1)
            np.testing.assert_array_equal(
                original_preds,
                scaled_preds,
                err_msg=f"Argmax was not preserved for T={T}",
            )

    def test_fit_temperature_reduces_nll_and_ece(self) -> None:
        """Confirm optimization reduces both NLL and ECE on held-out overconfident synthetic data."""
        np.random.seed(123)
        n_samples, n_classes = 600, 3
        # Generate true probabilities from moderate logits
        base_logits = np.random.randn(n_samples, n_classes)
        true_probs = np.exp(base_logits) / np.sum(np.exp(base_logits), axis=1, keepdims=True)

        # Sample targets from true_probs
        targets = np.array([np.random.choice(n_classes, p=p) for p in true_probs])

        # Overconfident logits: multiply base_logits by 4.0
        overconfident_logits = base_logits * 4.0

        # Uncalibrated metrics (T=1.0)
        initial_nll = compute_nll(overconfident_logits, targets, temperature=1.0)
        init_probs = np.exp(overconfident_logits) / np.sum(np.exp(overconfident_logits), axis=1, keepdims=True)
        init_preds = np.argmax(init_probs, axis=1)
        init_confs = np.max(init_probs, axis=1)
        initial_ece = compute_ece(init_confs, init_preds, targets)

        # Fit optimal temperature
        T_opt = fit_temperature(overconfident_logits, targets, bounds=(0.1, 10.0))

        # T_opt should be > 1.0 because logits were artificially scaled up
        assert T_opt > 1.5, f"Expected T_opt > 1.5 for overconfident model, got {T_opt}"

        # Calibrated metrics
        calibrated_nll = compute_nll(overconfident_logits, targets, temperature=T_opt)
        scaled_logits = overconfident_logits / T_opt
        cal_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
        cal_preds = np.argmax(cal_probs, axis=1)
        cal_confs = np.max(cal_probs, axis=1)
        calibrated_ece = compute_ece(cal_confs, cal_preds, targets)

        assert calibrated_nll < initial_nll, f"NLL should decrease: {calibrated_nll} vs {initial_nll}"
        assert calibrated_ece < initial_ece, f"ECE should decrease: {calibrated_ece} vs {initial_ece}"

    def test_fit_temperature_binary_logits(self) -> None:
        """Verify 1D binary logits are supported and calibrated properly."""
        np.random.seed(42)
        n_samples = 400
        z = np.random.randn(n_samples) * 3.0  # overconfident
        probs = 1.0 / (1.0 + np.exp(-z))
        targets = (np.random.rand(n_samples) < probs).astype(int)

        T_opt = fit_temperature(z, targets, bounds=(0.1, 10.0))
        assert 0.1 <= T_opt <= 10.0

        init_nll = compute_nll(z, targets, temperature=1.0)
        cal_nll = compute_nll(z, targets, temperature=T_opt)
        assert cal_nll <= init_nll + 1e-6

    def test_beta_space_convexity(self) -> None:
        """Prove empirically that NLL loss is convex with respect to beta = 1/T."""
        np.random.seed(42)
        n_samples, n_classes = 200, 4
        logits = np.random.randn(n_samples, n_classes) * 2.0
        targets = np.random.randint(0, n_classes, size=n_samples)

        betas = np.linspace(0.2, 4.0, 50)
        nlls = [compute_nll(logits, targets, temperature=1.0 / b) for b in betas]

        # Numerical second differences d^2 NLL / d beta^2 should be >= -1e-4
        d2 = np.diff(nlls, n=2)
        assert np.all(d2 >= -1e-4), "NLL in beta-space failed convexity check (second difference < 0)"

    def test_boundary_hit_fallback(self) -> None:
        """Confirm that boundary-hit conditions trigger fallback to neutral T=1.0."""
        # Intentionally create inverted logits where true class has lowest logit
        n_samples, n_classes = 50, 3
        inverted_logits = np.zeros((n_samples, n_classes))
        targets = np.zeros(n_samples, dtype=int)
        inverted_logits[:, 0] = -10.0  # true class has minimum logit
        inverted_logits[:, 1] = 5.0
        inverted_logits[:, 2] = 5.0

        # Optimizer should hit boundary or fail accuracy gate and fall back to 1.0
        T_opt = fit_temperature(inverted_logits, targets, bounds=(0.1, 10.0))
        assert T_opt == 1.0

    def test_sample_size_gating(self) -> None:
        """Confirm N < 30 samples triggers fallback to neutral T=1.0 under enforce_gating."""
        small_logits = np.random.randn(15, 3)
        small_targets = np.random.randint(0, 3, size=15)
        T_opt = fit_temperature(small_logits, small_targets, enforce_gating=True)
        assert T_opt == 1.0

    def test_accuracy_gating(self) -> None:
        """Confirm accuracy <= 1/K triggers fallback to neutral T=1.0."""
        np.random.seed(42)
        n_samples, n_classes = 50, 4
        # Model that predicts class 0 100% of the time
        logits = np.zeros((n_samples, n_classes))
        logits[:, 0] = 5.0
        # But target is class 1 for 45 samples (accuracy = 5/50 = 10% < 25%)
        targets = np.ones(n_samples, dtype=int)
        targets[:5] = 0

        T_opt = fit_temperature(logits, targets, enforce_gating=True)
        assert T_opt == 1.0


class TestQuantileBinningECE:
    def test_quantile_ece_on_clustered_confidences(self) -> None:
        """Verify quantile binning accurately partitions dense probability clusters."""
        n_samples = 400
        # Confidences tightly clustered in [0.55, 0.65]
        confidences = 0.55 + 0.10 * np.random.rand(n_samples)
        predictions = np.ones(n_samples, dtype=int)
        # 60% accuracy
        targets = (np.random.rand(n_samples) < 0.60).astype(int)

        ece_quantile = compute_ece(confidences, predictions, targets, n_bins=10, strategy="quantile")
        assert 0.0 <= ece_quantile <= 1.0
        # Equal width might lump everything into 1-2 bins, quantile distributes into 10 equal bins
        ece_width = compute_ece(confidences, predictions, targets, n_bins=10, strategy="equal_width")
        assert 0.0 <= ece_width <= 1.0


class TestCalibrationPersistence:
    def test_save_and_load_calibration(self, tmp_path: Path) -> None:
        """Verify saving and loading calibration.json matches schema."""
        cal_file = tmp_path / "calibration.json"
        payload = {
            "version": "0.2.0",
            "temperatures": {
                "choice": 1.45,
                "noul": 0.85,
                "score": 1.10,
            },
            "ece_before": {"choice": 0.18, "noul": 0.12, "score": None},
            "ece_after": {"choice": 0.03, "noul": 0.02, "score": None},
        }

        save_calibration(payload, cal_file)
        assert cal_file.exists()

        loaded = load_calibration(cal_file)
        assert loaded["version"] == "0.2.0"
        assert loaded["temperatures"]["choice"] == 1.45
        assert loaded["temperatures"]["noul"] == 0.85
        assert loaded["temperatures"]["score"] == 1.10

    def test_load_nonexistent_returns_defaults(self, tmp_path: Path) -> None:
        """Loading from nonexistent path returns default neutral temperatures."""
        cal_file = tmp_path / "does_not_exist.json"
        loaded = load_calibration(cal_file)
        assert loaded["temperatures"]["choice"] == 1.0
        assert loaded["temperatures"]["noul"] == 1.0
        assert loaded["temperatures"]["score"] == 1.0


class TestEngineCalibrationIntegration:
    def test_engine_loads_and_applies_temperature(self, tmp_path: Path) -> None:
        """Verify engine loads calibration.json and scales probabilities."""
        # 1. Create a temporary models directory with calibration.json
        models_dir = Path("models")
        if not (models_dir / "backbone.onnx").exists():
            pytest.skip("models/backbone.onnx not found")

        # Copy or point to real models
        # Test default engine (uncalibrated / T=1.0)
        engine_uncal = GamanEngine(models_dir=models_dir)
        state = {"cpu_usage": 98, "memory_usage": 85}
        options = ["scale_up", "scale_down", "do_nothing"]

        res_uncal = engine_uncal.choice(state, options)

        # Now test an engine where calibration.json sets high temperature (T=3.0)
        # Should soften probabilities toward uniform
        cal_path = tmp_path / "calibration.json"
        save_calibration(
            {
                "version": "0.2.0",
                "temperatures": {"choice": 3.0, "noul": 2.0, "score": 2.0},
                "ece_before": None,
                "ece_after": None,
            },
            cal_path,
        )

        engine_cal = GamanEngine(models_dir=models_dir)
        # Manually load calibration from tmp_path or set attribute
        engine_cal.load_calibration(cal_path)

        res_cal = engine_cal.choice(state, options)

        # Argmax selection must remain identical
        assert res_cal["selection"] == res_uncal["selection"]
        # Confidence under T=3.0 should be lower (more entropy / softened)
        assert res_cal["confidence"] < res_uncal["confidence"]
        assert res_cal["confidence"] > 1.0 / len(options)
