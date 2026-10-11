"""Tests for Uncertainty-Aware Conformal Prediction in Gaman AI."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.conformal import ConformalPredictionError, ConformalPredictor


class TestConformalPredictionMath:
    def test_alpha_bounds_validation(self):
        with pytest.raises(ConformalPredictionError):
            ConformalPredictor(alpha=0.0)
        with pytest.raises(ConformalPredictionError):
            ConformalPredictor(alpha=1.0)
        with pytest.raises(ConformalPredictionError):
            ConformalPredictor(alpha=-0.1)

    def test_fit_and_theoretical_coverage(self):
        """Verify empirical coverage meets or exceeds 1 - alpha on holdout distribution."""
        np.random.seed(42)
        n_samples = 500
        n_classes = 5
        alpha = 0.10  # 90% target coverage

        # Simulate calibrated probability distributions
        logits = np.random.randn(n_samples, n_classes)
        # Give true class a positive bias
        true_labels = np.random.randint(0, n_classes, size=n_samples)
        logits[np.arange(n_samples), true_labels] += 2.0
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        predictor = ConformalPredictor(alpha=alpha)
        q = predictor.fit(probs[:300], true_labels[:300])
        assert 0.0 <= q <= 1.0
        assert predictor.is_calibrated

        # Test coverage on remaining 200 holdout samples
        classes = [f"class_{i}" for i in range(n_classes)]
        covered = 0
        for i in range(300, n_samples):
            res = predictor.predict_set(probs[i], classes)
            pred_set = res["prediction_set"]
            true_class_name = classes[true_labels[i]]
            if true_class_name in pred_set:
                covered += 1

        empirical_coverage = covered / (n_samples - 300)
        # Statistical coverage guarantee: empirical coverage >= (1 - alpha) - epsilon
        assert empirical_coverage >= (1.0 - alpha) - 0.05

    def test_single_option_safe_prediction_set(self):
        predictor = ConformalPredictor(alpha=0.05, quantile=0.30)
        classes = ["scale_up", "scale_down", "do_nothing"]
        # Very high certainty for scale_up
        probs = np.array([0.95, 0.03, 0.02])
        res = predictor.predict_set(probs, classes)

        assert res["certified_safe"] is True
        assert res["escalate"] is False
        assert res["prediction_set"] == ["scale_up"]
        assert res["set_size"] == 1
        assert res["coverage_guarantee"] == 0.95

    def test_ambiguous_prediction_set_triggers_escalation(self):
        predictor = ConformalPredictor(alpha=0.05, quantile=0.80)
        classes = ["billing_dispute", "wire_recall", "general_inquiry"]
        # Split decision between billing and wire_recall
        probs = np.array([0.48, 0.45, 0.07])
        res = predictor.predict_set(probs, classes)

        assert res["certified_safe"] is False
        assert res["escalate"] is True
        assert set(res["prediction_set"]) == {"billing_dispute", "wire_recall"}
        assert res["set_size"] == 2

    def test_out_of_distribution_empty_set(self):
        # Very high threshold (q=0.05 -> threshold=0.95)
        predictor = ConformalPredictor(alpha=0.05, quantile=0.05)
        classes = ["opt_a", "opt_b", "opt_c"]
        # Flat unentailed distribution
        probs = np.array([0.34, 0.33, 0.33])
        res = predictor.predict_set(probs, classes)

        assert res["ood_detected"] is True
        assert res["escalate"] is True
        assert res["certified_safe"] is False
        assert res["set_size"] == 0
        assert res["prediction_set"] == []

    def test_conformal_serialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "conformal.json"
            p1 = ConformalPredictor(alpha=0.08, quantile=0.45)
            p1.save(path)

            p2 = ConformalPredictor.load(path)
            assert p2.alpha == 0.08
            assert p2.quantile == 0.45
            assert p2.is_calibrated
