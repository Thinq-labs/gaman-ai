"""
tests/test_salience_hardening.py — Salience Cleaning, Adversative Reweighting & Logit Regularization
====================================================================================================
Verifies:
1. Conversational boilerplate stripping (Sandwich Trap mitigation).
2. Adversative and counterfactual clause reweighting (Counterfactual bias mitigation).
3. Logit clipping and anti-saturation temperature scaling.
4. End-to-end routing resilience on adversarial structures.
"""

from pathlib import Path

import numpy as np
import pytest

from src.calibration import apply_logit_regularization, regularize_and_scale_logits
from src.engine import GamanEngine
from src.serializer import (
    reweight_adversative_clauses,
    strip_conversational_boilerplate,
)

MODELS_DIR = Path("models")
BACKBONE_ONNX = MODELS_DIR / "backbone.onnx"

_requires_model = pytest.mark.skipif(
    not BACKBONE_ONNX.exists(),
    reason=f"Model artifact not found at '{BACKBONE_ONNX}'. Run export script first.",
)


@pytest.fixture(scope="module")
def engine() -> GamanEngine:
    eng = GamanEngine(models_dir=MODELS_DIR)
    eng.temp_choice = 1.0
    eng.temp_noul = 1.0
    eng.temp_score = 1.0
    return eng


class TestSalienceCleaningUnit:
    """Unit tests for boilerplate stripping and adversative clause reweighting."""

    def test_strip_conversational_boilerplate_leading(self) -> None:
        raw = "Hello team, Hope you are well. The database connection pool is exhausted."
        stripped = strip_conversational_boilerplate(raw)
        assert "Hello team" not in stripped
        assert "Hope you are well" not in stripped
        assert "The database connection pool is exhausted." in stripped

    def test_strip_conversational_boilerplate_trailing(self) -> None:
        raw = "The database connection pool is exhausted. Let me know when you fix it. Thanks in advance"
        stripped = strip_conversational_boilerplate(raw)
        assert "Let me know when you fix it" not in stripped
        assert "Thanks in advance" not in stripped
        assert "The database connection pool is exhausted." in stripped

    def test_strip_conversational_boilerplate_sandwich(self) -> None:
        raw = "Hello team... border patrol seized the box... fix tracking link"
        stripped = strip_conversational_boilerplate(raw)
        assert "Hello team" not in stripped
        assert "fix tracking link" not in stripped
        assert "border patrol seized the box" in stripped

    def test_reweight_adversative_clauses_if_instead(self) -> None:
        raw = "If I wanted return_policy... Instead credit card declined"
        reweighted = reweight_adversative_clauses(raw)
        assert reweighted.startswith("credit card declined.")
        assert "Instead" in reweighted or "instead" in reweighted

    def test_reweight_adversative_clauses_however(self) -> None:
        raw = "I thought about return_policy, however my credit card was declined"
        reweighted = reweight_adversative_clauses(raw)
        assert reweighted.startswith("my credit card was declined.")


class TestLogitRegularizationUnit:
    """Unit tests for logit clamping and anti-saturation temperature scaling."""

    def test_logit_clamping_bounds(self) -> None:
        raw_logits = np.array([25.0, 18.0, -25.0])
        clamped, t_eff = apply_logit_regularization(raw_logits, temp=1.0, n_conflicts=0)
        assert np.all(clamped >= -8.0)
        assert np.all(clamped <= 8.0)
        assert clamped[0] == 8.0
        assert clamped[2] == -8.0
        assert t_eff == 1.0

    def test_conflict_temperature_scaling(self) -> None:
        raw_logits = np.array([4.0, 2.0, 0.0])
        _, t_eff = apply_logit_regularization(raw_logits, temp=1.0, n_conflicts=2)
        # T_eff = T * (1.0 + 0.25 * 2) = 1.5
        assert t_eff == pytest.approx(1.5)

    def test_logit_clamping_prevents_false_saturation(self) -> None:
        """Verify extreme logits with narrow margin do not produce >95% probability saturation."""
        raw_logits = np.array([20.0, 19.8, -15.0])
        probs = regularize_and_scale_logits(raw_logits, temp=1.0, n_conflicts=2)
        assert probs.max() < 0.95


@_requires_model
class TestSalienceHardeningIntegration:
    """Integration tests verifying end-to-end resilience in GamanEngine."""

    def test_sandwich_trap_mitigation(self, engine: GamanEngine) -> None:
        """
        Sandwich Trap Test:
        'Hello team... border patrol seized the box... fix tracking link'
        Must strip boilerplate padding and route to CUSTOMS_DELAY instead of ADDRESS_VALIDATION.
        """
        state = {"text": "Hello team... border patrol seized the box... fix tracking link"}
        options = {
            "CUSTOMS_DELAY": "Shipment held, delayed, or seized by customs or border patrol.",
            "ADDRESS_VALIDATION": "Customer asking to update delivery address or fix tracking link.",
        }
        res = engine.choice(state, options)
        assert res["selection"] == "CUSTOMS_DELAY"
        assert res["probabilities"]["CUSTOMS_DELAY"] > res["probabilities"]["ADDRESS_VALIDATION"]

    def test_counterfactual_adversative_mitigation(self, engine: GamanEngine) -> None:
        """
        Counterfactual Test:
        'If I wanted return_policy... Instead credit card declined'
        Must select PAYMENT_FAILURE rather than being trapped by the conditional return_policy.
        """
        state = {"text": "If I wanted return_policy... Instead credit card declined"}
        options = ["PAYMENT_FAILURE", "RETURN_POLICY", "SHIPPING_DELAY", "GENERAL_INQUIRY"]
        res = engine.choice(state, options)
        assert res["selection"] == "PAYMENT_FAILURE"
        assert res["probabilities"]["PAYMENT_FAILURE"] > res["probabilities"]["RETURN_POLICY"]

    def test_extreme_logit_regularization_in_engine(self, engine: GamanEngine) -> None:
        """Verify logit regularization applies during engine inference without probability saturation."""
        state = {"text": "credit card payment failed repeatedly"}
        options = ["PAYMENT_FAILURE", "BILLING_INQUIRY", "GENERAL_INQUIRY"]
        res = engine.choice(state, options)
        assert res["selection"] == "PAYMENT_FAILURE"
        assert "margin" in res
        assert "entropy" in res
