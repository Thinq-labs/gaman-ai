"""
tests/test_audit_remediation.py — Audit Remediation (Patch Directives 1, 2, & 3)
================================================================================
Verifies:
1. Directive 1: Generalized Pre-Tokenizer De-Obfuscation (r.e.f.u.n.d -> refund, p-a-y -> pay).
2. Directive 2: Structural Multi-Clause Intent Gate (sentence split & imperative conjunctions).
3. Directive 3: Tight max_length=64 sequence truncation and thread affinity optimization.
"""

from pathlib import Path

import pytest

from src.calibration import detect_multi_clause_disjoint_intent
from src.engine import GamanEngine
from src.serializer import collapse_spaced_tokens

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


class TestDirective1DeobfuscationUnit:
    """Unit tests for Directive 1 generalized delimiter collapsing."""

    def test_collapse_spaced_tokens_dots(self) -> None:
        raw = "I demand a r.e.f.u.n.d immediately"
        collapsed = collapse_spaced_tokens(raw)
        assert "refund" in collapsed
        assert "r.e.f.u.n.d" not in collapsed

    def test_collapse_spaced_tokens_dashes(self) -> None:
        raw = "The portal will not let me p-a-y for my bill"
        collapsed = collapse_spaced_tokens(raw)
        assert "pay" in collapsed
        assert "p-a-y" not in collapsed

    def test_collapse_spaced_tokens_underscores(self) -> None:
        raw = "please send the w_i_r_e transfer"
        collapsed = collapse_spaced_tokens(raw)
        assert "wire" in collapsed

    def test_collapse_spaced_tokens_invariance(self) -> None:
        assert collapse_spaced_tokens("10.5% discount") == "10.5% discount"
        assert collapse_spaced_tokens("a-b testing") == "a-b testing"
        assert collapse_spaced_tokens("traveling to U.S.A.") == "traveling to USA."


class TestDirective2MultiClauseUnit:
    """Unit tests for Directive 2 multi-clause disjoint intent gate."""

    def test_detect_multi_clause_sentence_split(self) -> None:
        text = "dispute this unauthorized $400 charge... In a separate matter, cancel my subscription."
        options = ["FRAUD_DISPUTE", "CANCEL_SUBSCRIPTION", "GENERAL_INQUIRY"]
        assert detect_multi_clause_disjoint_intent(text, options) is True

    def test_detect_multi_clause_imperative_conjunction(self) -> None:
        text = "Schedule a cardiology consultation... and dispute my last co-pay charge."
        options = ["CARDIOLOGY_CONSULTATION", "BILLING_DISPUTE", "GENERAL_INQUIRY"]
        assert detect_multi_clause_disjoint_intent(text, options) is True

    def test_detect_multi_clause_single_intent_is_false(self) -> None:
        text = "The product arrived completely shattered. I demand a refund for this item."
        options = ["REFUND_REQUEST", "GENERAL_INQUIRY", "TECHNICAL_SUPPORT"]
        assert detect_multi_clause_disjoint_intent(text, options) is False


@_requires_model
class TestAuditRemediationIntegration:
    """End-to-end integration tests verifying audit vulnerability remediation."""

    def test_audit_test_03_fix_dot_obfuscated_refund(self, engine: GamanEngine) -> None:
        """
        Test 03 Fix:
        'The product arrived completely shattered. I demand a r.e.f.u.n.d...'
        Must select REFUND_REQUEST over incidental damaged delivery description.
        """
        state = {
            "text": "The product arrived completely shattered. I demand a r.e.f.u.n.d..."
        }
        options = ["REFUND_REQUEST", "DAMAGED_DELIVERY", "GENERAL_INQUIRY"]
        res = engine.choice(state, options)
        assert res["selection"] == "REFUND_REQUEST"
        assert res["selection"] != "DAMAGED_DELIVERY"
        assert res["probabilities"]["REFUND_REQUEST"] > res["probabilities"]["DAMAGED_DELIVERY"]

    def test_audit_test_04_fix_hyphen_obfuscated_pay(self, engine: GamanEngine) -> None:
        """
        Test 04 Fix:
        'The portal will not let me p-a-y for my recurring invoice...'
        Must select INVOICE_PAYMENT.
        """
        state = {
            "text": "The portal will not let me p-a-y for my recurring invoice..."
        }
        options = ["INVOICE_PAYMENT", "ACCOUNT_INQUIRY", "GENERAL_INQUIRY"]
        res = engine.choice(state, options)
        assert res["selection"] == "INVOICE_PAYMENT"
        assert res["probabilities"]["INVOICE_PAYMENT"] > 0.80

    def test_audit_test_07_fix_sentence_split_dual_intent(self, engine: GamanEngine) -> None:
        """
        Test 07 Fix:
        'dispute this unauthorized $400 charge... In a separate matter, cancel my subscription.'
        Must yield escalate_to_system2 == True and tier == 'LOW'.
        """
        state = {
            "text": "dispute this unauthorized $400 charge... In a separate matter, cancel my subscription."
        }
        options = ["FRAUD_DISPUTE", "CANCEL_SUBSCRIPTION", "GENERAL_INQUIRY"]
        res = engine.choice(state, options)
        assert res["escalate_to_system2"] is True
        assert res["tier"] == "LOW"

    def test_audit_test_08_fix_imperative_conjunction_dual_intent(self, engine: GamanEngine) -> None:
        """
        Test 08 Fix:
        'Schedule a cardiology consultation... and dispute my last co-pay charge.'
        Must yield escalate_to_system2 == True and tier == 'LOW'.
        """
        state = {
            "text": "Schedule a cardiology consultation... and dispute my last co-pay charge."
        }
        options = ["CARDIOLOGY_CONSULTATION", "BILLING_DISPUTE", "GENERAL_INQUIRY"]
        res = engine.choice(state, options)
        assert res["escalate_to_system2"] is True
        assert res["tier"] == "LOW"

    def test_directive_3_tight_sequence_truncation_bound(self, engine: GamanEngine) -> None:
        """
        Directive 3:
        Verify that premise pairs are strictly capped at max_length=64 tokens.
        """
        long_text = "word " * 150
        pairs = [(long_text, "Option hypothesis")]
        enc = engine.tokenizer.encode_batch(pairs, max_length=64)
        assert enc["input_ids"].shape[1] <= 64
