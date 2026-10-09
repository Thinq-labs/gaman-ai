"""
tests/test_domains.py — Cross-Domain Integration Verification (Phase 5)
=========================================================================
Verifies that GamanEngine evaluates three completely distinct domains:
1. Security Audit Log (DevSecOps)
2. Customer Chat Message (Agentic / Support routing)
3. Source Code Snippet (CI/CD static analysis)
using the exact same universal pipeline without hardcoded domain schemas.
"""

from pathlib import Path

import pytest

from src.engine import GamanEngine

MODELS_DIR = Path("models")
BACKBONE_ONNX = MODELS_DIR / "backbone.onnx"

_requires_model = pytest.mark.skipif(
    not BACKBONE_ONNX.exists(),
    reason="Model artifact not found. Run export script first.",
)


@pytest.fixture(scope="module")
def engine():
    return GamanEngine(models_dir=MODELS_DIR)


@_requires_model
class TestCrossDomainEvaluation:
    """Verifies universal domain-agnostic behavior across three different data modalities."""

    def test_domain_1_security_audit_log(self, engine: GamanEngine) -> None:
        """Domain 1: High-volume security telemetry."""
        state = {
            "source_ip": "198.51.100.23",
            "event": "repeated brute force authentication failure",
            "failed_attempts": 42,
            "target_resource": "/admin/vault/keys",
        }

        # 1. Guardrail / predicate
        noul_res = engine.noul(state, "A security incident occurred.")
        assert noul_res["primitive"] == "noul"
        assert noul_res["passed"] is True
        assert noul_res["probability"] > 0.5
        assert noul_res["latency_ms"] >= 0.0

        # 2. Semantic routing
        options = ["quarantine_ip", "send_2fa_challenge", "ignore_event"]
        choice_res = engine.choice(state, options)
        assert choice_res["primitive"] == "choice"
        assert choice_res["selection"] in options
        assert choice_res["confidence"] > 0.33

        # 3. Context evaluation
        score_res = engine.score(state, "Severity level of security incident")
        assert score_res["primitive"] == "score"
        assert 0.0 <= score_res["value"] <= 1.0

    def test_domain_2_customer_chat_message(self, engine: GamanEngine) -> None:
        """Domain 2: Natural language conversation event."""
        state = {
            "sender_id": "cust_4821",
            "channel": "live_chat",
            "message": "I was double-billed for invoice #8921. Please issue a financial refund immediately.",
            "account_tier": "enterprise",
        }

        # 1. Semantic routing
        options = ["billing_refund_queue", "technical_support_queue", "sales_inquiry_queue"]
        choice_res = engine.choice(state, options)
        assert choice_res["primitive"] == "choice"
        assert choice_res["selection"] == "billing_refund_queue"
        assert choice_res["confidence"] > 0.33

        # 2. Guardrail / predicate
        noul_res = engine.noul(state, "This customer is requesting a financial refund.")
        assert noul_res["primitive"] == "noul"
        assert noul_res["passed"] is True
        assert noul_res["probability"] > 0.5

        # 3. Context evaluation
        score_res = engine.score(state, "Urgency and customer dissatisfaction level")
        assert score_res["primitive"] == "score"
        assert 0.0 <= score_res["value"] <= 1.0

    def test_domain_3_source_code_snippet(self, engine: GamanEngine) -> None:
        """Domain 3: Static code analysis / CI pipeline check."""
        state = {
            "file": "src/api/auth.py",
            "issue": "critical SQL injection vulnerability in database query",
            "author": "external_contributor",
        }

        # 1. Guardrail / predicate
        noul_res = engine.noul(state, "This code contains a security vulnerability.")
        assert noul_res["primitive"] == "noul"
        assert noul_res["passed"] is True
        assert noul_res["probability"] > 0.5

        # 2. Semantic routing
        options = ["block_pull_request", "approve_merge", "request_documentation"]
        choice_res = engine.choice(state, options)
        assert choice_res["primitive"] == "choice"
        assert choice_res["selection"] in options

        # 3. Context evaluation
        score_res = engine.score(state, "Criticality of software vulnerability")
        assert score_res["primitive"] == "score"
        assert 0.0 <= score_res["value"] <= 1.0
