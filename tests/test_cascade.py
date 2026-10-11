"""
tests/test_cascade.py — Unit Tests for Speculative Cascade Engine
==================================================================
Tests speculative decision cascading between draft and verifier tiers.
"""

from unittest.mock import MagicMock

from src.cascade import SpeculativeCascadeEngine


def _make_mock_engine(tier: str = "small") -> MagicMock:
    engine = MagicMock()
    engine.slab_config = MagicMock()
    engine.slab_config.tier = tier
    return engine


class TestSpeculativeCascadeEngine:
    def test_speculative_hit_on_certified_singleton(self) -> None:
        """When draft produces certified singleton and no escalation, verifier is never called."""
        draft = _make_mock_engine("small")
        draft.choice.return_value = {
            "primitive": "choice",
            "selection": "REFUND",
            "confidence": 0.96,
            "prediction_set": ["REFUND"],
            "certified_safe": True,
            "escalate_to_system2": False,
        }

        verifier = _make_mock_engine("large")

        cascade = SpeculativeCascadeEngine(
            draft_engine=draft,
            verifier_engine=verifier,
        )

        res = cascade.choice(
            state={"message": "I want a refund"},
            options=["REFUND", "SUPPORT"],
        )

        assert res["selection"] == "REFUND"
        assert res["speculative_hit"] is True
        assert res["cascaded"] is False
        assert res["draft_tier"] == "small"
        assert res["verifier_tier"] is None
        assert "cascade_latency_ms" in res
        # Verifier must not have been invoked
        verifier.choice.assert_not_called()

    def test_cascade_triggered_on_ambiguous_prediction_set(self) -> None:
        """When draft prediction set contains > 1 classes, verifier is called."""
        draft = _make_mock_engine("small")
        draft.choice.return_value = {
            "primitive": "choice",
            "selection": "REFUND",
            "confidence": 0.55,
            "prediction_set": ["REFUND", "SUPPORT"],
            "certified_safe": True,
            "escalate_to_system2": False,
        }

        verifier = _make_mock_engine("base")
        verifier.choice.return_value = {
            "primitive": "choice",
            "selection": "REFUND",
            "confidence": 0.94,
            "prediction_set": ["REFUND"],
            "certified_safe": True,
            "escalate_to_system2": False,
        }

        cascade = SpeculativeCascadeEngine(
            draft_engine=draft,
            verifier_engine=verifier,
        )

        res = cascade.choice(
            state={"message": "Help me with refund or support"},
            options=["REFUND", "SUPPORT"],
        )

        assert res["selection"] == "REFUND"
        assert res["speculative_hit"] is False
        assert res["cascaded"] is True
        assert res["draft_tier"] == "small"
        assert res["verifier_tier"] == "base"
        assert res["draft_prediction_set"] == ["REFUND", "SUPPORT"]
        verifier.choice.assert_called_once()

    def test_cascade_triggered_on_system2_escalation(self) -> None:
        """When draft triggers escalate_to_system2, verifier is called even if singleton."""
        draft = _make_mock_engine("small")
        draft.choice.return_value = {
            "primitive": "choice",
            "selection": "SECURITY_ALERT",
            "confidence": 0.40,
            "prediction_set": ["SECURITY_ALERT"],
            "certified_safe": True,
            "escalate_to_system2": True,
        }

        verifier = _make_mock_engine("large")
        verifier.choice.return_value = {
            "primitive": "choice",
            "selection": "SECURITY_ALERT",
            "confidence": 0.88,
            "prediction_set": ["SECURITY_ALERT"],
            "certified_safe": True,
            "escalate_to_system2": False,
        }

        cascade = SpeculativeCascadeEngine(
            draft_engine=draft,
            verifier_engine=verifier,
        )

        res = cascade.choice(
            state={"message": "Suspicious login attempt"},
            options=["SECURITY_ALERT", "NORMAL"],
        )

        assert res["cascaded"] is True
        assert res["speculative_hit"] is False
        verifier.choice.assert_called_once()

    def test_fallback_when_verifier_is_none(self) -> None:
        """When verifier is None, ambiguous draft result is returned gracefully."""
        draft = _make_mock_engine("small")
        draft.choice.return_value = {
            "primitive": "choice",
            "selection": "REFUND",
            "confidence": 0.50,
            "prediction_set": ["REFUND", "SUPPORT"],
            "certified_safe": True,
            "escalate_to_system2": True,
        }

        cascade = SpeculativeCascadeEngine(
            draft_engine=draft,
            verifier_engine=None,
            verifier_slab="",
        )

        res = cascade.choice(
            state={"message": "Unclear request"},
            options=["REFUND", "SUPPORT"],
        )

        assert res["cascaded"] is False
        assert res["speculative_hit"] is False
        assert res["verifier_tier"] is None

    def test_decide_alias_matches_choice(self) -> None:
        """decide() forwards to choice()."""
        draft = _make_mock_engine("small")
        draft.choice.return_value = {
            "primitive": "choice",
            "selection": "A",
            "prediction_set": ["A"],
            "certified_safe": True,
            "escalate_to_system2": False,
        }

        cascade = SpeculativeCascadeEngine(draft_engine=draft, verifier_engine=None, verifier_slab="")
        res = cascade.decide(state={"x": 1}, options=["A", "B"])
        assert res["selection"] == "A"
        assert res["speculative_hit"] is True

    def test_noul_high_confidence_speculative_hit(self) -> None:
        """High confidence predicate returns draft without invoking verifier."""
        draft = _make_mock_engine("small")
        draft.noul.return_value = {
            "primitive": "noul",
            "verdict": True,
            "probability": 0.98,
        }
        verifier = _make_mock_engine("base")

        cascade = SpeculativeCascadeEngine(draft_engine=draft, verifier_engine=verifier)
        res = cascade.noul(state={"action": "delete_all"}, predicate="Is this destructive?")

        assert res["verdict"] is True
        assert res["speculative_hit"] is True
        assert res["cascaded"] is False
        verifier.noul.assert_not_called()

    def test_noul_borderline_cascades_to_verifier(self) -> None:
        """Borderline predicate (p=0.52) cascades to verifier."""
        draft = _make_mock_engine("small")
        draft.noul.return_value = {
            "primitive": "noul",
            "verdict": True,
            "probability": 0.52,
        }
        verifier = _make_mock_engine("base")
        verifier.noul.return_value = {
            "primitive": "noul",
            "verdict": True,
            "probability": 0.89,
        }

        cascade = SpeculativeCascadeEngine(draft_engine=draft, verifier_engine=verifier)
        res = cascade.noul(state={"action": "update_meta"}, predicate="Is this destructive?")

        assert res["probability"] == 0.89
        assert res["cascaded"] is True
        assert res["speculative_hit"] is False
        verifier.noul.assert_called_once()

    def test_score_attaches_telemetry(self) -> None:
        """score() attaches cascade telemetry."""
        draft = _make_mock_engine("small")
        draft.score.return_value = {
            "primitive": "score",
            "score": 0.82,
        }

        cascade = SpeculativeCascadeEngine(draft_engine=draft, verifier_engine=None, verifier_slab="")
        res = cascade.score(state={"text": "Great"}, criterion="Helpfulness")

        assert res["score"] == 0.82
        assert res["speculative_hit"] is True
        assert res["cascaded"] is False
        assert "cascade_latency_ms" in res
