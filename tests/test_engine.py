"""
tests/test_engine.py — Unit & Integration Tests for GamanEngine
================================================================
Verifies initialization, dynamic hardware dispatch, schema compliance,
determinism, and zero-shot reasoning for the three core primitives.
"""

from pathlib import Path
import numpy as np
import pytest

from src.engine import GamanEngine

MODELS_DIR = Path("models")
BACKBONE_ONNX = MODELS_DIR / "backbone.onnx"

_requires_model = pytest.mark.skipif(
    not BACKBONE_ONNX.exists(),
    reason=f"Model artifact not found at '{BACKBONE_ONNX}'. Run export script first.",
)


@pytest.fixture(scope="module")
def engine():
    """Module-scoped GamanEngine instance loaded from local models directory."""
    return GamanEngine(models_dir=MODELS_DIR)


class TestEngineInit:
    """Verifies initialization and hardware dispatch."""

    def test_missing_model_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Model artifact not found"):
            GamanEngine(models_dir=tmp_path)

    @_requires_model
    def test_dynamic_hidden_dim_loaded(self, engine: GamanEngine) -> None:
        assert isinstance(engine.hidden_dim, int)
        assert engine.hidden_dim > 0

    @_requires_model
    def test_active_provider_detected(self, engine: GamanEngine) -> None:
        assert isinstance(engine.active_provider, str)
        assert "ExecutionProvider" in engine.active_provider


@_requires_model
class TestChoicePrimitive:
    """Tests for the choice (semantic routing) primitive."""

    def test_choice_schema_and_types(self, engine: GamanEngine) -> None:
        state = {"cpu_usage": 98, "memory_usage": 85}
        options = ["scale_up", "scale_down", "do_nothing"]
        result = engine.choice(state, options)

        assert result["primitive"] == "choice"
        assert result["selection"] == "scale_up"
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["latency_ms"], float)
        assert result["latency_ms"] >= 0.0

    def test_choice_empty_options_raises_error(self, engine: GamanEngine) -> None:
        with pytest.raises(ValueError, match="options list cannot be empty"):
            engine.choice({"key": "val"}, [])

    def test_choice_determinism(self, engine: GamanEngine) -> None:
        state = {"endpoint": "/api/v1/auth", "status": 401}
        options = ["retry", "block", "log"]
        res1 = engine.choice(state, options)
        res2 = engine.choice(state, options)

        assert res1["selection"] == res2["selection"]
        assert res1["confidence"] == res2["confidence"]

    def test_choice_single_option(self, engine: GamanEngine) -> None:
        state = {"task": "cleanup"}
        options = ["archive"]
        result = engine.choice(state, options)
        assert result["selection"] == "archive"
        assert result["confidence"] == 1.0


@_requires_model
class TestNoulPrimitive:
    """Tests for the noul (guardrail / predicate) primitive."""

    def test_noul_schema_and_types(self, engine: GamanEngine) -> None:
        state = {"user_id": 123, "action": "delete_all"}
        predicate = "Is this a destructive action?"
        result = engine.noul(state, predicate)

        assert result["primitive"] == "noul"
        assert isinstance(result["passed"], bool)
        assert isinstance(result["probability"], float)
        assert 0.0 <= result["probability"] <= 1.0
        assert isinstance(result["latency_ms"], float)
        assert result["latency_ms"] >= 0.0

    def test_noul_destructive_action_detected(self, engine: GamanEngine) -> None:
        state = {"user_id": 123, "action": "delete_all"}
        result = engine.noul(state, "Is this a destructive action?")
        assert result["passed"] is True
        assert result["probability"] > 0.5

    def test_noul_determinism(self, engine: GamanEngine) -> None:
        state = {"role": "guest", "access": "read_only"}
        predicate = "Does the user have admin privileges?"
        res1 = engine.noul(state, predicate)
        res2 = engine.noul(state, predicate)

        assert res1["passed"] == res2["passed"]
        assert res1["probability"] == res2["probability"]


@_requires_model
class TestScorePrimitive:
    """Tests for the score (context evaluator) primitive."""

    def test_score_schema_and_types(self, engine: GamanEngine) -> None:
        state = {"review": "The product broke after two days of use."}
        criterion = "Severity of hardware failure"
        result = engine.score(state, criterion)

        assert result["primitive"] == "score"
        assert isinstance(result["value"], float)
        assert 0.0 <= result["value"] <= 1.0
        assert isinstance(result["latency_ms"], float)
        assert result["latency_ms"] >= 0.0

    def test_score_determinism(self, engine: GamanEngine) -> None:
        state = {"temperature": 105, "pressure": 450}
        criterion = "Risk of catastrophic reactor overheat"
        res1 = engine.score(state, criterion)
        res2 = engine.score(state, criterion)

        assert res1["value"] == res2["value"]


@_requires_model
class TestEmbedMethod:
    """Tests for the raw representation vector embedding."""

    def test_embed_returns_numpy_array(self, engine: GamanEngine) -> None:
        state = {"system": "auth", "latency": 120}
        emb = engine.embed(state)
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1
        assert len(emb) > 0


@_requires_model
class TestShiftInvariance:
    """Verifies that 3-class normalized Softmax is mathematically invariant to constant logit shifts."""

    def test_shift_invariance_on_primitives(
        self, engine: GamanEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = {"user_id": 123, "action": "delete_all"}
        options = ["approve", "reject", "flag"]

        # Run unshifted baseline
        choice_base = engine.choice(state, options)
        noul_base = engine.noul(state, "Is this a destructive action?")
        score_base = engine.score(state, "Severity of action")

        # Mock forward pass to add constant +10.0 to all logits
        orig_forward = engine._forward

        def shifted_forward(encodings: dict[str, np.ndarray]) -> np.ndarray:
            logits = orig_forward(encodings)
            return logits + 10.0

        monkeypatch.setattr(engine, "_forward", shifted_forward)

        # Run shifted pass
        choice_shifted = engine.choice(state, options)
        noul_shifted = engine.noul(state, "Is this a destructive action?")
        score_shifted = engine.score(state, "Severity of action")

        assert choice_shifted["selection"] == choice_base["selection"]
        assert pytest.approx(choice_shifted["confidence"], abs=1e-4) == choice_base["confidence"]
        assert pytest.approx(noul_shifted["probability"], abs=1e-4) == noul_base["probability"]
        assert pytest.approx(score_shifted["value"], abs=1e-4) == score_base["value"]
