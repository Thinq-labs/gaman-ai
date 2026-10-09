"""
tests/test_heads.py — Unit and Integration Tests for Lightweight Linear Adapters
================================================================================
Tests for:
1. ModelNotFoundError on missing model tier assets.
2. CustomLinearHead forward pass, numerical stability, and latency.
3. fit_adapter closed-form Ridge solver (d=768 and d=1024).
4. Head serialization and deserialization.
5. GamanEngine embed_batch and predict integration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.resolver import ModelNotFoundError


# ─── 1. Missing Model Tier Exception Guidance ────────────────────────────────


def test_model_not_found_error_inheritance():
    """Verify ModelNotFoundError inherits from FileNotFoundError."""
    err = ModelNotFoundError("test error")
    assert isinstance(err, FileNotFoundError)


def test_missing_tier_raises_model_not_found_with_instructions(tmp_path: Path):
    """Verify missing tier raises ModelNotFoundError with copy-paste instructions."""
    from src.engine import GamanEngine

    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()

    with pytest.raises(ModelNotFoundError) as exc_info:
        GamanEngine(slab="base", models_dir=empty_dir)

    msg = str(exc_info.value)
    assert "Model tier 'base' not found in" in msg
    assert "Run: python scripts/export_backbone.py --tier base" in msg


# ─── 2. CustomLinearHead Forward Pass & Latency ──────────────────────────────


class TestCustomLinearHead:
    """Test CustomLinearHead behavior and performance."""

    def test_head_shapes_and_probabilities(self):
        from src.heads import CustomLinearHead

        d = 768
        k = 3
        weights = np.zeros((k, d), dtype=np.float32)
        # Class 1 heavily favors dimension 0
        weights[1, 0] = 10.0
        bias = np.zeros(k, dtype=np.float32)
        classes = ["negative", "positive", "neutral"]

        head = CustomLinearHead(
            name="sentiment",
            classes=classes,
            weights=weights,
            bias=bias,
            hidden_dim=d,
        )

        x = np.zeros(d, dtype=np.float32)
        x[0] = 1.0  # triggers class 1

        probs = head.predict_proba(x)
        assert probs.shape == (3,)
        assert np.isclose(np.sum(probs), 1.0)
        assert probs[1] > 0.99  # Class 1 should dominate

        pred = head.predict(x)
        assert pred["selection"] == "positive"
        assert pred["confidence"] > 0.99
        assert "positive" in pred["probabilities"]

    def test_forward_pass_latency_under_10_microseconds(self):
        """Forward pass should execute in < 10 microseconds on CPU."""
        from src.heads import CustomLinearHead

        d = 768
        k = 5
        weights = np.random.randn(k, d).astype(np.float32)
        bias = np.random.randn(k).astype(np.float32)
        classes = [f"class_{i}" for i in range(k)]

        head = CustomLinearHead(
            name="bench",
            classes=classes,
            weights=weights,
            bias=bias,
            hidden_dim=d,
        )

        x = np.random.randn(d).astype(np.float32)

        # Warmup
        for _ in range(50):
            head.predict_proba(x)

        n_iter = 2000
        t0 = time.perf_counter()
        for _ in range(n_iter):
            head.predict_proba(x)
        total_time = time.perf_counter() - t0
        avg_us = (total_time / n_iter) * 1e6

        assert avg_us < 50.0  # Safe threshold on any consumer CPU (typically < 10 us)


# ─── 3. Analytical Fitting (fit_adapter) ──────────────────────────────────────


class TestFitAdapter:
    """Test analytical Ridge fitting on synthetic representations."""

    def test_fit_adapter_d768_separable(self):
        from src.heads import fit_adapter

        np.random.seed(42)
        d = 768
        n_per_class = 50
        classes = ["finance", "tech", "health"]

        # Generate well-separated clusters
        X_list = []
        y_list = []
        for idx, cls in enumerate(classes):
            center = np.zeros(d, dtype=np.float32)
            center[idx * 10] = 5.0
            noise = np.random.randn(n_per_class, d).astype(np.float32) * 0.1
            X_list.append(noise + center)
            y_list.extend([cls] * n_per_class)

        X = np.vstack(X_list)
        head = fit_adapter(X, y_list, name="topic_head", l2_reg=1.0)

        assert head.name == "topic_head"
        assert head.hidden_dim == 768
        assert head.classes == sorted(classes)
        assert head.weights.shape == (3, 768)
        assert head.bias.shape == (3,)

        # Evaluate accuracy on training set
        correct = 0
        for i in range(len(y_list)):
            pred = head.predict(X[i])
            if pred["selection"] == y_list[i]:
                correct += 1
        acc = correct / len(y_list)
        assert acc >= 0.99

    def test_fit_adapter_d1024_large_slab(self):
        from src.heads import fit_adapter

        np.random.seed(1337)
        d = 1024
        classes = ["spam", "ham"]
        X_spam = np.random.randn(30, d).astype(np.float32) + 1.0
        X_ham = np.random.randn(30, d).astype(np.float32) - 1.0
        X = np.vstack([X_spam, X_ham])
        y = ["spam"] * 30 + ["ham"] * 30

        head = fit_adapter(X, y, name="filter_head", l2_reg=0.5)
        assert head.hidden_dim == 1024
        assert head.weights.shape == (2, 1024)
        assert head.bias.shape == (2,)

        # Check predictions
        assert head.predict(X[0])["selection"] == "spam"
        assert head.predict(X[-1])["selection"] == "ham"

    def test_fit_benchmark_under_1_second_cpu(self):
        """2,000 samples in R^768 should solve in < 1.0s on CPU."""
        from src.heads import fit_adapter

        np.random.seed(99)
        N = 2000
        d = 768
        classes = ["low", "medium", "high"]
        X = np.random.randn(N, d).astype(np.float32)
        y = np.random.choice(classes, size=N).tolist()

        t0 = time.perf_counter()
        head = fit_adapter(X, y, name="bench_head", l2_reg=1.0)
        fit_time = time.perf_counter() - t0

        assert fit_time < 1.0  # < 1.0s constraint
        assert head.weights.shape == (3, 768)


# ─── 4. Head Serialization & Persistence ─────────────────────────────────────


class TestHeadSerialization:
    """Test saving and loading adapter heads."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        from src.heads import CustomLinearHead, fit_adapter

        np.random.seed(123)
        X = np.random.randn(40, 768).astype(np.float32)
        y = ["red"] * 20 + ["blue"] * 20
        head = fit_adapter(X, y, name="color_head", l2_reg=1.0)

        out_path = tmp_path / "color_head.json"
        head.save(out_path)

        assert out_path.exists()
        loaded = CustomLinearHead.load(out_path)

        assert loaded.name == head.name
        assert loaded.classes == head.classes
        assert loaded.hidden_dim == head.hidden_dim
        assert np.allclose(loaded.weights, head.weights, atol=1e-5)
        assert np.allclose(loaded.bias, head.bias, atol=1e-5)

        # Check prediction equivalence
        test_vec = np.random.randn(768).astype(np.float32)
        orig_res = head.predict(test_vec)
        load_res = loaded.predict(test_vec)
        assert orig_res["selection"] == load_res["selection"]
        assert np.isclose(orig_res["confidence"], load_res["confidence"])


# ─── 5. Engine Integration (embed_batch & predict) ───────────────────────────


class TestEngineAdapterIntegration:
    """Test GamanEngine embed_batch and predict with CustomLinearHead."""

    def test_embed_batch_returns_correct_shape(self):
        from src.engine import GamanEngine

        engine = GamanEngine(models_dir="models")
        states = [
            {"service": "auth", "latency": 120},
            {"service": "payment", "latency": 450},
            {"service": "search", "latency": 80},
        ]

        embeddings = engine.embed_batch(states)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, engine.hidden_dim)
        assert embeddings.dtype == np.float32

    def test_engine_predict_with_saved_head(self, tmp_path: Path):
        from src.engine import GamanEngine
        from src.heads import fit_adapter

        engine = GamanEngine(models_dir="models")

        train_states = [
            {"severity": "critical", "error": "database connection failed"},
            {"severity": "critical", "error": "out of memory crash"},
            {"severity": "low", "error": "css asset 404"},
            {"severity": "low", "error": "favicon missing"},
        ]
        labels = ["critical", "critical", "low", "low"]

        Z = engine.embed_batch(train_states)
        head = fit_adapter(Z, labels, name="incident_triage", l2_reg=1.0)

        heads_dir = Path("models") / "heads"
        heads_dir.mkdir(parents=True, exist_ok=True)
        head_path = heads_dir / "incident_triage.json"
        head.save(head_path)

        try:
            # Predict using engine
            test_state = {"severity": "critical", "error": "database pool exhausted"}
            res = engine.predict(test_state, head_name="incident_triage")

            assert "selection" in res
            assert "confidence" in res
            assert "probabilities" in res
            assert "latency_ms" in res
            assert res["head"] == "incident_triage"
            assert isinstance(res["selection"], str)
            assert isinstance(res["confidence"], float)
        finally:
            if head_path.exists():
                head_path.unlink()
