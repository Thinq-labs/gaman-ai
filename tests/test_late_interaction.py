"""Tests for Decoupled Late-Interaction and State Caching in Gaman AI."""

import numpy as np
import pytest

from src.late_interaction import (
    LateInteractionScorer,
    StateRepresentationCache,
)


class TestStateRepresentationCache:
    def test_cache_put_get_hit(self):
        cache = StateRepresentationCache(max_size=3)
        h1 = StateRepresentationCache.compute_hash("status: 503 | retry: 4")
        emb1 = np.ones((10, 768), dtype=np.float32)

        cache.put(h1, emb1, timestamp_ms=100.0)
        cached = cache.get(h1)
        assert cached is not None
        assert cached.state_hash == h1
        assert cached.seq_len == 10
        assert np.array_equal(cached.token_embeddings, emb1)

    def test_cache_miss_returns_none(self):
        cache = StateRepresentationCache(max_size=3)
        assert cache.get("nonexistent_hash") is None

    def test_lru_eviction(self):
        cache = StateRepresentationCache(max_size=2)
        h1 = "hash_1"
        h2 = "hash_2"
        h3 = "hash_3"

        cache.put(h1, np.ones((5, 64)))
        cache.put(h2, np.ones((5, 64)))
        assert len(cache) == 2

        # Access h1 to make it most recently used
        _ = cache.get(h1)

        # Insert h3, which should evict h2 (oldest)
        cache.put(h3, np.ones((5, 64)))
        assert len(cache) == 2
        assert cache.get(h1) is not None
        assert cache.get(h2) is None
        assert cache.get(h3) is not None

    def test_hash_determinism(self):
        h1 = StateRepresentationCache.compute_hash("query: SELECT * FROM users")
        h2 = StateRepresentationCache.compute_hash("query: SELECT * FROM users")
        h3 = StateRepresentationCache.compute_hash("query: SELECT id FROM users")
        assert h1 == h2
        assert h1 != h3


class TestLateInteractionScorer:
    def test_maxsim_identical_vectors_gives_unity_alignment(self):
        scorer = LateInteractionScorer()
        # Single token matching exactly
        v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        score = scorer.compute_maxsim_score(v, v)
        assert pytest.approx(score, rel=1e-4) == 1.0

    def test_orthogonal_vectors_gives_zero_alignment(self):
        scorer = LateInteractionScorer()
        v1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        v2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        score = scorer.compute_maxsim_score(v1, v2)
        assert pytest.approx(score, abs=1e-5) == 0.0

    def test_score_options_linear_scaling_and_selection(self):
        scorer = LateInteractionScorer(temperature=0.1)
        hidden_dim = 128
        state_tokens = np.random.randn(20, hidden_dim).astype(np.float32)

        # Create 5 candidate options where candidate 2 is aligned with state token 0
        candidates = []
        options = ["opt_0", "opt_1", "opt_winner", "opt_3", "opt_4"]
        for idx in range(5):
            if idx == 2:
                # Aligned candidate
                opt = np.tile(state_tokens[0:1], (3, 1)) + 0.01 * np.random.randn(3, hidden_dim)
            else:
                opt = np.random.randn(3, hidden_dim).astype(np.float32)
            candidates.append(opt)

        res = scorer.score_options(state_tokens, candidates, options)
        assert res["selection"] == "opt_winner"
        assert res["confidence"] > 0.50
        assert sum(res["probabilities"].values()) == pytest.approx(1.0, rel=1e-3)
        assert len(res["probabilities"]) == 5

    def test_mismatched_candidate_count_raises_error(self):
        scorer = LateInteractionScorer()
        state = np.ones((5, 32))
        cands = [np.ones((2, 32))]
        with pytest.raises(ValueError):
            scorer.score_options(state, cands, ["opt_1", "opt_2"])

    def test_score_candidates_default_keys(self):
        scorer = LateInteractionScorer()
        state = np.ones((4, 16), dtype=np.float32)
        cands = [np.ones((2, 16), dtype=np.float32), np.zeros((2, 16), dtype=np.float32)]
        res = scorer.score_candidates(state, cands)
        assert res["selection"] in ["opt_0", "opt_1"]
        assert len(res["probabilities"]) == 2


class TestEngineLateInteraction:
    def test_engine_late_interaction_initialization(self):
        from src.engine import GamanEngine

        engine = GamanEngine(late_interaction=True)
        assert engine.late_interaction is True
        assert engine.state_cache is not None
        assert engine.late_scorer is not None

    def test_engine_late_interaction_choice_routing(self):
        from src.engine import GamanEngine

        engine = GamanEngine(late_interaction=True)
        state = {"intent": "refund", "amount": 50}
        options = ["REFUND", "SUPPORT", "FEEDBACK"]
        result = engine.choice(state, options)

        assert result["primitive"] == "choice"
        assert result["selection"] in options
        assert 0.0 <= result["confidence"] <= 1.0
        assert "latency_ms" in result
        assert len(result["probabilities"]) == 3

    def test_engine_late_interaction_state_caching_hit(self):
        from src.engine import GamanEngine

        engine = GamanEngine(late_interaction=True)
        state = {"ticket_id": "TCK-9921", "issue": "connection timeout"}
        options = ["NETWORK_ERROR", "DATABASE_ERROR"]

        # First call: populates cache
        _ = engine.choice(state, options)
        initial_cache_len = len(engine.state_cache)
        assert initial_cache_len >= 1

        # Second call with same state: hits cache
        _ = engine.choice(state, options)
        assert len(engine.state_cache) == initial_cache_len

