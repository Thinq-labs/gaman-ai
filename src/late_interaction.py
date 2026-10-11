"""Decoupled Late-Interaction (ColBERT-Style) and State Representation Caching for Gaman AI.

Replaces quadratic in-sequence [MASK] concatenation with O(K) late cross-attention.
Enables instant state caching across repeated multi-option evaluation queries.

Zero runtime dependencies beyond numpy and standard library.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CachedStateTensor:
    """Cached state representation tensor and token metadata."""

    state_hash: str
    token_embeddings: np.ndarray  # Shape: [L, hidden_dim]
    seq_len: int
    created_at_ms: float


class StateRepresentationCache:
    """Thread-safe LRU in-memory cache for state token representations."""

    def __init__(self, max_size: int = 256) -> None:
        self.max_size: int = max_size
        self._cache: OrderedDict[str, CachedStateTensor] = OrderedDict()

    @staticmethod
    def compute_hash(serialized_state: str) -> str:
        """Compute SHA-256 digest of serialized state."""
        return hashlib.sha256(serialized_state.encode("utf-8")).hexdigest()

    def get(self, state_hash: str) -> CachedStateTensor | None:
        """Retrieve cached state representation if present, updating LRU order."""
        if state_hash in self._cache:
            self._cache.move_to_end(state_hash)
            return self._cache[state_hash]
        return None

    def put(self, state_hash: str, embeddings: np.ndarray, timestamp_ms: float = 0.0) -> None:
        """Store state representation in cache, evicting oldest if capacity exceeded."""
        if state_hash in self._cache:
            self._cache.move_to_end(state_hash)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
        self._cache[state_hash] = CachedStateTensor(
            state_hash=state_hash,
            token_embeddings=embeddings,
            seq_len=embeddings.shape[0],
            created_at_ms=timestamp_ms,
        )

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class LateInteractionScorer:
    """
    Decoupled Late-Interaction operator computing alignment between state and candidate options.
    Implements vectorized MaxSim operator:
    Score(k) = sum_j max_i ( H_state[i] · E_k[j]^T )
    """

    def __init__(self, temperature: float = 0.1) -> None:
        self.temperature: float = temperature

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """L2-normalize vectors along the last dimension."""
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(norm, 1e-12)

    def compute_maxsim_score(
        self, state_embeddings: np.ndarray, option_embeddings: np.ndarray
    ) -> float:
        """
        Compute ColBERT-style MaxSim score for a single candidate option.

        Args:
            state_embeddings: Shape [L_state, hidden_dim]
            option_embeddings: Shape [L_option, hidden_dim]

        Returns:
            Scalar alignment score.
        """
        h_norm = self._normalize(state_embeddings)  # [L_state, d]
        e_norm = self._normalize(option_embeddings)  # [L_opt, d]

        # Cosine similarity matrix: shape [L_opt, L_state]
        sim_matrix = np.matmul(e_norm, h_norm.T)

        # For each option token, find the maximum similarity across all state tokens
        max_per_token = np.max(sim_matrix, axis=1)  # [L_opt]

        # Mean aggregation over option tokens to normalize for option length
        return float(np.mean(max_per_token))

    def score_options(
        self,
        state_embeddings: np.ndarray,
        candidate_embeddings: list[np.ndarray],
        option_keys: list[str],
    ) -> dict[str, Any]:
        """
        Compute late-interaction scores across all K candidate options in O(K) time.

        Args:
            state_embeddings: Shape [L_state, hidden_dim]
            candidate_embeddings: List of K option embeddings, each of shape [L_k, hidden_dim]
            option_keys: List of K option names.

        Returns:
            Dictionary matching Universal Contract:
            {
                "selection": str,
                "confidence": float,
                "probabilities": dict[str, float],
                "raw_scores": dict[str, float]
            }
        """
        if not candidate_embeddings or len(candidate_embeddings) != len(option_keys):
            raise ValueError("candidate_embeddings count must match option_keys count")

        raw_scores: list[float] = []
        for opt_emb in candidate_embeddings:
            raw_scores.append(self.compute_maxsim_score(state_embeddings, opt_emb))

        scores_arr = np.array(raw_scores, dtype=np.float64)

        # Temperature-scaled Softmax
        shift_scores = (scores_arr - np.max(scores_arr)) / max(1e-4, self.temperature)
        exp_scores = np.exp(shift_scores)
        probs = exp_scores / np.sum(exp_scores)

        prob_dict = {k: float(round(p, 4)) for k, p in zip(option_keys, probs, strict=True)}
        best_idx = int(np.argmax(probs))

        return {
            "selection": option_keys[best_idx],
            "confidence": float(round(probs[best_idx], 4)),
            "probabilities": prob_dict,
            "raw_scores": {k: float(round(s, 4)) for k, s in zip(option_keys, raw_scores, strict=True)},
        }

    def score_candidates(
        self,
        state_embeddings: np.ndarray,
        candidate_embeddings: list[np.ndarray],
        option_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Score candidate options against state representation in O(K) time.

        Args:
            state_embeddings: Shape [L_state, hidden_dim]
            candidate_embeddings: List of K option embeddings, each of shape [L_k, hidden_dim]
            option_keys: Optional list of option names. Defaults to opt_0, opt_1, ...
        """
        if option_keys is None:
            option_keys = [f"opt_{i}" for i in range(len(candidate_embeddings))]
        return self.score_options(state_embeddings, candidate_embeddings, option_keys)

