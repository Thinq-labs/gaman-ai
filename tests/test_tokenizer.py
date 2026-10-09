"""
tests/test_tokenizer.py
========================
Pytest unit tests for ``src/tokenizer.py``.

**Tokenizer-dependent tests** require ``models/tokenizer.json`` to exist.
If the file is absent, those test classes are SKIPPED (not failed) with an
actionable message. Run ``python scripts/export_backbone.py`` to generate it.

**Error-path tests** (``TestFileNotFoundError``) always run — they test the
constructor's failure mode using a fresh empty ``tmp_path`` directory.

Test matrix:
    - TestEncode        — single encode output shape, dtype, determinism, truncation
    - TestEncodeBatch   — batch shape, padding, dtype, consistency with single encode
    - TestFileNotFoundError — FileNotFoundError raised when artifact is missing
"""

import numpy as np
import pytest

from pathlib import Path

# ─── Guard: skip tokenizer-dependent tests if artifact is absent ──────────────
MODELS_DIR = Path("models")
TOKENIZER_JSON = MODELS_DIR / "tokenizer.json"

_requires_tokenizer = pytest.mark.skipif(
    not TOKENIZER_JSON.exists(),
    reason=(
        f"'{TOKENIZER_JSON}' not found. "
        "Run: python scripts/export_backbone.py"
    ),
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tokenizer():
    """
    Module-scoped fixture: loads GamanTokenizer once for the whole test session.
    Skipped automatically if tokenizer.json is absent (guard above handles it).
    """
    from src.tokenizer import GamanTokenizer
    return GamanTokenizer(models_dir=MODELS_DIR)


# ─── Single-string encode ─────────────────────────────────────────────────────

@_requires_tokenizer
class TestEncode:
    """Tests for GamanTokenizer.encode() — single string input."""

    def test_returns_dict(self, tokenizer) -> None:
        result = tokenizer.encode("Hello world")
        assert isinstance(result, dict)

    def test_has_all_required_keys(self, tokenizer) -> None:
        result = tokenizer.encode("Hello world")
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "token_type_ids" in result

    def test_all_values_are_numpy_arrays(self, tokenizer) -> None:
        result = tokenizer.encode("Hello world")
        for key, arr in result.items():
            assert isinstance(arr, np.ndarray), f"'{key}' must be np.ndarray"

    def test_all_values_are_int64(self, tokenizer) -> None:
        result = tokenizer.encode("Hello world")
        for key, arr in result.items():
            assert arr.dtype == np.int64, (
                f"'{key}' must be int64 for ONNX Runtime compatibility"
            )

    def test_batch_dimension_is_one(self, tokenizer) -> None:
        result = tokenizer.encode("Hello world")
        for key, arr in result.items():
            assert arr.shape[0] == 1, f"'{key}' must have batch dim = 1 for single encode"

    def test_all_arrays_have_same_seq_len(self, tokenizer) -> None:
        result = tokenizer.encode("Hello world")
        seq_len = result["input_ids"].shape[1]
        assert result["attention_mask"].shape[1] == seq_len
        assert result["token_type_ids"].shape[1] == seq_len

    def test_attention_mask_has_nonzero_ones(self, tokenizer) -> None:
        result = tokenizer.encode("Short text here")
        mask = result["attention_mask"][0]
        assert mask.sum() > 0, "Attention mask must have at least some active tokens"

    def test_attention_mask_values_binary(self, tokenizer) -> None:
        result = tokenizer.encode("Test sentence")
        mask = result["attention_mask"][0]
        assert set(mask.tolist()).issubset({0, 1}), "Mask values must be 0 or 1"

    def test_determinism_same_text_same_ids(self, tokenizer) -> None:
        text = "[STATE] cpu_usage: 98 | memory_usage: 85 [QUERY] scale_up"
        r1 = tokenizer.encode(text)
        r2 = tokenizer.encode(text)
        np.testing.assert_array_equal(
            r1["input_ids"], r2["input_ids"],
            err_msg="input_ids must be identical for the same input (determinism)"
        )
        np.testing.assert_array_equal(
            r1["attention_mask"], r2["attention_mask"],
            err_msg="attention_mask must be identical for the same input"
        )

    def test_different_texts_produce_different_ids(self, tokenizer) -> None:
        r1 = tokenizer.encode("Text A with distinct content")
        r2 = tokenizer.encode("Completely different sequence B")
        # They may differ in length or content — at least one array differs
        same = np.array_equal(r1["input_ids"], r2["input_ids"])
        assert not same, "Different texts must produce different token IDs"

    def test_truncation_respects_max_length(self, tokenizer) -> None:
        # 1000 repetitions of a word guarantees exceeding MAX_LENGTH tokens
        long_text = " ".join(["tokenization"] * 1000)
        result = tokenizer.encode(long_text)
        assert result["input_ids"].shape[1] <= tokenizer.MAX_LENGTH, (
            f"Sequence must be truncated to MAX_LENGTH={tokenizer.MAX_LENGTH}"
        )

    def test_nli_format_tokenizes_without_error(self, tokenizer) -> None:
        """Smoke test: the actual [STATE]...[QUERY] format tokenizes cleanly."""
        nli_text = (
            "[STATE] action: delete_all | user_id: 123 "
            "[QUERY] Is this a destructive action?"
        )
        result = tokenizer.encode(nli_text)
        assert result["input_ids"].shape[0] == 1
        assert result["input_ids"].shape[1] > 0

    def test_pair_encoding_token_type_ids(self, tokenizer) -> None:
        """Verify (premise, hypothesis) encoding outputs correct token_type_ids transition."""
        premise = "cpu_usage: 98 | memory_usage: 85"
        hypothesis = "This state corresponds to: scale_up"
        result = tokenizer.encode(premise, pair=hypothesis)
        type_ids = result["token_type_ids"][0]
        # Should have both 0s (premise) and 1s (hypothesis)
        assert 0 in type_ids
        assert 1 in type_ids
        # First token is 0 (CLS) and last non-padded token is 1
        assert type_ids[0] == 0
        assert type_ids[-1] == 1


# ─── Batch encode ─────────────────────────────────────────────────────────────

@_requires_tokenizer
class TestEncodeBatch:
    """Tests for GamanTokenizer.encode_batch() — list of strings input."""

    def test_returns_dict(self, tokenizer) -> None:
        result = tokenizer.encode_batch(["text one", "text two"])
        assert isinstance(result, dict)

    def test_has_all_required_keys(self, tokenizer) -> None:
        result = tokenizer.encode_batch(["text one", "text two"])
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "token_type_ids" in result

    def test_all_values_are_numpy_arrays(self, tokenizer) -> None:
        result = tokenizer.encode_batch(["Hello", "World"])
        for key, arr in result.items():
            assert isinstance(arr, np.ndarray), f"'{key}' must be np.ndarray"

    def test_all_values_are_int64(self, tokenizer) -> None:
        result = tokenizer.encode_batch(["Hello", "World"])
        for key, arr in result.items():
            assert arr.dtype == np.int64, f"'{key}' must be int64"

    def test_batch_dimension_matches_input_count(self, tokenizer) -> None:
        texts = ["alpha", "beta", "gamma"]
        result = tokenizer.encode_batch(texts)
        assert result["input_ids"].shape[0] == 3

    def test_all_arrays_have_same_shape(self, tokenizer) -> None:
        texts = ["short", "a much longer sentence with additional words here"]
        result = tokenizer.encode_batch(texts)
        seq_len = result["input_ids"].shape[1]
        assert result["attention_mask"].shape[1] == seq_len, (
            "All output arrays must have the same seq_len after padding"
        )
        assert result["token_type_ids"].shape[1] == seq_len

    def test_padding_aligns_sequences(self, tokenizer) -> None:
        """Shorter sequences must be padded so all have the same seq_len."""
        texts = ["short", "a much longer sentence with many more words in it"]
        result = tokenizer.encode_batch(texts)
        # The short sequence will have trailing 0s in the attention mask
        short_mask = result["attention_mask"][0]
        long_mask = result["attention_mask"][1]
        assert short_mask.sum() < long_mask.sum(), (
            "Short sequence should have fewer active tokens than the long one"
        )

    def test_determinism(self, tokenizer) -> None:
        texts = ["first sequence here", "second sequence there"]
        r1 = tokenizer.encode_batch(texts)
        r2 = tokenizer.encode_batch(texts)
        np.testing.assert_array_equal(r1["input_ids"], r2["input_ids"])
        np.testing.assert_array_equal(r1["attention_mask"], r2["attention_mask"])

    def test_single_batch_consistent_with_encode(self, tokenizer) -> None:
        """encode([text]) must produce identical token IDs as encode_batch([text])."""
        text = "Consistency check between encode and encode_batch."
        single = tokenizer.encode(text)
        batch = tokenizer.encode_batch([text])
        np.testing.assert_array_equal(
            single["input_ids"],
            batch["input_ids"],
            err_msg="encode and encode_batch must agree for a single-element input",
        )

    def test_choice_nli_batch_smoke_test(self, tokenizer) -> None:
        """Simulate the K-way batched NLI input used by GamanEngine.choice."""
        state_prefix = "[STATE] cpu_usage: 98 | memory_usage: 85 [QUERY] This state corresponds to: "
        options = ["scale_up", "scale_down", "do_nothing"]
        texts = [f"{state_prefix}{opt}" for opt in options]
        result = tokenizer.encode_batch(texts)
        assert result["input_ids"].shape[0] == 3, "Batch size must equal number of options"
        assert result["input_ids"].shape[1] > 0, "Sequence length must be non-zero"


# ─── Error path (always runs, no artifact needed) ─────────────────────────────

class TestFileNotFoundError:
    """Tests the constructor failure mode — no tokenizer.json required."""

    def test_missing_tokenizer_raises_file_not_found(self, tmp_path: Path) -> None:
        """GamanTokenizer must raise FileNotFoundError for an empty models dir."""
        from src.tokenizer import GamanTokenizer
        with pytest.raises(FileNotFoundError, match="tokenizer.json"):
            GamanTokenizer(models_dir=tmp_path)

    def test_error_message_mentions_export_script(self, tmp_path: Path) -> None:
        """The error message must guide the user toward the fix."""
        from src.tokenizer import GamanTokenizer
        with pytest.raises(FileNotFoundError, match="export_backbone"):
            GamanTokenizer(models_dir=tmp_path)

    def test_nonexistent_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        """A completely missing models dir must also raise FileNotFoundError."""
        from src.tokenizer import GamanTokenizer
        ghost_dir = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            GamanTokenizer(models_dir=ghost_dir)
