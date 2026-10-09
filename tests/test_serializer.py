"""
tests/test_serializer.py
========================
Pytest unit tests for ``src/serializer.py``.

These tests have zero external dependencies — they run on a fresh clone
without any model artifacts downloaded.

Coverage areas:
    - flatten_state: primitive types, nesting, lists, bool/None disambiguation
    - serialize_state: key sorting, pipe separator, idempotency, API examples
    - build_nli_input: structural markers, ordering, exact format, determinism
"""

import pytest

from src.serializer import build_nli_input, flatten_state, serialize_state


# ─── flatten_state ────────────────────────────────────────────────────────────

class TestFlattenState:
    """Unit tests for the recursive JSON flattener."""

    def test_flat_dict_preserves_values(self) -> None:
        result = flatten_state({"b": 2, "a": 1})
        assert result == {"a": "1", "b": "2"}

    def test_nested_dict_uses_dot_notation(self) -> None:
        result = flatten_state({"cpu": {"usage": 98}})
        assert result == {"cpu.usage": "98"}

    def test_deeply_nested_three_levels(self) -> None:
        result = flatten_state({"a": {"b": {"c": "deep"}}})
        assert result == {"a.b.c": "deep"}

    def test_mixed_depth(self) -> None:
        result = flatten_state({"top": 1, "nested": {"key": "val"}})
        assert result == {"top": "1", "nested.key": "val"}

    def test_empty_dict_returns_empty(self) -> None:
        assert flatten_state({}) == {}

    # ── Type serialization ────────────────────────────────────────────────────

    def test_none_becomes_null(self) -> None:
        assert flatten_state({"key": None}) == {"key": "null"}

    def test_true_becomes_lowercase_true(self) -> None:
        # Must be JSON-consistent "true", NOT Python's "True"
        assert flatten_state({"flag": True}) == {"flag": "true"}

    def test_false_becomes_lowercase_false(self) -> None:
        assert flatten_state({"flag": False}) == {"flag": "false"}

    def test_bool_not_treated_as_int(self) -> None:
        # Python: bool is a subclass of int. True == 1, but must serialize as "true"
        assert flatten_state({"v": True})["v"] == "true"
        assert flatten_state({"v": True})["v"] != "1"

    def test_integer(self) -> None:
        assert flatten_state({"count": 42}) == {"count": "42"}

    def test_float(self) -> None:
        assert flatten_state({"score": 0.95}) == {"score": "0.95"}

    def test_zero_int(self) -> None:
        assert flatten_state({"n": 0}) == {"n": "0"}

    def test_zero_float(self) -> None:
        assert flatten_state({"n": 0.0}) == {"n": "0.0"}

    def test_string_passthrough(self) -> None:
        assert flatten_state({"msg": "hello world"}) == {"msg": "hello world"}

    # ── List serialization ────────────────────────────────────────────────────

    def test_list_of_strings(self) -> None:
        assert flatten_state({"tags": ["a", "b", "c"]}) == {"tags": "a, b, c"}

    def test_list_of_numbers(self) -> None:
        assert flatten_state({"vals": [1, 2, 3]}) == {"vals": "1, 2, 3"}

    def test_empty_list(self) -> None:
        assert flatten_state({"tags": []}) == {"tags": ""}

    def test_list_with_none(self) -> None:
        assert flatten_state({"vals": [None, 1]}) == {"vals": "null, 1"}

    def test_list_with_booleans(self) -> None:
        assert flatten_state({"flags": [True, False]}) == {"flags": "true, false"}

    def test_singleton_list(self) -> None:
        assert flatten_state({"x": [42]}) == {"x": "42"}


# ─── serialize_state ─────────────────────────────────────────────────────────

class TestSerializeState:
    """Unit tests for the deterministic key-value string serializer."""

    def test_keys_sorted_alphabetically(self) -> None:
        result = serialize_state({"z": 1, "a": 2, "m": 3})
        assert result == "a: 2 | m: 3 | z: 1"

    def test_pipe_separator_between_pairs(self) -> None:
        result = serialize_state({"a": 1, "b": 2})
        assert " | " in result

    def test_no_trailing_pipe(self) -> None:
        result = serialize_state({"a": 1, "b": 2})
        assert not result.endswith(" | ")

    def test_single_key_no_pipe(self) -> None:
        result = serialize_state({"key": "value"})
        assert result == "key: value"
        assert " | " not in result

    def test_empty_dict_returns_empty_string(self) -> None:
        assert serialize_state({}) == ""

    def test_nested_keys_sorted_with_dot_notation(self) -> None:
        # flat keys after flattening: "a", "z.b" → sorted: ["a", "z.b"]
        result = serialize_state({"z": {"b": 2}, "a": 1})
        assert result == "a: 1 | z.b: 2"

    def test_idempotent_same_input(self) -> None:
        state = {"user_id": 123, "action": "delete", "nested": {"level": 2}}
        assert serialize_state(state) == serialize_state(state)

    def test_output_independent_of_insertion_order(self) -> None:
        # Python dicts preserve insertion order; serialize_state must not
        s1 = serialize_state({"z": 1, "a": 2})
        s2 = serialize_state({"a": 2, "z": 1})
        assert s1 == s2

    # ── API contract examples from docs/api_contract.md ──────────────────────

    def test_api_contract_noul_example(self) -> None:
        state = {"user_id": 123, "action": "delete_all"}
        assert serialize_state(state) == "action: delete_all | user_id: 123"

    def test_api_contract_choice_example(self) -> None:
        state = {"cpu_usage": 98, "memory_usage": 85}
        assert serialize_state(state) == "cpu_usage: 98 | memory_usage: 85"

    def test_api_contract_score_example(self) -> None:
        state = {"review": "The product broke after two days of use."}
        assert serialize_state(state) == (
            "review: The product broke after two days of use."
        )

    def test_different_states_produce_different_strings(self) -> None:
        assert serialize_state({"a": 1}) != serialize_state({"a": 2})

    def test_nested_vs_flat_same_key_same_value(self) -> None:
        # {"a": {"b": 1}} and {"a.b": 1} should produce the same string
        assert serialize_state({"a": {"b": 1}}) == serialize_state({"a.b": 1})


# ─── build_nli_input ─────────────────────────────────────────────────────────

class TestBuildNliInput:
    """Unit tests for the full [STATE]...[QUERY] format builder."""

    def test_starts_with_state_marker(self) -> None:
        result = build_nli_input({"key": "val"}, "query")
        assert result.startswith("[STATE]")

    def test_contains_query_marker(self) -> None:
        result = build_nli_input({"key": "val"}, "query")
        assert "[QUERY]" in result

    def test_state_marker_before_query_marker(self) -> None:
        result = build_nli_input({"key": "val"}, "my query")
        assert result.index("[STATE]") < result.index("[QUERY]")

    def test_query_text_is_last(self) -> None:
        query = "Is this a destructive action?"
        result = build_nli_input({"key": "val"}, query)
        assert result.endswith(query)

    def test_exact_format_api_noul(self) -> None:
        state = {"action": "delete_all", "user_id": 123}
        result = build_nli_input(state, "Is this destructive?")
        expected = "[STATE] action: delete_all | user_id: 123 [QUERY] Is this destructive?"
        assert result == expected

    def test_exact_format_api_choice(self) -> None:
        state = {"cpu_usage": 98, "memory_usage": 85}
        query = "This state corresponds to: scale_up"
        result = build_nli_input(state, query)
        expected = "[STATE] cpu_usage: 98 | memory_usage: 85 [QUERY] This state corresponds to: scale_up"
        assert result == expected

    def test_empty_state_produces_double_space(self) -> None:
        # serialize_state({}) == "" → "[STATE]  [QUERY] query"
        result = build_nli_input({}, "Is anything here?")
        assert result == "[STATE]  [QUERY] Is anything here?"

    def test_deterministic_repeated_calls(self) -> None:
        state = {"z": 99, "a": 1, "m": {"x": True}}
        query = "Is this consistent?"
        assert build_nli_input(state, query) == build_nli_input(state, query)

    def test_different_states_different_output(self) -> None:
        query = "same query"
        assert build_nli_input({"a": 1}, query) != build_nli_input({"a": 2}, query)

    def test_different_queries_different_output(self) -> None:
        state = {"key": "val"}
        assert build_nli_input(state, "query A") != build_nli_input(state, "query B")

    def test_choice_option_format(self) -> None:
        """Simulate the exact format used by GamanEngine.choice per option."""
        state = {"cpu_usage": 98, "memory_usage": 85}
        option = "scale_up"
        result = build_nli_input(state, f"This state corresponds to: {option}")
        assert "This state corresponds to: scale_up" in result
        assert result.startswith("[STATE]")
