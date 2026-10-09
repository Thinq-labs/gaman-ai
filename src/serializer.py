"""
src/serializer.py — Universal Deterministic JSON Flattener
===========================================================
Converts an arbitrary JSON-compatible dict (any depth, any primitive types)
into the canonical ``[STATE] ... [QUERY] ...`` string format consumed by the
Gaman AI NLI engine.

Serialization format (spec §2.4):
    [STATE] key1: value1 | key2: value2 | nested.key: value3 [QUERY] <query>

Determinism guarantees:
    - Keys are sorted alphabetically at each nesting level after flattening.
    - Nested dict keys use dot-notation (``parent.child``).
    - Lists are serialized as comma-separated primitive values.
    - ``None``  → ``"null"``
    - ``True``  → ``"true"``   (JSON-consistent casing, not Python's "True")
    - ``False`` → ``"false"``
    - Numeric types use Python's ``str()`` — no trailing zeros introduced.
"""

from __future__ import annotations

from typing import Any


# ─── Primitive value serializer ──────────────────────────────────────────────

def _serialize_value(value: Any) -> str:
    """
    Convert a single leaf value to its canonical string representation.

    Lists are expanded recursively as comma-separated values.
    ``bool`` is checked before ``int`` because ``bool`` is a subclass of ``int``
    in Python; without this order, ``True`` would serialize as ``"1"``.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(_serialize_value(item) for item in value)
    return str(value)


# ─── Recursive key flattener ─────────────────────────────────────────────────

def flatten_state(state: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """
    Recursively flatten a nested dict using dot-notation for nested keys.

    All leaf values are converted to strings via :func:`_serialize_value`.
    The returned dict preserves insertion order from the recursion, but keys
    are NOT yet sorted — sorting is the caller's responsibility (done in
    :func:`serialize_state`).

    Args:
        state:  Arbitrary JSON-compatible dict (any depth).
        prefix: Internal — tracks the current key path during recursion.
                Pass ``""`` (default) for the top-level call.

    Returns:
        A flat ``dict[str, str]`` mapping dot-notation keys to string values.

    Examples:
        >>> flatten_state({"cpu": {"usage": 98}, "memory": 85})
        {'cpu.usage': '98', 'memory': '85'}

        >>> flatten_state({"a": {"b": {"c": "deep"}}})
        {'a.b.c': 'deep'}

        >>> flatten_state({"tags": ["x", "y"], "active": True})
        {'tags': 'x, y', 'active': 'true'}
    """
    result: dict[str, str] = {}
    for key, value in state.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_state(value, prefix=full_key))
        else:
            result[full_key] = _serialize_value(value)
    return result


# ─── Public API ──────────────────────────────────────────────────────────────

def serialize_state(state: dict[str, Any]) -> str:
    """
    Serialize a state dict into a canonical key-value string.

    Keys are sorted alphabetically after flattening for strict determinism.
    The same input dict will always produce the exact same output string,
    regardless of Python dict insertion order.

    Args:
        state: Arbitrary JSON-compatible dict (any depth, any primitive values).

    Returns:
        Canonical key-value string: ``"key1: v1 | key2: v2 | ..."``.
        Returns an empty string ``""`` for an empty state dict.

    Examples:
        >>> serialize_state({"user_id": 123, "action": "delete_all"})
        'action: delete_all | user_id: 123'

        >>> serialize_state({"cpu": {"usage": 98}, "memory": 85})
        'cpu.usage: 98 | memory: 85'

        >>> serialize_state({})
        ''
    """
    flat = flatten_state(state)
    pairs = [f"{k}: {v}" for k, v in sorted(flat.items())]
    return " | ".join(pairs)


def build_nli_input(state: dict[str, Any], query: str) -> str:
    """
    Build the full NLI input string for the DeBERTa model.

    Maps the API contract to the model's expected premise-hypothesis format:
    - **Premise** = serialized state (the factual context).
    - **Hypothesis** = ``query`` (the predicate, criterion, or option label).

    Format (spec §2.4):
        ``"[STATE] key1: v1 | key2: v2 [QUERY] <query_text>"``

    Args:
        state: Arbitrary JSON state dict — the NLI premise.
        query: Hypothesis text. This is:
               - The ``predicate`` string for ``noul`` calls.
               - The ``criterion`` string for ``score`` calls.
               - A single ``"This state corresponds to: {option}"`` string
                 for each option in a ``choice`` call.

    Returns:
        Full NLI input string, ready for tokenization by :class:`GamanTokenizer`.

    Examples:
        >>> build_nli_input({"action": "delete_all"}, "Is this destructive?")
        '[STATE] action: delete_all [QUERY] Is this destructive?'

        >>> build_nli_input({}, "Is anything here?")
        '[STATE]  [QUERY] Is anything here?'
    """
    state_str = serialize_state(state)
    return f"[STATE] {state_str} [QUERY] {query}"
