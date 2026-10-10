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

import re
from typing import Any

# ─── Pre-Tokenization Adversarial Normalizer ─────────────────────────────────

def collapse_spaced_tokens(text: str) -> str:
    """
    Collapse sequences of single characters separated by dots, dashes, underscores,
    slashes, or whitespace designed to break subword tokenization
    (e.g., 'r.e.f.u.n.d' -> 'refund', 'p-a-y' -> 'pay', 'w_i_r_e' -> 'wire',
    'R-E-F-U-N-D' -> 'REFUND', 'S P A M' -> 'SPAM', 'V_I_P' -> 'VIP',
    'p-a-s-s-w-o-r-d-c-h-a-n-g-e' -> 'passwordchange').

    Preserves standard hyphenated words like 'a-b testing' and decimals like '10.5%'.
    """
    # 1. Non-whitespace delimiters: dots, dashes, underscores, slashes (length >= 3 single chars)
    text = re.sub(
        r"\b([A-Za-z](?:[\.\-_/][A-Za-z]){2,})\b",
        lambda m: re.sub(r"[\.\-_/]", "", m.group(0)),
        text,
    )
    # 2. Whitespace delimiter: single characters separated by spaces (length >= 3 single chars)
    text = re.sub(
        r"\b([A-Za-z](?:\s[A-Za-z]){2,})\b",
        lambda m: re.sub(r"\s", "", m.group(0)),
        text,
    )
    return text


def extract_zero_percentage_dampeners(text: str) -> list[str]:
    """
    Extract category labels explicitly cancelled or assigned zero weight in the text
    (e.g., '0% spam', '0 % refund', 'no intention of asking for a refund').
    """
    targets: list[str] = []
    # 0% <word>
    for m in re.finditer(r"\b0\s*%\s*([A-Za-z_]+)", text, re.IGNORECASE):
        targets.append(m.group(1).lower())
    # zero <word>
    for m in re.finditer(r"\bzero\s+([A-Za-z_]+)", text, re.IGNORECASE):
        targets.append(m.group(1).lower())
    # no intention of [asking for|requesting] a <word>
    for m in re.finditer(
        r"\bno\s+intention\s+of\s+(?:[\w\s]+\s+)?([A-Za-z_]+)", text, re.IGNORECASE
    ):
        targets.append(m.group(1).lower())
    return targets


def extract_figurative_modifiers(text: str) -> list[tuple[str, str]]:
    """
    Detect figurative qualifiers preceding contractual or intent tokens
    (e.g., 'emotional refund', 'metaphorical override', 'figurative spam').
    """
    pattern = r"\b(emotional|metaphorical|figurative|symbolic|allegorical|philosophical)\s+([A-Za-z_]+)"
    return [
        (m.group(1).lower(), m.group(2).lower())
        for m in re.finditer(pattern, text, re.IGNORECASE)
    ]


def strip_conversational_boilerplate(text: str) -> str:
    """
    Strip leading pleasantries, greetings, and trailing boilerplate/signoffs
    to prevent transformer attention from smearing across conversational padding.
    Retains core clauses containing domain verbs and operational evidence.
    """
    prefix_match = re.match(r"^([a-zA-Z_]+:\s+)", text)
    prefix = prefix_match.group(1) if prefix_match else ""
    body = text[len(prefix) :] if prefix else text

    leading_patterns = [
        r"^\s*(hello|hi|hey)(\s+(team|there|all|support|everyone))?(\s*\.{2,}|\s*[,.!;:–-])*\s*",
        r"^\s*(good\s+(morning|afternoon|evening|day))(\s*\.{2,}|\s*[,.!;:–-])*\s*",
        r"^\s*to\s+whom\s+it\s+may\s+concern(\s*\.{2,}|\s*[,.!;:–-])*\s*",
        r"^\s*dear\s+(team|support|customer\s+service|sir|madam|all|sir\/madam)(\s*\.{2,}|\s*[,.!;:–-])*\s*",
        r"^\s*hope\s+(you\s+are|this\s+finds\s+you)\s+well(\s*\.{2,}|\s*[,.!;:–-])*\s*",
        r"^\s*hope\s+you(\x27re|\x20are)\s+doing\s+well(\s*\.{2,}|\s*[,.!;:–-])*\s*",
    ]

    trailing_patterns = [
        r"(?:[,\s;–-]|\.{2,})*(thanks\s+in\s+advance|thank\s+you(\s+so\s+much|\s+very\s+much)?|thanks|best\s+regards|warm\s+regards|regards|sincerely|cheers|yours\s+truly)\s*[.!]?\s*$",
        r"(?:[,\s;–-]|\.{2,})*(let\s+me\s+know\s+(when|if|how)\s+.*)$",
        r"(?:[,\s;–-]|\.{2,})*(please\s+(advise|help|update|let\s+me\s+know).*)$",
        r"(?:[,\s;–-]|\.{2,})*(fix\s+tracking\s+link.*)$",
    ]

    cleaned = body.strip()
    changed = True
    while changed:
        changed = False
        for p in leading_patterns:
            m = re.match(p, cleaned, re.IGNORECASE)
            if m:
                sub = cleaned[m.end() :].strip()
                if sub:
                    cleaned = sub
                    changed = True

    changed = True
    while changed:
        changed = False
        for p in trailing_patterns:
            m = re.search(p, cleaned, re.IGNORECASE)
            if m:
                sub = cleaned[: m.start()].strip()
                if sub:
                    cleaned = sub
                    changed = True

    return f"{prefix}{cleaned}"


def reweight_adversative_clauses(text: str) -> str:
    """
    Detect adversative conjunctions ('instead', 'however', 'rather than',
    'in reality', 'actually') and counterfactual patterns.
    Isolates the adversative resolution clause and prepends it to the front
    of the context so positional attention heads prioritize the authentic intent.
    """
    adv_pattern = r"\b(instead|however|rather\s+than|in\s+reality|actually)\b"
    prefix_match = re.match(r"^([a-zA-Z_]+:\s+)", text)
    prefix = prefix_match.group(1) if prefix_match else ""
    body = text[len(prefix) :] if prefix else text

    if not re.search(adv_pattern, body, re.IGNORECASE):
        return text

    norm = re.sub(r"\.{2,}", ", ", body).strip()

    # Pattern 1: If [condition] [,;.]* (adversative) [resolution]
    m_if = re.search(
        r"\bif\b\s+([^,;]+?)\s*[,;.]*\s*" + adv_pattern + r"\s+([^,;.!?]+)",
        norm,
        re.IGNORECASE,
    )
    if m_if:
        cond = m_if.group(1).strip()
        adv = m_if.group(2).lower()
        res = m_if.group(3).strip()
        rest = re.sub(
            r"\bif\b\s+([^,;]+?)\s*[,;.]*\s*" + adv_pattern + r"\s+([^,;.!?]+)",
            f"If {cond}, {adv} {res}",
            norm,
            flags=re.IGNORECASE,
        )
        return f"{prefix}{res}. {rest}"

    # Pattern 2: [clause A] [,;.]* (adversative) [clause B]
    m_adv = re.search(
        r"([^,;.!?]+?)\s*[,;.]*\s*" + adv_pattern + r"\s+([^,;.!?]+)",
        norm,
        re.IGNORECASE,
    )
    if m_adv:
        clause_a = m_adv.group(1).strip()
        adv = m_adv.group(2).lower()
        clause_b = m_adv.group(3).strip()
        if len(clause_b.split()) >= 2:
            rest = re.sub(
                r"([^,;.!?]+?)\s*[,;.]*\s*" + adv_pattern + r"\s+([^,;.!?]+)",
                f"{clause_a}, {adv} {clause_b}",
                norm,
                flags=re.IGNORECASE,
            )
            return f"{prefix}{clause_b}. {rest}"

    return text


def sanitize_adversarial_input(
    text: str,
) -> tuple[str, list[str], list[tuple[str, str]]]:
    """
    Sanitize text input:
    1. Collapses spaced/delimeter tokens.
    2. Strips conversational boilerplate.
    3. Reweights adversative and counterfactual clauses.
    4. Extracts zero-percentage/cancelled dampeners.
    5. Extracts figurative modifiers.
    """
    collapsed = collapse_spaced_tokens(text)
    stripped = strip_conversational_boilerplate(collapsed)
    reweighted = reweight_adversative_clauses(stripped)
    dampeners = extract_zero_percentage_dampeners(reweighted)
    figuratives = extract_figurative_modifiers(reweighted)
    return reweighted, dampeners, figuratives


def encapsulate_payload(text: str) -> str:
    """
    Encapsulate text within lightweight structural fences («...») to sandbox
    prompt injections and delimiter breakout attacks without adding verbose
    preamble tokens.
    """
    clean = text.replace("«", "").replace("»", "")
    return f"«{clean}»"


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

    Handles:
    - Command Semantic Framing: Keys 'command', 'cmd', or 'query' map to 'action: {val} (execute command)'.
    - Numerical Limit Normalizer: When a metric count exceeds its limit/max, appends a relational clause.

    Args:
        state: Arbitrary JSON-compatible dict (any depth, any primitive values).

    Returns:
        Canonical key-value string: ``"key1: v1 | key2: v2 | ..."``.
        Returns an empty string ``""`` for an empty state dict.

    Examples:
        >>> serialize_state({"user_id": 123, "action": "delete_all"})
        'action: delete_all | user_id: 123'

        >>> serialize_state({"command": "drop database production"})
        'action: drop database production (execute command)'

        >>> serialize_state({"cpu": {"usage": 98}, "memory": 85})
        'cpu.usage: 98 | memory: 85'

        >>> serialize_state({})
        ''
    """
    flat = flatten_state(state)

    # Normalize snake_case in action values to natural words (e.g. drop_table -> drop table)
    for k in list(flat.keys()):
        if "action" in k.lower() and isinstance(flat[k], str):
            flat[k] = flat[k].replace("_", " ")

    # 1. DDL & Command Semantic Frame Bridge
    command_keys = {"command", "cmd", "query"}
    keys_to_remap = [
        k for k in flat if k in command_keys or k.rsplit(".", 1)[-1] in command_keys
    ]
    for k in keys_to_remap:
        val = flat.pop(k)
        action_key = f"{k.rsplit('.', 1)[0]}.action" if "." in k else "action"
        if action_key in flat:
            flat[action_key] = f"{flat[action_key]} | {val} (execute command)"
        else:
            flat[action_key] = f"{val} (execute command)"

    # 2. Numerical Limit Normalizer
    relational_clauses: list[str] = []
    metric_keys = [
        k
        for k in flat
        if (
            "_per_" in k.lower()
            or k.lower().startswith("current_")
            or k.lower() == "current"
            or k.lower().endswith("_count")
            or k.lower() == "count"
            or k.lower().endswith("_usage")
            or k.lower() == "usage"
        )
    ]
    limit_keys = [
        k
        for k in flat
        if (
            k.lower() == "limit"
            or k.lower().endswith("_limit")
            or k.lower().startswith("limit_")
            or k.lower() == "max"
            or k.lower().endswith("_max")
            or k.lower().startswith("max_")
        )
    ]

    for mk in metric_keys:
        try:
            m_val = float(flat[mk])
        except (ValueError, TypeError):
            continue
        for lk in limit_keys:
            try:
                l_val = float(flat[lk])
            except (ValueError, TypeError):
                continue
            if m_val > l_val:
                relational_clauses.append(
                    f"{mk} of {flat[mk]} exceeds {lk} of {flat[lk]}."
                )

    pairs = [f"{k}: {v}" for k, v in sorted(flat.items())]
    if relational_clauses:
        pairs.extend(sorted(relational_clauses))
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


def build_nli_pair(state: dict[str, Any], query: str) -> tuple[str, str]:
    """
    Build the premise and hypothesis pair for cross-encoder tokenization.

    Returns:
        tuple[str, str]: (premise, hypothesis), where premise is the
        deterministically serialized state string, and hypothesis is the query string.
    """
    return serialize_state(state), query
