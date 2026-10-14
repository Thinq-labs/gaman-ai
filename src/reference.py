"""
src/reference.py — Reference Grounding Engine for Gaman AI
===========================================================
Extracts, segments, caches, and matches reference documents (.docx, .txt, .md, .json),
tabular datasets (.csv, .jsonl), and inline policy prompt strings for grounded decision making.

Constraints:
- ZERO heavy runtime dependencies: pure standard library (zipfile, xml.etree.ElementTree,
  csv, json, pathlib, hashlib, re, math).
- Sub-millisecond clause retrieval (< 1ms).
- In-memory SHA-256 caching for zero-I/O repeated document evaluations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "did", "do", "does",
    "doing", "don", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
    "same", "she", "should", "so", "some", "such", "t", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
})


def _tokenize(text: str) -> list[str]:
    """Extract lowercased alphanumeric tokens of length >= 2."""
    return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())


def extract_text_from_docx(docx_path: str | Path) -> str:
    """
    Extract text paragraphs from a .docx file using standard library zipfile and XML parsing.

    Reads word/document.xml and extracts all text inside <w:t> tags grouped by <w:p> paragraphs.
    """
    p = Path(docx_path)
    if not p.is_file():
        raise FileNotFoundError(f"DOCX file not found: {p}")

    with zipfile.ZipFile(p, "r") as z:
        if "word/document.xml" not in z.namelist():
            raise ValueError(f"Invalid DOCX file (missing word/document.xml): {p}")
        xml_bytes = z.read("word/document.xml")

    paragraphs: list[str] = []
    try:
        root = ET.fromstring(xml_bytes)
        for elem in root.iter():
            if elem.tag.endswith("p"):
                p_text = "".join(
                    t.text for t in elem.iter()
                    if t.tag.endswith("t") and t.text and t.text.strip()
                )
                if p_text.strip():
                    paragraphs.append(p_text.strip())
    except Exception:
        # Resilient fallback: regex extraction over XML text
        xml_str = xml_bytes.decode("utf-8", errors="replace")
        p_blocks = re.findall(r"<w:p\b[^>]*>(.*?)</w:p>", xml_str, flags=re.DOTALL)
        if p_blocks:
            for block in p_blocks:
                t_matches = re.findall(r"<w:t\b[^>]*>(.*?)</w:t>", block, flags=re.DOTALL)
                p_text = "".join(t_matches).strip()
                if p_text:
                    paragraphs.append(p_text)
        else:
            t_matches = re.findall(r"<w:t\b[^>]*>(.*?)</w:t>", xml_str, flags=re.DOTALL)
            paragraphs = [t.strip() for t in t_matches if t.strip()]

    return "\n".join(paragraphs)


def extract_text_from_json(json_path: str | Path) -> str:
    """Extract text from a JSON file."""
    p = Path(json_path)
    if not p.is_file():
        raise FileNotFoundError(f"JSON file not found: {p}")

    with open(p, encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    def _extract_strings(val: Any) -> list[str]:
        if isinstance(val, str):
            return [val.strip()] if val.strip() else []
        if isinstance(val, list):
            res = []
            for item in val:
                res.extend(_extract_strings(item))
            return res
        if isinstance(val, dict):
            res = []
            for v in val.values():
                res.extend(_extract_strings(v))
            return res
        return []

    strs = _extract_strings(data)
    return "\n".join(strs)


def segment_clauses(text: str, max_chars: int = 128) -> list[str]:
    """
    Chunk reference text into concise, atomic policy propositions or clauses (max 128 chars each).

    Splits on newlines, bullet points, and sentence boundaries (. ! ? ;), further subdividing
    clauses exceeding max_chars on clause boundaries or words.
    """
    raw_lines = text.splitlines()
    clauses: list[str] = []

    for line in raw_lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Strip list markers like '*', '-', '•', '1.', '1)'
        line_clean = re.sub(r"^[\*\-•\d\.\)\s]+", "", line_clean).strip()
        if not line_clean:
            continue

        # Split on sentence punctuation or semicolons
        sentences = re.split(r"(?<=[.!?])\s+|;\s*", line_clean)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean or len(s_clean) < 4:
                continue

            if len(s_clean) <= max_chars:
                clauses.append(s_clean)
            else:
                # Subdivide long sentences on commas, conjunctions, or word boundaries
                sub_parts = re.split(r",\s*|\s+(?:and|or|but|while|whereas)\s+", s_clean)
                current = ""
                for part in sub_parts:
                    p_strip = part.strip()
                    if not p_strip:
                        continue
                    candidate = f"{current}, {p_strip}" if current else p_strip
                    if len(candidate) <= max_chars:
                        current = candidate
                    else:
                        if current:
                            clauses.append(current)
                        # If a single subpart still exceeds max_chars, split on words
                        if len(p_strip) > max_chars:
                            words = p_strip.split()
                            chunk = ""
                            for w in words:
                                cand_w = f"{chunk} {w}" if chunk else w
                                if len(cand_w) <= max_chars:
                                    chunk = cand_w
                                else:
                                    if chunk:
                                        clauses.append(chunk)
                                    chunk = w
                            if chunk:
                                current = chunk
                        else:
                            current = p_strip
                if current and len(current) >= 4:
                    clauses.append(current)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_clauses: list[str] = []
    for c in clauses:
        c_norm = c.strip().rstrip(".").lower()
        if c_norm not in seen:
            seen.add(c_norm)
            unique_clauses.append(c.strip())

    return unique_clauses


def find_relevant_clauses(query: str, clauses: list[str], top_k: int = 3) -> list[str]:
    """
    Retrieve top_k matching clauses for a query string in < 1ms using BM25 token overlap.

    Args:
        query: Query string (input state or predicate).
        clauses: Candidate segmented policy clauses.
        top_k: Number of top clauses to return.

    Returns:
        List of the top_k most relevant clauses.
    """
    if not clauses:
        return []
    if len(clauses) <= top_k:
        return list(clauses)

    query_tokens = [w for w in _tokenize(query) if w not in STOP_WORDS]
    if not query_tokens:
        query_tokens = _tokenize(query)
    if not query_tokens:
        return clauses[:top_k]

    num_docs = len(clauses)
    df: dict[str, int] = {}
    clause_token_lists: list[list[str]] = []
    for c in clauses:
        toks = _tokenize(c)
        clause_token_lists.append(toks)
        for term in set(toks):
            df[term] = df.get(term, 0) + 1

    avgdl = sum(len(toks) for toks in clause_token_lists) / num_docs if num_docs > 0 else 1.0
    k1 = 1.2
    b = 0.75

    scores: list[tuple[float, int]] = []
    for idx, (clause, toks) in enumerate(zip(clauses, clause_token_lists, strict=False)):
        score = 0.0
        doc_len = len(toks)
        tf: dict[str, int] = {}
        for tok in toks:
            tf[tok] = tf.get(tok, 0) + 1

        for q_term in query_tokens:
            if q_term in tf:
                term_df = df.get(q_term, 1)
                idf = math.log((num_docs - term_df + 0.5) / (term_df + 0.5) + 1.0)
                term_tf = tf[q_term]
                num = term_tf * (k1 + 1.0)
                denom = term_tf + k1 * (1.0 - b + b * (doc_len / (avgdl + 1e-6)))
                score += idf * (num / denom)

        # Exact substring or bigram boost
        c_lower = clause.lower()
        for q_term in query_tokens:
            if q_term in c_lower:
                score += 0.5
        for i in range(len(query_tokens) - 1):
            bigram = f"{query_tokens[i]} {query_tokens[i+1]}"
            if bigram in c_lower:
                score += 2.0

        scores.append((score, idx))

    scores.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    if scores[0][0] <= 0.0:
        return clauses[:top_k]

    selected_indices = [idx for _, idx in scores[:top_k]]
    return [clauses[idx] for idx in selected_indices]


class ReferenceExtractor:
    """
    High-performance, zero-dependency ingestion and retrieval for reference knowledge.

    Features:
    - DOCX parsing via zipfile + ElementTree.
    - Plain text, markdown, and JSON parsing.
    - Tabular dataset loading (.csv, .jsonl).
    - SHA-256 in-memory caching for zero-overhead repeated lookups.
    - Sub-millisecond BM25 clause retrieval.
    """

    _clause_cache: dict[str, list[str]] = {}
    _dataset_cache: dict[str, tuple[list[str], list[str]]] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """Clear in-memory clause and dataset caches."""
        cls._clause_cache.clear()
        cls._dataset_cache.clear()

    @classmethod
    def is_file_path(cls, source: str | Path | None) -> bool:
        """Safely check if source refers to an existing file without raising OS/path errors."""
        if source is None:
            return False
        try:
            p = Path(source)
            return p.is_file()
        except (OSError, ValueError):
            return False

    @classmethod
    def is_tabular_file(cls, source: str | Path | None) -> bool:
        """Check if source is a tabular dataset file (.csv or .jsonl)."""
        if not cls.is_file_path(source):
            return False
        assert source is not None
        p = Path(source)
        return p.suffix.lower() in (".csv", ".jsonl")

    @classmethod
    def compute_hash(cls, source: str | Path) -> str:
        """Compute SHA-256 hash of file contents or inline string."""
        if cls.is_file_path(source):
            p = Path(source)
            hasher = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()
        return hashlib.sha256(str(source).encode("utf-8")).hexdigest()

    @classmethod
    def load_dataset(
        cls,
        source: str | Path,
        text_column: str | None = None,
        target_column: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """
        Load tabular dataset (.csv or .jsonl) into lists of (texts, labels).
        """
        h = cls.compute_hash(source)
        if h in cls._dataset_cache:
            return cls._dataset_cache[h]

        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {p}")

        ext = p.suffix.lower()
        texts: list[str] = []
        labels: list[str] = []

        if ext == ".csv":
            with open(p, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                t_col = text_column
                if not t_col:
                    for cand in ["text", "query", "sentence", "state", "prompt", "message", "utterance", "input"]:
                        if cand in fieldnames:
                            t_col = cand
                            break
                if not t_col and fieldnames:
                    t_col = fieldnames[0]

                l_col = target_column
                if not l_col:
                    for cand in ["label", "target", "category", "intent", "class", "option"]:
                        if cand in fieldnames:
                            l_col = cand
                            break
                if not l_col and len(fieldnames) > 1:
                    l_col = fieldnames[1]

                for row in reader:
                    t_val = str(row.get(t_col, "")).strip() if t_col else ""
                    l_val = str(row.get(l_col, "")).strip() if l_col else ""
                    if t_val:
                        texts.append(t_val)
                        labels.append(l_val)

        elif ext == ".jsonl":
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    item = json.loads(line_s)
                    t_col = text_column or next(
                        (k for k in ["text", "query", "sentence", "state", "prompt", "message"] if k in item),
                        None,
                    )
                    l_col = target_column or next(
                        (k for k in ["label", "target", "category", "intent", "class", "option"] if k in item),
                        None,
                    )
                    t_val = str(item.get(t_col, "")).strip() if t_col else ""
                    l_val = str(item.get(l_col, "")).strip() if l_col else ""
                    if t_val:
                        texts.append(t_val)
                        labels.append(l_val)

        cls._dataset_cache[h] = (texts, labels)
        return texts, labels

    @classmethod
    def extract_clauses(cls, source: str | Path, max_chars: int = 128) -> list[str]:
        """
        Extract and segment reference source into atomic propositions/clauses.
        Uses in-memory cache keyed by SHA-256.
        """
        h = cls.compute_hash(source)
        if h in cls._clause_cache:
            return cls._clause_cache[h]

        if cls.is_file_path(source):
            p = Path(source)
            ext = p.suffix.lower()
            if ext == ".docx":
                raw_text = extract_text_from_docx(p)
            elif ext in (".csv", ".jsonl"):
                texts, _ = cls.load_dataset(p)
                raw_text = "\n".join(texts)
            elif ext == ".json":
                raw_text = extract_text_from_json(p)
            else:
                raw_text = p.read_text(encoding="utf-8", errors="replace")
        else:
            raw_text = str(source)

        clauses = segment_clauses(raw_text, max_chars=max_chars)
        cls._clause_cache[h] = clauses
        return clauses

    @staticmethod
    def find_relevant_clauses(query: str, clauses: list[str], top_k: int = 3) -> list[str]:
        """Retrieve the top matching clauses in < 1ms."""
        return find_relevant_clauses(query, clauses, top_k=top_k)
