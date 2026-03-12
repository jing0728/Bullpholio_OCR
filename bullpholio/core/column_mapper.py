"""
core/column_mapper.py
---------------------
Fuzzy column name matching: maps raw DataFrame headers to canonical DTO fields.
"""

import difflib
from typing import Optional


def _fuzzy_score(a: str, b: str) -> float:
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def map_columns(
    raw_columns: list[str],
    alias_map: dict[str, list[str]],
    threshold: float = 0.6,
) -> dict[str, Optional[str]]:
    """
    Match raw column headers to canonical DTO field names.

    After a canonical picks a raw column via fuzzy match, that raw column
    is removed from the candidate pool — guaranteeing one-to-one assignment
    (no two canonicals map to the same raw column).
    """
    alias_lookup: dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            alias_lookup[alias.lower()] = canonical

    result: dict[str, Optional[str]] = {k: None for k in alias_map}
    used_raw: set[str] = set()

    # First pass: exact match (case-insensitive)
    for raw_col in raw_columns:
        key = raw_col.lower().strip()
        if key in alias_lookup:
            canonical = alias_lookup[key]
            if result[canonical] is None:
                result[canonical] = raw_col
                used_raw.add(raw_col)

    # Second pass: fuzzy match — unmatched_raw shrinks as we assign
    unresolved = [c for c, v in result.items() if v is None]
    unmatched_raw = [c for c in raw_columns if c not in used_raw]

    for canonical in unresolved:
        aliases = alias_map[canonical]
        best_raw, best_score = None, 0.0
        for raw_col in unmatched_raw:
            for alias in aliases:
                score = _fuzzy_score(raw_col, alias)
                if score > best_score:
                    best_score = score
                    best_raw = raw_col
        if best_score >= threshold and best_raw is not None:
            result[canonical] = best_raw
            used_raw.add(best_raw)
            unmatched_raw.remove(best_raw)

    return result
