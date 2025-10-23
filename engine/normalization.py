"""Normalization helpers for ISO identifiers, regions, and headers."""

from __future__ import annotations
import re
import logging
from functools import lru_cache
from typing import Iterable

from common.regions_schema import REGION_MAP

def _sanitize(value: str | None) -> str:
    """Return a normalized lowercase token with non-alphanumerics replaced."""
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")

# ---------------------------------------------------------------------------
# ISO alias normalization
# ---------------------------------------------------------------------------

_ISO_ALIAS_GROUPS: dict[str, Iterable[str]] = {
    "nyiso": ("nyiso", "ny iso", "ny", "new york"),
    "iso_ne": ("iso-ne", "isone", "nepool", "new england"),
    "pjm": ("pjm", "midatlantic"),
    "miso": ("miso", "midwest iso", "midcontinent iso"),
    "spp": ("spp", "southwest power pool"),
    "ercot": ("ercot", "texas"),
    "caiso": ("caiso", "california iso", "california"),
    "southeast": ("southeast", "se"),
    "canada": ("canada", "canadian"),
}

_ISO_ALIAS_MAP: dict[str, str] = {}
for canonical, aliases in _ISO_ALIAS_GROUPS.items():
    canonical_token = _sanitize(canonical)
    if not canonical_token:
        continue
    _ISO_ALIAS_MAP[canonical_token] = canonical_token
    for alias in aliases:
        alias_token = _sanitize(alias)
        if alias_token:
            _ISO_ALIAS_MAP.setdefault(alias_token, canonical_token)

def normalize_iso_name(value: str | None) -> str:
    """Return canonical ISO key for value or sanitized token."""
    token = _sanitize(value)
    if not token:
        return ""
    compact = token.replace("_", "")
    return _ISO_ALIAS_MAP.get(token) or _ISO_ALIAS_MAP.get(compact, token)

def canon_header(value: str | None) -> str:
    return _sanitize(value)

def normalize_token(value: str | None) -> str:
    token = _sanitize(value)
    if not token:
        return ""
    compact = token.replace("_", "")
    return _ISO_ALIAS_MAP.get(token) or _ISO_ALIAS_MAP.get(compact, token) or token

# ---------------------------------------------------------------------------
# Region normalization
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _region_alias_map() -> dict[str, str]:
    """Build mapping from normalized tokens to canonical region IDs."""
    alias_map: dict[str, str] = {}

    def _record_alias(token: str | None, canonical: str) -> None:
        norm = _sanitize(token)
        if not norm:
            return
        alias_map.setdefault(norm, canonical)
        compact = norm.replace("_", "")
        if compact:
            alias_map.setdefault(compact, canonical)

    for region_id, display_name in REGION_MAP.items():
        _record_alias(region_id, region_id)
        _record_alias(display_name, region_id)

    # manual fallbacks for common shorthand
    manual_aliases = {
        "frcc": "FRCC_SYS",
        "duke": "DUK_SYS",
        "tva": "TVA_SYS",
        "soco": "SOCO_SYS",
        "canada_ontario": "ONTARIO",
        "canada_quebec": "QUEBEC",
    }
    for token, canonical in manual_aliases.items():
        if canonical in REGION_MAP:
            _record_alias(token, canonical)

    return alias_map

def normalize_region_id(value: str | None, *, iso: str | None = None) -> str:
    """Return canonical region_id or raise ValueError with clear message."""
    token = _sanitize(value)
    if not token:
        return ""

    alias_map = _region_alias_map()
    iso_token = normalize_iso_name(iso) if iso else ""
    compact = token.replace("_", "")
    candidates: list[str] = []

    if iso_token:
        for suffix in (token, compact):
            if not suffix:
                continue
            candidates.append(f"{iso_token}_{suffix}")
            candidates.append(f"{iso_token}{suffix}")

    candidates.extend([token, compact])

    for cand in candidates:
        match = alias_map.get(cand)
        if match:
            return match

    # direct uppercase match to registry
    upper = token.upper()
    if upper in REGION_MAP:
        return upper

    # strip known suffixes and retry
    for suffix in ("_baseline", "_reference", "_scenario", "_high", "_low"):
        if token.endswith(suffix):
            trimmed = token[: -len(suffix)].strip("_")
            if trimmed:
                try:
                    return normalize_region_id(trimmed, iso=iso)
                except ValueError:
                    pass

    raise ValueError(
        f"Unknown region identifier: {value!r}. "
        "Check that forecast CSV filenames match region_ids in regions/registry "
        "and run tools/validate_regions.py."
    )

def normalize_region(value: str | None) -> str:
    return normalize_region_id(value)

def canonical_region_value(value: object) -> object:
    """Return canonical region identifier when possible.

    The helper mirrors the behaviour of the GUI canonicalization utility but
    stays tolerant of unexpected values.  Non-string inputs and values that
    cannot be normalized are passed through unchanged so caller code can decide
    how to handle them.
    """

    if not isinstance(value, str):
        return value

    try:
        return normalize_region(value)
    except ValueError:
        return value


__all__ = [
    "canon_header",
    "canonical_region_value",
    "normalize_iso_name",
    "normalize_region",
    "normalize_region_id",
    "normalize_token",
]
