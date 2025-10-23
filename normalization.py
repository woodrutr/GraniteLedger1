from __future__ import annotations

import re
from typing import Iterable


def _sanitize(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


_ALIAS_GROUPS: dict[str, Iterable[str]] = {
    "iso_ne": (
        "ISO-NE",
        "ISO NE",
        "ISO New England",
        "isone",
        "iso-ne",
        "iso ne",
    ),
    "nyiso": (
        "NYISO",
        "NY ISO",
        "New York ISO",
        "NY Independent System Operator",
    ),
    "pjm": ("PJM", "PJM ISO", "PJM Interconnection", "P J M"),
    "miso": (
        "MISO",
        "Midcontinent ISO",
        "Midcontinent Independent System Operator",
        "MI SO",
    ),
    "spp": ("SPP", "Southwest Power Pool", "S P P"),
    "southeast": (
        "Southeast",
        "South East",
        "South-East",
        "SE United States",
        "SE",
    ),
    "canada": ("Canada", "Can Ada", "Can-Ada"),
}

ISO_ALIASES: dict[str, str] = {}
for canonical, aliases in _ALIAS_GROUPS.items():
    canonical_token = _sanitize(canonical)
    ISO_ALIASES[canonical_token] = canonical_token
    for alias in aliases:
        ISO_ALIASES[_sanitize(alias)] = canonical_token


def normalize_token(value: str | None) -> str:
    token = _sanitize(value)
    return ISO_ALIASES.get(token, token)


def normalize_iso_name(value: str | None) -> str:
    token = _sanitize(value)
    return ISO_ALIASES.get(token, token)


__all__ = ["ISO_ALIASES", "normalize_token", "normalize_iso_name"]
