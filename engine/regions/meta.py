
"""Region metadata and canonicalization helpers.

Replaces prior GUI dependency `gui.region_metadata` with engine-local helpers.
Attempts to source canonicalization from `common.regions_schema` and
`engine.normalization`. Falls back to identity where unavailable.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

# Best-effort imports. No GUI imports.
try:
    from common.regions_schema import REGION_ALIASES, REGION_METADATA  # type: ignore
except Exception:  # pragma: no cover
    REGION_ALIASES = {}   # type: ignore[assignment]
    REGION_METADATA = {}  # type: ignore[assignment]

try:
    from engine.normalization import normalize_region  # type: ignore
except Exception:  # pragma: no cover
    def normalize_region(x: str) -> str:  # type: ignore[override]
        return x

@lru_cache(maxsize=4096)
def canonical_region_value(val: str) -> str:
    """Return canonical region id for *val* using aliases and normalization.

    Order of application:
      1) normalize_region from engine.normalization if available
      2) REGION_ALIASES mapping (case-insensitive)
    Fallback: return input unchanged.
    """
    if not isinstance(val, str):
        return val  # type: ignore[return-value]
    s = normalize_region(val)
    if not REGION_ALIASES:
        return s
    key = s.lower()
    return REGION_ALIASES.get(key, s)

@lru_cache(maxsize=1)
def region_metadata() -> Dict[str, Dict[str, object]]:
    """Return region metadata dict keyed by canonical id.

    If REGION_METADATA is not available, returns an empty dict.
    """
    if isinstance(REGION_METADATA, dict):
        return REGION_METADATA  # type: ignore[return-value]
    return {}
