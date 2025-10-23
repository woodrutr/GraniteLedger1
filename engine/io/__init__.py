"""Convenience exports for engine I/O helpers."""

from __future__ import annotations

from .load_forecast import ISO_DIR_CANON, load_load_forecasts
from .transmission import load_ei_edges, validate_exported_interface_caps

# Re-export the strict loader module so callers can rely on ``engine.io`` as the
# canonical entry point.  Importing inside ``try`` keeps namespace packages
# behaving even if optional dependencies are absent.
try:  # pragma: no cover - import guard
    from . import load_forecasts_strict  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional dependency may be missing
    load_forecasts_strict = None  # type: ignore[assignment]

__all__ = [
    "ISO_DIR_CANON",
    "load_ei_edges",
    "load_load_forecasts",
    "load_forecasts_strict",
    "validate_exported_interface_caps",
]

