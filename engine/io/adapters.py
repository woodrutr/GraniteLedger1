"""Adapters bridging GUI expectations with IO helpers."""

from __future__ import annotations
from typing import Any
import pandas as pd

# optional strict loader (may be unavailable in some deployments)
try:  # pragma: no cover - import guard
    from engine.io import load_forecasts_strict as strict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - dependency optional
    strict = None

REQUIRED_LOAD_COLS = {"region_id", "iso", "scenario", "year", "load_gwh"}

_ALIASES = {
    "region": "region_id",
    "zone": "region_id",
    "balancing_area": "iso",
    "balancing_authority": "iso",
    "ba": "iso",
    "gwh": "load_gwh",
    "load_gw": "load_gwh",
}

# Columns provided in megawatt-hours which need to be scaled to gigawatt-hours
_MWH_COLUMNS = ("load_mwh", "demand_mwh", "demand")


def normalize_load_frame(frame: Any) -> pd.DataFrame:
    """Return a load forecast frame normalized for GUI consumption.

    - Coerces any input convertible to a DataFrame.
    - Normalizes aliases to canonical schema.
    - Fills in defaults for missing required columns.
    - Defers to strict finalization for dtype enforcement.
    """

    if frame is None:
        base = pd.DataFrame(columns=list(REQUIRED_LOAD_COLS))
    elif not isinstance(frame, pd.DataFrame):
        base = pd.DataFrame(frame)
    else:
        base = frame.copy()

    base.columns = [c.strip().lower() for c in base.columns]

    # Apply alias renaming
    ren: dict[str, str] = {}
    for source, target in _ALIASES.items():
        if source not in base.columns or target in base.columns:
            continue

        # ``zone`` values should flow through to ``region_id`` without losing the
        # original column.  This mirrors the behaviour of the forecast loaders,
        # which expose both identifiers.  Earlier implementations renamed the
        # column, inadvertently erasing the zone information before it reached
        # the UI cache.
        if source == "zone" and target == "region_id":
            base[target] = base[source]
            continue

        ren[source] = target

    if ren:
        base = base.rename(columns=ren)

    mwh_source = next((column for column in _MWH_COLUMNS if column in base.columns), None)
    if mwh_source is not None:
        base["load_gwh"] = pd.to_numeric(base[mwh_source], errors="coerce") / 1000.0

    # Scenario default
    if "scenario" not in base.columns:
        base["scenario"] = "Baseline"

    # ISO default/derivation
    if "iso" not in base.columns:
        if "region_id" in base.columns:
            base["iso"] = base["region_id"].apply(
                lambda r: str(r).split("_", 1)[0] if isinstance(r, str) and "_" in str(r) else "UNKNOWN"
            )
        else:
            base["iso"] = "UNKNOWN"

    # Coerce types
    if "year" in base.columns:
        base["year"] = pd.to_numeric(base["year"], errors="coerce").astype("Int64")
    if "load_gwh" in base.columns:
        base["load_gwh"] = pd.to_numeric(base["load_gwh"], errors="coerce")

    # Column order for UI expectations
    ordered = ["iso", "region_id", "scenario", "year", "load_gwh"]
    cols = [c for c in ordered if c in base.columns] + [c for c in base.columns if c not in ordered]
    base = base[cols]

    if strict is not None:
        finalized = strict._finalize_frame(base)  # type: ignore[attr-defined]
        finalized.columns = [c.strip().lower() for c in finalized.columns]
        return finalized
    return base


__all__ = ["normalize_load_frame"]
