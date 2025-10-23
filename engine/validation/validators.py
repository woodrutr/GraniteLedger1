from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

__all__ = ["assert_edges_valid", "assert_units_valid"]


def _ensure_zone_subset(series: pd.Series, zones: Iterable[str], column: str) -> None:
    missing = set(series) - set(zones)
    assert not missing, f"{column} has non-canonical IDs: {sorted(missing)}"


def assert_edges_valid(df: pd.DataFrame, zones: Iterable[str]) -> None:
    """Validate the core structural assumptions for transmission edges."""

    assert not df.empty, "ei_edges is empty"
    for column in ["capacity_mw", "reverse_capacity_mw", "added_cost_per_mwh"]:
        assert column in df.columns, f"missing required column: {column}"
        series = pd.to_numeric(df[column], errors="coerce")
        if column == "added_cost_per_mwh":
            series = series.fillna(0.0)
        else:
            assert series.notna().all(), f"{column} must be numeric"
        assert (series >= 0).all(), f"{column} must be non-negative"

    for column in ["contracted_flow_mw_forward", "contracted_flow_mw_reverse"]:
        assert column in df.columns, f"missing required column: {column}"
        series = pd.to_numeric(df[column], errors="coerce").fillna(0)
        assert (series >= 0).all(), f"{column} must be non-negative"

    eff_series = pd.to_numeric(df.get("efficiency"), errors="coerce").fillna(1.0)
    assert (eff_series > 0).all(), "efficiency must be positive"
    assert (eff_series <= 1.0 + 1e-9).all(), "efficiency cannot exceed 1.0"

    _ensure_zone_subset(df["from_region"], zones, "from_region")
    _ensure_zone_subset(df["to_region"], zones, "to_region")

    capacity = pd.to_numeric(df["capacity_mw"], errors="coerce").fillna(0)
    reverse_capacity = pd.to_numeric(df["reverse_capacity_mw"], errors="coerce").fillna(0)
    contracted_forward = pd.to_numeric(df["contracted_flow_mw_forward"], errors="coerce").fillna(0)
    contracted_reverse = pd.to_numeric(df["contracted_flow_mw_reverse"], errors="coerce").fillna(0)

    assert (contracted_forward <= capacity + 1e-6).all(), (
        "contracted forward flow exceeds capacity"
    )
    assert (contracted_reverse <= reverse_capacity + 1e-6).all(), (
        "contracted reverse flow exceeds reverse capacity"
    )

    fwd = set(zip(df["from_region"], df["to_region"]))
    bwd = set(zip(df["to_region"], df["from_region"]))
    missing_back = {(a, b) for (a, b) in fwd if (b, a) not in bwd}
    assert not missing_back, (
        f"missing reverse edges for: {sorted(list(missing_back))[:10]}"
    )


def assert_units_valid(df: pd.DataFrame, zones: Iterable[str]) -> None:
    """Validate that the unit inventory contains consistent records."""

    assert not df.empty, "ei_units is empty"
    _ensure_zone_subset(df["zone"], zones, "units.zone")
    assert df["nameplate_mw"].gt(0).all(), "nonpositive nameplate_mw"
    assert df["net_max_mw"].gt(0).all(), "nonpositive net_max_mw"
    hr = df["heat_rate_mmbtu_per_mwh"]
    assert hr.between(4, 35).all(), "heat rates outside [4,35] MMBtu/MWh"
