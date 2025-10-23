"""Post-run audit checks for engine outputs."""
from __future__ import annotations

from typing import Any, Mapping

import math

try:  # pragma: no cover - optional dependency guard
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore[assignment]

from engine.constants import FLOW_TOL

DEFAULT_TOLERANCE = FLOW_TOL


def _ensure_pandas() -> None:
    if pd is None:  # pragma: no cover - exercised indirectly
        raise ImportError(
            "pandas is required for engine audits; install it with `pip install pandas`."
        )


def _coerce_numeric_series(series: "pd.Series") -> "pd.Series":
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return numeric.astype(float)


def _aggregate_by_year(df: "pd.DataFrame", value_column: str) -> "pd.Series":
    if df.empty or value_column not in df.columns:
        return pd.Series(dtype=float)
    working = df.copy()
    year_series = pd.to_numeric(working.get("year"), errors="coerce")
    working = working.assign(year=year_series).dropna(subset=["year"])
    if working.empty:
        return pd.Series(dtype=float)
    working["year"] = working["year"].astype(int)
    working[value_column] = _coerce_numeric_series(working[value_column])
    return working.groupby("year")[value_column].sum()


def _max_abs_difference(a: "pd.Series", b: "pd.Series") -> float:
    if a.empty and b.empty:
        return 0.0
    combined_index = sorted(set(a.index) | set(b.index))
    aligned_a = a.reindex(combined_index, fill_value=0.0)
    aligned_b = b.reindex(combined_index, fill_value=0.0)
    if aligned_a.empty and aligned_b.empty:
        return 0.0
    return float((aligned_a - aligned_b).abs().max())


def run_audits(
    *,
    annual_df: "pd.DataFrame",
    emissions_by_region_df: "pd.DataFrame",
    emissions_by_fuel_df: "pd.DataFrame",
    generation_by_fuel_df: "pd.DataFrame",
    capacity_by_fuel_df: "pd.DataFrame",
    cost_by_fuel_df: "pd.DataFrame",
    stranded_units_df: "pd.DataFrame",
    demand_by_year: Mapping[int, float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Return a structured audit report for the supplied output tables."""

    _ensure_pandas()

    report: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Emissions reconciliation
    # ------------------------------------------------------------------
    emissions_total = _aggregate_by_year(annual_df, "emissions_tons")
    regional_total = _aggregate_by_year(emissions_by_region_df, "emissions_tons")
    fuel_total = _aggregate_by_year(emissions_by_fuel_df, "emissions_tons")

    emissions_section: dict[str, Any] = {
        "passed": True,
        "max_region_gap": 0.0,
        "max_fuel_gap": 0.0,
        "issues": [],
    }

    if emissions_total.empty or regional_total.empty:
        emissions_section["passed"] = False
        emissions_section["issues"].append("emissions_data_missing")
    else:
        region_gap = _max_abs_difference(regional_total, emissions_total)
        emissions_section["max_region_gap"] = region_gap
        if region_gap > tolerance:
            emissions_section["passed"] = False
            emissions_section["issues"].append("regional_total_mismatch")

    if fuel_total.empty:
        emissions_section["passed"] = False
        emissions_section["issues"].append("fuel_emissions_missing")
    else:
        fuel_gap = _max_abs_difference(fuel_total, regional_total)
        emissions_section["max_fuel_gap"] = fuel_gap
        if fuel_gap > tolerance:
            emissions_section["passed"] = False
            emissions_section["issues"].append("fuel_total_mismatch")

    report["emissions"] = emissions_section

    # ------------------------------------------------------------------
    # Generation and capacity
    # ------------------------------------------------------------------
    generation_total = _aggregate_by_year(generation_by_fuel_df, "generation_mwh")
    capacity_total = _aggregate_by_year(capacity_by_fuel_df, "capacity_mwh")
    demand_series = pd.Series(
        {int(year): float(value) for year, value in demand_by_year.items()}, dtype=float
    ).sort_index()

    generation_section: dict[str, Any] = {
        "passed": True,
        "generation_demand_gap": 0.0,
        "capacity_margin": {},
        "issues": [],
        "stranded_units": [],
    }

    if generation_total.empty or demand_series.empty:
        generation_section["passed"] = False
        generation_section["issues"].append("generation_or_demand_missing")
    else:
        gen_gap = _max_abs_difference(generation_total, demand_series)
        generation_section["generation_demand_gap"] = gen_gap
        if gen_gap > tolerance:
            generation_section["passed"] = False
            generation_section["issues"].append("generation_does_not_meet_demand")

    capacity_margin: dict[int, float] = {}
    if not capacity_total.empty and not demand_series.empty:
        combined_years = sorted(set(capacity_total.index) | set(demand_series.index))
        for year in combined_years:
            available = float(capacity_total.reindex([year], fill_value=0.0).iloc[0])
            demand_value = float(demand_series.reindex([year], fill_value=0.0).iloc[0])
            margin = available - demand_value
            capacity_margin[int(year)] = margin
            if margin < -tolerance:
                generation_section["passed"] = False
                generation_section["issues"].append("capacity_below_demand")
    elif capacity_total.empty:
        generation_section["issues"].append("capacity_data_missing")
        generation_section["passed"] = False

    generation_section["capacity_margin"] = capacity_margin

    if not stranded_units_df.empty:
        stranded_records = stranded_units_df.to_dict("records")
        generation_section["stranded_units"] = stranded_records
        if stranded_records:
            generation_section.setdefault("issues", []).append("stranded_capacity_present")
    else:
        generation_section["stranded_units"] = []

    report["generation_capacity"] = generation_section

    # ------------------------------------------------------------------
    # Cost reconciliation
    # ------------------------------------------------------------------
    cost_section: dict[str, Any] = {
        "passed": True,
        "max_system_gap": 0.0,
        "max_allowance_gap": 0.0,
        "issues": [],
    }

    if cost_by_fuel_df.empty:
        cost_section["passed"] = False
        cost_section["issues"].append("cost_data_missing")
        report["cost"] = cost_section
        return report

    cost_totals = cost_by_fuel_df.groupby("year")[
        ["variable_cost", "allowance_cost", "carbon_price_cost", "total_cost"]
    ].sum()
    system_gap_series = (
        cost_totals["total_cost"]
        - (
            cost_totals["variable_cost"]
            + cost_totals["allowance_cost"]
            + cost_totals["carbon_price_cost"]
        )
    ).abs()
    if not system_gap_series.empty:
        max_system_gap = float(system_gap_series.max())
        cost_section["max_system_gap"] = max_system_gap
        if max_system_gap > tolerance:
            cost_section["passed"] = False
            cost_section["issues"].append("system_cost_mismatch")

    allowance_series = pd.Series(dtype=float)
    if {
        "year",
        "allowance_price",
        "surrender",
    }.issubset(annual_df.columns):
        working = annual_df[["year", "allowance_price", "surrender"]].copy()
        working["year"] = pd.to_numeric(working["year"], errors="coerce")
        working = working.dropna(subset=["year"])
        if not working.empty:
            working["year"] = working["year"].astype(int)
            working["allowance_price"] = _coerce_numeric_series(working["allowance_price"])
            working["surrender"] = _coerce_numeric_series(working["surrender"])
            allowance_series = (
                working["allowance_price"] * working["surrender"]
            ).groupby(working["year"]).sum()
    else:
        cost_section["issues"].append("allowance_revenue_missing")

    if not allowance_series.empty:
        allowance_cost_series = cost_totals["allowance_cost"].reindex(
            allowance_series.index, fill_value=0.0
        )
        allowance_gap = (allowance_cost_series - allowance_series).abs()
        if not allowance_gap.empty:
            max_allowance_gap = float(allowance_gap.max())
            cost_section["max_allowance_gap"] = max_allowance_gap
            if max_allowance_gap > tolerance:
                cost_section["passed"] = False
                cost_section["issues"].append("allowance_revenue_mismatch")
    else:
        cost_section["passed"] = False

    report["cost"] = cost_section

    return report


__all__ = ["run_audits"]
