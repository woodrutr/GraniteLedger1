"""Emissions accounting utilities used by the Granite Ledger engine."""
from __future__ import annotations

from typing import Mapping, Tuple

import pandas as pd

from engine.regions.shares import load_zone_to_state_share


def apply_declining_cap(annual: pd.DataFrame, decline_rate: float = 0.05) -> pd.DataFrame:
    """Ensure allowance totals remain consistent with minted quantities."""

    # ``decline_rate`` is retained for API compatibility but no longer used to
    # synthesize artificial caps. Previously this helper imposed a synthetic
    # decline regardless of the policy inputs which overwrote the actual
    # allowance supply computed by the market solver.
    _ = decline_rate

    if annual.empty or "allowances_minted" not in annual.columns:
        return annual

    working = annual.copy()
    working = working.sort_values("year").reset_index(drop=True)

    minted_series = pd.to_numeric(working["allowances_minted"], errors="coerce").fillna(0.0)

    if "allowances_available" in working.columns and "bank_start" in working.columns:
        allowances_total = pd.to_numeric(
            working["allowances_available"], errors="coerce"
        ).fillna(0.0)
        bank_start = pd.to_numeric(working["bank_start"], errors="coerce").fillna(0.0)
        implied_minted = (allowances_total - bank_start).clip(lower=0.0)
        if len(implied_minted) == len(minted_series):
            delta = (implied_minted - minted_series).abs()
            if (delta > 1e-6).any():
                minted_series = implied_minted

    working["allowances_minted"] = minted_series.astype(float)
    return working


def summarize_emissions(emissions_df: pd.DataFrame) -> Tuple[Mapping[int, float], Mapping[str, Mapping[int, float]]]:
    """Return total and regional emissions mappings from ``emissions_df``."""

    if emissions_df.empty:
        return {}, {}

    working = emissions_df.copy()
    if "year" not in working.columns:
        working["year"] = 0
    working["year"] = pd.to_numeric(working["year"], errors="coerce").fillna(0).astype(int)
    if "region" not in working.columns:
        working["region"] = "system"
    working["region"] = working["region"].astype(str)
    working["emissions_tons"] = pd.to_numeric(
        working.get("emissions_tons", 0.0), errors="coerce"
    ).fillna(0.0)

    totals = working.groupby("year")["emissions_tons"].sum().to_dict()

    region_map: dict[str, dict[int, float]] = {}
    for region, region_frame in working.groupby("region"):
        values = {
            int(year): float(total)
            for year, total in region_frame.groupby("year")["emissions_tons"].sum().items()
        }
        region_map[str(region)] = values

    return totals, region_map


def summarize_emissions_by_state(emissions_by_region: pd.DataFrame) -> pd.DataFrame:
    """Return state-level emissions derived from ``emissions_by_region``."""

    columns = ["year", "state", "emissions_tons"]
    if emissions_by_region.empty:
        return pd.DataFrame(columns=columns)

    shares = load_zone_to_state_share()
    if shares.empty:
        return pd.DataFrame(columns=columns)

    working = emissions_by_region.copy()
    if "region" in working.columns and "region_id" not in working.columns:
        working = working.rename(columns={"region": "region_id"})
    if "region_id" not in working.columns:
        return pd.DataFrame(columns=columns)

    working["year"] = pd.to_numeric(working.get("year", 0), errors="coerce")
    working = working.dropna(subset=["year"])
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["year"] = working["year"].astype(int)
    working["region_id"] = working["region_id"].astype(str)
    working["emissions_tons"] = pd.to_numeric(
        working.get("emissions_tons", 0.0), errors="coerce"
    ).fillna(0.0)

    regional_totals = (
        working.groupby(["year", "region_id"], as_index=False)["emissions_tons"].sum()
    )
    merged = regional_totals.merge(shares, on="region_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)

    merged["emissions_tons"] = merged["emissions_tons"] * merged["share"]
    result = (
        merged.groupby(["year", "state"], as_index=False)["emissions_tons"].sum()
        .sort_values(["year", "state"])
        .reset_index(drop=True)
    )
    result["state"] = result["state"].astype(str)
    return result[columns]


__all__ = ["apply_declining_cap", "summarize_emissions", "summarize_emissions_by_state"]
