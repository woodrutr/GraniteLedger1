"""Helper functions for attaching fuel price data to dispatch units."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


FuelPricePath = Union[str, Path]


def attach_fuel_costs(
    units_frame: pd.DataFrame,
    year: int,
    scenario_id: str,
    fuel_price_path: FuelPricePath = "input/fuel_prices/fuel_prices_annual.csv",
) -> pd.DataFrame:
    """Attach fuel and dispatch cost data to the provided units frame.

    Parameters
    ----------
    units_frame
        DataFrame containing unit characteristics. Must include ``unit_id``,
        ``region_id``, ``fuel``, and ``hr_mmbtu_per_mwh`` columns.
    year
        Model year to filter the fuel price data.
    scenario_id
        Scenario identifier to select matching fuel prices.
    fuel_price_path
        Path to the CSV containing annual fuel price forecasts.

    Returns
    -------
    pandas.DataFrame
        Copy of ``units_frame`` with fuel price and dispatch cost columns
        attached.
    """

    prices = pd.read_csv(fuel_price_path)
    prices = prices.loc[
        (prices["year"] == year)
        & (prices["scenario_id"].str.upper() == scenario_id.upper())
    ]
    prices["region_id"] = prices["region_id"].str.upper()
    prices["fuel"] = prices["fuel"].str.upper()

    merged = units_frame.copy()
    merged["region_id"] = merged["region_id"].str.upper()
    merged["fuel"] = merged["fuel"].str.upper()
    merged["hr_mmbtu_per_mwh"] = pd.to_numeric(
        merged["hr_mmbtu_per_mwh"], errors="coerce"
    )

    merged = merged.merge(
        prices[["region_id", "fuel", "price_per_mmbtu"]],
        on=["region_id", "fuel"],
        how="left",
    )

    missing_prices_mask = merged["price_per_mmbtu"].isna()
    if missing_prices_mask.any():
        missing_groups = (
            merged.loc[missing_prices_mask]
            .groupby(["region_id", "fuel"])
            .agg({"unit_id": lambda ids: ", ".join(sorted(map(str, set(ids))))})
        )
        missing_details = [
            f"region={region}, fuel={fuel} (units: {unit_ids})"
            for (region, fuel), unit_ids in missing_groups["unit_id"].items()
        ]
        raise ValueError(
            "Missing fuel price entries for the following region/fuel combinations: "
            + "; ".join(missing_details)
        )

    merged["fuel_price_per_mmbtu"] = merged["price_per_mmbtu"]
    merged["dispatch_fuel_cost_per_mwh"] = (
        merged["hr_mmbtu_per_mwh"] * merged["fuel_price_per_mmbtu"]
    )
    merged.drop(columns=["price_per_mmbtu"], inplace=True)

    merged["carbon_cost_per_mwh"] = 0.0
    merged["total_dispatch_cost_per_mwh"] = (
        merged["dispatch_fuel_cost_per_mwh"] + merged["carbon_cost_per_mwh"]
    )

    return merged
