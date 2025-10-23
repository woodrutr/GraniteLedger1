"""Tests for the fuel derivation helper exposed by :mod:`engine.data_loaders`."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data_loaders import derive_fuels


def test_derive_fuels_aggregates_boolean_and_emissions() -> None:
    units = pd.DataFrame(
        [
            {
                "unit_id": "a",
                "unique_id": "a",
                "fuel": "gas",
                "heat_rate_mmbtu_mwh": 7.0,
                "emission_rate_ton_mwh": 0.7,
                "covered": True,
            },
            {
                "unit_id": "b",
                "unique_id": "b",
                "fuel": "coal",
                "heat_rate_mmbtu_mwh": 9.0,
                "emission_rate_ton_mwh": 0.9,
                "covered": False,
            },
            {
                "unit_id": "c",
                "unique_id": "c",
                "fuel": "gas",
                "heat_rate_mmbtu_mwh": 7.5,
                "emission_rate_ton_mwh": 0.75,
                "covered": True,
            },
        ]
    )

    fuels = derive_fuels(units)

    assert set(fuels.columns) == {"fuel", "covered", "co2_short_ton_per_mwh"}
    assert dict(zip(fuels["fuel"], fuels["covered"])) == {"coal": False, "gas": True}

    emission_lookup = dict(zip(fuels["fuel"], fuels["co2_short_ton_per_mwh"]))
    assert emission_lookup["gas"] == pytest.approx(0.1)
    assert emission_lookup["coal"] == pytest.approx(0.1)


def test_derive_fuels_defaults_coverage_when_missing() -> None:
    units = pd.DataFrame(
        [
            {
                "Unit ID": "x",
                "Technology": "wind",
                "Heat Rate (mmBtu/MWh)": 0.0,
                "Capacity (MW)": 10.0,
                "region_id": "R1",
                "Emission Rate (short ton/MWh)": 0.0,
            }
        ]
    )

    fuels = derive_fuels(units, default_coverage=False)

    assert len(fuels) == 1
    assert fuels.loc[0, "fuel"] == "wind"
    assert bool(fuels.loc[0, "covered"]) is False
    assert pd.isna(fuels.loc[0, "co2_short_ton_per_mwh"])
