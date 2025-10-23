from pathlib import Path

import pandas as pd
import pytest

from engine.data_loaders.units import load_unit_fleet, load_units


def test_load_unit_fleet_prefers_unique_ids(tmp_path: Path) -> None:
    csv_path = tmp_path / "ei_units.csv"
    pd.DataFrame(
        [
            {
                "unit_id": "legacy-1",
                "unique_id": "preferred-1",
                "region_id": "R1",
                "heat_rate_mmbtu_per_mwh": 9.5,
                "capacity_mw": 50.0,
                "emission_rate_ton_mwh": 0.4,
            },
            {
                "unit_id": "legacy-2",
                "unique_id": "preferred-2",
                "region_id": "R2",
                "heat_rate_mmbtu_per_mwh": 7.1,
                "capacity_mw": 40.0,
                "emission_rate_ton_mwh": 0.0,
            },
        ]
    ).to_csv(csv_path, index=False)

    fleet = load_unit_fleet(path=csv_path)

    assert list(fleet["unique_id"]) == ["preferred-1", "preferred-2"]
    assert list(fleet["unit_id"]) == ["preferred-1", "preferred-2"]
    assert list(fleet["hr_mmbtu_per_mwh"]) == [9.5, 7.1]
    assert list(fleet["cap_mw"]) == [50.0, 40.0]


def test_load_units_exposes_unique_id_column(tmp_path: Path) -> None:
    catalog_path = tmp_path / "units.csv"
    pd.DataFrame(
        [
            {
                "unit_id": "legacy-a",
                "unique_id": "preferred-a",
                "region_id": "NYISO_A",
                "fuel": "gas",
                "cap_mw": 25.0,
                "hr_mmbtu_per_mwh": 7.2,
                "ef_ton_per_mwh": 0.3,
                "vom_per_mwh": 1.5,
                "availability": 0.95,
            }
        ]
    ).to_csv(catalog_path, index=False)

    missing_prices = tmp_path / "fuel_prices.csv"
    units = load_units(unit_catalog=catalog_path, fuel_price_catalog=missing_prices)

    assert list(units["unique_id"]) == ["preferred-a"]
    assert list(units["unit_id"]) == ["preferred-a"]
    assert float(units.loc[0, "hr_mmbtu_per_mwh"]) == 7.2
    assert float(units.loc[0, "ef_ton_per_mwh"]) == 0.3


def test_load_units_filters_active_regions(tmp_path: Path) -> None:
    catalog_path = tmp_path / "units.csv"
    pd.DataFrame(
        [
            {
                "unit_id": "legacy-a",
                "unique_id": "preferred-a",
                "region_id": "NYISO_A",
                "fuel": "gas",
                "cap_mw": 25.0,
                "hr_mmbtu_per_mwh": 7.2,
                "ef_ton_per_mwh": 0.3,
                "vom_per_mwh": 1.5,
                "availability": 0.95,
            },
            {
                "unit_id": "legacy-b",
                "unique_id": "preferred-b",
                "region_id": "NYISO_B",
                "fuel": "gas",
                "cap_mw": 30.0,
                "hr_mmbtu_per_mwh": 7.5,
                "ef_ton_per_mwh": 0.35,
                "vom_per_mwh": 1.6,
                "availability": 0.94,
            },
        ]
    ).to_csv(catalog_path, index=False)

    missing_prices = tmp_path / "fuel_prices.csv"
    units = load_units(
        active_regions=["NYISO_B"],
        unit_catalog=catalog_path,
        fuel_price_catalog=missing_prices,
    )

    assert units.shape[0] == 1
    assert units.loc[0, "unique_id"] == "preferred-b"
    assert units.loc[0, "region"] == "NYISO_B"


def test_load_units_raises_when_regions_missing(tmp_path: Path) -> None:
    catalog_path = tmp_path / "units.csv"
    pd.DataFrame(
        [
            {
                "unit_id": "legacy-a",
                "unique_id": "preferred-a",
                "region_id": "NYISO_A",
                "fuel": "gas",
                "cap_mw": 25.0,
                "hr_mmbtu_per_mwh": 7.2,
                "ef_ton_per_mwh": 0.3,
                "vom_per_mwh": 1.5,
                "availability": 0.95,
            }
        ]
    ).to_csv(catalog_path, index=False)

    missing_prices = tmp_path / "fuel_prices.csv"

    with pytest.raises(
        ValueError,
        match=(
            rf"Unit catalog {catalog_path} has no rows for regions: \['NYISO_B'\]"
        ),
    ):
        load_units(
            active_regions=["NYISO_B"],
            unit_catalog=catalog_path,
            fuel_price_catalog=missing_prices,
        )
