from __future__ import annotations

from pathlib import Path

import pytest

from engine.data_loaders.ei_units import load_ei_units


def test_load_ei_units_normalizes_and_coerces(tmp_path: Path) -> None:
    csv_path = tmp_path / "ei_units.csv"
    csv_path.write_text(
        "region_id,heat_rate_mmbtu_per_mwh,capacity_mw,co2_short_ton_per_mwh\n"
        "north-east,10.5,150,0.9\n"
        "south ,not_a_number, ,1.2\n"
    )

    result = load_ei_units(csv_path)

    assert list(result.columns) == [
        "region_id",
        "heat_rate_mmbtu_per_mwh",
        "capacity_mw",
        "co2_short_ton_per_mwh",
    ]
    assert result.iloc[0]["region_id"] == "NORTH_EAST"
    assert result.iloc[0]["heat_rate_mmbtu_per_mwh"] == 10.5
    assert result.iloc[0]["capacity_mw"] == 150.0
    assert result.iloc[1]["region_id"] == "SOUTH"
    assert result.iloc[1]["capacity_mw"] == 0.0
    assert result.iloc[1]["heat_rate_mmbtu_per_mwh"] == 0.0
    assert result.iloc[1]["co2_short_ton_per_mwh"] == 1.2


def test_load_ei_units_tab_delimited(tmp_path: Path) -> None:
    csv_path = tmp_path / "ei_units.tsv"
    csv_path.write_text(
        "\t".join(
            [
                "region_id",
                "heat_rate_mmbtu_per_mwh",
                "capacity_mw",
                "co2_short_ton_per_mwh",
            ]
        )
        + "\nA-1\t5\t10\t0.5\n"
    )

    result = load_ei_units(csv_path, delimiter="\t")

    assert result.loc[0, "region_id"] == "A_1"
    assert result.loc[0, "capacity_mw"] == 10.0


def test_load_ei_units_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "ei_units_missing.csv"
    csv_path.write_text("region_id,capacity_mw\nA,10\n")

    with pytest.raises(ValueError):
        load_ei_units(csv_path)
