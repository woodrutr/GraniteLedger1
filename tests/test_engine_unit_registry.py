from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

import engine
from engine.orchestrate import prepare_units


def test_prepare_units_registers_dataframe(tmp_path: Path) -> None:
    engine.clear_units()

    csv_path = tmp_path / "ei_units.csv"
    csv_path.write_text(
        "region_id,heat_rate_mmbtu_per_mwh,capacity_mw,co2_short_ton_per_mwh\n"
        "north-east,10.5,150,0.9\n"
        "south ,not_a_number, ,1.2\n"
    )

    config = {"ei_units_csv": csv_path}

    df = prepare_units(config)

    assert isinstance(df, pd.DataFrame)
    registered = engine.registered_units().reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "region_id": ["NORTH_EAST", "SOUTH"],
            "heat_rate": [10.5, 0.0],
            "capacity": [150.0, 0.0],
            "co2_rate": [0.9, 1.2],
        }
    )

    pdt.assert_frame_equal(registered, expected)


def test_set_units_requires_expected_columns() -> None:
    engine.clear_units()

    with pytest.raises(ValueError):
        engine.set_units(pd.DataFrame({"region_id": ["A"], "capacity_mw": [10]}))
