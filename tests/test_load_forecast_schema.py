from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.api.types as ptypes
import pytest

from common.schemas.load_forecast import parse_load_forecast_csv


@pytest.mark.parametrize(
    "payload, expected",
    [
        (
            {
                "region_id": ["nyiso_a"],
                "state_or_province": ["NY"],
                "scenario_name": ["Baseline"],
                "year": [2025],
                "load_gwh": [100.0],
            },
            {
                "region_id": "NYISO_A",
                "state_or_province": "NY",
                "scenario_name": "Baseline",
                "scenario": "baseline",
                "year": 2025,
                "load_gwh": 100.0,
            },
        ),
        (
            {
                "region": ["nyiso a"],
                "state": ["ny"],
                "scenario": ["High Growth"],
                "timestamp": ["2025-06-01"],
                "demand": [150.5],
            },
            {
                "region_id": "NYISO_A",
                "state_or_province": "ny",
                "scenario_name": "High Growth",
                "scenario": "high_growth",
                "year": 2025,
                "load_gwh": 150.5,
            },
        ),
        (
            {
                "iso_zone": ["test zone"],
                "timestamp": ["2030-01-01"],
                "load_mwh": [5000],
            },
            {
                "region_id": "TEST_ZONE",
                "state_or_province": pd.NA,
                "scenario_name": "DEFAULT",
                "scenario": "default",
                "year": 2030,
                "load_gwh": 5.0,
            },
        ),
    ],
)
def test_parse_load_forecast_csv_synonyms(
    tmp_path: Path, payload: dict[str, list[object]], expected: dict[str, object]
) -> None:
    csv_path = tmp_path / "forecast.csv"
    pd.DataFrame(payload).to_csv(csv_path, index=False)

    frame = parse_load_forecast_csv(csv_path)

    assert frame.columns.tolist() == [
        "region_id",
        "state_or_province",
        "scenario_name",
        "scenario",
        "year",
        "load_gwh",
    ]
    assert len(frame) == 1

    row = frame.iloc[0]
    for column, value in expected.items():
        if value is pd.NA:
            assert pd.isna(row[column])
        else:
            assert row[column] == value

    assert ptypes.is_string_dtype(frame["region_id"])
    assert ptypes.is_string_dtype(frame["scenario_name"])
    assert ptypes.is_string_dtype(frame["scenario"])
    assert ptypes.is_string_dtype(frame["state_or_province"])
    assert frame["year"].dtype == "int64"
    assert ptypes.is_float_dtype(frame["load_gwh"])
