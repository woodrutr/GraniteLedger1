from pathlib import Path

import logging
import pandas as pd
import pytest

from engine.data_loaders.load_forecasts import load_table, scenario_index, zones_for
from engine.io import load_forecasts_strict as strict


def _write_csv(path: Path, *, rows: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Year,Load_GWh\n")
        for year, load in rows:
            handle.write(f"{year},{load}\n")


def test_load_table_prefers_strict(tmp_path: Path) -> None:
    root = tmp_path / "load_forecasts"
    _write_csv(root / "NYISO" / "Scenario_One" / "NYISO_A.csv", rows=[(2025, 1.5)])

    expected = strict.build_table(base_path=root)
    result = load_table(base_path=root)

    pd.testing.assert_frame_equal(result, expected)
    for column in ("iso", "scenario", "zone"):
        assert str(result[column].dtype) == "category"


def test_load_table_validation_error_logs_and_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "load_forecasts"
    target = root / "NYISO" / "Scenario_One" / "NYISO_A.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("Bad,Header\n", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(strict.ValidationError) as excinfo:
            load_table(base_path=root)

    # Check exception details
    err = excinfo.value
    assert err.file == target

    # Check log output
    assert any(
        "Strict load forecast validation failed" in record.getMessage()
        for record in caplog.records
    )



def test_helper_utilities() -> None:
    df = pd.DataFrame(
        {
            "iso": pd.Categorical(["iso_a", "iso_a", "iso_b"]),
            "scenario": pd.Categorical(["Baseline", "High", "Baseline"]),
            "zone": pd.Categorical(["Zone1", "Zone2", "Zone3"]),
            "region_id": pd.Categorical(["ISO_A_ZONE1", "ISO_A_ZONE2", "ISO_B_ZONE3"]),
            "Year": [2025, 2026, 2025],
            "Load_GWh": [1.0, 2.0, 3.0],
            "iso_norm": ["iso_a", "iso_a", "iso_b"],
            "scenario_norm": ["baseline", "high", "baseline"],
            "region_norm": ["iso_a_zone1", "iso_a_zone2", "iso_b_zone3"],
        }
    )

    scenarios = scenario_index(df)
    assert scenarios == {"iso_a": ["Baseline", "High"], "iso_b": ["Baseline"]}

    zones = zones_for(df, "iso_a", "Baseline")
    assert zones == ["Zone1"]
