from __future__ import annotations

from pathlib import Path

import pytest

from engine.constants import CSV_LOAD_COL, CSV_YEAR_COL
from engine.data_loaders.load_forecasts import load_table
from engine.io import load_forecasts_strict as strict
from engine.settings import input_root


def _expected_root() -> Path:
    return Path(__file__).resolve().parents[1] / "input" / "electricity" / "load_forecasts"


def test_input_root_matches_repo_layout() -> None:
    root = input_root()
    assert root == _expected_root().resolve()


def test_load_table_includes_repo_zone() -> None:
    frame = load_table()

    expected_columns = [
        "iso",
        "scenario",
        "zone",
        "region_id",
        CSV_YEAR_COL,
        CSV_LOAD_COL,
        "iso_norm",
        "scenario_norm",
        "region_norm",
    ]
    assert list(frame.columns) == expected_columns

    subset = frame[
        (frame["iso"].astype(str) == "nyiso")
        & (frame["scenario"].astype(str) == "goldbook_2025_baseline")
        & (frame["zone"].astype(str) == "NYISO_J")
    ]
    assert not subset.empty
    assert set(subset["region_id"].astype(str)) == {"NYISO_J"}

    zone_path = _expected_root() / "nyiso" / "goldbook_2025_baseline" / "NYISO_J.csv"
    assert zone_path.exists()

    zone_frame = strict.read_zone_csv(zone_path)
    assert list(zone_frame.columns) == [CSV_YEAR_COL, CSV_LOAD_COL, "year", "demand_mwh"]
    assert not zone_frame.empty
    assert zone_frame[CSV_YEAR_COL].iloc[0] == zone_frame["year"].iloc[0]
    assert zone_frame["demand_mwh"].iloc[0] == pytest.approx(
        zone_frame[CSV_LOAD_COL].iloc[0] * 1_000.0
    )
