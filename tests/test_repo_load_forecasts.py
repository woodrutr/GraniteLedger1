from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from common.data_access.load_forecasts import repo_load_forecasts


def test_repo_load_forecasts_uses_canonical_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path
    csv_dir = root / "input" / "electricity" / "load_forecasts"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "load_forecasts.csv"
    csv_path.write_text(
        "\n".join(
            [
                "region_id,state_or_province,scenario_name,year,load_gwh",
                "ISO-NE_CT,CT,Baseline Scenario,2030,1.25",
                "ISO-NE_CT,CT,Baseline Scenario,2031,1.50",
            ]
        )
    )

    def _fake_input_root() -> Path:
        return root / "input"

    monkeypatch.setattr("common.data_access.load_forecasts.input_root", _fake_input_root)

    repo_load_forecasts.cache_clear()

    frame_first = repo_load_forecasts()
    frame_second = repo_load_forecasts()

    expected_columns = [
        "region_id",
        "state_or_province",
        "scenario_name",
        "scenario",
        "year",
        "load_gwh",
        "load_mwh",
    ]
    assert list(frame_first.columns) == expected_columns
    assert frame_first["scenario"].iloc[0] == "baseline_scenario"
    pd.testing.assert_series_equal(
        frame_first["load_mwh"], frame_first["load_gwh"] * 1000.0, check_names=False
    )

    assert frame_first is frame_second

    repo_load_forecasts.cache_clear()
