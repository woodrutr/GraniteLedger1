from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.deepdoc import build_deep_doc, deep_doc_to_markdown


class StubFrames:
    def __init__(self, units: pd.DataFrame, transmission: pd.DataFrame):
        self._units = units
        self._transmission = transmission

    def units(self) -> pd.DataFrame:
        return self._units

    def transmission(self) -> pd.DataFrame:
        return self._transmission


@pytest.fixture
def sample_manifest(tmp_path: Path) -> dict[str, object]:
    directory = tmp_path / "nyiso" / "GoldBook_2025_Baseline"
    directory.mkdir(parents=True)
    csv_path = directory / "NYISO_J.csv"
    csv_path.write_text("year,demand_mwh,region\n2025,1000,NYISO_J\n2026,1050,NYISO_J\n")

    frame = pd.DataFrame(
        {
            "year": [2025, 2026],
            "demand_mwh": [1000.0, 1050.0],
            "region": ["NYISO_J", "NYISO_J"],
        }
    )
    return {
        "iso": "nyiso",
        "scenario": "Baseline",
        "manifest": "NYISO Gold Book 2025 – Baseline",
        "source": "NYISO Gold Book",
        "vintage": 2025,
        "zones": ["NYISO_J"],
        "path": str(directory),
        "years": [2025, 2026],
        "frames": {"NYISO_J": frame},
        "source_files": {"NYISO_J": csv_path},
    }


def test_build_deep_doc_includes_forecast_and_dispatch(sample_manifest: dict[str, object]) -> None:
    manifest = {
        "load_forecasts": [
            {
                "iso": "nyiso",
                "manifest": "NYISO Gold Book 2025 – Baseline",
                "path": str(Path(sample_manifest["path"])) ,
                "years": sample_manifest.get("years"),
            }
        ]
    }

    units = pd.DataFrame(
        {
            "unit_id": ["u1", "u2", "u3"],
            "unique_id": ["u1", "u2", "u3"],
            "fuel": ["Gas", "Gas", "Coal"],
            "region": ["NYISO_J", "NYISO_K", "NYISO_J"],
            "cap_mw": [100.0, 150.0, 80.0],
            "hr_mmbtu_per_mwh": [7.5, 7.0, 10.0],
        }
    )
    transmission = pd.DataFrame(
        {
            "from_region": ["NYISO_J"],
            "to_region": ["NYISO_K"],
            "limit_mw": [500.0],
        }
    )
    frames = StubFrames(units, transmission)

    emissions_by_fuel = pd.DataFrame(
        {"year": [2025], "fuel": ["Gas"], "emissions_tons": [123.0]}
    )
    outputs = SimpleNamespace(emissions_by_fuel=emissions_by_fuel)

    deep_doc = build_deep_doc(manifest, frames, outputs)

    load_forecasts = deep_doc["load_forecasts"]
    assert isinstance(load_forecasts, pd.DataFrame)
    assert not load_forecasts.empty
    assert set(["Year", "Load_GWh", "iso_norm", "scenario_norm", "region_norm"]).issubset(
        load_forecasts.columns
    )
    assert load_forecasts["Year"].min() == 2025

    dispatch = deep_doc["dispatch"]
    assert dispatch["units_by_fuel"]["Gas"] == 2
    assert any(record["fuel"] == "Coal" for record in dispatch["capacity_mw_by_fuel_region"])

    transmission_records = deep_doc["transmission"]
    assert transmission_records[0]["from_region"] == "NYISO_J"

    emissions = deep_doc["emissions_by_fuel"]
    assert emissions and emissions[0]["fuel"] == "Gas"

    markdown = deep_doc_to_markdown(deep_doc)
    assert "Load Forecast Data" in markdown
