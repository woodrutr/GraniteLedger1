from __future__ import annotations
import pytest
pd = pytest.importorskip("pandas")
import engine.outputs as outputs_module
from engine.outputs import EngineOutputs, _ensure_pandas
from regions.registry import REGIONS

REGION_IDS = list(REGIONS)
REGION_A = REGION_IDS[0]
REGION_B = REGION_IDS[1]


@pytest.fixture
def engine_outputs() -> EngineOutputs:
    annual = pd.DataFrame(
        [
            {"year": 2025, "emissions_tons": 500.0, "iterations": 3},
            {"year": 2026, "emissions_tons": 480.0, "iterations": 2},
        ]
    )
    emissions_by_region = pd.DataFrame(
        [
            {"year": 2025, "region": REGION_A, "emissions_tons": 300.0},
            {"year": 2025, "region": REGION_B, "emissions_tons": 200.0},
            {"year": 2026, "region": REGION_A, "emissions_tons": 280.0},
        ]
    )
    price_by_region = pd.DataFrame(
        [
            {"year": 2025, "region": REGION_A, "price": 30.0},
            {"year": 2025, "region": REGION_B, "price": 29.0},
        ]
    )
    flows = pd.DataFrame(
        [
            {"from": REGION_A, "to": REGION_B, "flow_mwh": 10.0},
        ]
    )
    return EngineOutputs(
        annual=annual,
        emissions_by_region=emissions_by_region,
        price_by_region=price_by_region,
        flows=flows,
    )


def test_ensure_pandas_raises_when_module_missing(monkeypatch) -> None:
    monkeypatch.setattr(outputs_module, "pd", None, raising=False)
    with pytest.raises(ImportError):
        _ensure_pandas()
    monkeypatch.setattr(outputs_module, "pd", pd, raising=False)


def test_engine_outputs_to_csv_writes_expected_files(engine_outputs, tmp_path) -> None:
    engine_outputs.to_csv(tmp_path)
    for filename in {"annual.csv", "emissions_by_region.csv", "price_by_region.csv", "flows.csv"}:
        assert (tmp_path / filename).exists()


def test_emissions_summary_table_normalises_fields(engine_outputs) -> None:
    summary = engine_outputs.emissions_summary_table()
    assert list(summary.columns) == ["year", "region", "emissions_tons"]
    assert summary.iloc[0]["year"] == 2025
    assert summary.iloc[0]["region"] == REGION_A
