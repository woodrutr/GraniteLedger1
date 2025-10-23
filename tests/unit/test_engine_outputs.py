"""Unit tests for :mod:`engine.outputs`."""
import importlib
from pathlib import Path
import pytest

EngineOutputs = importlib.import_module("engine.outputs").EngineOutputs
REGION_IDS = list(importlib.import_module("regions.registry").REGIONS)
REGION_A = REGION_IDS[0]
REGION_B = REGION_IDS[1]
from tests.carbon_price_utils import CarbonPriceFixture
pd = pytest.importorskip("pandas")


@pytest.fixture
def sample_outputs():
    annual = pd.DataFrame([
        {"year": 2025, "cp_last": 12.0, "emissions": 100.0},
        {"year": 2026, "cp_last": 14.0, "emissions": 95.0},
    ])
    emissions = pd.DataFrame(
        [
            {"region": REGION_A, "year": "2025", "emissions_tons": "50"},
            {"region": REGION_B, "year": "2026", "emissions_tons": "45"},
        ]
    )
    price = pd.DataFrame(
        [
            {"region": REGION_A, "year": 2025, "price": 32.0},
            {"region": REGION_B, "year": 2026, "price": 31.5},
        ]
    )
    flows = pd.DataFrame(
        [
            {"from_region": REGION_A, "to_region": REGION_B, "flow_mwh": 100.0},
        ]
    )
    demand_region = pd.DataFrame(
        [{"year": 2025, "region": "north", "demand_mwh": 120.0}]
    )
    generation_region = pd.DataFrame(
        [
            {"year": 2025, "region": "north", "fuel": "wind", "generation_mwh": 80.0}
        ]
    )
    capacity_region = pd.DataFrame(
        [
            {
                "year": 2025,
                "region": "north",
                "fuel": "wind",
                "capacity_mwh": 700.0,
                "capacity_mw": 80.0,
            }
        ]
    )
    cost_region = pd.DataFrame(
        [
            {
                "year": 2025,
                "region": "north",
                "fuel": "wind",
                "variable_cost": 1000.0,
                "allowance_cost": 200.0,
                "carbon_price_cost": 150.0,
                "total_cost": 1350.0,
            }
        ]
    )
    return EngineOutputs(
        annual=annual,
        emissions_by_region=emissions,
        price_by_region=price,
        flows=flows,
        limiting_factors=["policy"],
        emissions_total={2025: 100.0},
        emissions_by_region_map={REGION_A: {2025: 50.0}},
        demand_by_region=demand_region,
        generation_by_region=generation_region,
        capacity_by_region=capacity_region,
        cost_by_region=cost_region,
    )



def test_to_csv_writes_all_outputs(tmp_path: Path, sample_outputs: EngineOutputs) -> None:
    sample_outputs.to_csv(tmp_path)
    expected_files = [
        "annual.csv",
        "emissions_by_region.csv",
        "price_by_region.csv",
        "flows.csv",
        "demand_by_region.csv",
        "generation_by_region.csv",
        "capacity_by_region.csv",
        "cost_by_region.csv",
    ]
    for filename in expected_files:
        assert (tmp_path / filename).exists()


def test_emissions_summary_table_normalises_columns(sample_outputs: EngineOutputs) -> None:
    summary = sample_outputs.emissions_summary_table()
    assert list(summary.columns) == ["year", "region", "emissions_tons"]
    assert summary.iloc[0]["region"] == REGION_A
    assert summary.iloc[0]["emissions_tons"] == pytest.approx(50.0)


def test_engine_outputs_generates_state_tables(monkeypatch):
    share_df = pd.DataFrame(
        [
            {"region_id": REGION_A, "state": "NY", "share": 0.75},
            {"region_id": REGION_A, "state": "MA", "share": 0.25},
            {"region_id": REGION_B, "state": "NY", "share": 0.50},
            {"region_id": REGION_B, "state": "MA", "share": 0.50},
        ]
    )
    monkeypatch.setattr(
        "engine.outputs.load_zone_to_state_share", lambda: share_df.copy(deep=True)
    )
    monkeypatch.setattr(
        "engine.emissions.load_zone_to_state_share", lambda: share_df.copy(deep=True)
    )

    emissions = pd.DataFrame(
        [
            {"year": 2030, "region": REGION_A, "emissions_tons": 100.0},
            {"year": 2030, "region": REGION_B, "emissions_tons": 60.0},
        ]
    )
    demand = pd.DataFrame(
        [
            {"year": 2030, "region": REGION_A, "demand_mwh": 120.0},
            {"year": 2030, "region": REGION_B, "demand_mwh": 80.0},
        ]
    )

    outputs = EngineOutputs(
        annual=pd.DataFrame({"year": [2030]}),
        emissions_by_region=emissions,
        price_by_region=pd.DataFrame({"year": [2030], "region": [REGION_A], "price": [0.0]}),
        flows=pd.DataFrame({"year": [2030], "from_region": [REGION_A], "to_region": [REGION_B], "flow_mwh": [0.0]}),
        demand_by_region=demand,
        states=["ny", "ma"],
    )

    assert outputs.states == ("NY", "MA")
    demand_state = outputs.demand_by_state.sort_values(["state"]).reset_index(drop=True)
    emissions_state = outputs.emissions_by_state.sort_values(["state"]).reset_index(drop=True)

    assert pytest.approx(demand_state.loc[demand_state["state"] == "NY", "demand_mwh"].iloc[0]) == 130.0
    assert pytest.approx(demand_state.loc[demand_state["state"] == "MA", "demand_mwh"].iloc[0]) == 70.0
    assert pytest.approx(emissions_state.loc[emissions_state["state"] == "NY", "emissions_tons"].iloc[0]) == 105.0
    assert pytest.approx(emissions_state.loc[emissions_state["state"] == "MA", "emissions_tons"].iloc[0]) == 55.0
