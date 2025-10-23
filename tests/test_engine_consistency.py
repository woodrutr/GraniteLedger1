import pytest

pd = pytest.importorskip("pandas")

from engine.allowance import enforce_bank_trajectory
from engine.emissions import (
    apply_declining_cap,
    summarize_emissions,
    summarize_emissions_by_state,
)
from engine.outputs import EngineOutputs


def test_allowance_bank_compliance():
    annual = pd.DataFrame(
        {
            "year": [2030, 2031],
            "allowances_minted": [1000.0, 100.0],
            "emissions_tons": [800.0, 950.0],
            "bank_start": [500.0, None],
            "allowance_price": [5.0, 5.0],
            "floor": [4.0, 4.0],
        }
    )
    enforced = enforce_bank_trajectory(annual)
    totals = enforced["allowances_available"].tolist()
    banks = enforced["bank"].tolist()
    expected_total_first = 1500.0
    expected_bank_first = expected_total_first - 800.0
    expected_total_second = expected_bank_first + 100.0
    expected_bank_second = max(expected_total_second - 950.0, 0.0)
    assert totals[0] == pytest.approx(expected_total_first, rel=1e-6)
    assert banks[0] == pytest.approx(expected_bank_first, rel=1e-6)
    assert totals[1] == pytest.approx(expected_total_second, rel=1e-6)
    assert banks[1] == pytest.approx(expected_bank_second, rel=1e-6)
    shortage_row = enforced.iloc[1]
    assert shortage_row["allowance_price"] > shortage_row["floor"] - 1e-9


def test_cap_declines_five_percent():
    annual = pd.DataFrame({"year": [2030, 2031, 2032], "allowances_minted": [1000.0, 1000.0, 1000.0]})
    declined = apply_declining_cap(annual)
    minted = declined["allowances_minted"].to_numpy(dtype=float)
    for idx in range(1, len(minted)):
        assert minted[idx] == pytest.approx(minted[idx - 1] * 0.95, rel=1e-6)
    assert "cap" not in declined.columns


def test_emissions_reconcile():
    emissions_df = pd.DataFrame(
        {
            "year": [2030, 2030, 2031],
            "region": ["north", "south", "north"],
            "emissions_tons": [120.0, 80.0, 90.0],
        }
    )
    totals, region_map = summarize_emissions(emissions_df)
    assert totals == {2030: 200.0, 2031: 90.0}
    assert region_map["north"] == {2030: 120.0, 2031: 90.0}
    assert region_map["south"] == {2030: 80.0}


def test_summarize_emissions_by_state(monkeypatch):
    share_df = pd.DataFrame(
        [
            {"region_id": "north", "state": "NY", "share": 0.75},
            {"region_id": "north", "state": "MA", "share": 0.25},
            {"region_id": "south", "state": "NY", "share": 0.50},
            {"region_id": "south", "state": "MA", "share": 0.50},
        ]
    )
    monkeypatch.setattr(
        "engine.emissions.load_zone_to_state_share", lambda: share_df.copy(deep=True)
    )

    emissions_df = pd.DataFrame(
        [
            {"year": 2030, "region": "north", "emissions_tons": 100.0},
            {"year": 2030, "region": "south", "emissions_tons": 60.0},
        ]
    )

    summary = summarize_emissions_by_state(emissions_df)
    assert list(summary.columns) == ["year", "state", "emissions_tons"]
    ny_row = summary[summary["state"] == "NY"].iloc[0]
    ma_row = summary[summary["state"] == "MA"].iloc[0]
    assert ny_row["emissions_tons"] == pytest.approx(105.0, rel=1e-6)
    assert ma_row["emissions_tons"] == pytest.approx(55.0, rel=1e-6)


def test_dispatch_outputs_populated():
    annual = pd.DataFrame({"year": [2030]})
    emissions = pd.DataFrame({"year": [2030], "region": ["north"], "emissions_tons": [100.0]})
    price = pd.DataFrame({"year": [2030], "region": ["north"], "price": [30.0]})
    flows = pd.DataFrame({"year": [2030], "from_region": ["north"], "to_region": ["south"], "flow_mwh": [25.0]})
    demand_region = pd.DataFrame({"year": [2030, 2030], "region": ["north", "south"], "demand_mwh": [120.0, 80.0]})
    generation_region = pd.DataFrame(
        {
            "year": [2030, 2030],
            "region": ["north", "south"],
            "fuel": ["gas", "wind"],
            "generation_mwh": [100.0, 80.0],
        }
    )
    outputs = EngineOutputs(
        annual=annual,
        emissions_by_region=emissions,
        price_by_region=price,
        flows=flows,
        demand_by_region=demand_region,
        generation_by_region=generation_region,
        capacity_by_region=pd.DataFrame(),
        cost_by_region=pd.DataFrame(),
        generation_by_fuel=pd.DataFrame(),
        capacity_by_fuel=pd.DataFrame(),
        cost_by_fuel=pd.DataFrame(),
        emissions_by_fuel=pd.DataFrame(),
        stranded_units=pd.DataFrame(),
    )
    assert not outputs.generation_by_region.empty
    assert outputs.generation_by_region["generation_mwh"].sum() > 0.0
    assert not outputs.flows.empty
    assert outputs.flows["flow_mwh"].abs().sum() > 0.0
