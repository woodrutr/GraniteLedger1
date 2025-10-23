from __future__ import annotations
import math
import pytest
pd = pytest.importorskip("pandas")
from dispatch.interface import DispatchResult
from dispatch import stub
from dispatch.lp_single import (
    _aggregate_generation_by_fuel,
    _aggregate_generation_by_region,
    _dispatch_merit_order,
    _validate_units_df,
    solve as lp_single_solve,
)
from regions.registry import REGIONS
from tests.fixtures.dispatch_single_minimal import (
    DEFAULT_REGION_ID,
    baseline_frames,
    baseline_units,
)

REGION_IDS = list(REGIONS)
SECOND_REGION_ID = REGION_IDS[1]


@pytest.fixture
def units_df() -> pd.DataFrame:
    return baseline_units().copy(deep=True)


def test_dispatch_result_summary_properties() -> None:
    baseline = DispatchResult(
        gen_by_fuel={"wind": 40.0, "gas": 60.0},
        region_prices={"r1": 32.0},
        emissions_tons=100.0,
        generation_by_coverage={"covered": 80.0, "non_covered": 20.0},
    )
    scenario = DispatchResult(
        gen_by_fuel={"wind": 50.0, "gas": 70.0},
        region_prices={"r1": 30.0},
        emissions_tons=90.0,
        generation_by_coverage={"covered": 90.0, "non_covered": 30.0},
    )
    assert baseline.total_generation == pytest.approx(100.0)
    assert baseline.covered_generation == pytest.approx(80.0)
    assert baseline.non_covered_generation == pytest.approx(20.0)
    assert scenario.leakage_percent(baseline) == pytest.approx(50.0)


def test_validate_units_df_requires_numeric_columns(units_df: pd.DataFrame) -> None:
    units_df = units_df.astype({"cap_mw": "object"})
    units_df.loc[0, "cap_mw"] = "not-a-number"
    with pytest.raises(ValueError):
        _validate_units_df(units_df)


def test_dispatch_merit_order_prioritises_low_cost_units(units_df: pd.DataFrame) -> None:
    units_df.loc[units_df["fuel"] == "gas", "vom_per_mwh"] = 100.0
    dispatch = _dispatch_merit_order(units_df, load_mwh=400_000.0, allowance_cost=0.0)
    generation = dispatch["generation"]
    assert generation.loc["wind-1"] > 0.0
    assert generation.loc["coal-1"] > 0.0
    assert generation.loc["gas-1"] == pytest.approx(0.0)
    assert math.isclose(dispatch["price"], dispatch["units"].loc["coal-1", "marginal_cost"])


def test_generation_aggregation_helpers_group_by_labels(units_df: pd.DataFrame) -> None:
    dispatch = _dispatch_merit_order(units_df, load_mwh=900_000.0, allowance_cost=0.0)
    generation = dispatch["generation"]
    units = dispatch["units"]
    by_fuel = _aggregate_generation_by_fuel(generation, units)
    assert set(by_fuel) == {"wind", "coal", "gas"}
    units = units.copy()
    units.loc[:, "region"] = [SECOND_REGION_ID, SECOND_REGION_ID, DEFAULT_REGION_ID]
    by_region = _aggregate_generation_by_region(generation, units)
    assert by_region[SECOND_REGION_ID] > 0.0
    assert by_region[DEFAULT_REGION_ID] > 0.0


def test_single_region_solve_returns_dispatch_frames() -> None:
    frames = baseline_frames(year=2025, load_mwh=900_000.0)
    result = lp_single_solve(2025, allowance_cost=5.0, frames=frames, carbon_price=2.5)
    assert result.gen_by_fuel
    assert result.region_prices
    assert result.emissions_tons >= 0.0
    assert DEFAULT_REGION_ID in result.region_coverage


def test_stub_solver_emissions_decline_with_price() -> None:
    baseline = stub.solve(2025, allowance_cost=0.0, carbon_price=0.0)
    priced = stub.solve(2025, allowance_cost=10.0, carbon_price=5.0)
    assert priced.emissions_tons < baseline.emissions_tons
    assert priced.total_generation > 0.0
