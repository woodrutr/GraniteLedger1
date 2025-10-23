"""Unit tests exercising demand scaling within :mod:`engine.run_loop`."""
from __future__ import annotations
import importlib
from typing import Any
import pytest

DispatchResult = importlib.import_module("dispatch.interface").DispatchResult
run_loop = importlib.import_module("engine.run_loop")
fixtures = importlib.import_module("tests.fixtures.dispatch_single_minimal")
baseline_frames = fixtures.baseline_frames
DEFAULT_REGION_ID = fixtures.DEFAULT_REGION_ID
pd = pytest.importorskip("pandas")


@pytest.fixture
def frames():
    return baseline_frames(year=2025, load_mwh=1_000_000.0)


def test_dispatch_respects_period_weights(frames, monkeypatch) -> None:
    observed_loads: list[float] = []
    observed_prices: list[float] = []

    def fake_solve(year: Any, allowance_cost: float, *, frames, carbon_price: float = 0.0) -> DispatchResult:
        demand_map = frames.demand_for_year(year)
        total_load = sum(demand_map.values())
        observed_loads.append(total_load)
        observed_prices.append(carbon_price)
        return DispatchResult(
            gen_by_fuel={"total": total_load},
            region_prices={DEFAULT_REGION_ID: allowance_cost + carbon_price},
            emissions_tons=total_load * 0.1,
            emissions_by_region={DEFAULT_REGION_ID: total_load * 0.1},
            flows={},
            generation_by_region={DEFAULT_REGION_ID: total_load},
            generation_by_coverage={"covered": total_load},
            imports_to_covered=0.0,
            exports_from_covered=0.0,
            region_coverage={DEFAULT_REGION_ID: True},
        )


    monkeypatch.setattr("dispatch.lp_single.solve", fake_solve)
    monkeypatch.setattr(run_loop, "solve_single", fake_solve)
    weighted_dispatch = run_loop._dispatch_from_frames(
        frames,
        period_weights={2025: 2.0},
        carbon_price_schedule={2025: 15.0},
    )
    result = weighted_dispatch(2025, 25.0)
    assert observed_loads[0] == pytest.approx(500_000.0)
    assert result.total_generation == pytest.approx(1_000_000.0)
    assert observed_prices[0] == pytest.approx(15.0)
    assert result.region_prices[DEFAULT_REGION_ID] == pytest.approx(40.0)
    unweighted_dispatch = run_loop._dispatch_from_frames(frames)
    baseline = unweighted_dispatch(2025, 25.0)
    assert baseline.total_generation == pytest.approx(1_000_000.0)
    assert observed_loads[-1] == pytest.approx(1_000_000.0)
