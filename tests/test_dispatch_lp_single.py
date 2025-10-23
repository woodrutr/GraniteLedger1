"""Tests for the single-region merit-order dispatch implementation."""

from __future__ import annotations

import importlib
from typing import Iterable

import pytest

pytest.importorskip("pandas")

from dispatch.interface import DispatchResult
from dispatch.lp_single import _dispatch_merit_order, solve

_fixtures = importlib.import_module("tests.fixtures.dispatch_single_minimal")
baseline_frames = _fixtures.baseline_frames
baseline_units = _fixtures.baseline_units
infeasible_frames = _fixtures.infeasible_frames
frames_with_expansion = _fixtures.frames_with_expansion
expansion_options = _fixtures.expansion_options
DEFAULT_REGION_ID = _fixtures.DEFAULT_REGION_ID


def _collect_emissions(costs: Iterable[float]) -> list[float]:
    """Helper returning emissions for the supplied allowance price path."""

    emissions: list[float] = []
    for cost in costs:
        frames = baseline_frames()
        result = solve(2030, cost, frames=frames)
        emissions.append(result.emissions_tons)
    return emissions


def test_merit_order_shifts_with_allowance_cost() -> None:
    """Low allowance prices favour coal while high prices shift toward gas."""

    units = baseline_units()
    load = 1_000_000.0

    low_cost = _dispatch_merit_order(units, load, allowance_cost=0.0)
    high_cost = _dispatch_merit_order(units, load, allowance_cost=50.0)

    wind_cap = low_cost["units"].loc["wind-1", "cap_mwh"]

    assert low_cost["generation"].loc["wind-1"] == pytest.approx(wind_cap)
    assert high_cost["generation"].loc["wind-1"] == pytest.approx(wind_cap)
    assert low_cost["generation"].loc["coal-1"] > high_cost["generation"].loc["coal-1"]
    assert low_cost["generation"].loc["gas-1"] < high_cost["generation"].loc["gas-1"]

    result = solve(2030, 0.0, frames=baseline_frames())

    assert isinstance(result, DispatchResult)
    assert set(result.gen_by_fuel) == {"wind", "coal", "gas"}
    assert result.emissions_by_region[DEFAULT_REGION_ID] == pytest.approx(
        result.emissions_tons
    )
    assert result.flows == {}


def test_price_matches_marginal_cost_of_marginal_unit() -> None:
    """The reported price equals the marginal cost of the last dispatched unit."""

    units = baseline_units()
    summary = _dispatch_merit_order(units, 1_000_000.0, allowance_cost=0.0)

    used = summary["generation"][summary["generation"] > 0.0]
    last_unit = used.index[-1]
    expected_price = summary["units"].loc[last_unit, "marginal_cost"]

    assert summary["price"] == pytest.approx(expected_price)

    result = solve(2030, 0.0, frames=baseline_frames())

    assert result.region_prices[DEFAULT_REGION_ID] == pytest.approx(expected_price)


def test_variable_cost_includes_carbon_surcharge() -> None:
    """Carbon cost per MWh contributes to the marginal cost of the marginal unit."""

    units = baseline_units()
    units.loc[units["unit_id"] == "gas-1", "carbon_cost_per_mwh"] = 5.0

    summary = _dispatch_merit_order(units, 1_000_000.0, allowance_cost=0.0)

    gas_row = summary["units"].loc["gas-1"]
    expected_price = (
        float(gas_row["vom_per_mwh"])
        + float(gas_row["hr_mmbtu_per_mwh"]) * float(gas_row["fuel_price_per_mmbtu"])
        + float(gas_row["carbon_cost_per_mwh"])
    )

    assert summary["price"] == pytest.approx(expected_price)

    frames = baseline_frames()
    frames = frames.with_frame("units", units)
    result = solve(2030, 0.0, frames=frames)

    assert result.region_prices[DEFAULT_REGION_ID] == pytest.approx(expected_price)


def test_emissions_decline_with_allowance_cost() -> None:
    """Increasing the allowance price should not raise emissions."""

    costs = [0.0, 10.0, 30.0, 60.0]
    emissions = _collect_emissions(costs)

    assert all(a >= b - 1e-9 for a, b in zip(emissions, emissions[1:]))


def test_emissions_use_short_ton_factor_when_available() -> None:
    """Emissions calculations prefer the short-ton intensity column when supplied."""

    frames = baseline_frames()
    units = frames.units()
    units["co2_short_ton_per_mwh"] = units["ef_ton_per_mwh"] * 1.75
    frames = frames.with_frame("units", units)

    demand = frames.demand_for_year(2030)
    load = float(sum(demand.values()))
    summary = _dispatch_merit_order(units, load, allowance_cost=0.0)
    expected_emissions = float(
        (summary["generation"] * summary["units"]["ef_ton_per_mwh"]).sum()
    )

    result = solve(2030, 0.0, frames=frames)
    assert result.emissions_tons == pytest.approx(expected_emissions, rel=1e-6)


def test_emissions_decline_with_carbon_price() -> None:
    """An exogenous carbon price should suppress emissions."""

    prices = [0.0, 15.0, 45.0]
    emissions: list[float] = []
    for price in prices:
        result = solve(2030, 0.0, frames=baseline_frames(), carbon_price=price)
        emissions.append(result.emissions_tons)

    assert all(a >= b - 1e-9 for a, b in zip(emissions, emissions[1:]))


def test_generation_respects_capacity_limits() -> None:
    """No unit may exceed its annual energy capability."""

    units = baseline_units()
    summary = _dispatch_merit_order(units, 1_000_000.0, allowance_cost=20.0)

    caps = summary["units"]["cap_mwh"]
    for unit_id, dispatched in summary["generation"].items():
        assert dispatched <= caps.loc[unit_id] + 1e-6

    assert summary["generation"].sum() == pytest.approx(1_000_000.0)


def test_infeasible_load_reports_shortfall_and_price() -> None:
    """Loads above total capability return the correct price and shortfall."""

    frames = infeasible_frames()
    demand = frames.demand()
    year = int(demand.iloc[0]["year"])
    load = float(demand.loc[demand["year"] == year, "demand_mwh"].sum())

    summary = _dispatch_merit_order(frames.units(), load, allowance_cost=10.0)
    caps = summary["units"]["cap_mwh"]
    total_cap = caps.sum()
    shortfall_expected = load - total_cap

    assert shortfall_expected > 0.0
    assert summary["shortfall_mwh"] == pytest.approx(shortfall_expected)
    assert summary["generation"].equals(caps)

    used = summary["generation"][summary["generation"] > 0.0]
    last_unit = used.index[-1]
    expected_price = summary["units"].loc[last_unit, "marginal_cost"]

    assert summary["price"] == pytest.approx(expected_price)

    result = solve(year, 10.0, frames=frames)

    assert pytest.approx(total_cap) == sum(result.gen_by_fuel.values())


def test_capacity_expansion_inactive_without_trigger() -> None:
    """Expansion audit should not build capacity when economics are unfavourable."""

    frames = frames_with_expansion()
    result = solve(2030, 0.0, frames=frames, capacity_expansion=True)

    assert result.capacity_builds == []


def test_capacity_expansion_resolves_shortage() -> None:
    """Shortages trigger builds even when NPV would otherwise be negative."""

    frames = infeasible_frames()
    frames = frames.with_frame("expansion", expansion_options())
    demand = frames.demand()
    year = int(demand.iloc[0]["year"])
    load = float(demand.loc[demand["year"] == year, "demand_mwh"].sum())

    result = solve(year, 0.0, frames=frames, capacity_expansion=True)

    assert result.capacity_builds, "expected at least one build to resolve shortage"
    assert any(entry["reason"] == "supply_shortage" for entry in result.capacity_builds)
    assert sum(result.gen_by_fuel.values()) == pytest.approx(load)

    for entry in result.capacity_builds:
        assert {"capex_cost", "opex_cost", "emissions_tons"}.issubset(entry)
        if entry["reason"] == "supply_shortage":
            assert entry["npv_positive"] in {True, False}


def test_capacity_expansion_responds_to_carbon_price() -> None:
    """Positive NPV triggered by high carbon prices induces clean builds."""

    frames = frames_with_expansion()

    baseline = solve(2030, 0.0, frames=frames, capacity_expansion=True, carbon_price=0.0)
    high_price = solve(2030, 0.0, frames=frames, capacity_expansion=True, carbon_price=80.0)

    assert baseline.capacity_builds == []
    assert high_price.capacity_builds, "expected build under elevated carbon price"

    clean_build = next(
        entry for entry in high_price.capacity_builds if entry["reason"] == "npv_positive"
    )
    assert clean_build["npv_positive"] is True
    assert clean_build["capex_cost"] > 0.0
    assert clean_build["opex_cost"] >= 0.0
    assert clean_build["emissions_tons"] == pytest.approx(0.0, abs=1e-9)
    assert high_price.emissions_tons < baseline.emissions_tons


def test_capacity_expansion_multiple_builds_regression() -> None:
    """Multiple-build scenarios retain their generation mix and build ordering."""

    shortage_frames = frames_with_expansion(load_mwh=3_382_600.0)
    shortage_result = solve(2030, 0.0, frames=shortage_frames, capacity_expansion=True)

    expected_shortage_builds = [
        {
            "candidate": "solar",
            "unit_id": "solar_build1",
            "unique_id": "solar_build1",
            "capacity_mw": 60.0,
            "generation_mwh": 210_240.0,
            "reason": "supply_shortage",
            "npv_positive": False,
        },
        {
            "candidate": "solar",
            "unit_id": "solar_build2",
            "unique_id": "solar_build2",
            "capacity_mw": 60.0,
            "generation_mwh": 210_240.0,
            "reason": "supply_shortage",
            "npv_positive": False,
        },
        {
            "candidate": "solar",
            "unit_id": "solar_build3",
            "unique_id": "solar_build3",
            "capacity_mw": 60.0,
            "generation_mwh": 210_240.0,
            "reason": "supply_shortage",
            "npv_positive": False,
        },
        {
            "candidate": "fast_gas",
            "unit_id": "fast_gas_build1",
            "unique_id": "fast_gas_build1",
            "capacity_mw": 100.0,
            "generation_mwh": 788_400.0,
            "reason": "supply_shortage",
            "npv_positive": False,
        },
        {
            "candidate": "fast_gas",
            "unit_id": "fast_gas_build2",
            "unique_id": "fast_gas_build2",
            "capacity_mw": 100.0,
            "generation_mwh": 167_680.0,
            "reason": "supply_shortage",
            "npv_positive": False,
        },
    ]

    assert len(shortage_result.capacity_builds) == len(expected_shortage_builds)
    for actual, expected in zip(shortage_result.capacity_builds, expected_shortage_builds, strict=True):
        assert actual["candidate"] == expected["candidate"]
        assert actual["unit_id"] == expected["unit_id"]
        assert actual["reason"] == expected["reason"]
        assert actual["npv_positive"] is expected["npv_positive"]
        assert actual["capacity_mw"] == pytest.approx(expected["capacity_mw"])
        assert actual["generation_mwh"] == pytest.approx(expected["generation_mwh"])

    expected_shortage_generation = {
        "coal": 630_720.0,
        "gas": 1_902_160.0,
        "solar": 630_720.0,
        "wind": 219_000.0,
    }
    assert set(shortage_result.gen_by_fuel) == set(expected_shortage_generation)
    for fuel, expected in expected_shortage_generation.items():
        assert shortage_result.gen_by_fuel[fuel] == pytest.approx(expected)

    npv_frames = frames_with_expansion()
    npv_result = solve(
        2030,
        0.0,
        frames=npv_frames,
        capacity_expansion=True,
        carbon_price=120.0,
    )

    expected_npv_builds = [
        {
            "candidate": "solar",
            "unit_id": "solar_build1",
            "unique_id": "solar_build1",
            "capacity_mw": 60.0,
            "generation_mwh": 210_240.0,
            "reason": "npv_positive",
            "npv_positive": True,
        },
        {
            "candidate": "solar",
            "unit_id": "solar_build2",
            "unique_id": "solar_build2",
            "capacity_mw": 60.0,
            "generation_mwh": 210_240.0,
            "reason": "npv_positive",
            "npv_positive": True,
        },
        {
            "candidate": "solar",
            "unit_id": "solar_build3",
            "unique_id": "solar_build3",
            "capacity_mw": 60.0,
            "generation_mwh": 210_240.0,
            "reason": "npv_positive",
            "npv_positive": True,
        },
    ]

    assert len(npv_result.capacity_builds) == len(expected_npv_builds)
    for actual, expected in zip(npv_result.capacity_builds, expected_npv_builds, strict=True):
        assert actual["candidate"] == expected["candidate"]
        assert actual["unit_id"] == expected["unit_id"]
        assert actual["reason"] == expected["reason"]
        assert actual["npv_positive"] is expected["npv_positive"]
        assert actual["capacity_mw"] == pytest.approx(expected["capacity_mw"])
        assert actual["generation_mwh"] == pytest.approx(expected["generation_mwh"])

    expected_npv_generation = {
        "gas": 150_280.0,
        "solar": 630_720.0,
        "wind": 219_000.0,
    }
    assert set(npv_result.gen_by_fuel) == set(expected_npv_generation)
    for fuel, expected in expected_npv_generation.items():
        assert npv_result.gen_by_fuel[fuel] == pytest.approx(expected)
