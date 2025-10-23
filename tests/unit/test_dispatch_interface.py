"""Unit tests for :mod:`dispatch.interface` convenience helpers."""
import importlib
from dataclasses import replace

import pytest

DispatchResult = importlib.import_module("dispatch.interface").DispatchResult


def _sample_result() -> DispatchResult:
    return DispatchResult(
        gen_by_fuel={"coal": 100.0, "wind": 50.0},
        region_prices={"north": 30.0, "south": 32.0},
        emissions_tons=120.0,
        emissions_by_region={"north": 70.0, "south": 50.0},
        flows={("north", "south"): 10.0},
        generation_by_region={
            "north": {"coal": 60.0, "wind": 30.0},
            "south": {"coal": 40.0, "wind": 20.0},
        },
        generation_by_coverage={"covered": 110.0, "non_covered": 40.0},
        imports_to_covered=5.0,
        exports_from_covered=2.0,
        region_coverage={"north": True, "south": False},
    )


def test_total_generation_matches_sum() -> None:
    result = _sample_result()
    assert result.total_generation == 150.0
    totals = result.generation_total_by_region
    assert totals["north"] == pytest.approx(90.0)
    assert totals["south"] == pytest.approx(60.0)


def test_generation_by_coverage_helpers() -> None:
    result = _sample_result()
    assert result.covered_generation == 110.0
    assert result.non_covered_generation == 40.0


def test_leakage_percent_compares_against_baseline() -> None:
    baseline = _sample_result()
    scenario = DispatchResult(
        gen_by_fuel={"coal": 120.0, "wind": 60.0},
        region_prices=baseline.region_prices,
        emissions_tons=baseline.emissions_tons,
        emissions_by_region=baseline.emissions_by_region,
        flows=baseline.flows,
        generation_by_region=baseline.generation_by_region,
        generation_by_coverage={"covered": 90.0, "non_covered": 90.0},
        imports_to_covered=baseline.imports_to_covered,
        exports_from_covered=baseline.exports_from_covered,
        region_coverage=baseline.region_coverage,
    )
    leakage = scenario.leakage_percent(baseline)
    assert isinstance(leakage, float)
    assert leakage > 0.0


def test_has_unserved_capacity_flag() -> None:
    result = _sample_result()
    assert not result.has_unserved_capacity

    updated = replace(result, unserved_capacity_by_region={"north": 5.0})
    assert updated.has_unserved_capacity
