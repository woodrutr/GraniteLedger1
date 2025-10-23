"""Tests for the dispatch stub implementation."""

from __future__ import annotations

import importlib
import math

import pytest

pd = pytest.importorskip('pandas')

DispatchResult = importlib.import_module('dispatch.interface').DispatchResult
_stub = importlib.import_module('dispatch.stub')
EMISSIONS_INTERCEPT = _stub.EMISSIONS_INTERCEPT
EMISSIONS_SLOPE = _stub.EMISSIONS_SLOPE
solve = _stub.solve


@pytest.mark.parametrize(
    'year, allowance_cost',
    [
        (2020, 0.0),
        (2025, 10.0),
        (2015, 60.0),
    ],
)
def test_emissions_follow_linear_formula(year: int, allowance_cost: float) -> None:
    """Emissions should follow the linear rule and never fall below zero."""

    result = solve(year, allowance_cost)

    expected = max(0.0, EMISSIONS_INTERCEPT - EMISSIONS_SLOPE * allowance_cost)

    assert isinstance(result, DispatchResult)
    assert math.isclose(result.emissions_tons, expected)


def test_region_prices_are_copied() -> None:
    """Mutating the returned prices must not impact subsequent runs."""

    first = solve(2024, 0.0)
    first.region_prices['east'] = 0.0

    second = solve(2024, 0.0)

    assert second.region_prices['east'] == pytest.approx(35.0)


def test_generation_scales_uniformly_with_year() -> None:
    """The generation mix should scale linearly with the model year."""

    base_year = 2020
    later_year = 2024
    allowance_cost = 5.0

    base = solve(base_year, allowance_cost).gen_by_fuel
    later = solve(later_year, allowance_cost).gen_by_fuel

    scale = 1.0 + 0.01 * (later_year - base_year)

    assert set(base) == set(later)

    for fuel, base_output in base.items():
        assert math.isclose(later[fuel], base_output * scale)

