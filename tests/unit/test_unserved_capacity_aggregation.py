"""Tests for peak demand shortfall aggregation in :mod:`engine.run_loop`."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from dispatch.interface import DispatchResult
from engine.run_loop import _build_engine_outputs


class _PolicyStub(SimpleNamespace):
    """Stub policy providing minimal interface for :func:`_build_engine_outputs`."""

    enabled: bool = True
    banking_enabled: bool = True

    def compliance_year_for(self, period: object) -> int:
        return 2030


def _dispatch_not_called(*_args, **_kwargs) -> None:
    raise AssertionError("dispatch solver should not be invoked in this test")


def _make_dispatch_result(shortfall: float, peak: float = 100.0) -> DispatchResult:
    """Create a dispatch result capturing the supplied shortfall."""

    return DispatchResult(
        gen_by_fuel={},
        region_prices={},
        emissions_tons=0.0,
        demand_by_region={"north": 0.0},
        peak_demand_by_region={"north": peak},
        unserved_capacity_by_region={"north": shortfall},
        unserved_capacity_total=shortfall,
    )


def test_unserved_capacity_uses_peak_shortfall() -> None:
    """The aggregated shortfall should reflect the peak across periods, not the sum."""

    policy = _PolicyStub()

    raw_results = {
        "period-a": {
            "cp_last": 0.0,
            "cp_exempt": 0.0,
            "emissions": 0.0,
            "_dispatch_result": _make_dispatch_result(shortfall=5.0),
        },
        "period-b": {
            "cp_last": 0.0,
            "cp_exempt": 0.0,
            "emissions": 0.0,
            "_dispatch_result": _make_dispatch_result(shortfall=10.0),
        },
    }

    outputs = _build_engine_outputs(
        years=list(raw_results),
        raw_results=raw_results,
        dispatch_solver=_dispatch_not_called,
        policy=policy,
    )

    peak_shortfall = (
        outputs.peak_demand_by_region.set_index(["year", "region"])
        .loc[(2030, "north"), "shortfall_mw"]
    )
    assert peak_shortfall == pytest.approx(10.0)

    peak_requirement = (
        outputs.peak_demand_by_region.set_index(["year", "region"])
        .loc[(2030, "north"), "peak_demand_mw"]
    )
    assert peak_requirement == pytest.approx(100.0)


@pytest.mark.parametrize(
    "first, second, expected",
    [
        (0.0, 0.0, 0.0),
        (12.5, 7.5, 12.5),
        (1.0, 3.5, 3.5),
    ],
)
def test_shortfall_reflects_maximum(first: float, second: float, expected: float) -> None:
    """Different period combinations should always select the maximum shortfall."""

    policy = _PolicyStub()

    raw_results = {
        "p1": {
            "cp_last": 0.0,
            "cp_exempt": 0.0,
            "emissions": 0.0,
            "_dispatch_result": _make_dispatch_result(shortfall=first),
        },
        "p2": {
            "cp_last": 0.0,
            "cp_exempt": 0.0,
            "emissions": 0.0,
            "_dispatch_result": _make_dispatch_result(shortfall=second),
        },
    }

    outputs = _build_engine_outputs(
        years=list(raw_results),
        raw_results=raw_results,
        dispatch_solver=_dispatch_not_called,
        policy=policy,
    )

    peak_shortfall = (
        outputs.peak_demand_by_region.set_index(["year", "region"])
        .loc[(2030, "north"), "shortfall_mw"]
    )
    assert peak_shortfall == pytest.approx(expected)

    # Ensure the DataFrame materialised correctly.
    assert isinstance(outputs.peak_demand_by_region, pd.DataFrame)
    assert not outputs.peak_demand_by_region.empty
