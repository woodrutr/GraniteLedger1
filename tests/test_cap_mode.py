"""Unit tests for the endogenous cap allowance pricing engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from engine.cap_mode import CapParams, align_series, run_cap_mode
from tests.carbon_price_utils import assert_aliases_match_canonical, with_carbon_vector_columns


@dataclass
class SimpleDispatchFrame:
    """Minimal dispatch frame used to exercise price responsiveness."""

    base_costs: pd.Series
    emission_rates: pd.Series
    capacities: pd.Series


BASE_COSTS = pd.Series(
    {
        "coal1": 18.0,
        "coal2": 22.0,
        "gas1": 28.0,
        "gas2": 32.0,
        "wind": 38.0,
    }
)
EMISSION_RATES = pd.Series(
    {
        "coal1": 1.0,
        "coal2": 1.0,
        "gas1": 0.5,
        "gas2": 0.5,
        "wind": 0.0,
    }
)
CAPACITIES = pd.Series(
    {
        "coal1": 30.0,
        "coal2": 30.0,
        "gas1": 30.0,
        "gas2": 30.0,
        "wind": 40.0,
    }
)


def _build_frames(years: list[int]) -> dict[int, SimpleDispatchFrame]:
    frames: dict[int, SimpleDispatchFrame] = {}
    for year in years:
        frames[year] = SimpleDispatchFrame(
            base_costs=BASE_COSTS,
            emission_rates=EMISSION_RATES,
            capacities=CAPACITIES,
        )
    return frames


def run_case(
    *,
    cap: float = 70.0,
    demand: float = 1.0,
    banking: bool = True,
    bank0: float = 0.0,
) -> tuple[pd.DataFrame, list[dict]]:
    years = list(range(2025, 2031))
    frames = _build_frames(years)

    demand_base = {year: 120.0 * demand for year in years}
    demand_sched = align_series(demand_base, years, 120.0)

    budgets = align_series({years[0]: cap}, years, cap)
    reserve = align_series({years[0]: 12.0}, years, 12.0)
    ccr1_trigger = align_series({2030: 15.0}, years, float("inf"))
    ccr1_amount = align_series({2030: 15.0}, years, 0.0)
    ccr2_trigger = align_series({}, years, float("inf"))
    ccr2_amount = align_series({}, years, 0.0)

    params = CapParams(
        budgets=budgets,
        reserve=reserve,
        ccr1_trigger=ccr1_trigger,
        ccr1_amount=ccr1_amount,
        ccr2_trigger=ccr2_trigger,
        ccr2_amount=ccr2_amount,
        banking=banking,
    )

    result = run_cap_mode(years, frames, demand_sched, params, bank0)
    return result.table, result.diagnostics


def test_cap_responsiveness() -> None:
    table_lo, _ = run_case(cap=70.0)
    table_hi, _ = run_case(cap=60.0)
    assert table_hi.loc[2028, "cp_last"] > table_lo.loc[2028, "cp_last"]


def test_demand_responsiveness() -> None:
    table_base, _ = run_case(cap=65.0, demand=1.0)
    table_high, _ = run_case(cap=65.0, demand=1.1)
    assert table_high.loc[2027, "cp_last"] > table_base.loc[2027, "cp_last"]


def test_reserve_floor_binds() -> None:
    table, _ = run_case(cap=80.0)
    assert table.loc[2025, "cp_last"] >= 12.0


def test_ccr_triggers_when_price_crosses() -> None:
    table, _ = run_case(cap=55.0)
    allowances_total = table.loc[2030, "allowances_total"]
    assert allowances_total >= 70.0


def test_banking_toggle_affects_price() -> None:
    table_on, _ = run_case(cap=60.0, banking=True, bank0=30.0)
    table_off, _ = run_case(cap=60.0, banking=False, bank0=0.0)
    assert table_on.loc[2027, "cp_last"] <= table_off.loc[2027, "cp_last"]


def test_annual_coverage() -> None:
    table, _ = run_case()
    years = list(range(2025, 2031))
    assert list(table["year"]) == years


def test_cap_outputs_include_allowance_price_and_enforce_limit() -> None:
    table, _ = run_case()
    assert "allowance_price" in table.columns
    pd.testing.assert_series_equal(
        table["allowance_price"], table["cp_last"], check_names=False, rtol=0.0, atol=0.0
    )
    assert (table["emissions_tons"] <= table["available_allowances"] + 1e-6).all()
