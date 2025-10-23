"""Tests verifying behaviour when the carbon policy module is disabled."""

from __future__ import annotations

import importlib
from typing import Iterable

import pytest

from tests.carbon_price_utils import assert_aliases_match_canonical, with_carbon_vector_columns

pd = pytest.importorskip("pandas")


run_end_to_end_from_frames = importlib.import_module(
    "engine.run_loop"
).run_end_to_end_from_frames
solve_single = importlib.import_module("dispatch.lp_single").solve
fixtures = importlib.import_module("tests.fixtures.dispatch_single_minimal")
baseline_frames = fixtures.baseline_frames
DEFAULT_REGION_ID = fixtures.DEFAULT_REGION_ID


def _policy_frame(
    years: Iterable[int],
    *,
    policy_enabled: bool = True,
    bank_enabled: bool = True,
    ccr1_enabled: bool = True,
    ccr2_enabled: bool = True,
    cap_tons: float = 500_000.0,
    floor_dollars: float = 0.0,
    ccr1_trigger: float = 5.0,
    ccr1_qty: float = 0.0,
    ccr2_trigger: float = 15.0,
    ccr2_qty: float = 0.0,
    bank0: float = 0.0,
    annual_surrender_frac: float = 0.5,
    carry_pct: float = 1.0,
) -> pd.DataFrame:
    """Construct a policy frame spanning ``years`` with configurable toggles."""

    records: list[dict[str, object]] = []
    years_list = list(years)
    for idx, year in enumerate(years_list):
        records.append(
            {
                "year": int(year),
                "cap_tons": float(cap_tons),
                "floor_dollars": float(floor_dollars),
                "ccr1_trigger": float(ccr1_trigger),
                "ccr1_qty": float(ccr1_qty),
                "ccr2_trigger": float(ccr2_trigger),
                "ccr2_qty": float(ccr2_qty),
                "cp_id": "CP-1",
                "full_compliance": idx == len(years_list) - 1,
                "bank0": float(bank0),
                "annual_surrender_frac": float(annual_surrender_frac),
                "carry_pct": float(carry_pct),
                "policy_enabled": bool(policy_enabled),
                "bank_enabled": bool(bank_enabled),
                "ccr1_enabled": bool(ccr1_enabled),
                "ccr2_enabled": bool(ccr2_enabled),
            }
        )
    return pd.DataFrame(records)


def _frames_with_policy(years: Iterable[int], policy_df: pd.DataFrame) -> object:
    years_list = list(years)
    base = baseline_frames(year=years_list[0], load_mwh=600_000.0)
    demand = pd.DataFrame(
        [
            {"year": year, "region": DEFAULT_REGION_ID, "demand_mwh": 600_000.0}
            for year in years_list
        ]
    )
    return base.with_frame("demand", demand).with_frame("policy", policy_df)


def test_engine_matches_dispatch_when_policy_disabled():
    years = [2030, 2031]
    policy_df = pd.DataFrame({"year": years, "policy_enabled": False})
    frames = _frames_with_policy(years, policy_df)

    outputs = run_end_to_end_from_frames(
        frames,
        years=years,
        price_initial=20.0,
        tol=1e-5,
        relaxation=0.7,
    )
    annual = outputs.annual.set_index("year")
    assert annual["cp_last"].eq(0.0).all()
    assert annual["bank"].eq(0.0).all()
    assert annual["obligation"].eq(0.0).all()

    for year in years:
        dispatch = solve_single(year, 0.0, frames=frames)
        emissions = float(dispatch.emissions_tons)
        assert annual.loc[year, "emissions_tons"] == pytest.approx(emissions)
        assert annual.loc[year, "allowances_minted"] == pytest.approx(emissions)
        assert annual.loc[year, "allowances_available"] == pytest.approx(emissions)
        assert not bool(annual.loc[year, "finalized"])
        assert not bool(annual.loc[year, "shortage_flag"])


def test_policy_disabled_with_minimal_inputs():
    years = [2040]
    policy_df = pd.DataFrame({"year": years, "policy_enabled": False})
    frames = _frames_with_policy(years, policy_df)

    outputs = run_end_to_end_from_frames(frames, years=years)

    annual = outputs.annual.set_index("year")
    assert annual.loc[years[0], "cp_last"] == pytest.approx(0.0)
    assert annual.loc[years[0], "bank"] == pytest.approx(0.0)
    assert annual.loc[years[0], "obligation"] == pytest.approx(0.0)


def test_policy_disabled_applies_carbon_price_schedule():
    years = [2035]
    policy_df = pd.DataFrame({"year": years, "policy_enabled": False})
    frames = _frames_with_policy(years, policy_df)

    price_value = 42.5
    outputs = run_end_to_end_from_frames(
        frames, years=years, carbon_price_schedule={years[0]: price_value}
    )

    annual = outputs.annual.set_index("year")
    assert annual.loc[years[0], "cp_last"] == pytest.approx(price_value)

    dispatch = solve_single(
        years[0],
        0.0,
        frames=frames,
        carbon_price=price_value,
    )
    assert annual.loc[years[0], "emissions_tons"] == pytest.approx(dispatch.emissions_tons)



def test_ccr_toggles_control_allowance_issuance():
    years = [2050]
    base_policy = _policy_frame(
        years,
        cap_tons=50_000.0,
        ccr1_trigger=0.0,
        ccr1_qty=40_000.0,
        ccr2_trigger=0.0,
        ccr2_qty=60_000.0,
        annual_surrender_frac=1.0,
        carry_pct=0.0,
        bank0=0.0,
    )

    frames_all = _frames_with_policy(years, base_policy)
    frames_no_ccr1 = _frames_with_policy(
        years,
        base_policy.assign(ccr1_enabled=False, ccr2_enabled=True),
    )
    frames_no_ccr2 = _frames_with_policy(
        years,
        base_policy.assign(ccr1_enabled=True, ccr2_enabled=False),
    )

    enabled_outputs = run_end_to_end_from_frames(frames_all, years=years)
    no_ccr1_outputs = run_end_to_end_from_frames(frames_no_ccr1, years=years)
    no_ccr2_outputs = run_end_to_end_from_frames(frames_no_ccr2, years=years)

    annual_enabled = enabled_outputs.annual.set_index("year")
    annual_no_ccr1 = no_ccr1_outputs.annual.set_index("year")
    annual_no_ccr2 = no_ccr2_outputs.annual.set_index("year")

    year = years[0]
    cap_value = float(base_policy.loc[base_policy['year'] == year, 'cap_tons'].iloc[0])

    extra_enabled = annual_enabled.loc[year, "allowances_minted"] - cap_value
    extra_no_ccr1 = annual_no_ccr1.loc[year, "allowances_minted"] - cap_value
    extra_no_ccr2 = annual_no_ccr2.loc[year, "allowances_minted"] - cap_value

    assert extra_enabled == pytest.approx(100_000.0)
    assert extra_no_ccr1 == pytest.approx(60_000.0)
    assert extra_no_ccr2 == pytest.approx(40_000.0)
