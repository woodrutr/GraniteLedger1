"""Tests for allowance supply behaviour with year-specific inputs."""
from __future__ import annotations

import pytest

from engine import run_loop
from io_loader import Frames
from tests.carbon_price_utils import assert_aliases_match_canonical, with_carbon_vector_columns

pd = pytest.importorskip("pandas")


class DummyDispatchResult(dict):
    """Lightweight dispatch output carrying emissions information."""

    def __init__(self, emissions: float):
        super().__init__({"emissions_tons": float(emissions)})
        self.emissions_tons = float(emissions)
        self.emissions_by_region = {"system": float(emissions)}
        self.region_prices: dict[str, float] = {}
        self.flows: dict[tuple[str, str], float] = {}


def _linear_dispatch(emission_params: dict[int, tuple[float, float]]):
    def dispatch(
        year: int,
        price: float,
        *,
        carbon_price: float | None = None,
    ) -> DummyDispatchResult:
        _ = carbon_price  # carbon price adjustments are ignored in this stub
        base, slope = emission_params[int(year)]
        emissions = max(base - slope * price, 0.0)
        return DummyDispatchResult(emissions)

    return dispatch


def _run_supply_simulation(
    monkeypatch: pytest.MonkeyPatch,
    policy_rows: list[dict[str, object]],
    emission_params: dict[int, tuple[float, float]],
    *,
    enable_floor: bool = True,
    enable_ccr: bool = True,
    price_cap: float = 500.0,
    tol: float = 1e-6,
    max_iter: int = 60,
):
    frames = Frames({"policy": pd.DataFrame(policy_rows)})
    dispatch = _linear_dispatch(emission_params)

    def fake_dispatch_from_frames(  # pragma: no cover - simple shim
        frames_obj,
        *,
        use_network=False,
        period_weights=None,
        carbon_price_schedule=None,
        deep_carbon_pricing=False,
    ):
        _ = deep_carbon_pricing
        return dispatch

    monkeypatch.setattr(run_loop, "_dispatch_from_frames", fake_dispatch_from_frames)
    years = sorted(emission_params)
    return run_loop.run_end_to_end_from_frames(
        frames,
        years=years,
        enable_floor=enable_floor,
        enable_ccr=enable_ccr,
        price_cap=price_cap,
        tol=tol,
        max_iter=max_iter,
    )


def _annual_table(outputs) -> pd.DataFrame:
    annual_df = with_carbon_vector_columns(outputs.annual)
    assert_aliases_match_canonical(annual_df)
    return annual_df.set_index("year")


def test_ccr_tiers_unlock_by_year(monkeypatch: pytest.MonkeyPatch):
    policy_rows = [
        {
            "year": 2025,
            "cap_tons": 100.0,
            "floor_dollars": 0.0,
            "ccr1_trigger": 20.0,
            "ccr1_qty": 30.0,
            "ccr2_trigger": 60.0,
            "ccr2_qty": 80.0,
            "cp_id": "CP1",
            "full_compliance": False,
            "bank0": 0.0,
            "annual_surrender_frac": 1.0,
            "carry_pct": 1.0,
            "policy_enabled": True,
        },
        {
            "year": 2026,
            "cap_tons": 100.0,
            "floor_dollars": 0.0,
            "ccr1_trigger": 30.0,
            "ccr1_qty": 50.0,
            "ccr2_trigger": 50.0,
            "ccr2_qty": 80.0,
            "cp_id": "CP1",
            "full_compliance": False,
            "bank0": 0.0,
            "annual_surrender_frac": 1.0,
            "carry_pct": 1.0,
            "policy_enabled": True,
        },
    ]
    emission_params = {2025: (150.0, 1.0), 2026: (260.0, 1.0)}

    outputs = _run_supply_simulation(
        monkeypatch,
        policy_rows,
        emission_params,
        enable_floor=False,
        enable_ccr=True,
        price_cap=200.0,
    )
    annual = _annual_table(outputs)

    assert annual.loc[2025, "cp_last"] == pytest.approx(20.0, abs=1e-6)
    assert annual.loc[2025, "allowances_minted"] == pytest.approx(130.0)
    assert annual.loc[2026, "cp_last"] == pytest.approx(50.0, abs=1e-6)
    assert annual.loc[2026, "allowances_minted"] == pytest.approx(230.0)


def test_floor_changes_by_year(monkeypatch: pytest.MonkeyPatch):
    policy_rows = [
        {
            "year": 2027,
            "cap_tons": 200.0,
            "floor_dollars": 5.0,
            "ccr1_trigger": 90.0,
            "ccr1_qty": 0.0,
            "ccr2_trigger": 120.0,
            "ccr2_qty": 0.0,
            "cp_id": "CP1",
            "full_compliance": False,
            "bank0": 0.0,
            "annual_surrender_frac": 1.0,
            "carry_pct": 1.0,
            "policy_enabled": True,
        },
        {
            "year": 2028,
            "cap_tons": 200.0,
            "floor_dollars": 15.0,
            "ccr1_trigger": 90.0,
            "ccr1_qty": 0.0,
            "ccr2_trigger": 120.0,
            "ccr2_qty": 0.0,
            "cp_id": "CP1",
            "full_compliance": False,
            "bank0": 0.0,
            "annual_surrender_frac": 1.0,
            "carry_pct": 1.0,
            "policy_enabled": True,
        },
    ]
    emission_params = {2027: (120.0, 0.0), 2028: (120.0, 0.0)}

    outputs = _run_supply_simulation(
        monkeypatch,
        policy_rows,
        emission_params,
        enable_floor=True,
        enable_ccr=False,
        price_cap=100.0,
    )
    annual = _annual_table(outputs)

    assert annual.loc[2027, "cp_last"] == pytest.approx(5.0, abs=1e-6)
    assert annual.loc[2028, "cp_last"] == pytest.approx(15.0, abs=1e-6)
    assert annual.loc[2027, "allowances_minted"] == pytest.approx(200.0)
    assert annual.loc[2028, "allowances_minted"] == pytest.approx(200.0)


def test_toggles_disable_features(monkeypatch: pytest.MonkeyPatch):
    policy_ccr = [
        {
            "year": 2029,
            "cap_tons": 100.0,
            "floor_dollars": 0.0,
            "ccr1_trigger": 20.0,
            "ccr1_qty": 30.0,
            "ccr2_trigger": 120.0,
            "ccr2_qty": 0.0,
            "cp_id": "CP1",
            "full_compliance": False,
            "bank0": 0.0,
            "annual_surrender_frac": 1.0,
            "carry_pct": 1.0,
            "policy_enabled": True,
        }
    ]
    emission_ccr = {2029: (150.0, 1.0)}

    with_ccr = _run_supply_simulation(
        monkeypatch,
        policy_ccr,
        emission_ccr,
        enable_floor=False,
        enable_ccr=True,
        price_cap=200.0,
    )
    without_ccr = _run_supply_simulation(
        monkeypatch,
        policy_ccr,
        emission_ccr,
        enable_floor=False,
        enable_ccr=False,
        price_cap=200.0,
    )

    annual_with = _annual_table(with_ccr)
    annual_without = _annual_table(without_ccr)

    assert annual_with.loc[2029, "allowances_minted"] == pytest.approx(130.0)
    assert annual_with.loc[2029, "cp_last"] == pytest.approx(20.0, abs=1e-6)
    assert annual_without.loc[2029, "allowances_minted"] == pytest.approx(100.0)
    assert annual_without.loc[2029, "cp_last"] == pytest.approx(50.0, abs=1e-6)

    policy_floor = [
        {
            "year": 2030,
            "cap_tons": 120.0,
            "floor_dollars": 20.0,
            "ccr1_trigger": 200.0,
            "ccr1_qty": 0.0,
            "ccr2_trigger": 250.0,
            "ccr2_qty": 0.0,
            "cp_id": "CP1",
            "full_compliance": False,
            "bank0": 0.0,
            "annual_surrender_frac": 1.0,
            "carry_pct": 1.0,
            "policy_enabled": True,
        }
    ]
    emission_floor = {2030: (130.0, 1.0)}

    with_floor = _run_supply_simulation(
        monkeypatch,
        policy_floor,
        emission_floor,
        enable_floor=True,
        enable_ccr=False,
        price_cap=200.0,
    )
    without_floor = _run_supply_simulation(
        monkeypatch,
        policy_floor,
        emission_floor,
        enable_floor=False,
        enable_ccr=False,
        price_cap=200.0,
    )

    annual_with_floor = _annual_table(with_floor)
    annual_without_floor = _annual_table(without_floor)

    assert annual_with_floor.loc[2030, "cp_last"] == pytest.approx(20.0, abs=1e-6)
    assert annual_without_floor.loc[2030, "cp_last"] == pytest.approx(10.0, abs=1e-6)
    assert annual_with_floor.loc[2030, "allowances_minted"] == pytest.approx(120.0)
    assert annual_without_floor.loc[2030, "allowances_minted"] == pytest.approx(120.0)


def test_supply_disabled_returns_baseline(monkeypatch: pytest.MonkeyPatch):
    policy_rows = [
        {
            "year": 2031,
            "cap_tons": 100.0,
            "floor_dollars": 25.0,
            "ccr1_trigger": 0.0,
            "ccr1_qty": 75.0,
            "ccr2_trigger": 0.0,
            "ccr2_qty": 125.0,
            "cp_id": "CP1",
            "full_compliance": False,
            "bank0": 0.0,
            "annual_surrender_frac": 1.0,
            "carry_pct": 1.0,
            "policy_enabled": False,
        }
    ]
    emission_params = {2031: (150.0, 0.0)}

    outputs = _run_supply_simulation(
        monkeypatch,
        policy_rows,
        emission_params,
        enable_floor=True,
        enable_ccr=True,
        price_cap=200.0,
    )
    annual = _annual_table(outputs)
    row = annual.loc[2031]

    assert row["cp_last"] == pytest.approx(0.0, abs=1e-6)
    assert row["allowances_minted"] == pytest.approx(150.0)
    assert row["allowances_available"] == pytest.approx(150.0)
    assert row["allowances_minted"] == pytest.approx(row["emissions_tons"])
