from __future__ import annotations

import importlib
import math

import pytest

from tests.carbon_price_utils import assert_aliases_match_canonical, with_carbon_vector_columns

pd = pytest.importorskip("pandas")

run_loop = importlib.import_module("engine.run_loop")
run_end_to_end_from_frames = run_loop.run_end_to_end_from_frames
run_sensitivity_audit = importlib.import_module("engine.sensitivity").run_sensitivity_audit
fixtures = importlib.import_module("tests.fixtures.engine_audit")
YEARS = fixtures.YEARS
audit_frames = fixtures.audit_frames


def _annual(outputs):
    df = with_carbon_vector_columns(outputs.annual.copy())
    assert_aliases_match_canonical(df)
    return df


def test_zero_demand_results_zero_emissions():
    frames = audit_frames(loads=[0.0, 0.0, 0.0], cap_scale=1.0, bank0=0.0)
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    annual = _annual(outputs)
    assert math.isclose(float(annual["emissions_tons"].max()), 0.0, abs_tol=1e-6)
    assert math.isclose(float(annual["cp_last"].max()), 0.0, abs_tol=1e-6)
    assert not annual["shortage_flag"].any()


def test_zero_cap_triggers_shortage():
    frames = audit_frames(cap_scale=0.0, bank0=0.0)
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    annual = _annual(outputs)
    assert (annual["allowances_minted"] == 0.0).all()
    assert annual["shortage_flag"].all()
    assert (annual["cp_last"] > 0.0).all()


def test_large_bank_prevents_shortage():
    frames = audit_frames(bank0=10_000_000.0)
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    annual = _annual(outputs)
    assert not annual["shortage_flag"].any()
    assert (annual["cp_last"] >= 0.0).all()
    assert annual["bank"].iloc[-1] > 0.0


def test_fuel_price_spike_produces_finite_outputs():
    baseline = run_end_to_end_from_frames(
        audit_frames(),
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    spike = run_end_to_end_from_frames(
        audit_frames(gas_price_multiplier=10.0),
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    annual_spike = _annual(spike)
    assert annual_spike.isna().sum().sum() == 0
    assert (annual_spike["cp_last"] >= 0.0).all()
    assert (annual_spike["allowances_available"] >= 0.0).all()
    assert (annual_spike["emissions_tons"] >= 0.0).all()
    baseline_annual = _annual(baseline)
    assert annual_spike["emissions_tons"].max() <= baseline_annual["emissions_tons"].max() * 1.05


def test_sensitivity_audit_expected_direction():
    frames = audit_frames()
    report = run_sensitivity_audit(
        frames,
        YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    summary = report.summary()
    baseline = summary["baseline"]
    demand_up = summary["demand_up"]
    demand_down = summary["demand_down"]
    gas_up = summary["gas_up"]
    gas_down = summary["gas_down"]
    cap_up = summary["cap_up"]
    cap_down = summary["cap_down"]

    assert demand_up["total_emissions"] > baseline["total_emissions"]
    assert demand_down["total_emissions"] < baseline["total_emissions"]
    assert gas_up["avg_allowance_price"] >= baseline["avg_allowance_price"]
    assert gas_down["avg_allowance_price"] <= baseline["avg_allowance_price"]
    assert cap_up["avg_allowance_price"] <= baseline["avg_allowance_price"]
    assert cap_down["avg_allowance_price"] >= baseline["avg_allowance_price"]

    table = report.to_dataframe()
    assert not table.empty
    assert set(table["scenario"]) >= {
        "baseline",
        "demand_up",
        "gas_up",
        "cap_up",
    }
