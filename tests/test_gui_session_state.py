"""Tests for the Streamlit session state helpers."""

from __future__ import annotations

from gui.session_state import CarbonSessionState


def _base_state() -> CarbonSessionState:
    return CarbonSessionState(
        enabled=True,
        price_enabled=False,
        enable_floor=True,
        enable_ccr=True,
        ccr1_enabled=True,
        ccr2_enabled=False,
        banking_enabled=True,
        bank0=1000.0,
        control_override=True,
        control_years=3,
        ccr1_price=50.0,
        ccr1_escalator=1.5,
        ccr2_price=75.0,
        ccr2_escalator=2.5,
        coverage_regions=["All"],
        price_value=10.0,
        price_escalator=0.0,
        cap_start=100_000,
        cap_reduction_mode="percent",
        cap_reduction_percent=3.0,
        cap_reduction_fixed=0.0,
        floor_value_input="5.00",
        floor_mode="fixed",
        floor_escalator_input="0.50",
    )


def test_apply_defaults_refreshes_state_when_config_changes() -> None:
    session: dict[str, object] = {}
    first = _base_state()

    first.apply_defaults(session)
    assert session["carbon_price_value"] == 10.0
    assert session["carbon_coverage_regions"] == ["All"]

    # User modifies a value in the active session.
    session["carbon_price_value"] = 25.0
    first.apply_defaults(session)
    assert session["carbon_price_value"] == 25.0

    updated = CarbonSessionState(
        enabled=False,
        price_enabled=True,
        enable_floor=False,
        enable_ccr=False,
        ccr1_enabled=False,
        ccr2_enabled=False,
        banking_enabled=False,
        bank0=500.0,
        control_override=False,
        control_years=5,
        ccr1_price=60.0,
        ccr1_escalator=2.0,
        ccr2_price=90.0,
        ccr2_escalator=3.0,
        coverage_regions=["NYISO"],
        price_value=75.0,
        price_escalator=1.0,
        cap_start=80_000,
        cap_reduction_mode="fixed",
        cap_reduction_percent=0.0,
        cap_reduction_fixed=1500.0,
        floor_value_input="12.00",
        floor_mode="percent",
        floor_escalator_input="1.50",
    )

    updated.apply_defaults(session)

    assert session["carbon_enable"] is False
    assert session["carbon_price_enable"] is True
    assert session["carbon_price_value"] == 75.0
    assert session["carbon_price_escalator"] == 1.0
    assert session["carbon_coverage_regions"] == ["NYISO"]
    assert session["carbon_coverage_regions"] is not updated.coverage_regions
    assert session["carbon_floor_mode"] == "percent"
    assert session["carbon_floor_escalator_input"] == "1.50"
    assert session["carbon_bank0"] == 500.0
    assert session["_carbon_defaults_signature"] == updated._signature()


def test_override_for_lock_resets_to_defaults() -> None:
    session: dict[str, object] = {}
    defaults = _base_state()

    defaults.apply_defaults(session)
    session["carbon_enable"] = False
    session["carbon_bank0"] = 250.0
    session["carbon_coverage_regions"] = ["Custom"]

    defaults.override_for_lock(session)

    assert session["carbon_enable"] is True
    assert session["carbon_bank0"] == 1000.0
    assert session["carbon_coverage_regions"] == ["All"]
    assert session["_carbon_defaults_signature"] == defaults._signature()
