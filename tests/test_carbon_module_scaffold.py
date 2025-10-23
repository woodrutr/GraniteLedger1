from __future__ import annotations

import pytest

from policy import carbon
from policy.carbon import CarbonPolicyError


def test_coerce_float_handles_missing_values() -> None:
    """Optional numeric fields fall back to defaults when missing."""

    assert carbon._coerce_float(None, default=2.5) == pytest.approx(2.5)
    assert carbon._coerce_float("7.5") == pytest.approx(7.5)
    with pytest.raises(CarbonPolicyError):
        carbon._coerce_float("invalid")


def test_required_float_raises_for_missing_keys() -> None:
    """Missing required configuration keys raise a friendly error."""

    config = {"cap": "100"}
    assert carbon._required_float(config, "cap") == pytest.approx(100.0)
    with pytest.raises(CarbonPolicyError):
        carbon._required_float({}, "cap")


def test_validate_trigger_enforces_numeric_inputs() -> None:
    """Trigger and quantity validation enforces numeric entries when enabled."""

    config = {"ccr1_trigger_price": 15, "ccr1_quantity": 2000}
    trigger, quantity = carbon._validate_trigger(
        enabled=True,
        trigger_keys=("ccr1_trigger_price",),
        quantity_keys=("ccr1_quantity",),
        config=config,
    )
    assert trigger == pytest.approx(15.0)
    assert quantity == pytest.approx(2000.0)

    with pytest.raises(CarbonPolicyError):
        carbon._validate_trigger(
            enabled=True,
            trigger_keys=("missing",),
            quantity_keys=("ccr1_quantity",),
            config={},
        )


def test_extract_year_hint_parses_embedded_digits() -> None:
    """Year hints are extracted from numeric and string tokens."""

    assert carbon._extract_year_hint("CY2028 compliance") == 2028
    assert carbon._extract_year_hint(2035) == 2035
    assert carbon._extract_year_hint("no year") is None


def test_resolve_reserve_price_prefers_year_specific_values() -> None:
    """Reserve price lookup falls back through state, config, and year hints."""

    state = {"year": 2030, "period_label": "CP1"}
    config = {
        "reserve_price": {
            "2030": 12.0,
            "CP1": 10.0,
        }
    }
    assert carbon._resolve_reserve_price(state, config) == pytest.approx(12.0)

    assert carbon._resolve_reserve_price({}, {"reserve_price": 9.5}) == pytest.approx(9.5)


def test_apply_carbon_policy_handles_enabled_and_disabled() -> None:
    """The primary policy function computes allowances and price adjustments."""

    state = {"emissions": 1000.0, "allowances": 900.0, "price": 5.0}
    config = {
        "enabled": True,
        "cap": 950.0,
        "enable_floor": True,
        "price_floor": 6.0,
        "allowance_banking_enabled": True,
    }
    result = carbon.apply_carbon_policy(state, config)
    assert result["allowances_minted"] == pytest.approx(900.0)
    assert result["price"] == pytest.approx(6.0)
    assert result["bank_balance"] >= 0.0

    disabled = carbon.apply_carbon_policy({"emissions": 100.0}, {"enabled": False})
    assert disabled["shortage"] is False
