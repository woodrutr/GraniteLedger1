"""Unit tests for :mod:`policy.carbon`."""
from __future__ import annotations

import pytest

from policy import apply_carbon_policy, CarbonPolicyError


@pytest.fixture()
def base_config() -> dict[str, float | bool]:
    return {
        "enabled": True,
        "cap": 100.0,
        "enable_floor": False,
        "price_floor": 0.0,
        "enable_ccr": True,
        "ccr1_enabled": True,
        "ccr1_trigger_price": 30.0,
        "ccr1_qty": 10.0,
        "ccr2_enabled": True,
        "ccr2_trigger_price": 50.0,
        "ccr2_qty": 20.0,
        "allowance_banking_enabled": True,
    }


def test_apply_carbon_policy_enforces_cap_and_banking(base_config: dict[str, float | bool]) -> None:
    state = {
        "emissions": 80.0,
        "allowances": 120.0,
        "price": 25.0,
        "bank_balance": 15.0,
    }

    result = apply_carbon_policy(state, base_config)

    assert result["allowances_minted"] == pytest.approx(100.0)
    assert result["total_allowances"] == pytest.approx(115.0)
    assert result["surrendered"] == pytest.approx(80.0)
    assert result["bank_balance"] == pytest.approx(35.0)
    assert result["shortage"] is False


def test_apply_carbon_policy_applies_floor_and_ccr(base_config: dict[str, float | bool]) -> None:
    base_config.update({"enable_floor": True, "price_floor": 40.0})
    state = {
        "emissions": 140.0,
        "allowances": 100.0,
        "price": 35.0,
        "bank_balance": 0.0,
    }

    result = apply_carbon_policy(state, base_config)

    assert result["price"] == pytest.approx(40.0)
    assert result["ccr1_issued"] == pytest.approx(10.0)
    assert result["ccr2_issued"] == pytest.approx(0.0)
    assert result["allowances_minted"] == pytest.approx(110.0)
    assert result["shortage"] is True


def test_apply_carbon_policy_invalid_ccr_config(base_config: dict[str, float | bool]) -> None:
    base_config.pop("ccr1_trigger_price")
    base_config.pop("ccr1_price", None)
    with pytest.raises(CarbonPolicyError):
        apply_carbon_policy({"emissions": 10.0, "bank_balance": 0.0}, base_config)


def test_apply_carbon_policy_accepts_price_alias(base_config: dict[str, float | bool]) -> None:
    base_config.pop("ccr1_trigger_price")
    base_config.pop("ccr2_trigger_price")
    base_config["ccr1_price"] = 45.0
    base_config["ccr2_price"] = 65.0

    result = apply_carbon_policy(
        {"emissions": 90.0, "allowances": 100.0, "bank_balance": 0.0, "price": 70.0},
        base_config,
    )

    assert result["ccr1_issued"] == pytest.approx(10.0)
    assert result["ccr2_issued"] == pytest.approx(20.0)


def test_apply_carbon_policy_enforces_reserve_price(base_config: dict[str, float | bool]) -> None:
    base_config["reserve_price"] = {2025: 42.0}
    state = {
        "year": 2025,
        "emissions": 90.0,
        "allowances": 100.0,
        "price": 10.0,
        "bank_balance": 0.0,
    }

    result = apply_carbon_policy(state, base_config)

    assert result["price"] == pytest.approx(42.0)
    assert result["ccr1_issued"] == pytest.approx(10.0)
    assert result["ccr2_issued"] == pytest.approx(0.0)
    assert result["allowances_minted"] == pytest.approx(110.0)
    assert result["shortage"] is False
