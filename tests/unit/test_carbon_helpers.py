"""Unit tests for helper functions in :mod:`policy.carbon`."""
import importlib
import pytest

carbon = importlib.import_module("policy.carbon")
CarbonPolicyError = carbon.CarbonPolicyError
_coerce_float = carbon._coerce_float
_extract_year_hint = carbon._extract_year_hint
_required_float = carbon._required_float
_resolve_reserve_price = carbon._resolve_reserve_price
_validate_trigger = carbon._validate_trigger


def test_coerce_float_handles_missing_values() -> None:
    assert _coerce_float(None, default=5.0) == 5.0
    assert _coerce_float("", default=3.0) == 3.0
    assert _coerce_float("7.25") == pytest.approx(7.25)


def test_coerce_float_raises_for_invalid_values() -> None:
    with pytest.raises(CarbonPolicyError):
        _coerce_float({"not": "numeric"})


def test_required_float_fetches_existing_entry() -> None:
    payload = {"cap": "123.4"}
    assert _required_float(payload, "cap") == pytest.approx(123.4)
    with pytest.raises(CarbonPolicyError):
        _required_float({}, "cap")


def test_validate_trigger_interprets_aliases() -> None:
    config = {"trigger": "15", "quantity": 12}
    trigger, quantity = _validate_trigger(
        enabled=True,
        trigger_keys=("trigger", "trigger_alias"),
        quantity_keys=("quantity", "quantity_alias"),
        config=config,
    )
    assert trigger == pytest.approx(15.0)
    assert quantity == pytest.approx(12.0)


def test_validate_trigger_raises_when_enabled_but_missing() -> None:
    with pytest.raises(CarbonPolicyError):
        _validate_trigger(
            enabled=True,
            trigger_keys=("missing",),
            quantity_keys=("quantity",),
            config={},
        )


def test_extract_year_hint_from_strings() -> None:
    assert _extract_year_hint("2024 compliance") == 2024
    assert _extract_year_hint(2026.9) == 2026
    assert _extract_year_hint("no year") is None


def test_resolve_reserve_price_prefers_specific_matches() -> None:
    state = {"year": 2030}
    config = {"reserve_price": {2030: "17.5", None: "10"}}
    assert _resolve_reserve_price(state, config) == pytest.approx(17.5)
    state = {"year": 2040}
    assert _resolve_reserve_price(state, config) is None
    config = {"reserve_price": 12}
    assert _resolve_reserve_price(state, config) == pytest.approx(12.0)
