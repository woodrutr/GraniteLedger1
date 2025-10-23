"""Tests for demand module settings coercion helpers."""

from __future__ import annotations

from gui.app import (
    _DEMAND_SETTINGS_FALLBACK_ERROR,
    _coerce_demand_module_settings,
    DemandModuleSettings,
)


def test_coerce_demand_settings_passthrough() -> None:
    original = DemandModuleSettings(
        enabled=True,
        curve_by_region={"a": "b"},
        forecast_by_region={"a": "forecast"},
        load_forecasts={"iso": "bundle"},
        custom_load_forecasts={"state": {"records": []}},
        errors=["error"],
    )

    result = _coerce_demand_module_settings(original)

    assert result is original


def test_coerce_demand_settings_from_mapping() -> None:
    mapping = {
        "enabled": True,
        "curve_by_region": {"region": "curve"},
        "forecast_by_region": {"region": "forecast"},
        "load_forecasts": {"iso": "manifest"},
        "custom_load_forecasts": {"state": {"records": []}},
        "errors": ["note"],
    }

    result = _coerce_demand_module_settings(mapping)

    assert isinstance(result, DemandModuleSettings)
    assert result.enabled is True
    assert result.curve_by_region == {"region": "curve"}
    assert result.forecast_by_region == {"region": "forecast"}
    assert result.load_forecasts == {"iso": "manifest"}
    assert result.custom_load_forecasts == {"state": {"records": []}}
    assert result.errors == ["note"]


def test_coerce_demand_settings_from_bool() -> None:
    result = _coerce_demand_module_settings(False)

    assert isinstance(result, DemandModuleSettings)
    assert result.enabled is False
    assert result.curve_by_region == {}
    assert result.forecast_by_region == {}
    assert result.load_forecasts == {}
    assert result.custom_load_forecasts == {}
    assert result.errors == [_DEMAND_SETTINGS_FALLBACK_ERROR]

