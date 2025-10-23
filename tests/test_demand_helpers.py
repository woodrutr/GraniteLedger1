"""Tests for GUI demand helper utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from gui import demand_helpers as dh


def test_normalize_state_forecast_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """State forecast selection should return clean mappings and ISO pairs."""

    monkeypatch.setattr(dh, "_known_state_codes", lambda: {"NY", "CA"})

    selection = {"NY": "NYISO::Baseline", "CAISO": "Scenario"}
    state_map, iso_pairs = dh._normalize_state_forecast_selection(selection)

    assert state_map == {"NY": {"iso": "NYISO", "scenario": "Baseline"}}
    assert ("CAISO", "Scenario") in iso_pairs


def test_default_scenario_manifests_includes_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default manifest selection should surface every discovered scenario."""

    frame = pd.DataFrame(
        {
            "iso": ["NYISO", "NYISO"],
            "scenario": ["nyiso_2025_forecast", "nyiso_2025_goldbook_high_forecast"],
        }
    )

    captured: list[tuple[str, str]] = []

    def fake_build(
        iso_label: str,
        scenario_label: str,
        *,
        frame: pd.DataFrame,
        base_path: str | None,
    ) -> dh._ScenarioSelection | None:
        captured.append((iso_label, scenario_label))
        return dh._ScenarioSelection(iso=iso_label, scenario=scenario_label, zones=[], years=[])

    monkeypatch.setattr(dh, "_build_scenario_manifest", fake_build)

    manifests = dh._default_scenario_manifests(frame, base_path=None)

    assert [(entry.iso, entry.scenario) for entry in manifests] == captured == [
        ("nyiso", "nyiso_2025_forecast"),
        ("nyiso", "nyiso_2025_goldbook_high_forecast"),
    ]


def test_build_scenario_manifest_uses_zone_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario manifest construction should fall back to available zones when frames empty."""

    monkeypatch.setattr(dh, "_scenario_frame_subset", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(dh, "_load_iso_scenario_frame", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(dh, "_regions_available_zones", lambda *args, **kwargs: ["ZoneA", "ZoneB"])

    manifest = dh._build_scenario_manifest("ISO", "Scenario", base_path="/tmp/base")

    assert manifest is not None
    assert manifest.zones == ["ZoneA", "ZoneB"]


def test_demand_frame_from_manifests_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Demand frame builder should aggregate rows across manifests."""

    manifest = dh._ScenarioSelection(iso="ISO", scenario="Scenario", zones=["Z1"], years=[2030])

    data = pd.DataFrame(
        {
            "region_id": ["NYISO_NYUP", "NYISO_NYUP"],
            "year": [2030, 2031],
            "load_gwh": [1.5, 2.0],
        }
    )
    monkeypatch.setattr(dh, "_load_iso_scenario_frame", lambda *args, **kwargs: data)
    frame = dh._demand_frame_from_manifests([manifest], years=[2030, 2031], base_path="/tmp/base")

    assert set(frame.columns) == {"region", "year", "demand_mwh"}
    ordered = frame.sort_values(['region', 'year']).reset_index(drop=True)
    indexed = ordered.set_index(['region', 'year'])
    assert indexed.loc[("NYISO_NYUP", 2030), 'demand_mwh'] == pytest.approx(1500.0)
    assert indexed.loc[("NYISO_NYUP", 2031), 'demand_mwh'] == pytest.approx(2000.0)


def test_demand_frame_from_manifests_collects_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Demand frame builder should return load frames when requested."""

    manifest = dh._ScenarioSelection(iso="ISO", scenario="Scenario", zones=["Z1"], years=[2030])
    scenario_frame = pd.DataFrame(
        {
            "iso": ["ISO", "ISO"],
            "zone": ["ISO_Z1", "ISO_Z1"],
            "region_id": ["ISO_Z1", "ISO_Z1"],
            "scenario": ["Scenario", "Scenario"],
            "year": [2030, 2031],
            "load_gwh": [1.5, 2.0],
        }
    )

    monkeypatch.setattr(dh, "_load_iso_scenario_frame", lambda *args, **kwargs: scenario_frame)

    demand_df, load_df = dh._demand_frame_from_manifests(
        [manifest], years=[2030, 2031], base_path="/tmp/base", collect_load_frames=True
    )

    assert not demand_df.empty
    assert set(load_df.columns) >= {"iso", "zone", "scenario", "year", "load_gwh"}
    assert len(load_df) == 2
    assert load_df.loc[load_df["year"] == 2030, "load_gwh"].iloc[0] == pytest.approx(1.5)


def test_build_demand_output_frame_prefers_iso_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demand output frame should use ISO scenario lookups when available."""

    scenario_frame = pd.DataFrame(
        {
            "iso": ["ISO"],
            "region_id": ["NYISO_NYUP"],
            "zone": ["NYISO_NYUP"],
            "scenario": ["Scenario"],
            "year": [2030],
            "load_gwh": [1.0],
        }
    )

    monkeypatch.setattr(dh, "_cached_input_root", lambda: "/tmp/base")
    monkeypatch.setattr(dh, "_load_iso_scenario_frame", lambda *args, **kwargs: scenario_frame)
    monkeypatch.setattr(dh, "_load_demand_curve_catalog", lambda: {})
    monkeypatch.setattr(dh, "normalize_region", lambda value: value)
    monkeypatch.setattr(dh, "canonical_region_value", lambda value: value)
    monkeypatch.setattr(dh, "DEFAULT_REGION_METADATA", {"NYISO_NYUP": {}})
    import warnings

    demand_module = {
        "enabled": True,
        "load_forecasts": {"NYISO_NYUP": "ISO::Scenario"},
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        result = dh._build_demand_output_frame([2030], ["NYISO_NYUP"], demand_module)

    assert result is not None
    assert set(result.columns) == {"region", "year", "demand_mwh", "curve_key", "curve_label"}
    row = result.sort_values(['region', 'year']).iloc[0]
    assert row["region"] == "NYISO_NYUP"
    assert row["demand_mwh"] == pytest.approx(1000.0)
    assert "ISO" in row["curve_key"]


def test_build_demand_output_frame_handles_load_mwh(monkeypatch: pytest.MonkeyPatch) -> None:
    """ISO scenario frames expressed in MWh should be converted to GWh and back."""

    scenario_frame = pd.DataFrame(
        {
            "iso": ["ISO"],
            "region_id": ["NYISO_NYUP"],
            "zone": ["NYISO_NYUP"],
            "scenario": ["Scenario"],
            "year": [2030],
            "load_mwh": [1500.0],
            "load_gwh": [1.5],
            "state": ["NY"],
            "state_or_province": ["NY"],
            "scenario_name": ["Scenario"],
            "Year": [2030],
            "Load_GWh": [1.5],
            "iso_norm": ["ISO"],
            "scenario_norm": ["scenario"],
            "region_norm": ["NYISO_NYUP"],
        }
    )

    monkeypatch.setattr(dh, "_cached_input_root", lambda: "/tmp/base")
    monkeypatch.setattr(dh, "_load_iso_scenario_frame", lambda *args, **kwargs: scenario_frame)
    monkeypatch.setattr(dh, "_load_demand_curve_catalog", lambda: {})
    monkeypatch.setattr(dh, "normalize_region", lambda value: value)
    monkeypatch.setattr(dh, "canonical_region_value", lambda value: value)
    monkeypatch.setattr(dh, "DEFAULT_REGION_METADATA", {"NYISO_NYUP": {}})
    import warnings

    demand_module = {
        "enabled": True,
        "load_forecasts": {"NYISO_NYUP": "ISO::Scenario"},
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = dh._build_demand_output_frame([2030], ["NYISO_NYUP"], demand_module)

    assert result is not None
    row = result.sort_values(["region", "year"]).iloc[0]
    assert row["demand_mwh"] == pytest.approx(1500.0)
    assert "ISO" in row["curve_key"]


def test_scenario_frame_to_demand_prefers_non_empty_load_mwh() -> None:
    """Scenario demand conversion should fall back to GWh when MWh is empty."""

    scenario_frame = pd.DataFrame(
        {
            "region_id": ["NYISO_A"],
            "year": [2030],
            "load_mwh": [float("nan")],
            "load_gwh": [1.5],
        }
    )

    result = dh._scenario_frame_to_demand(scenario_frame)

    assert result.shape == (1, 3)
    row = result.iloc[0]
    assert row["region"] == "NYISO_A"
    assert row["year"] == 2030
    assert row["demand_mwh"] == pytest.approx(1500.0)


def test_discovered_region_names_uses_cached_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Region discovery should rely on the cached façade forecast frame."""

    forecast_frame = pd.DataFrame({"zone": ["Region_1", ""]})
    monkeypatch.setattr(dh, "_cached_forecast_frame", lambda base: forecast_frame)
    monkeypatch.setattr(dh, "DEFAULT_REGION_METADATA", {"Region_1": {}})
    monkeypatch.setattr(dh, "canonical_region_value", lambda value: value)

    regions = dh._discovered_region_names()

    assert regions == ["Region_1"]


def test_build_demand_output_frame_defaults_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no curves are available the demand builder should default to zero load."""

    monkeypatch.setattr(dh, "normalize_region", lambda value: value)
    monkeypatch.setattr(dh, "canonical_region_value", lambda value: value)
    monkeypatch.setattr(dh, "canonical_region_label", lambda value: value)
    monkeypatch.setattr(dh, "DEFAULT_REGION_METADATA", {"R1": {}})
    monkeypatch.setattr(dh, "_load_demand_curve_catalog", lambda: {})

    result = dh._build_demand_output_frame([2030, 2031], ["R1"], demand_module={})

    assert result is not None
    assert set(result["curve_key"]) == {dh._DEMAND_CURVE_FALLBACK_KEY}
    assert set(result["curve_label"]) == {dh._FALLBACK_DEMAND_LABEL}
    assert (result["demand_mwh"] == dh._FALLBACK_DEMAND_MWH).all()


def test_build_demand_output_frame_supports_flat_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    """The explicit flat curve selection should continue to yield the 100k load."""

    monkeypatch.setattr(dh, "normalize_region", lambda value: value)
    monkeypatch.setattr(dh, "canonical_region_value", lambda value: value)
    monkeypatch.setattr(dh, "canonical_region_label", lambda value: value)
    monkeypatch.setattr(dh, "DEFAULT_REGION_METADATA", {"R1": {}})
    monkeypatch.setattr(dh, "_load_demand_curve_catalog", lambda: {})

    demand_module = {
        "enabled": True,
        "curve_by_region": {"R1": dh._DEMAND_CURVE_FLAT_KEY},
    }

    result = dh._build_demand_output_frame([2030, 2031], ["R1"], demand_module=demand_module)

    assert result is not None
    assert set(result["curve_key"]) == {dh._DEMAND_CURVE_FLAT_KEY}
    assert set(result["curve_label"]) == {dh._FLAT_DEMAND_LABEL}
    assert (result["demand_mwh"] == dh._FLAT_DEMAND_MWH).all()
