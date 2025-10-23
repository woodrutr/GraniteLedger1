"""Unit tests for :mod:`gui.forecast_helpers`."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gui import forecast_helpers as fh


def test_regions_scenario_index_normalizes_iso_labels(monkeypatch: Any) -> None:
    """Scenario index should include normalized ISO aliases."""

    monkeypatch.setattr(
        fh,
        "normalize_iso_name",
        lambda value: str(value).strip().lower() if value else "",
    )

    frame = pd.DataFrame(
        {
            "iso": [" NYISO ", "nyiso", "PJM"],
            "scenario": ["Baseline", "Baseline", "High"],
        }
    )

    mapping = fh._regions_scenario_index(frame)

    assert mapping["NYISO"] == ["Baseline"]
    assert mapping["nyiso"] == ["Baseline"]
    assert mapping["PJM"] == ["High"]


def test_cached_forecast_frame_uses_cache(monkeypatch: Any) -> None:
    """The cached forecast frame should avoid redundant loader calls."""

    calls: list[str | None] = []
    sample = pd.DataFrame(
        {
            "iso": ["ISO"],
            "region_id": ["Z1"],
            "state": [""],
            "zone": ["Z1"],
            "scenario": ["Baseline"],
            "year": [2030],
            "load_gwh": [1.5],
            "load_mwh": [1500.0],
            "state_or_province": [""],
            "scenario_name": ["Baseline"],
        }
    )

    def fake_loader(*, base_path: str | None) -> pd.DataFrame:
        calls.append(base_path)
        return sample.copy()

    monkeypatch.setattr(fh, "_regions_load_forecasts_frame", fake_loader)

    fh._clear_forecast_cache()

    first = fh._cached_forecast_frame("/tmp/root")
    second = fh._cached_forecast_frame("/tmp/root")

    assert list(first.columns) == [
        "iso",
        "region_id",
        "state",
        "zone",
        "scenario",
        "year",
        "load_gwh",
        "load_mwh",
        "state_or_province",
        "scenario_name",
    ]
    assert first["region_id"].tolist() == ["Z1"]
    assert first["state"].fillna("").eq("").all()
    assert first["load_mwh"].tolist() == [1500.0]

    assert calls == ["/tmp/root"]
    pd.testing.assert_frame_equal(first, second)

    fh._clear_forecast_cache()
    fh._cached_forecast_frame("/tmp/root")
    assert calls == ["/tmp/root", "/tmp/root"]

    fh._clear_forecast_cache()


def test_cached_forecast_frame_records_errors(monkeypatch: Any) -> None:
    """Load failures should be recorded and exposed via the error helper."""

    def fake_loader(*, base_path: str | None) -> pd.DataFrame:
        raise ValueError("sample failure")

    monkeypatch.setattr(fh, "_regions_load_forecasts_frame", fake_loader)

    fh._clear_forecast_cache()

    frame = fh._cached_forecast_frame("/tmp/root")

    assert frame.empty
    assert fh.forecast_frame_error("/tmp/root") == "sample failure"

    fh._clear_forecast_cache()
    assert fh.forecast_frame_error("/tmp/root") is None


def test_clear_forecast_cache_resets_local_cache(monkeypatch: Any) -> None:
    """Clearing the façade cache should force the loader to run again."""

    calls: list[str | None] = []
    sample = pd.DataFrame(
        {
            "iso": ["ISO"],
            "region_id": ["ISO_ZONE1"],
            "state": ["TX"],
            "zone": ["ISO_ZONE1"],
            "scenario": ["Baseline"],
            "year": [2030],
            "load_gwh": [1.5],
            "load_mwh": [1500.0],
            "state_or_province": ["TX"],
            "scenario_name": ["Baseline"],
        }
    )

    def fake_loader(*, base_path: str | None) -> pd.DataFrame:
        calls.append(base_path)
        return sample.copy()

    monkeypatch.setattr(fh, "_regions_load_forecasts_frame", fake_loader)

    fh._clear_forecast_cache()

    fh._cached_forecast_frame("/tmp/root")
    fh._cached_forecast_frame("/tmp/root")
    assert calls == ["/tmp/root"]

    fh._clear_forecast_cache()
    fh._cached_forecast_frame("/tmp/root")
    assert calls == ["/tmp/root", "/tmp/root"]

    fh._clear_forecast_cache()


def test_regions_zones_for_falls_back_to_region_id() -> None:
    """Zone discovery should use region_id when zone column is absent."""

    frame = pd.DataFrame(
        {
            "iso": ["ISO"],
            "region_id": ["ISO_ZONE1"],
            "scenario": ["Baseline"],
        }
    )

    zones = fh._regions_zones_for(frame, "ISO", "Baseline")

    assert zones == ["ISO_ZONE1"]


def test_load_iso_scenario_frame_returns_region_id(monkeypatch: Any) -> None:
    """Scenario frame helper should always include a region_id column."""

    sample = pd.DataFrame(
        {
            "iso": ["ISO"],
            "region_id": ["Z1"],
            "state": [""],
            "zone": ["Z1"],
            "scenario": ["Baseline"],
            "year": [2030],
            "load_gwh": [1.5],
            "load_mwh": [1500.0],
            "state_or_province": [""],
            "scenario_name": ["Baseline"],
        }
    )

    def fake_loader(*, base_path: str | None) -> pd.DataFrame:  # type: ignore[override]
        return sample.copy()

    monkeypatch.setattr(fh, "_regions_load_forecasts_frame", fake_loader)

    fh._clear_forecast_cache()
    frame = fh._load_iso_scenario_frame("/tmp/root", "ISO", "Baseline")

    assert list(frame.columns) == [
        "iso",
        "region_id",
        "state",
        "zone",
        "scenario",
        "year",
        "load_gwh",
        "load_mwh",
        "state_or_province",
        "scenario_name",
    ]
    assert frame["region_id"].tolist() == ["Z1"]
    assert frame["state"].fillna("").eq("").all()
