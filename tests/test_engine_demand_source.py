from __future__ import annotations

import pandas as pd
import pytest
from types import SimpleNamespace

from engine.inputs.demand_source import resolve_demand_frame
from engine import orchestrate


def test_resolve_from_gui_table():
    gui_table = pd.DataFrame(
        {
            "Region": [" zone_a", "Zone_B"],
            "Year": [2025, 2026],
            "demand_mwh": [1234, 5678.9],
        }
    )

    result = resolve_demand_frame({"demand_table": gui_table})

    assert list(result.columns) == ["region_id", "year", "mwh"]
    assert pd.api.types.is_string_dtype(result["region_id"])
    assert result["region_id"].tolist() == ["ZONE_A", "ZONE_B"]
    assert pd.api.types.is_integer_dtype(result["year"])
    assert result["year"].tolist() == [2025, 2026]
    assert pd.api.types.is_float_dtype(result["mwh"])
    assert result["mwh"].iloc[0] == pytest.approx(1234.0, rel=0, abs=1e-9)
    assert result["mwh"].iloc[1] == pytest.approx(5678.9, rel=0, abs=1e-9)


def test_resolve_from_repo_csv(tmp_path, monkeypatch):
    base = tmp_path / "inputs"
    csv_dir = base / "electricity" / "load_forecasts"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "load_forecasts.csv"
    frame = pd.DataFrame(
        {
            "region_id": ["zone_a", "zone_b"],
            "state_or_province": ["MA", "RI"],
            "scenario_name": ["Reference", "Reference"],
            "year": [2025, 2026],
            "load_gwh": [1.5, 2.75],
        }
    )
    frame.to_csv(csv_path, index=False)

    import engine.inputs.demand_source as demand_source

    monkeypatch.setattr(demand_source, "input_root", lambda: base)

    result = resolve_demand_frame({})

    assert list(result.columns) == ["region_id", "year", "mwh"]
    assert result["region_id"].tolist() == ["ZONE_A", "ZONE_B"]
    assert result["year"].tolist() == [2025, 2026]
    assert result["mwh"].iloc[0] == pytest.approx(1500.0, rel=0, abs=1e-9)
    assert result["mwh"].iloc[1] == pytest.approx(2750.0, rel=0, abs=1e-9)


def test_run_policy_simulation_skips_resolve_when_demand_present(monkeypatch):
    existing = pd.DataFrame({
        "region_id": ["ZONE_A"],
        "year": [2025],
        "mwh": [123.4],
    })

    monkeypatch.setattr(
        orchestrate,
        "resolve_demand_frame",
        lambda cfg: (_ for _ in ()).throw(AssertionError("resolve_demand_frame should not be called")),
    )

    fake_bundle = SimpleNamespace(
        frames={},
        vectors=SimpleNamespace(as_price_schedule=lambda _: {}),
    )
    monkeypatch.setattr(orchestrate, "_build_bundle", lambda inputs, **_: fake_bundle)
    monkeypatch.setattr(orchestrate, "_validate_inputs", lambda *_, **__: None)

    sentinel = object()
    monkeypatch.setattr(orchestrate, "run_end_to_end", lambda *_, **__: sentinel)

    result = orchestrate.run_policy_simulation({"years": [2025]}, {"demand": existing})

    assert result is sentinel
