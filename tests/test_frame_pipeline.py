from __future__ import annotations

import types

import pandas as pd

from engine.frames import FramePipeline
from engine.frames.vector import VectorColumn
from engine.orchestrate import _build_bundle
from engine.run_loop import run_end_to_end


def _basic_inputs() -> dict[str, pd.DataFrame]:
    load = pd.DataFrame(
        [
            {
                "iso": "ISO-NE",
                "zone": "ISO-NE_CT",
                "scenario": "Ref",
                "year": 2024,
                "load_gwh": 0.1,
            },
            {
                "iso": "ISO-NE",
                "zone": "ISO-NE_CT",
                "scenario": "Ref",
                "year": 2025,
                "load_gwh": 0.11,
            },
        ]
    )
    demand = pd.DataFrame(
        [
            {"year": 2024, "region": "ISO-NE_CT", "demand_mwh": 100.0},
            {"year": 2025, "region": "ISO-NE_CT", "demand_mwh": 110.0},
        ]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "sys_1",
                "unique_id": "sys_1",
                "region": "ISO-NE_CT",
                "fuel": "gas",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 9.0,
                "vom_per_mwh": 0.0,
                "fuel_price_per_mmbtu": 3.0,
                "ef_ton_per_mwh": 0.5,
            }
        ]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "gas", "covered": True, "co2_short_ton_per_mwh": 0.1},
        ]
    )
    return {"load": load, "demand": demand, "units": units, "fuels": fuels}


def test_frame_pipeline_registers_carbon_schedule():
    pipeline = FramePipeline()
    inputs = _basic_inputs()

    bundle = pipeline.build_bundle(
        inputs,
        years=[2024, 2025],
        carbon_price_schedule={2025: 15.0},
        carbon_price_value=5.0,
    )

    schedule = bundle.vectors.as_price_schedule("carbon_price")
    assert schedule is not None
    assert schedule[2024] == 5.0
    assert schedule[2025] == 15.0


def test_build_bundle_merges_metadata():
    inputs = _basic_inputs()
    inputs["region_weights"] = {"system": 1.0}

    bundle = _build_bundle(
        inputs,
        policy_enabled=True,
        banking_enabled=True,
        years=[2024, 2025],
        carbon_price_schedule=None,
        carbon_price_value=0.0,
    )

    assert bundle.meta is not None
    assert bundle.meta["region_weights"] == {"system": 1.0}


def test_run_end_to_end_prefers_bundle_schedule(monkeypatch):
    inputs = _basic_inputs()
    pipeline = FramePipeline()
    bundle = pipeline.build_bundle(inputs, years=[2024], carbon_price_schedule=None, carbon_price_value=10.0)

    def fake_run(frames, **kwargs):
        fake_run.called = types.SimpleNamespace(**kwargs)
        return "ok"

    monkeypatch.setattr("engine.run_loop.run_end_to_end_from_frames", fake_run)

    bundle.vectors.register(VectorColumn(name="carbon_price", values={2024: 99.0}))

    result = run_end_to_end(bundle, years=None, carbon_price_schedule=None)

    assert result == "ok"
    assert getattr(fake_run, "called").carbon_price_schedule == {2024: 99.0}


def test_bundle_preserves_core_frames():
    pipeline = FramePipeline()
    inputs = _basic_inputs()

    bundle = pipeline.build_bundle(inputs, years=[2024, 2025], carbon_price_schedule=None, carbon_price_value=0.0)

    demand = bundle.frames.demand()
    assert set(demand["year"]) == {2024, 2025}
    assert set(demand["region"]) == {"ISO-NE_CT"}

    units = bundle.frames.units()
    assert set(units["unit_id"]) == {"sys_1"}

    load = bundle.frames.optional_frame("load")
    assert load is not None
    assert set(load["scenario"].unique()) == {"Ref"}
