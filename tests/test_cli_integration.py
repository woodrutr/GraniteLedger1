from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd


def _install_region_stubs() -> None:
    if "regions" in sys.modules:
        return

    regions_pkg = types.ModuleType("regions")
    regions_pkg.__path__ = []  # type: ignore[attr-defined]
    regions_pkg.REGION_MAP = {"SYSTEM": "System"}  # type: ignore[attr-defined]
    sys.modules["regions"] = regions_pkg

    registry_mod = types.ModuleType("regions.registry")
    registry_mod.REGIONS = {  # type: ignore[attr-defined]
        "ISO-NE_CT": "Connecticut",
    }
    registry_mod.STATE_INDEX = {}  # type: ignore[attr-defined]
    registry_mod.validate_on_import = lambda: None  # type: ignore[attr-defined]
    sys.modules["regions.registry"] = registry_mod


_install_region_stubs()


def _install_gui_stubs() -> None:
    if "gui.region_metadata" in sys.modules:
        return

    region_metadata_mod = types.ModuleType("gui.region_metadata")

    def _canonical_region_value(value: str) -> str:
        return str(value)

    region_metadata_mod.canonical_region_value = _canonical_region_value  # type: ignore[attr-defined]
    region_metadata_mod.canonical_region_label = lambda value: str(value)  # type: ignore[attr-defined]
    region_metadata_mod.region_metadata = {}  # type: ignore[attr-defined]
    region_metadata_mod.DEFAULT_REGION_METADATA = {}  # type: ignore[attr-defined]

    def _region_alias_map() -> dict[str, str]:
        return {}

    region_metadata_mod.region_alias_map = _region_alias_map  # type: ignore[attr-defined]
    sys.modules["gui.region_metadata"] = region_metadata_mod

    if "gui.app" not in sys.modules:
        app_mod = types.ModuleType("gui.app")

        def _not_implemented(*_args, **_kwargs):  # pragma: no cover - stub
            raise RuntimeError("gui.app stub should be patched in tests")

        app_mod._build_default_frames = _not_implemented  # type: ignore[attr-defined]
        app_mod._build_policy_frame = _not_implemented  # type: ignore[attr-defined]
        app_mod._ensure_years_in_demand = lambda frames, years: frames  # type: ignore[attr-defined]
        app_mod._load_config_data = lambda source=None: {}  # type: ignore[attr-defined]
        app_mod._years_from_config = lambda config: []  # type: ignore[attr-defined]
        app_mod._cached_forecast_frame = _not_implemented  # type: ignore[attr-defined]
        app_mod._resolve_forecast_base_path = lambda: "root"  # type: ignore[attr-defined]
        sys.modules["gui.app"] = app_mod


_install_gui_stubs()

from cli import run as cli_run
from io_loader import Frames
from engine.outputs import EngineOutputs
from gui.demand_helpers import _ScenarioSelection


def _base_config() -> dict[str, object]:
    return {
        "allowance_market": {},
        "modules": {"demand": {"enabled": True, "load_forecasts": {}}},
    }


def test_run_engine_applies_forecast_bundles(monkeypatch):
    years = [2030]
    base_demand = pd.DataFrame({"year": years, "region": ["ISO-NE_CT"], "demand_mwh": [100.0]})

    def fake_build_default_frames(*args, **kwargs):
        return Frames({"demand": base_demand.copy()})

    def fake_ensure_years(frames, _years):
        return frames

    def fake_policy_frame(*args, **kwargs):
        return pd.DataFrame({"year": years, "policy": ["noop"]})

    def fake_run_end_to_end(frames, **kwargs):
        return EngineOutputs.empty()

    selection = _ScenarioSelection(iso="test", scenario="scenario", zones=["ISO-NE_CT"], years=years)

    monkeypatch.setattr(cli_run, "_build_default_frames", fake_build_default_frames)
    monkeypatch.setattr(cli_run, "_ensure_years_in_demand", fake_ensure_years)
    monkeypatch.setattr(cli_run, "_build_policy_frame", fake_policy_frame)
    monkeypatch.setattr(cli_run, "run_end_to_end_from_frames", fake_run_end_to_end)
    monkeypatch.setattr(
        cli_run,
        "_forecast_manifests_from_config",
        lambda _config: ([selection], "/tmp/test_bundle"),
    )

    frames, outputs, bundles = cli_run._run_engine(_base_config(), years)

    assert isinstance(outputs, EngineOutputs)
    assert bundles == [
        {
            "iso": "test",
            "scenario": "scenario",
            "zones": ["ISO-NE_CT"],
            "years": years,
            "manifest": "test::scenario",
            "path": str(Path("/tmp/test_bundle") / "test" / "scenario"),
        }
    ]
    demand = frames.demand()
    assert set(demand["region"]) == {"ISO-NE_CT"}


def test_cli_outputs_and_documentation_align(tmp_path):
    years = [2030, 2031]
    run_id = "test_run"
    out_dir = tmp_path / "out"
    outputs = EngineOutputs.empty()
    demand_frame = pd.DataFrame({"year": years, "region": ["A", "B"], "demand_mwh": [1.0, 2.0]})
    frames = Frames({"demand": demand_frame})

    manifest_entry = {
        "iso": "nyiso",
        "scenario": "baseline",
        "manifest": "nyiso::baseline",
        "zones": ["A"],
        "path": str(tmp_path / "bundle"),
        "years": years,
    }

    run_directory = cli_run._write_outputs(outputs, out_dir, run_id)
    documentation_paths = cli_run._write_documentation(
        run_id=run_id,
        run_directory=run_directory,
        config_path=None,
        config=_base_config(),
        years=years,
        use_network=False,
        frames=frames,
        outputs=outputs,
        forecast_manifests=[manifest_entry],
    )

    expected_files = {
        "allowance.csv",
        "emissions.csv",
        "prices.csv",
        "flows.csv",
        "run_manifest.json",
        "run_manifest.md",
        "model_documentation.md",
    }

    produced_files = {path.name for path in run_directory.iterdir()}
    assert expected_files.issubset(produced_files)

    manifest_json = documentation_paths["manifest_json"]
    data = json.loads(manifest_json.read_text())
    assert data["run_id"] == run_id
    assert data["forecast_bundles"][0]["manifest"] == "nyiso::baseline"
    assert data["forecast_bundles"][0]["path"] == manifest_entry["path"]
