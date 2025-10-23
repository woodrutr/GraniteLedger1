from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from engine.manifest import build_manifest, manifest_to_markdown


class StubFrames:
    def __init__(self, demand: pd.DataFrame, transmission: pd.DataFrame, policy: SimpleNamespace):
        self._demand = demand
        self._transmission = transmission
        self._policy = policy

    def demand(self) -> pd.DataFrame:
        return self._demand

    def transmission(self) -> pd.DataFrame:
        return self._transmission

    def policy(self) -> SimpleNamespace:
        return self._policy


def _sample_manifest(tmp_path: Path) -> dict[str, object]:
    directory = tmp_path / "nyiso" / "GoldBook_2025_Baseline"
    directory.mkdir(parents=True)
    csv_path = directory / "NYISO_J.csv"
    csv_path.write_text("year,demand_mwh,region\n2025,1000,NYISO_J\n")

    frame = pd.DataFrame(
        {"year": [2025], "demand_mwh": [1000.0], "region": ["NYISO_J"]}
    )
    return {
        "iso": "nyiso",
        "source": "NYISO Gold Book",
        "vintage": 2025,
        "scenario": "Baseline",
        "manifest": "NYISO Gold Book 2025 – Baseline",
        "zones": ["NYISO_J"],
        "path": str(directory),
        "years": [2025],
        "frames": {"NYISO_J": frame},
        "source_files": {"NYISO_J": csv_path},
    }


def test_manifest_includes_selected_bundle(tmp_path: Path) -> None:
    manifest_entry = _sample_manifest(tmp_path)

    demand = pd.DataFrame(
        {
            "year": [2025, 2026],
            "region": ["NYISO_J", "NYISO_J"],
            "demand_mwh": [1000.0, 1020.0],
        }
    )
    transmission = pd.DataFrame(
        {"from_region": ["NYISO_J"], "to_region": ["NYISO_K"], "limit_mw": [500.0]}
    )
    policy = SimpleNamespace(
        cap=pd.Series({2025: 500_000.0, 2026: 490_000.0}),
        floor=pd.Series({2025: 5.0, 2026: 6.0}),
        ccr1_trigger=pd.Series({2025: 8.0, 2026: 8.5}),
        ccr1_qty=pd.Series({2025: 10.0, 2026: 12.0}),
        ccr2_trigger=pd.Series({2025: 10.0, 2026: 10.5}),
        ccr2_qty=pd.Series({2025: 5.0, 2026: 6.0}),
        banking_enabled=True,
    )
    frames = StubFrames(demand, transmission, policy)

    outputs = SimpleNamespace(
        emissions_total={2025: 123.0},
        annual=pd.DataFrame({"year": [2025], "bank": [50.0]}),
    )

    run_config = {
        "regions": ["NYISO_J"],
        "modules": {
            "electricity_dispatch": {"deep_carbon_pricing": True},
            "carbon_policy": {"bank_enabled": True},
        },
    }

    manifest = build_manifest(
        run_config,
        frames,
        outputs,
        forecast_manifests=[manifest_entry],
        git_commit="abc123",
    )

    assert any(
        entry.get("manifest") == "NYISO Gold Book 2025 – Baseline"
        for entry in manifest["load_forecasts"]
    )

    markdown = manifest_to_markdown(manifest)
    assert "NYISO Gold Book 2025 – Baseline" in markdown

    run_id = "run-xyz"
    output_dir = tmp_path / "output" / run_id
    output_dir.mkdir(parents=True)

    manifest_json = output_dir / "run_manifest.json"
    manifest_md = output_dir / "run_manifest.md"

    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_md.write_text(markdown, encoding="utf-8")

    assert manifest_json.exists()
    assert manifest_md.exists()

    saved = json.loads(manifest_json.read_text())
    assert saved["load_forecasts"][0]["manifest"] == "NYISO Gold Book 2025 – Baseline"
