"""GUI forecast helper tests."""

from pathlib import Path

import pandas as pd
import pytest

from gui import app


def test_regions_available_zones_uses_bundle_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame = pd.DataFrame(columns=["iso", "scenario", "zone"])
    monkeypatch.setattr(app, "_regions_load_forecasts_frame", lambda base_path=None: frame)

    base_dir = tmp_path / "electricity" / "load_forecasts"
    bundle_dir = base_dir / "nyiso" / "GoldBook_2025_Baseline"
    bundle_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_discover(base_path: str | None) -> list[dict[str, object]]:
        captured["base_path"] = base_path
        return [
            {
                "iso": "nyiso",
                "scenario": "Baseline",
                "zones": ["NYISO_A ", "NYISO_B"],
                "path": str(bundle_dir),
            }
        ]

    monkeypatch.setattr(app, "_discover_bundle_records", fake_discover)

    zones = app._regions_available_zones(str(base_dir), "NYISO", "Baseline")

    assert zones == ["NYISO_A", "NYISO_B"]
    assert captured["base_path"] == str(base_dir)
