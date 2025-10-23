from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from engine.io.region_mapping import load_state_zone_maps


def test_load_state_zone_maps_builds_expected_mappings(tmp_path: Path) -> None:
    csv_path = tmp_path / "regions_states.csv"
    data = pd.DataFrame(
        {
            "state": ["ma", "MA", "ct"],
            "region_id": ["ISO-NE_CT", "ISO-NE_SEMA", "ISO-NE_CT"],
        }
    )
    data.to_csv(csv_path, index=False)

    json_path = tmp_path / "stats_zones.json"
    json_path.write_text(json.dumps({"ISO-NE_CT": {"foo": "bar"}}))

    result = load_state_zone_maps(states_csv=str(csv_path), stats_zones_json=str(json_path))

    assert result["state_to_regions"] == {
        "MA": ["ISO-NE_CT", "ISO-NE_SEMA"],
        "CT": ["ISO-NE_CT"],
    }
    assert result["region_to_state"] == {"ISO-NE_SEMA": "MA"}


def test_load_state_zone_maps_handles_missing_files(tmp_path: Path) -> None:
    missing_csv = tmp_path / "missing.csv"
    missing_json = tmp_path / "missing.json"

    result = load_state_zone_maps(states_csv=str(missing_csv), stats_zones_json=str(missing_json))

    assert result == {"state_to_regions": {}, "region_to_state": {}}
