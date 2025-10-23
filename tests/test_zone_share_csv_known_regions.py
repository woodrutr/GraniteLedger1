"""Contract tests for zone-to-state share CSV data."""

from __future__ import annotations

import csv
from pathlib import Path

from regions.registry import REGIONS

_DATA_DIR = Path(__file__).resolve().parents[1] / "input" / "regions"


def test_zone_share_csv_references_known_regions() -> None:
    csv_path = _DATA_DIR / "zone_to_state_share.csv"
    canonical_ids = set(REGIONS.keys())

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None, "zone_to_state_share.csv must include headers"
        assert "region_id" in reader.fieldnames, "zone_to_state_share.csv must include a 'region_id' column"

        unknown: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            region_id = (row.get("region_id") or "").strip()
            if not region_id:
                continue
            if region_id not in canonical_ids:
                unknown.append(f"{region_id} (line {row_number})")

    assert not unknown, (
        "zone_to_state_share.csv references unknown region_id(s): "
        + ", ".join(sorted(unknown))
    )
