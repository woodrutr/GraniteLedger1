"""Validate region identifiers used in configuration files."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Registry of valid region IDs
from regions.registry import REGIONS, STATE_INDEX  # must import cleanly
from tools.geo.build_zone_state_shares import simple_yaml_load


def _data_root() -> Path:
    return ROOT / "input" / "regions"


def _validate_iso_state_zones(regions: set[str]) -> dict[str, set[str]]:
    """Validate iso_state_zones.yaml region IDs."""
    payload = simple_yaml_load((_data_root() / "iso_state_zones.yaml").read_text())
    invalid: dict[str, set[str]] = {}
    for iso_data in payload.get("isos", {}).values():
        states = iso_data.get("states", {})
        for state, region_ids in states.items():
            missing = {rid for rid in region_ids if rid not in regions}
            if missing:
                invalid.setdefault(str(state), set()).update(missing)
    return invalid


def _validate_zone_to_state_share(regions: set[str]) -> set[str]:
    """Validate zone_to_state_share.csv region IDs."""
    invalid: set[str] = set()
    with (_data_root() / "zone_to_state_share.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            region_id = row.get("region_id", "").strip()
            if region_id and region_id not in regions:
                invalid.add(region_id)
    return invalid


def _iter_state_region_pairs():
    """Iterate region IDs from state_to_regions.json (handles dict/list/str payloads)."""
    data = json.loads((_data_root() / "state_to_regions.json").read_text())
    for state, payload in data.items():
        regions: list[str] = []
        if isinstance(payload, dict):
            raw_regions = payload.get("regions", [])
            if isinstance(raw_regions, list):
                regions.extend(raw_regions)
            weights = payload.get("weights", {})
            if isinstance(weights, dict):
                regions.extend(weights.keys())
        elif isinstance(payload, list):
            regions.extend(payload)
        elif isinstance(payload, str):
            regions.append(payload)
        for region in regions:
            rid = str(region).strip()
            if rid:
                yield state, rid


def _validate_state_to_regions(regions: set[str]) -> dict[str, set[str]]:
    """Validate state_to_regions.json region IDs."""
    invalid: dict[str, set[str]] = {}
    for state, rid in _iter_state_region_pairs():
        if rid not in regions:
            invalid.setdefault(state, set()).add(rid)
    return invalid


def main() -> int:
    regions = set(REGIONS.keys())
    iso_errors = _validate_iso_state_zones(regions)
    share_errors = _validate_zone_to_state_share(regions)
    state_errors = _validate_state_to_regions(regions)

    bad = []
    if iso_errors:
        for state, missing in sorted(iso_errors.items()):
            for rid in sorted(missing):
                bad.append(("iso_state_zones.yaml", state, rid))
    if share_errors:
        for rid in sorted(share_errors):
            bad.append(("zone_to_state_share.csv", "-", rid))
    if state_errors:
        for state, missing in sorted(state_errors.items()):
            for rid in sorted(missing):
                bad.append(("state_to_regions.json", state, rid))

    if bad:
        for src, st, rid in bad:
            print(f"{src}: {st} -> unknown region_id {rid}")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
