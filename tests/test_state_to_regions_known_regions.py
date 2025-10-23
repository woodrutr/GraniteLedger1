"""Contract tests for state-to-region metadata."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from regions.registry import REGIONS

_DATA_DIR = Path(__file__).resolve().parents[1] / "input" / "regions"


def _format_issue(state: str, issues: list[str]) -> str:
    return f"{state}: {', '.join(issues)}"


def test_state_to_regions_json_references_known_regions() -> None:
    json_path = _DATA_DIR / "state_to_regions.json"
    payload: dict[str, Any] = json.loads(json_path.read_text())

    canonical_ids = set(REGIONS.keys())
    problems: dict[str, list[str]] = {}

    for state, entry in sorted(payload.items()):
        issues: list[str] = []
        if not isinstance(entry, dict):
            issues.append("entry must be an object with 'regions' and 'weights'")
            problems[state] = issues
            continue

        regions = entry.get("regions", [])
        if not isinstance(regions, list):
            issues.append("'regions' must be a list")
            problems[state] = issues
            continue

        region_counter = Counter(regions)
        duplicates = sorted(rid for rid, count in region_counter.items() if count > 1)
        if duplicates:
            issues.append("duplicate region ids: " + ", ".join(duplicates))

        weights = entry.get("weights", {})
        if not isinstance(weights, dict):
            issues.append("'weights' must be a mapping")
            problems[state] = issues
            continue

        unknown_regions = sorted(set(regions) - canonical_ids)
        if unknown_regions:
            issues.append("unknown regions: " + ", ".join(unknown_regions))

        weight_keys = set(weights)
        unknown_weight_keys = sorted(weight_keys - canonical_ids)
        if unknown_weight_keys:
            issues.append("unknown weight regions: " + ", ".join(unknown_weight_keys))

        missing_weights = sorted(set(regions) - weight_keys)
        if missing_weights:
            issues.append("missing weights for: " + ", ".join(missing_weights))

        extra_weights = sorted(weight_keys - set(regions))
        if extra_weights:
            issues.append("weights provided for regions not listed: " + ", ".join(extra_weights))

        non_numeric = sorted(
            region_id for region_id, value in weights.items() if not isinstance(value, (int, float))
        )
        if non_numeric:
            issues.append("non-numeric weights for: " + ", ".join(non_numeric))

        if issues:
            problems[state] = issues

    assert not problems, "state_to_regions.json contains errors: " + "; ".join(
        _format_issue(state, issues) for state, issues in sorted(problems.items())
    )
