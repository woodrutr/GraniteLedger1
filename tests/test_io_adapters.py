"""Tests for load forecast frame normalization helpers."""

from __future__ import annotations

import pandas as pd

from engine.io.adapters import normalize_load_frame


def test_normalize_load_frame_preserves_zone_column() -> None:
    """Zone-only inputs should populate region identifiers without losing zones."""

    raw = pd.DataFrame(
        {
            "zone": ["ISO_A_ZONE1"],
            "scenario": ["Baseline"],
            "year": [2035],
            "load_mwh": [1_500.0],
        }
    )

    normalized = normalize_load_frame(raw)

    assert normalized.columns.tolist()[:6] == [
        "iso",
        "scenario",
        "zone",
        "region_id",
        "year",
        "load_gwh",
    ]
    assert normalized["zone"].tolist() == ["ISO_A_ZONE1"]
    assert normalized["region_id"].tolist() == ["ISO_A_ZONE1"]
    assert normalized.loc[0, "load_gwh"] == 1.5
