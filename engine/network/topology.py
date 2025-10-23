"""Network topology helpers for transmission modeling."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def build_transmission_caps(
    edges: pd.DataFrame, zones: list[str]
) -> tuple[pd.DataFrame, Dict[tuple[str, str], float]]:
    """Return dense capacity matrix and sparse map for declared edges."""

    caps: Dict[Tuple[str, str], float] = {
        (row.from_region, row.to_region): float(row.capacity_mw)
        for row in edges.itertuples(index=False)
    }

    mat = pd.DataFrame(0.0, index=zones, columns=zones)
    for (a, b), cap in caps.items():
        if a in mat.index and b in mat.columns:
            mat.at[a, b] = cap
    for zone in zones:
        mat.at[zone, zone] = 0.0
    return mat, caps


__all__ = ["build_transmission_caps"]
