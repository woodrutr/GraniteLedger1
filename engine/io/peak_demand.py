"""Utilities for loading peak demand datasets keyed to canonical regions."""

from __future__ import annotations

from pathlib import Path
from typing import List

import logging

import pandas as pd

from engine.normalization import normalize_region_id

LOGGER = logging.getLogger(__name__)

_COLUMN_ALIASES = {
    "year": "year",
    "years": "year",
    "region_id": "region",
    "region": "region",
    "zone": "region",
    "scenario": "scenario",
    "scenarios": "scenario",
    "peak_demand_mw": "peak_demand_mw",
    "peak_mw": "peak_demand_mw",
    "peak_demand": "peak_demand_mw",
}

_REQUIRED_COLUMNS = {"year", "region", "peak_demand_mw"}


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with canonical column labels and required data only."""

    renamed: dict[str, str] = {}
    for column in frame.columns:
        key = str(column).strip()
        alias = _COLUMN_ALIASES.get(key.lower())
        if alias:
            renamed[key] = alias

    working = frame.rename(columns=renamed)
    missing = _REQUIRED_COLUMNS - set(working.columns)
    if missing:
        raise ValueError(
            "Peak demand CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    optional = "scenario" if "scenario" in working.columns else None
    columns = ["year", "region", "peak_demand_mw"]
    if optional:
        columns.append(optional)

    return working.loc[:, columns]


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(col).strip() for col in frame.columns]
    frame = _standardize_columns(frame)

    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame.dropna(subset=["year", "region", "peak_demand_mw"])
    if frame.empty:
        return pd.DataFrame(columns=frame.columns)

    frame["year"] = frame["year"].astype(int)
    frame["peak_demand_mw"] = (
        pd.to_numeric(frame["peak_demand_mw"], errors="coerce").fillna(0.0)
    )
    frame["region"] = frame["region"].map(lambda value: normalize_region_id(str(value)))
    if "scenario" in frame.columns:
        frame["scenario"] = frame["scenario"].astype(str)
    else:
        frame["scenario"] = ""

    return frame


def _candidate_paths(root: Path) -> List[Path]:
    candidates: List[Path] = []

    if root.is_file():
        candidates.append(root)
    elif root.is_dir():
        candidates.extend(sorted(p for p in root.glob("*.csv") if p.is_file()))

    sibling = root.with_suffix(".csv")
    if sibling != root and sibling.is_file():
        candidates.append(sibling)

    default_csv = root / "peak_demand.csv"
    if default_csv.is_file():
        candidates.append(default_csv)

    sibling_dir = root.parent / "peak_demand"
    if sibling_dir.is_dir():
        candidates.extend(sorted(p for p in sibling_dir.glob("*.csv") if p.is_file()))

    sibling_csv = root.parent / "peak_demand.csv"
    if sibling_csv.is_file():
        candidates.append(sibling_csv)

    seen: dict[str, Path] = {}
    ordered: List[Path] = []
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen[key] = path
            ordered.append(path)
    return ordered


def load_peak_demand(root: Path | str, scenario: str | None = None) -> pd.DataFrame:
    """Return peak demand observations discovered relative to ``root``.

    Parameters
    ----------
    root:
        Directory containing the load forecast bundle (typically the
        ``load_forecasts`` directory).  The loader will look for peak demand
        CSVs alongside this directory (``peak_demand/*.csv`` or
        ``peak_demand.csv``).
    scenario:
        Optional scenario label used to filter the observations.  When provided
        the comparison is case-sensitive to match the scenario identifiers used
        elsewhere in the engine.
    """

    root_path = Path(root)
    frames: List[pd.DataFrame] = []
    for path in _candidate_paths(root_path):
        try:
            frame = _load_csv(path)
        except Exception as exc:
            LOGGER.warning("Skipping peak demand CSV %s: %s", path, exc)
            continue
        if frame.empty:
            continue
        frame["source_file"] = str(path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["year", "region", "peak_demand_mw", "scenario"])

    combined = pd.concat(frames, ignore_index=True)
    if scenario is not None:
        mask = combined["scenario"].astype(str) == str(scenario)
        combined = combined[mask]

    if combined.empty:
        return pd.DataFrame(columns=["year", "region", "peak_demand_mw", "scenario"])

    combined = combined.drop(columns=["source_file"]).sort_values(
        ["year", "region", "scenario"]
    )
    combined = combined.reset_index(drop=True)
    return combined


__all__ = ["load_peak_demand"]
