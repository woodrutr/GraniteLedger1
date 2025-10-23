from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from engine.settings import input_root
from common.schemas.load_forecast import parse_load_forecast_csv

__all__ = ["repo_load_forecasts"]


def _default_repo_csv(root: Path) -> Path:
    # canonical path first, then tolerant legacy fallback
    search_dirs = [
        root,
        root / "electricity" / "load_forecasts",
        root / "input" / "electricity" / "load_forecasts",
        root / "inputs" / "electricity" / "load_forecasts",
    ]

    for directory in search_dirs:
        try:
            if directory.is_file():
                return directory.resolve()
        except OSError:
            continue

        csv_path = directory / "load_forecasts.csv"
        if csv_path.exists():
            return csv_path.resolve()

    default_candidate = root / "electricity" / "load_forecasts" / "load_forecasts.csv"
    try:
        default_resolved = default_candidate.resolve()
    except FileNotFoundError:
        default_resolved = default_candidate
    raise FileNotFoundError(default_resolved)


@lru_cache(maxsize=1)
def repo_load_forecasts(base_path: str | Path | None = None) -> pd.DataFrame:
    root = Path(base_path) if base_path else input_root()
    csv_path = _default_repo_csv(root)
    frame = parse_load_forecast_csv(csv_path)
    # Provide both units for GUI convenience
    out = frame.copy()
    out["load_mwh"] = (out["load_gwh"] * 1000.0).astype(float)
    # Canonical order for downstream code that expects these keys
    return out.loc[:, ["region_id", "state_or_province", "scenario_name", "scenario", "year", "load_gwh", "load_mwh"]]
