from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from common.schemas.load_forecast import parse_load_forecast_csv
from engine.normalization import normalize_region_id
from engine.settings import input_root

__all__ = ["resolve_demand_frame"]


def _default_load_csv_path(root: Path) -> Path:
    candidates = [
        root / "electricity" / "load_forecasts" / "load_forecasts.csv",
        root / "load_forecasts.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    # keep canonical in error to guide users
    raise FileNotFoundError((root / "electricity" / "load_forecasts" / "load_forecasts.csv"))


def _coerce_gui_table(table: pd.DataFrame) -> pd.DataFrame:
    # Accepts GUI table with columns: region, year, demand_mwh (Step 2)
    columns = {str(column).strip().lower(): column for column in table.columns}
    region = columns.get("region") or columns.get("region_id")
    year = columns.get("year")
    demand = columns.get("demand_mwh") or columns.get("load_mwh")
    if not (region and year and demand):
        raise ValueError("GUI demand_table must include region/year/demand_mwh.")
    rid = table[region].astype("string").str.strip()

    def _normalize(value: Any) -> Any:
        if value is pd.NA or value is None:
            return value
        try:
            return normalize_region_id(value)
        except ValueError:
            text = str(value).strip()
            return text.upper() if text else pd.NA

    rid = rid.map(_normalize)
    df = pd.DataFrame(
        {
            "region_id": rid.astype("string"),
            "year": pd.to_numeric(table[year], errors="coerce").astype("Int64"),
            "mwh": pd.to_numeric(table[demand], errors="coerce"),
        }
    ).dropna(subset=["region_id", "year", "mwh"])
    df["year"] = df["year"].astype(int)
    df["mwh"] = df["mwh"].astype(float)
    return df.loc[:, ["region_id", "year", "mwh"]]


def _from_repo_csv(base_path: str | None) -> pd.DataFrame:
    root = Path(base_path) if base_path else input_root()
    csv_path = _default_load_csv_path(root)
    frame = parse_load_forecast_csv(csv_path)
    out = frame.loc[:, ["region_id", "year", "load_gwh"]].copy()
    out["mwh"] = (pd.to_numeric(out["load_gwh"], errors="coerce") * 1000.0).astype(float)
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    return out.loc[:, ["region_id", "year", "mwh"]]


def resolve_demand_frame(cfg: Mapping[str, Any]) -> pd.DataFrame:
    table = cfg.get("demand_table")
    if isinstance(table, pd.DataFrame) and not table.empty:
        return _coerce_gui_table(table)
    # fallback to repo CSV
    base_path = cfg.get("input_base_path")  # optional override
    return _from_repo_csv(base_path)
