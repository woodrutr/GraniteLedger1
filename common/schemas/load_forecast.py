"""Canonical parser for consolidated load forecast CSV files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

try:  # pragma: no cover - optional dependency during bootstrap
    from engine.normalization import normalize_region_id as _normalize_region_id  # type: ignore
except Exception:  # pragma: no cover - graceful fallback when engine unavailable
    _normalize_region_id = None  # type: ignore[assignment]

__all__ = ["parse_load_forecast_csv"]


_COLUMN_ORDER: Tuple[str, ...] = (
    "region_id",
    "state_or_province",
    "scenario_name",
    "scenario",
    "year",
    "load_gwh",
)

_STRING_DTYPES: dict[str, pd.api.extensions.ExtensionDtype | str] = {
    "region_id": pd.StringDtype(),
    "state_or_province": pd.StringDtype(),
    "scenario_name": pd.StringDtype(),
    "scenario": pd.StringDtype(),
}

_NUMERIC_DTYPES: dict[str, str] = {
    "year": "int64",
    "load_gwh": "float64",
}

_REGION_ALIASES: Tuple[str, ...] = ("region_id", "region", "iso_zone")
_STATE_ALIASES: Tuple[str, ...] = ("state_or_province", "state")
_SCENARIO_ALIASES: Tuple[str, ...] = ("scenario_name", "scenario")
_YEAR_ALIASES: Tuple[str, ...] = ("year", "timestamp")
_LOAD_GWH_ALIASES: Tuple[str, ...] = ("load_gwh", "demand_gwh", "load", "demand")
_LOAD_MWH_ALIASES: Tuple[str, ...] = ("load_mwh", "demand_mwh")

_SCENARIO_TOKEN_PATTERN = re.compile(r"[^0-9a-z]+")
_REGION_TOKEN_PATTERN = re.compile(r"[-\s]+")


def _empty_frame() -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for column, dtype in _STRING_DTYPES.items():
        data[column] = pd.Series([], dtype=dtype)
    for column, dtype in _NUMERIC_DTYPES.items():
        data[column] = pd.Series([], dtype=dtype)
    return pd.DataFrame(data, columns=_COLUMN_ORDER)


def _pick_column(frame: pd.DataFrame, options: Iterable[str]) -> tuple[pd.Series | None, str | None]:
    for option in options:
        normalized = option.strip().lower()
        if normalized in frame.columns:
            return frame[normalized].copy(), normalized
    return None, None


def _normalize_region(value: object) -> object:
    if value is pd.NA or value is None:
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    if _normalize_region_id is not None:
        try:
            normalized = _normalize_region_id(text)
            if normalized:
                return normalized
        except Exception:  # pragma: no cover - fallback to local normalisation
            pass
    collapsed = _REGION_TOKEN_PATTERN.sub("_", text)
    collapsed = re.sub(r"__+", "_", collapsed)
    collapsed = collapsed.replace("/", "_")
    token = collapsed.replace("__", "_").strip("_")
    if not token:
        return pd.NA
    return token.upper()


def _scenario_token(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "default"
    token = _SCENARIO_TOKEN_PATTERN.sub("_", text).strip("_")
    return token or "default"


def parse_load_forecast_csv(path: str | Path) -> pd.DataFrame:
    """Read ``path`` and return a canonical load forecast DataFrame."""

    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    if frame.empty:
        return _empty_frame()

    working = pd.DataFrame(index=frame.index)

    region_series, _ = _pick_column(frame, _REGION_ALIASES)
    if region_series is None:
        raise ValueError("Load forecast CSV missing required region column.")
    working["region_id"] = region_series

    state_series, _ = _pick_column(frame, _STATE_ALIASES)
    if state_series is not None:
        working["state_or_province"] = state_series

    scenario_series, _ = _pick_column(frame, _SCENARIO_ALIASES)
    if scenario_series is not None:
        working["scenario_name"] = scenario_series

    year_series, year_alias = _pick_column(frame, _YEAR_ALIASES)
    if year_series is None:
        raise ValueError("Load forecast CSV missing required year column.")
    working["year"] = year_series

    load_gwh_series, _ = _pick_column(frame, _LOAD_GWH_ALIASES)
    load_mwh_series, _ = _pick_column(frame, _LOAD_MWH_ALIASES)
    if load_gwh_series is not None:
        working["load_gwh"] = pd.to_numeric(load_gwh_series, errors="coerce")
    elif load_mwh_series is not None:
        working["load_gwh"] = pd.to_numeric(load_mwh_series, errors="coerce") / 1000.0
    else:
        raise ValueError("Load forecast CSV missing required demand column.")

    if year_alias == "timestamp":
        working["year"] = pd.to_datetime(working["year"], errors="coerce").dt.year
    else:
        working["year"] = pd.to_numeric(working["year"], errors="coerce")

    # Normalize and coerce textual columns
    region_normalized = working["region_id"].astype("string").map(_normalize_region)
    working["region_id"] = region_normalized.astype("string")

    if "state_or_province" in working.columns:
        state_series = working["state_or_province"].astype("string").str.strip()
        state_series = state_series.where(state_series != "", pd.NA)
    else:
        state_series = pd.Series(pd.NA, index=working.index, dtype=pd.StringDtype())
    working["state_or_province"] = state_series.astype(pd.StringDtype())

    scenario_series = working.get("scenario_name")
    if scenario_series is None:
        scenario_series = pd.Series("DEFAULT", index=working.index, dtype=pd.StringDtype())
    else:
        scenario_series = scenario_series.astype("string").str.strip()
        scenario_series = scenario_series.fillna("DEFAULT")
        scenario_series = scenario_series.replace("", "DEFAULT")
    working["scenario_name"] = scenario_series.astype(pd.StringDtype())

    working["scenario"] = working["scenario_name"].map(_scenario_token).astype(pd.StringDtype())

    working["load_gwh"] = pd.to_numeric(working["load_gwh"], errors="coerce")

    required = ["region_id", "year", "load_gwh"]
    working = working.replace({"region_id": {"": pd.NA}})
    working = working.dropna(subset=required)

    working["year"] = working["year"].astype(int)
    working["load_gwh"] = working["load_gwh"].astype(float)
    working["region_id"] = working["region_id"].astype(pd.StringDtype())
    working["state_or_province"] = working["state_or_province"].astype(pd.StringDtype())
    working["scenario_name"] = working["scenario_name"].astype(pd.StringDtype())
    working["scenario"] = working["scenario"].astype(pd.StringDtype())

    return working.loc[:, _COLUMN_ORDER].reset_index(drop=True)
