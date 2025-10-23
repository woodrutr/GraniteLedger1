from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from engine.io.transmission import load_ei_edges as _io_load_edges

def _data_root() -> Path:
    here = Path(__file__).resolve()
    for anc in (here, *here.parents):
        cand = anc / "input" / "engine"
        if cand.exists():
            return cand
    return (Path.cwd() / "input" / "engine").resolve()


def _resolve_csv_path(base_path: str | Path | None) -> Path:
    if base_path is None:
        roots = (_data_root(),)
    else:
        candidate = Path(base_path).expanduser().resolve()
        if candidate.is_file():
            return candidate
        roots = (
            candidate,
            candidate / "transmission",
        )

    for root in roots:
        csv_path = root / "ei_edges.csv"
        if csv_path.exists():
            return csv_path

    default = _data_root() / "transmission" / "ei_edges.csv"
    return default


def _zones_from_units_csv() -> List[str]:
    """Get zones from ei_units.csv (authoritative source)."""
    root = _data_root()
    # Only check ei_units.csv - it's the authoritative source
    for name in ("ei_units.csv",):
        p = root / "units" / name
        if p.exists():
            try:
                u = pd.read_csv(p, usecols=["zone"])
                return list(sorted(u["zone"].dropna().unique()))
            except Exception:
                pass
    return []

def load_edges(base_path: str | Path | None = None) -> pd.DataFrame:
    """Load transmission edges from ei_edges.csv - the authoritative source.
    
    No fallback or synthetic topology generation. If ei_edges.csv is not found,
    an error is raised.
    """
    csv_path = _resolve_csv_path(base_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Transmission edges file not found: {csv_path}. "
            "ei_edges.csv is required and no fallback topology is generated."
        )
    
    try:
        df = _io_load_edges(csv_path)
    except (FileNotFoundError, ValueError) as exc:
        raise ImportError(f"failed to load ei_edges.csv: {exc}") from exc
    
    return df


def load_ei_transmission(path: str | Path | None = None) -> pd.DataFrame:
    """Load the enhanced EI transmission topology with modern schema."""

    csv_path = Path(path) if path is not None else _data_root() / "transmission" / "ei_transmission.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"ei_transmission.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    rename_map = {
        "adder_cost_per_MWh": "added_cost_per_mwh",
        "type": "interface_type",
        "contracted_flow_mw_from_to": "contracted_flow_mw_forward",
        "contracted_flow_mw_to_from": "contracted_flow_mw_reverse",
    }
    df = df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns})

    if "contracted_flow_mw_forward" not in df.columns and "contracted_flow_mw" in df.columns:
        df["contracted_flow_mw_forward"] = df["contracted_flow_mw"]
    if "contracted_flow_mw_reverse" not in df.columns:
        df["contracted_flow_mw_reverse"] = df.get("uncontracted_flow_mw", 0.0)

    expected_columns = [
        "interface_id",
        "from_region",
        "to_region",
        "capacity_mw",
        "reverse_capacity_mw",
        "efficiency",
        "added_cost_per_mwh",
        "contracted_flow_mw_forward",
        "contracted_flow_mw_reverse",
        "notes",
        "profile_id",
        "in_service_year",
    ]
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise ImportError(f"ei_transmission.csv missing columns: {missing_list}")

    numeric_columns = [
        "capacity_mw",
        "reverse_capacity_mw",
        "efficiency",
        "added_cost_per_mwh",
        "contracted_flow_mw_forward",
        "contracted_flow_mw_reverse",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if column in (
            "capacity_mw",
            "reverse_capacity_mw",
            "contracted_flow_mw_forward",
            "contracted_flow_mw_reverse",
        ):
            if (df[column] < 0).any():
                raise ImportError(f"ei_transmission.csv has negative values in column '{column}'")

    df["efficiency"] = df["efficiency"].fillna(1.0)
    if (df["efficiency"] <= 0).any():
        raise ImportError("ei_transmission.csv has non-positive efficiency values")

    df["added_cost_per_mwh"] = df["added_cost_per_mwh"].fillna(0.0)
    df["contracted_flow_mw_forward"] = df["contracted_flow_mw_forward"].fillna(0.0)
    df["contracted_flow_mw_reverse"] = df["contracted_flow_mw_reverse"].fillna(0.0)

    df["interface_id"] = df["interface_id"].astype(str)
    df["from_region"] = df["from_region"].astype(str)
    df["to_region"] = df["to_region"].astype(str)

    if "limit_mw" not in df.columns:
        df["limit_mw"] = df["capacity_mw"]
    if "reverse_limit_mw" not in df.columns:
        df["reverse_limit_mw"] = df["reverse_capacity_mw"]

    if "notes" in df.columns:
        df["notes"] = df["notes"].astype(str)
    if "profile_id" in df.columns:
        df["profile_id"] = df["profile_id"].astype(str)
    if "interface_type" in df.columns:
        df["interface_type"] = df["interface_type"].astype(str)

    df["in_service_year"] = pd.to_numeric(df["in_service_year"], errors="coerce").astype("Int64")

    ordered_columns = [column for column in expected_columns if column in df.columns]
    extra_columns = [column for column in df.columns if column not in ordered_columns]
    return df[ordered_columns + extra_columns]
