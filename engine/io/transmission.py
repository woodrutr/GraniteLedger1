"""Transmission interface loaders and validators."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from engine.normalization import canonical_region_value
from engine.settings import input_root

REQUIRED = {
    "interface_id",
    "from_region",
    "to_region",
    "capacity_mw",
    "reverse_capacity_mw",
    "efficiency",
}

OPTIONAL_DEFAULTS: dict[str, object] = {
    "added_cost_per_mwh": 0.0,
    "contracted_flow_mw_forward": 0.0,
    "contracted_flow_mw_reverse": 0.0,
    "notes": None,
    "interface_type": None,
    "profile_id": None,
    "in_service_year": pd.NA,
}


def _iter_default_roots() -> Iterable[Path]:
    base = Path(input_root())
    for ancestor in (base, *base.parents):
        yield ancestor

    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        yield ancestor


def _resolve_csv_path(path: str | Path | None) -> Path:
    candidates: list[Path] = []

    if path is not None:
        supplied = Path(path).expanduser()
        if supplied.is_file():
            return supplied
        candidates.extend(
            [
                supplied / "ei_edges.csv",
                supplied / "engine" / "transmission" / "ei_edges.csv",
                supplied / "transmission" / "ei_edges.csv",
            ]
        )
    else:
        for root in _iter_default_roots():
            candidates.append(root / "engine" / "transmission" / "ei_edges.csv")
            candidates.append(root / "transmission" / "ei_edges.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if candidates:
        return candidates[0]

    raise FileNotFoundError("No path candidates found for ei_edges.csv")


def load_ei_edges(path: str | Path | None = None) -> pd.DataFrame:
    """Return transmission interface definitions from *path*.

    The loader normalizes region identifiers, removes self-loops, and ensures
    that each directed edge appears once.  Missing numerical values are treated
    as an error; use explicit zeros to indicate disabled interfaces.
    """

    csv_path = _resolve_csv_path(path)
    df = pd.read_csv(csv_path)

    rename_map = {
        "adder_cost_per_MWh": "added_cost_per_mwh",
        "interfaceID": "interface_id",
        "type": "interface_type",
        "contracted_flow_mw_from_to": "contracted_flow_mw_forward",
        "contracted_flow_mw_to_from": "contracted_flow_mw_reverse",
    }
    available = set(df.columns)
    df = df.rename(columns={key: value for key, value in rename_map.items() if key in available})

    missing = REQUIRED - set(df.columns)
    if missing:
        legacy_required = {"from_zone", "to_zone", "limit_mw"}
        if legacy_required <= set(df.columns):
            df = _upgrade_legacy_schema(df)
        else:
            raise ValueError(f"ei_edges.csv missing columns: {sorted(missing)}")

    df = df.copy()

    for column, default in OPTIONAL_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default

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

    if df[["capacity_mw", "reverse_capacity_mw"]].isna().any().any():
        raise ValueError("ei_edges.csv contains null capacity values; use explicit zeros")

    df["efficiency"] = df["efficiency"].fillna(1.0)
    if (df["efficiency"] <= 0.0).any():
        raise ValueError("ei_edges.csv has non-positive efficiency values")

    df["added_cost_per_mwh"] = df["added_cost_per_mwh"].fillna(0.0)
    df["contracted_flow_mw_forward"] = df["contracted_flow_mw_forward"].fillna(0.0)
    df["contracted_flow_mw_reverse"] = df["contracted_flow_mw_reverse"].fillna(0.0)

    invalid_contract = df["contracted_flow_mw_forward"] > df["capacity_mw"].clip(lower=0.0)
    if invalid_contract.any():
        sample = df.loc[invalid_contract, ["interface_id", "contracted_flow_mw_forward", "capacity_mw"]].head(3)
        raise ValueError(
            "contracted forward flow exceeds declared capacity: "
            + sample.to_dict("records").__repr__()
        )

    invalid_reverse_contract = df["contracted_flow_mw_reverse"] > df["reverse_capacity_mw"].clip(lower=0.0)
    if invalid_reverse_contract.any():
        sample = df.loc[
            invalid_reverse_contract,
            ["interface_id", "contracted_flow_mw_reverse", "reverse_capacity_mw"],
        ].head(3)
        raise ValueError(
            "contracted reverse flow exceeds declared reverse capacity: "
            + sample.to_dict("records").__repr__()
        )
    df["from_region"] = df["from_region"].map(canonical_region_value)
    df["to_region"] = df["to_region"].map(canonical_region_value)
    df = df[df["from_region"] != df["to_region"]]

    # Ensure a single declaration per directed edge, preferring the last entry.
    df = df.drop_duplicates(["from_region", "to_region"], keep="last")

    if "in_service_year" in df.columns:
        df["in_service_year"] = pd.to_numeric(df["in_service_year"], errors="coerce").astype("Int64")

    if "limit_mw" not in df.columns:
        df["limit_mw"] = df["capacity_mw"]
    if "reverse_limit_mw" not in df.columns:
        df["reverse_limit_mw"] = df["reverse_capacity_mw"]
    df["contracted_flow_mw"] = df["contracted_flow_mw_forward"]

    selected_columns = [
        "interface_id",
        "from_region",
        "to_region",
        "capacity_mw",
        "reverse_capacity_mw",
        "efficiency",
        "added_cost_per_mwh",
        "contracted_flow_mw_forward",
        "contracted_flow_mw_reverse",
        "interface_type",
        "notes",
        "in_service_year",
        "profile_id",
        "limit_mw",
        "reverse_limit_mw",
        "contracted_flow_mw",
    ]

    present = [column for column in selected_columns if column in df.columns]
    return df[present]


def validate_exported_interface_caps(
    export_df: pd.DataFrame, edges_df: pd.DataFrame
) -> pd.DataFrame:
    """Validate exported transmission capacities against declared edges."""

    key = ["from_region", "to_region"]
    truth_columns = key + ["capacity_mw"]

    merged = export_df.merge(
        edges_df[truth_columns],
        on=key,
        how="left",
        suffixes=("_export", "_truth"),
    )

    missing = merged["capacity_mw_truth"].isna()
    if missing.any():
        bad = merged.loc[missing, key].head(10).to_dict("records")
        raise ValueError(
            f"Export references non-listed interfaces: {bad}"
        )

    merged["capacity_mw_export"] = merged["capacity_mw_truth"]
    return merged


def _upgrade_legacy_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Translate legacy phase-1 schema to the modern interface layout."""

    working = df.rename(
        columns={
            "from_zone": "from_region",
            "to_zone": "to_region",
            "limit_mw": "capacity_mw",
        }
    ).copy()

    if "interface_id" not in working.columns:
        working["interface_id"] = (
            "legacy:" + working["from_region"].astype(str) + "->" + working["to_region"].astype(str)
        )

    working["capacity_mw"] = pd.to_numeric(working["capacity_mw"], errors="coerce")
    reverse_source = working.get(
        "reverse_limit_mw", pd.Series(0.0, index=working.index)
    )
    working["reverse_capacity_mw"] = (
        pd.to_numeric(reverse_source, errors="coerce").fillna(0.0)
    )

    loss = pd.to_numeric(working.get("loss_pct", 0.0), errors="coerce").fillna(0.0)
    efficiency = 1.0 - loss
    efficiency = efficiency.clip(lower=0.0, upper=1.0)
    working["efficiency"] = efficiency

    if "added_cost_per_mwh" not in working.columns:
        working["added_cost_per_mwh"] = 0.0
    if "contracted_flow_mw_forward" not in working.columns:
        working["contracted_flow_mw_forward"] = 0.0
    if "contracted_flow_mw_reverse" not in working.columns:
        working["contracted_flow_mw_reverse"] = 0.0
    if "interface_type" not in working.columns:
        working["interface_type"] = None
    if "notes" not in working.columns:
        working["notes"] = None
    if "profile_id" not in working.columns:
        working["profile_id"] = None
    if "in_service_year" not in working.columns:
        working["in_service_year"] = pd.NA

    return working


__all__ = ["load_ei_edges", "validate_exported_interface_caps"]
