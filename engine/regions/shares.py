"""Utilities for distributing region-level metrics to state-level shares."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import math

import pandas as pd

from engine.constants import FLOW_TOL
from common.regions_schema import REGION_MAP
from regions.registry import STATE_INDEX

_DATA_ROOT = Path(__file__).resolve().parents[2] / "input" / "regions"


def _data_path(filename: str) -> Path:
    path = _DATA_ROOT / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{filename} could not be found. Expected to find it at '{path}'."
        )
    return path


@lru_cache(maxsize=1)
def load_zone_to_state_share() -> pd.DataFrame:
    """Return zone→state share weights as a DataFrame."""

    path = _data_path("zone_to_state_share.csv")
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("zone_to_state_share.csv contains no share data.")

    expected_columns = {"region_id", "state", "share"}
    if set(df.columns) != expected_columns:
        raise ValueError(
            "zone_to_state_share.csv must contain exactly the columns 'region_id', 'state', and 'share'."
        )

    df["region_id"] = df["region_id"].astype(str)
    df["state"] = df["state"].astype(str).str.upper()
    df["share"] = pd.to_numeric(df["share"], errors="raise")

    if (df["share"] < 0).any():
        raise ValueError("Share values must be non-negative in zone_to_state_share.csv.")

    grouped = df.groupby("region_id")["share"].sum()
    invalid_regions = [
        region
        for region, total in grouped.items()
        if not math.isclose(float(total), 1.0, rel_tol=1e-9, abs_tol=FLOW_TOL)
    ]
    if invalid_regions:
        regions = ", ".join(sorted(invalid_regions))
        raise ValueError(
            "zone_to_state_share.csv must allocate a total share of 1.0 per region; "
            f"invalid regions: {regions}"
        )

    return df.sort_values(["region_id", "state"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def load_state_to_regions() -> dict[str, tuple[str, ...]]:
    """Return mapping of state codes to their configured regions from JSON overrides."""

    path = _DATA_ROOT / "state_to_regions.json"
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration error path
        raise ValueError(
            f"Failed to parse state_to_regions.json at '{path}': {exc}."
        ) from exc

    if not isinstance(raw, dict):
        raise ValueError("state_to_regions.json must contain a JSON object mapping states.")

    result: dict[str, tuple[str, ...]] = {}
    for state_code, payload in raw.items():
        state_key = str(state_code).upper()
        regions: list[str] = []

        if isinstance(payload, dict):
            raw_regions = payload.get("regions", [])
            if isinstance(raw_regions, dict):
                regions = [str(region) for region, enabled in raw_regions.items() if enabled]
            elif isinstance(raw_regions, (list, tuple, set)):
                regions = [str(region) for region in raw_regions]
            elif raw_regions is None:
                regions = []
            else:
                regions = [str(raw_regions)]
        elif isinstance(payload, (list, tuple, set)):
            regions = [str(region) for region in payload]
        elif payload is None:
            regions = []
        else:
            regions = [str(payload)]

        normalized: list[str] = []
        seen: set[str] = set()
        for region in regions:
            cleaned = region.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)

        result[state_key] = tuple(normalized)

    return result


def state_weights_for_region(state: str) -> dict[str, float]:
    """Return normalized weights for the regions that contribute to ``state``."""

    if not state:
        raise ValueError("State code is required to compute weights.")

    state_key = state.upper()
    overrides = load_state_to_regions()

    allowed_regions = set(STATE_INDEX.get(state_key, ()))
    if state_key in overrides:
        allowed_regions.update(overrides[state_key])

    if not allowed_regions:
        raise ValueError(f"No configured regions for state '{state_key}'.")

    share_df = load_zone_to_state_share()
    state_rows = share_df[share_df["state"] == state_key]
    state_rows = state_rows[state_rows["region_id"].isin(allowed_regions)]

    if state_rows.empty:
        raise ValueError(f"No configured shares for state '{state_key}'.")

    grouped = state_rows.groupby("region_id")["share"].sum()
    weights = {region: float(value) for region, value in grouped.items()}

    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"Configured shares for state '{state_key}' must sum to a positive number.")

    normalized = {region: value / total for region, value in sorted(weights.items())}
    return normalized


def expand_state_regions(state: str, min_share: float = 0.005) -> list[str]:
    """Return regions contributing to ``state`` filtered by ``min_share`` threshold."""

    if min_share < 0:
        raise ValueError("min_share must be a non-negative number.")

    weights = state_weights_for_region(state)
    ordered = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    filtered = [region for region, weight in ordered if weight >= min_share]

    if not filtered and ordered:
        filtered = [ordered[0][0]]

    return filtered


def apply_state_shares(df_zone_metric: pd.DataFrame, state: str, value_col: str) -> float:
    """Apply state share weights to ``value_col`` in ``df_zone_metric`` and return total."""

    if value_col not in df_zone_metric.columns:
        raise KeyError(f"Column '{value_col}' was not found in the provided DataFrame.")

    if "region_id" not in df_zone_metric.columns:
        raise KeyError("The DataFrame must include a 'region_id' column to match weights.")

    weights = state_weights_for_region(state)
    region_totals = (
        df_zone_metric[df_zone_metric["region_id"].isin(weights)]
        .groupby("region_id")[value_col]
        .sum()
    )

    if region_totals.empty:
        return 0.0

    total = 0.0
    for region, value in region_totals.items():
        total += float(value) * weights.get(region, 0.0)
    return total


def validate_shares_against_registry() -> None:
    """Ensure configured shares align with the canonical region registry."""

    share_df = load_zone_to_state_share()
    overrides = load_state_to_regions()

    known_regions = set(REGION_MAP.keys())
    state_index_regions = {
        region
        for regions in STATE_INDEX.values()
        for region in regions
    }
    override_regions: set[str] = set()
    for regions in overrides.values():
        override_regions.update(regions)

    recognized_regions = known_regions | state_index_regions | override_regions

    csv_regions = set(share_df["region_id"].astype(str))
    unknown_csv = sorted(csv_regions - recognized_regions)
    if unknown_csv:
        joined = ", ".join(unknown_csv)
        raise ValueError(
            "zone_to_state_share.csv references unknown regions: " + joined
        )

    json_regions = override_regions
    unknown_json = sorted(json_regions - recognized_regions)
    if unknown_json:
        joined = ", ".join(unknown_json)
        raise ValueError(
            "state_to_regions.json references unknown regions: " + joined
        )

    share_states = set(share_df["state"].astype(str).str.upper())
    missing_states = sorted(state for state in STATE_INDEX if state not in share_states)
    if missing_states:
        joined = ", ".join(missing_states)
        raise ValueError(
            "zone_to_state_share.csv is missing share rows for states: " + joined
        )


validate_shares_against_registry()


__all__ = [
    "apply_state_shares",
    "expand_state_regions",
    "load_zone_to_state_share",
    "load_state_to_regions",
    "state_weights_for_region",
]
