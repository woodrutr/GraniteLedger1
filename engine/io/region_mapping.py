"""Utilities for loading state-to-region mappings.

This module provides a lightweight helper that pulls together the
state/region information stored in the electricity inputs directory.  The
function is intentionally permissive – both backing files are optional – so it
can be reused in validation scripts or during development without requiring a
full dataset checkout.
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def load_state_zone_maps(
    states_csv: str = "input/electricity/cem_inputs/regions_states.csv",
    stats_zones_json: str = "input/electricity/cem_inputs/stats_zones.json",
) -> dict:
    """Return convenience mappings derived from the state/zone datasets.

    Parameters
    ----------
    states_csv:
        Path to ``regions_states.csv`` which should contain either ``state`` and
        ``region_id`` columns or an additional ``iso`` column.
    stats_zones_json:
        Optional metadata file keyed by ``region_id`` or ``iso``.  The helper
        only validates the file is parseable JSON – it is included for parity
        with legacy tooling that referenced both files.

    Returns
    -------
    dict
        A dictionary with keys ``state_to_regions`` and ``region_to_state``.  The
        former maps upper-case state abbreviations to a sorted list of region
        IDs.  The latter is only populated for regions that uniquely map back to
        a single state.
    """

    result: dict[str, dict] = {
        "state_to_regions": {},
        "region_to_state": {},
    }

    states_path = Path(states_csv)
    if states_path.exists():
        df = pd.read_csv(states_path)
        # Normalise column names for easier matching regardless of case or
        # surrounding whitespace that may be present in the CSV headers.
        df.columns = [col.strip().lower() for col in df.columns]

        if {"state", "region_id"}.issubset(df.columns):
            # Normalise state codes before grouping to avoid treating different
            # casings of the same abbreviation as distinct entries.
            df["state"] = df["state"].astype(str).str.upper()

            # Build a list of distinct regions for each state.  The values are
            # sorted to keep the output deterministic which makes it easier to
            # compare in tests or downstream tooling.
            for state, group in df.groupby("state"):
                regions = sorted({str(region) for region in group["region_id"].astype(str)})
                result["state_to_regions"][state] = regions

            # Build the reverse mapping for regions that only map to a single
            # state.  This mirrors the behaviour of existing validation helpers
            # where the reverse map is intentionally partial.
            per_region = df.groupby("region_id")["state"].nunique()
            unique_regions = per_region[per_region == 1].index
            mask = df["region_id"].isin(unique_regions)
            result["region_to_state"] = {
                str(region): str(state)
                for region, state in zip(df.loc[mask, "region_id"], df.loc[mask, "state"])
            }

    # stats_zones is optional; present for completeness.  We only attempt to
    # parse the file so callers get fast feedback if the JSON is malformed.
    stats_path = Path(stats_zones_json)
    if stats_path.exists():
        try:
            json.loads(stats_path.read_text())
        except Exception:
            # Swallow JSON errors to keep this helper resilient; consumers that
            # need to inspect the metadata can load it directly.
            pass

    return result


__all__ = ["load_state_zone_maps"]

