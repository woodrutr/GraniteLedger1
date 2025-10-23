from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pandas as pd


@dataclass(slots=True)
class DemandValidationError(Exception):
    message: str
    details: dict[str, object] | None = None

    def __str__(self) -> str:  # pragma: no cover
        return self.message


_REQUIRED_CANON = ("region_id", "year", "mwh")
_GUI_SHAPES = (
    ("region", "year", "demand_mwh"),
    ("region_id", "year", "demand_mwh"),
)


def _coerce_to_canon(df: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in df.columns}
    # GUI shapes
    for a, b, c in _GUI_SHAPES:
        if a in cols and b in cols and c in cols:
            out = pd.DataFrame(
                {
                    "region_id": (
                        df[cols[a]]
                        .astype("string")
                        .str.strip()
                        .str.upper()
                        .str.replace(r"[-\s]+", "_", regex=True)
                    ),
                    "year": pd.to_numeric(df[cols[b]], errors="coerce"),
                    "mwh": pd.to_numeric(df[cols[c]], errors="coerce"),
                }
            )
            return out
    # Canonical engine shape
    if all(k in cols for k in _REQUIRED_CANON):
        out = pd.DataFrame(
            {
                "region_id": (
                    df[cols["region_id"]]
                    .astype("string")
                    .str.strip()
                    .str.upper()
                    .str.replace(r"[-\s]+", "_", regex=True)
                ),
                "year": pd.to_numeric(df[cols["year"]], errors="coerce"),
                "mwh": pd.to_numeric(df[cols["mwh"]], errors="coerce"),
            }
        )
        return out
    raise DemandValidationError("Demand table missing required columns", {"have": list(df.columns)})


def validate_demand_table(
    demand_df: pd.DataFrame,
    ei_units_df: pd.DataFrame | None,
    *,
    required_years: Iterable[int] | None = None,
    aggregate_duplicates: bool = True,
    allow_nonpositive: bool = False,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Returns (normalized_df['region_id','year','mwh'], warnings).
    Raises DemandValidationError for fatal issues.
    """
    if not isinstance(demand_df, pd.DataFrame) or demand_df.empty:
        raise DemandValidationError("Demand table is empty")

    working = _coerce_to_canon(demand_df).dropna(subset=["region_id", "year", "mwh"])
    if working.empty:
        raise DemandValidationError("No valid region/year/mwh rows after coercion")

    # types
    working["year"] = working["year"].astype(int)
    working["mwh"] = working["mwh"].astype(float)
    working["region_id"] = working["region_id"].astype("string")

    warnings: list[str] = []

    # duplicates
    dup_mask = working.duplicated(subset=["region_id", "year"], keep=False)
    if dup_mask.any():
        if aggregate_duplicates:
            working = working.groupby(["region_id", "year"], as_index=False)["mwh"].sum()
            warnings.append("Duplicate (region_id, year) rows were summed.")
        else:
            raise DemandValidationError("Duplicate (region_id, year) rows found")

    # nonpositive
    if not allow_nonpositive:
        bad = working["mwh"] <= 0
        if bad.any():
            raise DemandValidationError("Non-positive demand values found", {"rows": int(bad.sum())})

    # required years coverage
    if required_years:
        have_years = set(working["year"].unique().tolist())
        need = set(int(y) for y in required_years)
        missing = sorted(need - have_years)
        if missing:
            warnings.append(f"Missing years in demand table: {missing}")

    # EI units region coverage
    if isinstance(ei_units_df, pd.DataFrame) and not ei_units_df.empty:
        unit_cols = {str(c).strip().lower(): c for c in ei_units_df.columns}
        units_region_col = unit_cols.get("region_id")
        if units_region_col is not None:
            units_regions = set(
                ei_units_df[units_region_col]
                .astype("string")
                .str.strip()
                .str.upper()
                .str.replace(r"[-\s]+", "_", regex=True)
                .unique()
                .tolist()
            )
            demand_regions = set(working["region_id"].unique().tolist())
            missing_in_units = sorted(demand_regions - units_regions)
            if missing_in_units:
                raise DemandValidationError(
                    "Demand regions not present in EI units",
                    {"missing_in_units": missing_in_units},
                )
        else:
            warnings.append("EI units did not include region_id column; skipped region cross-check.")

    return working.loc[:, ["region_id", "year", "mwh"]], warnings
