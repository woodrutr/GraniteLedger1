"""Test helpers for working with canonical carbon price columns."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Any
import math
import sys

import pandas as pd

from engine.prices.normalize import coerce_price_mapping
from gui.price_adapter import dataframe_to_carbon_vector

# Mapping of legacy alias columns to their canonical counterparts.
ALIAS_TO_CANONICAL: Mapping[str, str] = {
    "p_co2": "cp_last",
    "allowance_price": "cp_all",
    "p_co2_all": "cp_all",
    "p_co2_exc": "cp_exempt",
    "p_co2_eff": "cp_effective",
}

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class CarbonPriceFixture:
    """Convenience container for canonical carbon price components."""

    year: int
    all: float
    effective: float
    exempt: float
    last: float | None = None

    def as_row(self) -> dict[str, float | int | None]:
        """Return a mapping with canonical columns and derived aliases."""
        cp_last = self.last if self.last is not None else self.effective
        row: dict[str, float | int | None] = {
            "year": int(self.year),
            "cp_all": float(self.all),
            "cp_effective": float(self.effective),
            "cp_exempt": float(self.exempt),
            "cp_last": float(cp_last) if cp_last is not None else None,
        }
        for alias, canonical in ALIAS_TO_CANONICAL.items():
            if canonical in row and row[canonical] is not None:
                row[alias] = row[canonical]
        return row


def with_carbon_vector_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with canonical carbon price columns attached.

    The adapter preserves input order while adding ``cp_*`` columns derived from
    any legacy aliases that may still exist for backward compatibility.
    """

    result = dataframe_to_carbon_vector(df)
    # Ensure duplicate columns are removed (adapter may append cp_* even when present)
    result = result.loc[:, ~result.columns.duplicated()]
    required = {"cp_all", "cp_effective", "cp_exempt", "cp_last"}
    missing = required.difference(result.columns)
    if missing:
        raise AssertionError(f"Missing canonical columns: {sorted(missing)}")
    return result


def assert_aliases_match_canonical(df: pd.DataFrame) -> None:
    """Assert that legacy aliases equal their canonical carbon price columns."""

    for alias, canonical in ALIAS_TO_CANONICAL.items():
        if alias in df.columns:
            assert (
                canonical in df.columns
            ), f"Alias column '{alias}' is present without canonical '{canonical}'"
            alias_series = df[alias].astype(float)
            canonical_series = df[canonical].astype(float)
            pd.testing.assert_series_equal(
                alias_series,
                canonical_series,
                check_names=False,
                check_dtype=False,
                rtol=0.0,
                atol=0.0,
            )


def price_vector_from_mapping(record: Mapping[str, Any], *, year: int | None = None):
    """Return a CarbonPriceVector built from ``record`` using legacy fallbacks."""

    return coerce_price_mapping(
        record,
        year=year,
        default_last=record.get("cp_last") or record.get("p_co2"),
    )


def assert_aliases_match_mapping(record: Mapping[str, Any]) -> None:
    """Assert mapping aliases equal canonical price values when present."""

    for alias, canonical in ALIAS_TO_CANONICAL.items():
        if alias in record and canonical in record:
            alias_val = float(record[alias])
            canonical_val = float(record[canonical])
            if not math.isclose(alias_val, canonical_val, rel_tol=0.0, abs_tol=0.0):
                raise AssertionError(
                    f"Alias '{alias}'={alias_val} does not match canonical '{canonical}'={canonical_val}"
                )


def set_aliases_from_canonical(record: MutableMapping[str, Any]) -> None:
    """Populate legacy alias fields from canonical carbon price keys."""

    for alias, canonical in ALIAS_TO_CANONICAL.items():
        if canonical in record:
            record[alias] = record[canonical]
