"""Helpers for loading legacy EI-format unit CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

EXPECTED_COLUMNS: frozenset[str] = frozenset(
    {
        "region_id",
        "heat_rate_mmbtu_per_mwh",
        "capacity_mw",
        "co2_short_ton_per_mwh",
    }
)

OPTIONAL_COLUMNS: frozenset[str] = frozenset(
    {
        "unique_id",
        "plant_name",
        "plant_id",
        "unit_id",
        "generator_id",
        "state",
        "region",
        "technology",
    }
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "heat_rate_mmbtu_per_mwh",
    "capacity_mw",
    "co2_short_ton_per_mwh",
)


def _read_csv(path: Path, *, delimiters: Iterable[str]) -> pd.DataFrame:
    """Return a DataFrame read from ``path`` trying each ``delimiters`` value."""

    errors: list[Exception] = []
    for delimiter in delimiters:
        try:
            kwargs = {"sep": delimiter}
            if delimiter is None:
                kwargs = {"sep": None, "engine": "python"}
            return pd.read_csv(path, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive fallthrough
            errors.append(exc)
    # Re-raise the last error to preserve context when nothing succeeded.
    if errors:
        raise errors[-1]
    return pd.DataFrame()


def load_ei_units(path: str | Path | None = None, *, delimiter: str | None = ",") -> pd.DataFrame:
    """Load an EI-format unit fleet CSV returning normalized numeric columns."""

    if path is None:
        from engine.settings import input_root

        base = input_root()
        try:
            base_resolved = base.resolve()
        except OSError:  # pragma: no cover - defensive
            base_resolved = base

        if base_resolved.is_file():
            csv_path = base_resolved
        else:
            candidates: list[Path] = []
            if base_resolved.name.lower() == "load_forecasts":
                ancestors = list(base_resolved.parents)
                if len(ancestors) >= 2:
                    candidates.append(ancestors[1] / "ei_units.csv")
            candidates.extend(
                [
                    base_resolved / "ei_units.csv",
                    base_resolved / "electricity" / "load_forecasts" / "ei_units.csv",
                ]
            )
            existing = next((candidate for candidate in candidates if candidate.exists()), None)
            csv_path = existing.resolve() if existing is not None else candidates[0]
    else:
        csv_path = Path(path)

    csv_path = csv_path.expanduser()
    if not csv_path.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"Unit fleet file not found: {csv_path}")

    delimiters: tuple[str | None, ...]
    if delimiter is None:
        delimiters = (None, ",", "\t")
    else:
        delimiters = (delimiter, "\t") if delimiter != "\t" else ("\t",)

    df = _read_csv(csv_path, delimiters=delimiters)
    df.columns = df.columns.str.strip().str.lower()

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Process numeric columns
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    # Normalize region_id
    df["region_id"] = (
        df["region_id"]
        .astype(str)
        .str.strip()
        .str.replace("-", "_", regex=False)
        .str.upper()
    )

    # Include unique_id as the primary identifier if available
    ordered_columns = [
        "region_id",
        "heat_rate_mmbtu_per_mwh",
        "capacity_mw",
        "co2_short_ton_per_mwh",
    ]
    
    # Add optional metadata columns if present
    available_optional = [col for col in OPTIONAL_COLUMNS if col in df.columns]
    if available_optional:
        # Put unique_id first if present
        if "unique_id" in available_optional:
            ordered_columns = ["unique_id"] + ordered_columns
            available_optional.remove("unique_id")
        # Add remaining optional columns at the end
        ordered_columns.extend(sorted(available_optional))
    
    # Select only available columns
    final_columns = [col for col in ordered_columns if col in df.columns]

    return df.loc[:, final_columns].copy()


__all__ = ["load_ei_units"]
