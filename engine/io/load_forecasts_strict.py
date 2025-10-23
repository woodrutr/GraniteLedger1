"""Strict CSV loader for electricity demand forecasts.

This module provides a minimal, well-defined interface for reading zone-level
load forecast CSVs.  The implementation focuses on the behaviours exercised by
the unit tests:

* Rigorous validation with clear ``ValidationError`` exceptions.
* Optional Parquet caching backed by ``engine.io.cache``.
* Normalisation helpers that expose canonical ISO and region identifiers.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

from engine import settings as engine_settings
from engine.constants import CSV_LOAD_COL, CSV_YEAR_COL, LOAD_FORECASTS_STRICT_PARQUET
from engine.normalization import normalize_iso_name, normalize_region_id, normalize_token
from regions.registry import REGIONS

from . import cache

__all__ = [
    "ValidationError",
    "build_table",
    "read_zone_csv",
    "_finalize_frame",
    "_read_csv_records",
]


@dataclass
class ValidationError(Exception):
    """Raised when a CSV file fails strict validation rules."""

    file: Path
    row: int | None = None
    column: str | None = None
    reason: str | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial formatting helper
        details = self.reason or "validation failed"
        if self.row is None and self.column is None:
            return f"{self.file}: {details}"

        location: list[str] = []
        if self.row is not None:
            location.append(f"row {self.row}")
        if self.column is not None:
            location.append(f"column {self.column}")
        suffix = ", ".join(location)
        return f"{self.file}: {details} ({suffix})"


def _coerce_paths(base_path: object | None) -> list[Path]:
    """Return a list of forecast root paths from ``base_path`` input."""

    if base_path is None:
        return [engine_settings.input_root()]

    if isinstance(base_path, (str, Path)):
        return [Path(base_path)]

    if isinstance(base_path, Sequence):
        result: list[Path] = []
        for entry in base_path:
            if entry is None:
                continue
            if isinstance(entry, (str, Path)):
                result.append(Path(entry))
        return result

    raise TypeError("base_path must be a path-like object or sequence of paths")


def _iter_zone_files(roots: Iterable[Path]) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(iso_label, scenario_name, csv_path)`` triples for ``roots``."""

    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        try:
            iso_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            continue
        for iso_dir in iso_dirs:
            try:
                scenario_dirs = sorted(p for p in iso_dir.iterdir() if p.is_dir())
            except OSError:
                continue
            for scenario_dir in scenario_dirs:
                try:
                    csv_files = sorted(
                        p
                        for p in scenario_dir.iterdir()
                        if p.is_file() and p.suffix.lower() == ".csv"
                    )
                except OSError:
                    continue
                for csv_path in csv_files:
                    resolved = csv_path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    yield (iso_dir.name, scenario_dir.name, csv_path)


def _read_csv_records(path: Path) -> list[tuple[int, float]]:
    """Return ``(year, load_gwh)`` tuples from ``path`` with validation."""

    path = Path(path)
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:  # pragma: no cover - defensive guard
        raise ValidationError(path, reason=str(exc)) from exc

    with handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        normalized_header = {str(col).strip() for col in header if str(col).strip()}
        expected = {CSV_YEAR_COL, CSV_LOAD_COL}
        if normalized_header != expected:
            columns = ", ".join(sorted(expected))
            raise ValidationError(path, row=1, reason=f"expected columns {columns}")

        records: list[tuple[int, float]] = []
        previous_year: int | None = None
        for index, row in enumerate(reader, start=2):
            year_raw = str(row.get(CSV_YEAR_COL, "")).strip()
            load_raw = str(row.get(CSV_LOAD_COL, "")).strip()

            if not year_raw:
                raise ValidationError(path, row=index, column=CSV_YEAR_COL, reason="missing value")
            if not load_raw:
                raise ValidationError(path, row=index, column=CSV_LOAD_COL, reason="missing value")

            try:
                year = int(year_raw)
            except ValueError:
                raise ValidationError(path, row=index, column=CSV_YEAR_COL, reason="invalid integer")

            try:
                load = float(load_raw)
            except ValueError:
                raise ValidationError(path, row=index, column=CSV_LOAD_COL, reason="invalid number")

            if load < 0:
                raise ValidationError(
                    path,
                    row=index,
                    column=CSV_LOAD_COL,
                    reason="load must be non-negative",
                )

            if previous_year is not None and year <= previous_year:
                raise ValidationError(
                    path,
                    row=index,
                    column=CSV_YEAR_COL,
                    reason="Year values must be strictly increasing",
                )

            previous_year = year
            records.append((year, load))

        if not records:
            raise ValidationError(path, reason="no data rows")

        return records


def _resolve_region_id(zone_name: str, iso_label: str, csv_path: Path) -> str:
    """Return canonical region identifier for ``zone_name`` or raise."""

    zone_clean = zone_name.strip()
    iso_norm = normalize_iso_name(iso_label)
    try:
        region_id = normalize_region_id(zone_clean, iso=iso_norm or iso_label)
    except ValueError:
        raise ValidationError(
            csv_path,
            reason=f"Zone {zone_clean.strip().upper() or zone_name} missing from REGION registry",
        )

    if region_id not in REGIONS:
        raise ValidationError(
            csv_path,
            reason=f"Zone {region_id} missing from REGION registry",
        )

    return region_id


def _finalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with canonical ordering and categorical dtypes."""

    columns = [
        "iso",
        "scenario",
        "zone",
        "region_id",
        CSV_YEAR_COL,
        CSV_LOAD_COL,
        "iso_norm",
        "scenario_norm",
        "region_norm",
    ]

    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)

    base = frame.loc[:, columns].copy()
    base[CSV_YEAR_COL] = base[CSV_YEAR_COL].astype(int)
    base[CSV_LOAD_COL] = base[CSV_LOAD_COL].astype(float)
    base["iso_norm"] = base["iso_norm"].astype(str)
    base["scenario_norm"] = base["scenario_norm"].astype(str)
    base["region_norm"] = base["region_norm"].astype(str)

    sort_cols = ["iso_norm", "scenario_norm", "region_norm", CSV_YEAR_COL]
    base = base.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    for column in ("iso", "scenario", "zone", "region_id"):
        base[column] = pd.Categorical(base[column].astype(str))

    return base


def _build_rows(zone_files: Iterable[tuple[str, str, Path]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for iso_label, scenario_name, csv_path in zone_files:
        records = _read_csv_records(csv_path)
        iso_norm = normalize_iso_name(iso_label) or iso_label.strip().lower()
        scenario_norm = normalize_token(scenario_name) or scenario_name.strip().lower()
        region_label = csv_path.stem
        region_id = _resolve_region_id(region_label, iso_label, csv_path)
        region_norm = normalize_token(region_id) or region_id.strip().lower()

        for year, load in records:
            rows.append(
                {
                    "iso": iso_label,
                    "scenario": scenario_name,
                    "zone": region_label,
                    "region_id": region_id,
                    CSV_YEAR_COL: int(year),
                    CSV_LOAD_COL: float(load),
                    "iso_norm": iso_norm,
                    "scenario_norm": scenario_norm,
                    "region_norm": region_norm,
                }
            )
    return rows


def build_table(
    base_path: object | None = None,
    *,
    cache_path: Path | None = LOAD_FORECASTS_STRICT_PARQUET,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a validated demand forecast table for the provided ``base_path``."""

    roots = _coerce_paths(base_path)
    zone_files = list(_iter_zone_files(roots))

    manifest = cache.build_manifest([])
    if zone_files:
        manifest = cache.build_manifest(csv_path for *_iso, _scenario, csv_path in zone_files)

    if cache_path is not None and use_cache:
        cached = cache.load(Path(cache_path), manifest)
        if cached is not None:
            return _finalize_frame(cached)

    rows = _build_rows(zone_files)
    frame = pd.DataFrame(rows)
    finalized = _finalize_frame(frame)

    if cache_path is not None and zone_files:
        cache.write(Path(cache_path), finalized, manifest)

    return finalized


def read_zone_csv(path: Path | str) -> pd.DataFrame:
    """Return a DataFrame with the canonical schema for a single zone CSV."""

    records = _read_csv_records(Path(path))
    frame = pd.DataFrame(records, columns=[CSV_YEAR_COL, CSV_LOAD_COL])
    frame["year"] = frame[CSV_YEAR_COL].astype(int)
    frame["demand_mwh"] = frame[CSV_LOAD_COL].astype(float) * 1000.0
    return frame[[CSV_YEAR_COL, CSV_LOAD_COL, "year", "demand_mwh"]]

