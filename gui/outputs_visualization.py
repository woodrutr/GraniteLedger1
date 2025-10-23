from __future__ import annotations

import io
import logging
from typing import Mapping, Sequence

import pandas as pd

try:  # pragma: no cover - fallback when executed as a script
    from gui.region_metadata import (
        DEFAULT_REGION_METADATA,
        canonical_region_label,
        canonical_region_value,
        region_label_from_mapping,
        region_mapping_frame,
    )
except (ModuleNotFoundError, ImportError):  # pragma: no cover - compatibility fallback
    try:  # pragma: no cover - defensive import shim
        import importlib

        _metadata = importlib.import_module("gui.region_metadata")
    except ModuleNotFoundError:  # pragma: no cover - final fallback
        _metadata = None

    if _metadata is not None:
        DEFAULT_REGION_METADATA = getattr(_metadata, "DEFAULT_REGION_METADATA", {})
        canonical_region_label = getattr(_metadata, "canonical_region_label", lambda value: str(value))
        canonical_region_value = getattr(_metadata, "canonical_region_value", lambda value: value)
        region_mapping_frame = getattr(_metadata, "region_mapping_frame", lambda: pd.DataFrame())
    else:
        DEFAULT_REGION_METADATA = {}

        def canonical_region_label(value):  # type: ignore[no-redef]
            return str(value)

        def canonical_region_value(value):  # type: ignore[no-redef]
            return value

        def region_mapping_frame():  # type: ignore[no-redef]
            return pd.DataFrame()

    def region_label_from_mapping(value):  # type: ignore[no-redef]
        return canonical_region_label(value)


LOGGER = logging.getLogger(__name__)


_EMISSIONS_FRAME_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("emissions_by_region", "emissions_by_region.csv"),
    ("emissions", "emissions.csv"),
    ("emissions_region", "emissions_region.csv"),
)


def _annual_emissions_fallback(result: Mapping[str, object]) -> pd.DataFrame:
    """Return a system-level emissions frame derived from annual totals."""

    annual = result.get("annual")
    if not isinstance(annual, pd.DataFrame):
        return pd.DataFrame()

    required = {"year", "emissions_tons"}
    if not required.issubset(annual.columns):
        return pd.DataFrame()

    working = annual[list(required)].copy()
    working["year"] = pd.to_numeric(working["year"], errors="coerce")
    working = working.dropna(subset=["year"])
    if working.empty:
        return pd.DataFrame()

    working["year"] = working["year"].astype(int)
    working["emissions_tons"] = pd.to_numeric(
        working["emissions_tons"], errors="coerce"
    ).fillna(0.0)
    working["region"] = "system"
    working["region_canonical"] = working["region"].apply(canonical_region_value)
    working["region_label"] = working["region_canonical"].map(region_label_from_mapping)

    columns = ["year", "region", "region_label", "emissions_tons", "region_canonical"]
    return working[columns].reset_index(drop=True)


def load_emissions_data(result: Mapping[str, object] | None) -> pd.DataFrame:
    """Return the emissions-by-region frame from ``result`` when available."""

    if not isinstance(result, Mapping):
        return pd.DataFrame()

    frame: pd.DataFrame | None = None
    for key, _ in _EMISSIONS_FRAME_CANDIDATES:
        candidate = result.get(key)
        if isinstance(candidate, pd.DataFrame):
            frame = candidate.copy()
            break

    if frame is None:
        csv_files = result.get("csv_files")
        if isinstance(csv_files, Mapping):
            for key, csv_name in _EMISSIONS_FRAME_CANDIDATES:
                raw = csv_files.get(csv_name) or csv_files.get(f"{key}.csv")
                if isinstance(raw, (bytes, bytearray)):
                    try:
                        frame = pd.read_csv(io.BytesIO(raw))
                    except Exception:  # pragma: no cover - best effort load
                        continue
                    else:
                        break

    if frame is None or frame.empty:
        fallback = _annual_emissions_fallback(result)
        if fallback.empty:
            LOGGER.info(
                "Emissions data missing from engine outputs; no regional table available."
            )
            fallback.attrs["emissions_source"] = "missing"
        else:
            LOGGER.info(
                "Engine outputs missing regional emissions frame; constructed fallback from annual totals."
            )
            fallback.attrs["emissions_source"] = "fallback"
        return fallback

    frame = frame.copy()
    lower_lookup = {col.lower(): col for col in frame.columns}
    rename_map: dict[str, str] = {}
    column_aliases = {
        "year": ("year", "calendar_year", "period"),
        "region": ("region", "region_id", "region_code", "region_name"),
        "emissions_tons": (
            "emissions_tons",
            "tons",
            "emissions",
            "value",
        ),
    }
    for canonical, aliases in column_aliases.items():
        for alias in aliases:
            column = lower_lookup.get(alias.lower())
            if column is not None:
                rename_map[column] = canonical
                break

    if rename_map:
        frame = frame.rename(columns=rename_map)

    if "emissions_tons" not in frame.columns:
        LOGGER.warning(
            "Emissions frame lacks 'emissions_tons' column after normalisation; treating as missing."
        )
        fallback = _annual_emissions_fallback(result)
        fallback.attrs["emissions_source"] = "fallback" if not fallback.empty else "missing"
        return fallback

    frame["emissions_tons"] = pd.to_numeric(frame["emissions_tons"], errors="coerce")
    frame = frame.dropna(subset=["emissions_tons"])

    if frame.empty:
        LOGGER.info(
            "Regional emissions frame contains no numeric rows after coercion; attempting fallback."
        )
        fallback = _annual_emissions_fallback(result)
        fallback.attrs["emissions_source"] = "fallback" if not fallback.empty else "missing"
        return fallback

    if "region" not in frame.columns:
        frame["region"] = "system"

    frame["region"] = frame["region"].fillna("system")
    frame["region_canonical"] = frame["region"].apply(canonical_region_value)

    if "year" in frame.columns:
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        frame = frame.dropna(subset=["year"])
        frame["year"] = frame["year"].astype(int)
    else:
        frame["year"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    if frame.empty:
        LOGGER.info(
            "Regional emissions frame emptied after year normalisation; attempting fallback."
        )
        fallback = _annual_emissions_fallback(result)
        fallback.attrs["emissions_source"] = "fallback" if not fallback.empty else "missing"
        return fallback

    frame["region"] = frame["region"].astype(str)

    try:
        region_map = region_mapping_frame()
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("Unable to load region metadata for emissions table: %s", exc)
        region_map = pd.DataFrame()

    if not region_map.empty:
        metadata = region_map.copy()
        metadata["region_canonical"] = metadata["region"].apply(canonical_region_value)
        metadata = metadata.drop_duplicates("region_canonical")
        label_lookup = metadata.set_index("region_canonical")["region_label"]
        frame["region_label"] = frame["region_canonical"].map(label_lookup)
    else:
        frame["region_label"] = pd.NA

    frame["region_label"] = frame["region_label"].where(
        frame["region_label"].notna(),
        frame["region_canonical"].map(region_label_from_mapping),
    )

    frame = frame.dropna(subset=["region_label"])

    total_emissions = float(frame["emissions_tons"].sum())
    if abs(total_emissions) <= 1e-9:
        LOGGER.info("Regional emissions totals sum to zero after processing.")

    columns = ["year", "region", "region_label", "emissions_tons", "region_canonical"]
    frame = frame[columns].sort_values(["year", "region_label"]).reset_index(drop=True)

    frame.attrs["emissions_source"] = "engine"

    return frame


def region_selection_options(emissions_df: pd.DataFrame) -> list[tuple[str, int | str]]:
    """Return (label, value) pairs for all canonical regions present in ``emissions_df``."""

    if emissions_df.empty:
        return []

    if "region_canonical" in emissions_df.columns:
        canonical_series = emissions_df["region_canonical"]
    elif "region" in emissions_df.columns:
        canonical_series = emissions_df["region"]
    else:
        return []

    options: list[tuple[str, int | str]] = []
    seen: set[object] = set()
    for canonical in canonical_series:
        if pd.isna(canonical):
            continue

        canonical_value = canonical_region_value(canonical)
        if pd.isna(canonical_value):
            continue

        if isinstance(canonical_value, str):
            normalized = canonical_value.strip().lower()
            if not normalized or normalized in {"default", "<na>", "nan", "none"}:
                continue
        if canonical_value in seen:
            continue
        seen.add(canonical_value)

        display_label = canonical_region_label(canonical_value)
        if not display_label or display_label.strip().lower() in {"<na>", "nan", "none"}:
            continue

        if isinstance(canonical, str) and canonical.strip():
            output_value: int | str = canonical
        else:
            output_value = (
                canonical_value
                if isinstance(canonical_value, (int, float)) and not isinstance(canonical_value, bool)
                else canonical_value
            )

        options.append((display_label, output_value))

    options.sort(key=lambda item: item[0].lower())
    if not options:
        fallback: list[tuple[str, int | str]] = []
        for metadata in DEFAULT_REGION_METADATA.values():
            label = metadata.label or metadata.code or str(metadata.id)
            value: int | str = metadata.code or metadata.label or metadata.id
            fallback.append((label, value))
        fallback.sort(key=lambda item: item[0].lower())
        return fallback
    return options


def filter_emissions_by_regions(
    emissions_df: pd.DataFrame, selected_regions: Sequence[int | str] | None
) -> pd.DataFrame:
    """Return a copy of ``emissions_df`` filtered to the canonical ``selected_regions``."""

    if emissions_df.empty or "region_canonical" not in emissions_df.columns:
        return emissions_df.copy()

    if not selected_regions:
        return emissions_df.copy()

    resolved: set[int | str] = set()
    for value in selected_regions:
        resolved.add(canonical_region_value(value))

    return emissions_df[emissions_df["region_canonical"].isin(resolved)].copy()


def summarize_emissions_totals(emissions_df: pd.DataFrame) -> pd.DataFrame:
    """Return year/region emissions suitable for stacked bar visualisation."""

    if emissions_df.empty or "emissions_tons" not in emissions_df.columns:
        return pd.DataFrame(columns=["year", "region_label", "emissions_tons"])

    working = emissions_df.copy()
    if "region_label" not in working.columns:
        working["region_label"] = working.get("region", "region")
    working["year"] = pd.to_numeric(working.get("year", 0), errors="coerce")
    working = working.dropna(subset=["year", "region_label"])
    working["year"] = working["year"].astype(int)

    grouped = (
        working.groupby(["year", "region_label"], sort=True)["emissions_tons"]
        .sum(min_count=1)
        .reset_index()
    )

    return grouped.sort_values(["year", "region_label"]).reset_index(drop=True)
