"""Helper utilities for demand curve selection and bespoke forecast uploads."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from engine.constants import INPUT_DIR
from engine.normalization import normalize_iso_name, normalize_region, normalize_token
from common.schemas.load_forecast import parse_load_forecast_csv

from .forecast_helpers import (
    _cached_forecast_frame,
    _cached_input_root,
    _coerce_base_path,
    _load_iso_scenario_frame,
    _regions_available_zones,
    _scenario_frame_subset,
)

try:  # pragma: no cover - optional dependency wrapper
    import yaml
except Exception:  # pragma: no cover - optional dependency wrapper
    yaml = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency shim
    from main.definitions import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover - fallback for packaged app execution
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:  # pragma: no cover - optional dependency shim
    from engine.constants import INPUT_ROOT as _ENGINE_INPUT_ROOT
except Exception:  # pragma: no cover - compatibility guard for import order
    _ENGINE_INPUT_ROOT = PROJECT_ROOT / "input"

try:  # pragma: no cover - optional dependency shim
    from gui.region_metadata import (
        DEFAULT_REGION_METADATA,
        canonical_region_label,
        canonical_region_value,
    )
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    try:
        from region_metadata import (  # type: ignore[import-not-found]
            DEFAULT_REGION_METADATA,
            canonical_region_label,
            canonical_region_value,
        )
    except ModuleNotFoundError:  # pragma: no cover - final fallback for testing contexts
        DEFAULT_REGION_METADATA = {}

        def canonical_region_label(value: Any) -> str:  # type: ignore[return-type]
            return str(value)

        def canonical_region_value(value: Any) -> Any:  # type: ignore[return-type]
            return value
LOGGER = logging.getLogger(__name__)

def _resolve_electricity_input_dir() -> Path:
    """Return the electricity input directory, tolerating ``inputs`` layouts."""
    candidates = [
        INPUT_DIR / "electricity",
        Path(PROJECT_ROOT, "inputs", "electricity"),
        Path(PROJECT_ROOT, "input", "electricity"),
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return candidates[-1]


_DEMAND_CURVE_BASE_DIR = _resolve_electricity_input_dir()
_DEMAND_CURVE_FLAT_KEY = "flat_100k_mwh"
_FLAT_DEMAND_MWH = 100_000.0
_FLAT_DEMAND_LABEL = "Flat 100k MWh"
_DEMAND_CURVE_FALLBACK_KEY = "missing_data_default"
_FALLBACK_DEMAND_MWH = 0.0
_FALLBACK_DEMAND_LABEL = "Missing data (0 MWh)"
_ALL_REGIONS_LABEL = "All regions"



def load_user_demand_csv(path: str | Path) -> pd.DataFrame:
    """Parse user CSV via canonical schema and present as GUI ``region/year/demand_mwh``."""

    frame = parse_load_forecast_csv(path)
    out = frame.loc[:, ["region_id", "year", "load_gwh"]].copy()
    out.rename(columns={"region_id": "region", "load_gwh": "demand_mwh"}, inplace=True)
    out["year"] = out["year"].astype(int)
    out["demand_mwh"] = (out["demand_mwh"] * 1000.0).astype(float)
    return out


def normalize_gui_demand_table(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return a demand table with ``year``/``region``/``demand_mwh`` columns."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["region", "year", "demand_mwh"])

    column_lookup = {str(column).strip().lower(): column for column in frame.columns}
    year_column = column_lookup.get("year")
    if year_column is None:
        return pd.DataFrame(columns=["region", "year", "demand_mwh"])

    region_column = (
        column_lookup.get("region")
        or column_lookup.get("region_id")
        or column_lookup.get("zone")
    )

    demand_column: str | None = None
    demand_multiplier = 1.0
    if column_lookup.get("demand_mwh") is not None:
        demand_column = column_lookup["demand_mwh"]
    elif column_lookup.get("load_mwh") is not None:
        demand_column = column_lookup["load_mwh"]
    elif column_lookup.get("load_gwh") is not None:
        demand_column = column_lookup["load_gwh"]
        demand_multiplier = 1000.0

    if demand_column is None:
        return pd.DataFrame(columns=["region", "year", "demand_mwh"])

    year_series = pd.to_numeric(frame[year_column], errors="coerce")
    demand_series = pd.to_numeric(frame[demand_column], errors="coerce") * demand_multiplier

    if region_column is not None:
        region_series = frame[region_column].astype("string").str.strip()
        region_series = region_series.where(region_series != "", pd.NA)
    else:
        region_series = pd.Series(pd.NA, index=frame.index, dtype=pd.StringDtype())

    working = pd.DataFrame(
        {
            "region": region_series,
            "year": year_series,
            "demand_mwh": demand_series,
        }
    )

    working = working.dropna(subset=["year", "demand_mwh"])
    if working.empty:
        return pd.DataFrame(columns=["region", "year", "demand_mwh"])

    working["year"] = working["year"].astype(int)
    working["demand_mwh"] = working["demand_mwh"].astype(float)
    return working.loc[:, ["region", "year", "demand_mwh"]]


def _scenario_frame_to_demand(frame: pd.DataFrame) -> pd.DataFrame:
    """Return GUI demand table (region/year/demand_mwh) from an ISO scenario frame."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["region", "year", "demand_mwh"])

    columns: dict[str, Any] = {str(column).strip().lower(): column for column in frame.columns}

    region_column = (
        columns.get("region_id") or columns.get("zone") or columns.get("region")
    )

    year_column = columns.get("year")
    if year_column is None and columns.get("timestamp") is not None:
        year_series = pd.to_datetime(
            frame[columns["timestamp"]], errors="coerce"
        ).dt.year
    elif year_column is not None:
        year_series = pd.to_numeric(frame[year_column], errors="coerce")
    else:
        year_series = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    demand_series = None
    load_mwh_column = columns.get("load_mwh")
    load_gwh_column = columns.get("load_gwh")

    if load_mwh_column is not None:
        demand_series = pd.to_numeric(frame[load_mwh_column], errors="coerce")
        if not demand_series.notna().any():
            demand_series = None

    if demand_series is None and load_gwh_column is not None:
        demand_series = pd.to_numeric(frame[load_gwh_column], errors="coerce") * 1000.0

    if demand_series is None:
        return pd.DataFrame(columns=["region", "year", "demand_mwh"])

    if region_column is not None:
        region_series = (
            frame[region_column].astype("string").str.strip().replace("", pd.NA)
        )
    else:
        region_series = pd.Series(pd.NA, index=frame.index, dtype=pd.StringDtype())

    working = pd.DataFrame(
        {
            "region": region_series,
            "year": year_series,
            "demand_mwh": demand_series,
        }
    )

    working = working.dropna(subset=["year", "demand_mwh"])
    if working.empty:
        return pd.DataFrame(columns=["region", "year", "demand_mwh"])

    working["year"] = working["year"].astype(int)
    working["demand_mwh"] = working["demand_mwh"].astype(float)
    working["region"] = working["region"].astype("string").str.strip()
    working = working.dropna(subset=["region"])
    working = working[working["region"] != ""]

    if working.empty:
        return pd.DataFrame(columns=["region", "year", "demand_mwh"])

    aggregated = working.groupby(["region", "year"], as_index=False)["demand_mwh"].sum()
    return aggregated.loc[:, ["region", "year", "demand_mwh"]]


@dataclass(slots=True)
class _ScenarioSelection:
    iso: str
    scenario: str
    zones: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.iso = str(self.iso)
        self.scenario = str(self.scenario)

        zone_entries: list[str] = []
        for zone in self.zones:
            text = str(zone).strip()
            if text:
                zone_entries.append(text)
        self.zones = sorted(dict.fromkeys(zone_entries))

        year_entries: list[int] = []
        for year in self.years:
            try:
                year_entries.append(int(year))
            except (TypeError, ValueError):
                continue
        self.years = sorted(dict.fromkeys(year_entries))


class _FrameProxy:
    """Proxy ensuring DataFrame equality uses :meth:`pandas.DataFrame.equals`."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def __eq__(self, other: object) -> bool:  # pragma: no cover - exercised indirectly
        if isinstance(other, _FrameProxy):
            return self._frame.equals(other._frame)
        if isinstance(other, pd.DataFrame):
            return self._frame.equals(other)
        return NotImplemented  # type: ignore[return-value]

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple proxy shim
        return getattr(self._frame, name)

    def __repr__(self) -> str:  # pragma: no cover - repr aid
        return repr(self._frame)


def _extract_record_value(record: Any, *names: str) -> str:
    """Return the first matching attribute or mapping entry from ``record``."""

    for name in names:
        if isinstance(record, Mapping) and name in record:
            value = record[name]
        else:
            value = getattr(record, name, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _coerce_manifest_list(entries: Any) -> list[_ScenarioSelection]:
    """Return ``entries`` coerced to scenario selection objects."""

    manifests: list[_ScenarioSelection] = []
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)):
        for entry in entries:
            if isinstance(entry, _ScenarioSelection):
                manifests.append(entry)
                continue
            if isinstance(entry, Mapping):
                iso_value = _extract_record_value(entry, "iso", "iso_name")
                scenario_value = _extract_record_value(
                    entry, "scenario", "manifest", "manifest_name"
                )
                zones = _iter_strings(entry.get("zones")) if isinstance(entry, Mapping) else []
                years_raw = entry.get("years") if isinstance(entry, Mapping) else []
                years: list[int] = []
                for year in _iter_strings(years_raw):
                    try:
                        years.append(int(year))
                    except (TypeError, ValueError):
                        continue
                manifests.append(
                    _ScenarioSelection(
                        iso=iso_value,
                        scenario=scenario_value,
                        zones=list(zones),
                        years=years,
                    )
                )
    return manifests


def select_forecast_bundles(*args: Any, **kwargs: Any) -> list[_ScenarioSelection]:
    """Resolve scenario selections without relying on legacy bundle helpers."""

    selection: Mapping[str, Any] | None
    if args:
        candidate = args[0]
        selection = candidate if isinstance(candidate, Mapping) else None
    else:
        candidate = kwargs.get("selection")
        selection = candidate if isinstance(candidate, Mapping) else None

    base_path = kwargs.get("base_path")
    frame = kwargs.get("frame")
    if isinstance(frame, _FrameProxy):
        frame = frame._frame

    root = _coerce_base_path(base_path)
    working_frame = frame if isinstance(frame, pd.DataFrame) else None
    if working_frame is None:
        try:
            working_frame = _cached_forecast_frame(root)
        except Exception:  # pragma: no cover - defensive guard
            working_frame = pd.DataFrame()

    try:
        manifests = _manifests_from_selection(
            selection,
            frame=working_frame,
            base_path=root,
        )
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug(
            "Failed to resolve forecast manifests from selection", exc_info=True
        )
        manifests = []

    return manifests


def _iter_strings(value: Any) -> list[str]:
    """Return ``value`` coerced to a list of strings."""

    if value is None:
        return []
    if isinstance(value, Mapping):
        items = value.values()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        items = value
    else:
        return []
    result: list[str] = []
    for entry in items:
        text = str(entry).strip()
        if text:
            result.append(text)
    return result


def _build_scenario_manifest(
    iso: str,
    scenario: str,
    *,
    frame: pd.DataFrame | None = None,
    base_path: str | Path | None = None,
) -> _ScenarioSelection | None:
    iso_label = str(iso)
    scenario_label = str(scenario)

    subset = pd.DataFrame()
    if frame is not None:
        try:
            subset = _scenario_frame_subset(frame, iso_label, scenario_label)
        except Exception:
            subset = pd.DataFrame()

    root = base_path or _cached_input_root()
    if subset.empty:
        subset = _load_iso_scenario_frame(root, iso_label, scenario_label)

    zones: list[str] = []
    years: list[int] = []

    if not subset.empty:
        if "zone" in subset.columns:
            zone_series = subset["zone"].dropna().astype(str)
            zones = sorted(dict.fromkeys(zone_series.tolist()))
        if "year" in subset.columns:
            year_series = pd.to_numeric(subset["year"], errors="coerce").dropna().astype(int)
            years = sorted(dict.fromkeys(year_series.tolist()))

    if not zones:
        try:
            zone_candidates = _regions_available_zones(root, iso_label, scenario_label)
        except Exception:
            zone_candidates = []
        zones = sorted({str(value) for value in zone_candidates if value is not None})

    manifest_iso = normalize_iso_name(iso_label) or iso_label
    return _ScenarioSelection(
        iso=manifest_iso,
        scenario=scenario_label,
        zones=list(zones),
        years=list(years),
    )


def _manifests_from_selection(
    selection: Mapping[str, Any] | None,
    *,
    frame: pd.DataFrame | None,
    base_path: str | Path | None,
) -> list[_ScenarioSelection]:
    if not isinstance(selection, Mapping) or not selection:
        return []

    _, iso_pairs = _normalize_state_forecast_selection(selection)
    if not iso_pairs:
        return []

    manifests: list[_ScenarioSelection] = []
    seen: set[str] = set()
    for iso_label, scenario_label in iso_pairs:
        if not iso_label or not scenario_label:
            continue
        iso_token = normalize_iso_name(iso_label) or normalize_token(iso_label) or iso_label
        scenario_key = str(scenario_label)
        dedupe_key = f"{iso_token.lower()}::{normalize_token(scenario_key) or scenario_key.lower()}"
        if dedupe_key in seen:
            continue
        manifest = _build_scenario_manifest(
            iso_token,
            scenario_key,
            frame=frame,
            base_path=base_path,
        )
        if manifest is not None:
            manifests.append(manifest)
            seen.add(dedupe_key)

    return manifests


def _default_scenario_manifests(
    frame: pd.DataFrame,
    *,
    base_path: str | Path | None,
) -> list[_ScenarioSelection]:
    if frame.empty:
        return []

    manifests: list[_ScenarioSelection] = []
    iso_series = frame.get("iso")
    scenario_series = frame.get("scenario")
    if iso_series is None or scenario_series is None:
        return []

    seen_pairs: set[tuple[str, str]] = set()
    for iso_lower in sorted(dict.fromkeys(iso_series.astype(str).str.lower().tolist())):
        mask = iso_series.astype(str).str.lower() == iso_lower
        scenarios = scenario_series.loc[mask].astype(str).dropna().tolist()
        if not scenarios:
            continue
        ordered = _ordered_scenarios(scenarios)
        if not ordered:
            continue
        iso_display = iso_series.loc[mask].astype(str).dropna().tolist()[0]
        iso_token = normalize_iso_name(iso_display) or iso_display
        iso_key = str(iso_token).lower()
        for scenario_label in ordered:
            candidate_pair = (iso_key, scenario_label)
            if candidate_pair in seen_pairs:
                continue
            manifest = _build_scenario_manifest(
                iso_token,
                scenario_label,
                frame=frame,
                base_path=base_path,
            )
            if manifest is None:
                continue
            if not isinstance(manifest, _ScenarioSelection):
                coerced = _coerce_manifest_list([manifest])
                manifest = coerced[0] if coerced else None
            if manifest is None:
                continue
            manifests.append(manifest)
            seen_pairs.add((manifest.iso.lower(), manifest.scenario))

    return manifests


def _demand_frame_from_manifests(
    manifests: Sequence[_ScenarioSelection],
    *,
    years: Iterable[int] | None,
    base_path: str | Path | None,
    collect_load_frames: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    demand_columns = ["year", "region", "demand_mwh"]

    def _empty_result() -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        demand_empty = pd.DataFrame(columns=demand_columns)
        if collect_load_frames:
            load_columns = ["iso", "zone", "scenario", "year", "load_gwh"]
            load_empty = pd.DataFrame(columns=load_columns)
            return demand_empty, load_empty
        return demand_empty

    if not manifests:
        return _empty_result()

    root = base_path or _cached_input_root()
    frames: list[pd.DataFrame] = []
    load_frames: list[pd.DataFrame] = [] if collect_load_frames else []
    for manifest in manifests:
        iso_label = manifest.iso
        scenario_label = manifest.scenario
        scenario_frame = _load_iso_scenario_frame(root, iso_label, scenario_label)
        if scenario_frame.empty:
            LOGGER.debug(
                "Unable to load demand for %s/%s", iso_label, scenario_label, exc_info=True
            )
            continue

        if collect_load_frames:
            load_columns = [
                "iso",
                "zone",
                "region_id",
                "scenario",
                "year",
                "load_gwh",
                "load_mwh",
                "state",
                "state_or_province",
                "scenario_name",
            ]
            available = [column for column in load_columns if column in scenario_frame.columns]
            load_subset = scenario_frame.loc[:, available].copy()
            if not load_subset.empty:
                load_frames.append(load_subset)

        normalized = _scenario_frame_to_demand(scenario_frame)
        if normalized.empty:
            continue

        frames.append(normalized.loc[:, demand_columns])

    if not frames:
        return _empty_result()

    combined = pd.concat(frames, ignore_index=True)
    if years:
        target_years = {int(year) for year in years}
        combined = combined.loc[combined["year"].isin(target_years)]

    grouped = combined.groupby(["region", "year"], as_index=False)["demand_mwh"].sum()
    grouped = grouped.sort_values(["region", "year"]).reset_index(drop=True)

    if not collect_load_frames:
        return grouped

    if load_frames:
        load_df = pd.concat(load_frames, ignore_index=True)
    else:
        load_df = pd.DataFrame(
            columns=["iso", "zone", "region_id", "scenario", "year", "load_gwh", "load_mwh"]
        )

    if not load_df.empty:
        base_columns = ["iso", "zone", "scenario", "year", "load_gwh"]
        optional_columns = [
            column
            for column in ["region_id", "load_mwh", "state", "state_or_province", "scenario_name"]
            if column in load_df.columns
        ]
        ordered_columns = base_columns + [
            column for column in optional_columns if column not in base_columns
        ]
        load_df = load_df.loc[:, ordered_columns]
        load_df["iso"] = load_df["iso"].astype(str).str.strip()
        load_df["zone"] = load_df["zone"].astype(str).str.strip()
        load_df["scenario"] = load_df["scenario"].astype(str).str.strip()
        load_df["year"] = pd.to_numeric(load_df["year"], errors="coerce").astype("Int64")
        load_df["load_gwh"] = pd.to_numeric(load_df["load_gwh"], errors="coerce")
        load_df = load_df.dropna(
            subset=["iso", "zone", "scenario", "year", "load_gwh"], how="any"
        )
        if not load_df.empty:
            load_df = (
                load_df.sort_values(["iso", "zone", "scenario", "year"]).reset_index(drop=True)
            )
    else:
        load_df = pd.DataFrame(columns=["iso", "zone", "scenario", "year", "load_gwh"])

    return grouped, load_df


def _forecast_bundles_from_manifests(
    manifests: Sequence[_ScenarioSelection], *, base_path: str | Path | None
) -> list[dict[str, Any]]:
    """Return manifest records describing ``manifests``."""

    if not manifests:
        return []

    root = Path(_coerce_base_path(base_path))
    try:
        use_consolidated = (root / "load_forecasts.csv").exists()
    except OSError:
        use_consolidated = False
    csv_candidates = [
        root / "load_forecasts.csv",
        root / "electricity" / "load_forecasts" / "load_forecasts.csv",
    ]
    resolved_csv: Path | None = None
    for candidate in csv_candidates:
        try:
            if candidate.exists():
                resolved_csv = candidate.resolve()
                break
        except OSError:
            continue

    records: list[dict[str, Any]] = []
    for manifest in manifests:
        iso = manifest.iso
        scenario = manifest.scenario
        if not iso or not scenario:
            continue
        if use_consolidated:
            try:
                directory = root.resolve()
            except OSError:
                directory = root
        else:
            try:
                directory = (root / iso / scenario).resolve()
            except OSError:
                directory = root / iso / scenario
        try:
            directory = (root / iso / scenario).resolve()
        except OSError:
            directory = root / iso / scenario
        if not directory.exists() and resolved_csv is not None:
            directory = resolved_csv
        records.append(
            {
                "iso": iso,
                "scenario": scenario,
                "manifest": f"{iso}::{scenario}",
                "zones": list(manifest.zones),
                "years": list(manifest.years),
                "path": str(directory),
            }
        )
    return records


def _forecast_bundles_from_selection(
    selection: Any, *, base_path: str | Path | None
) -> list[dict[str, Any]]:
    """Return manifest records constructed from ``selection``."""

    if selection is None:
        return []

    manifests: Sequence[_ScenarioSelection] = []

    if isinstance(selection, Sequence) and not isinstance(selection, (str, bytes, bytearray)):
        manifests = [entry for entry in selection if isinstance(entry, _ScenarioSelection)]

    if not manifests:
        return []

    return _forecast_bundles_from_manifests(manifests, base_path=base_path)


def _select_scenario_manifests(
    selection: Mapping[str, Any] | None,
    *,
    base_path: str | None,
    frame: pd.DataFrame | None = None,
) -> list[_ScenarioSelection]:
    """Return manifests filtered by ``selection``."""

    manifests = _all_scenario_manifests(base_path)
    if not manifests:
        return []

    canonical = _canonical_forecast_selection(selection)
    if not canonical:
        return manifests

    selected: list[_ScenarioSelection] = []
    for manifest in manifests:
        iso_tokens = [manifest.iso, normalize_iso_name(manifest.iso)]
        match: str | None = None
        for token in iso_tokens:
            if not token:
                continue
            match = canonical.get(token)
            if match:
                break
        if not match:
            continue
        if normalize_token(match) == normalize_token(manifest.scenario):
            selected.append(manifest)

    return selected


def _all_scenario_manifests(base_path: str | Path | None) -> list[_ScenarioSelection]:
    try:
        frame = _cached_forecast_frame(base_path)
    except Exception:
        frame = pd.DataFrame()

    if isinstance(frame, pd.DataFrame):
        working = frame.copy()
    else:
        working = pd.DataFrame(frame)

    if working.empty:
        return []

    return _default_scenario_manifests(working, base_path=base_path)

def _load_demand_frame(
    years: Iterable[int] | None,
    *,
    base_path: str | None,
    manifests: Sequence[_ScenarioSelection],
) -> pd.DataFrame:
    """Return aggregated demand frame for ``manifests``."""

    if not manifests:
        return pd.DataFrame(columns=["year", "region", "demand_mwh"])

    year_filter = {int(year) for year in years} if years else None
    frames: list[pd.DataFrame] = []
    for manifest in manifests:
        scenario_frame = _load_iso_scenario_frame(base_path, manifest.iso, manifest.scenario)
        if scenario_frame.empty:
            continue
        normalized = _scenario_frame_to_demand(scenario_frame)
        if normalized.empty:
            continue
        frame_subset = normalized.loc[:, ["year", "region", "demand_mwh"]]
        if year_filter is not None:
            frame_subset = frame_subset[frame_subset["year"].isin(year_filter)]
        if frame_subset.empty:
            continue
        frames.append(frame_subset)

    if not frames:
        return pd.DataFrame(columns=["year", "region", "demand_mwh"])

    combined = pd.concat(frames, ignore_index=True)
    combined["year"] = combined["year"].astype(int)
    combined["region"] = combined["region"].astype(str).str.strip()
    combined = combined[combined["region"] != ""]
    aggregated = combined.groupby(["year", "region"], as_index=False)["demand_mwh"].sum()
    return aggregated.sort_values(["year", "region"]).reset_index(drop=True)


def _summarize_forecasts(
    frame: pd.DataFrame,
    *,
    base_path: str | None,
) -> list[dict[str, Any]]:
    """Return façade-derived summary rows for display in the UI."""

    if frame.empty:
        return []

    summary: list[dict[str, Any]] = []
    grouped = frame.groupby(["iso", "scenario"], sort=True, observed=True)
    for (iso_value, scenario_value), group in grouped:
        iso_str = str(iso_value)
        scenario_str = str(scenario_value)
        try:
            zones = _regions_available_zones(base_path, iso_str, scenario_str)
        except Exception:
            zones = []
        if not zones:
            zones = sorted(group["zone"].astype(str).unique().tolist())
        summary.append(
            {
                "iso": iso_str,
                "scenario": scenario_str,
                "zones": zones,
                "zone_count": len(zones),
            }
        )
    return summary


_SCENARIO_SELECTION_ORDER = {"baseline": 0, "high": 1, "low": 2}


def _ordered_scenarios(labels: Iterable[str]) -> list[str]:
    """Return scenario labels ordered using :data:`_SCENARIO_SELECTION_ORDER`."""

    unique: dict[str, None] = {}
    for label in labels:
        if label is None:
            continue
        unique[str(label)] = None

    return sorted(
        unique.keys(),
        key=lambda item: (
            _SCENARIO_SELECTION_ORDER.get(normalize_token(item), 99),
            item.lower(),
        ),
    )


def _discovered_region_names() -> list[str]:
    names: list[str] = []
    try:
        frame = _cached_forecast_frame(_cached_input_root())
    except Exception:
        return []
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    if frame.empty:
        return []

    zones = sorted(frame["zone"].astype(str).unique().tolist()) if "zone" in frame.columns else []
    for region in zones:
        label = str(region).strip()
        if not label:
            continue

        canonical = canonical_region_value(label)
        normalized: str | None = None
        if isinstance(canonical, (int, str)) and canonical in DEFAULT_REGION_METADATA:
            normalized = str(canonical)
        elif isinstance(canonical, str):
            secondary = canonical_region_value(canonical)
            if isinstance(secondary, (int, str)) and secondary in DEFAULT_REGION_METADATA:
                normalized = str(secondary)

        if not normalized:
            continue

        if normalized not in names:
            names.append(normalized)
    return names


def _encode_forecast_selection(iso_label: str, scenario_label: str) -> str:
    iso_text = str(iso_label).strip()
    scenario_text = str(scenario_label).strip()
    if iso_text and scenario_text:
        return f"{iso_text}::{scenario_text}"
    if scenario_text:
        return scenario_text
    return iso_text


def _decode_forecast_selection(value: Any) -> tuple[str | None, str | None]:
    if value in (None, ""):
        return (None, None)

    text = str(value).strip()
    if not text:
        return (None, None)

    for delimiter in ("::", "/", "|"):
        if delimiter in text:
            iso_part, _, scenario_part = text.partition(delimiter)
            iso_label = iso_part.strip() or None
            scenario_label = scenario_part.strip() or None
            return iso_label, scenario_label

    return (None, text)


@lru_cache(maxsize=1)
def _known_state_codes() -> set[str]:
    codes: set[str] = set()
    for entry in _iso_state_groups().values():
        states = entry.get("states")
        if isinstance(states, list):
            for state in states:
                if isinstance(state, str):
                    state_code = state.strip().upper()
                    if state_code:
                        codes.add(state_code)
    return codes


def _normalize_state_forecast_selection(
    selection: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, str]], set[tuple[str, str]]]:
    """Return state-level selections and ISO/scenario pairs from ``selection``."""

    state_codes = _known_state_codes()
    state_map: dict[str, dict[str, str]] = {}
    iso_pairs: set[tuple[str, str]] = set()

    if not isinstance(selection, Mapping):
        return state_map, iso_pairs

    for key, raw_value in selection.items():
        if raw_value in (None, ""):
            continue

        key_text = str(key).strip()
        if not key_text:
            continue

        value_text = str(raw_value).strip()
        if not value_text:
            continue

        iso_label, scenario_label = _decode_forecast_selection(value_text)
        key_upper = key_text.upper()

        if iso_label is None and key_upper not in state_codes:
            iso_label = key_text

        if scenario_label is None:
            continue

        iso_clean = str(iso_label).strip()
        scenario_clean = str(scenario_label).strip()
        if not scenario_clean:
            continue

        if iso_clean:
            LOGGER.debug("Looking for ISO: %s", iso_clean)
            iso_pairs.add((iso_clean, scenario_clean))

        if key_upper in state_codes and iso_clean:
            state_map[key_upper] = {"iso": iso_clean, "scenario": scenario_clean}

    return state_map, iso_pairs


def _canonical_forecast_selection(
    selection: Mapping[str, Any] | None,
) -> dict[str, str]:
    canonical: dict[str, str] = {}
    if not isinstance(selection, Mapping):
        return canonical

    state_codes = _known_state_codes()
    for key, value in selection.items():
        if value in (None, ""):
            continue
        key_text = str(key).strip()
        if not key_text:
            continue
        value_text = str(value).strip()
        if not value_text:
            continue

        key_upper = key_text.upper()
        if key_upper in state_codes:
            canonical[key_upper] = value_text
        else:
            canonical[key_text] = value_text
            token = normalize_iso_name(key_text)
            if token:
                canonical.setdefault(token, value_text)

    return canonical


def _normalize_state_codes(values: Iterable[Any]) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for entry in values:
        if entry in (None, ""):
            continue
        text = str(entry).strip()
        if not text:
            continue
        upper = text.upper()
        if upper in {"ALL", "ALL STATES", "ALL REGIONS"}:
            seen.clear()
            codes = []
            break
        if upper not in seen:
            seen.add(upper)
            codes.append(upper)
    return codes


def _states_from_config(config: Mapping[str, Any]) -> list[str]:
    raw_states = config.get("states")
    if isinstance(raw_states, Mapping):
        candidates = raw_states.keys()
    else:
        candidates = raw_states

    if isinstance(candidates, (list, tuple, set)):
        return _normalize_state_codes(candidates)
    if isinstance(candidates, str):
        return _normalize_state_codes([candidates])
    return []


def _iso_state_config_directory() -> Path:
    candidates = [
        INPUT_DIR / "regions",
        Path(__file__).resolve().parents[1] / "inputs" / "regions",
        Path(__file__).resolve().parents[1] / "input" / "regions",
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return candidates[-1]


@lru_cache(maxsize=1)
def _iso_state_groups() -> dict[str, dict[str, Any]]:
    """Return mapping of ISO tokens to metadata about covered states."""

    base_dir = _iso_state_config_directory()
    yaml_path = base_dir / "iso_state_zones.yaml"
    json_path = yaml_path.with_suffix(".json")

    payload: dict[str, Any] = {}
    if yaml_path.exists():
        if yaml is None:
            if json_path.exists():
                try:
                    payload = json.loads(json_path.read_text()) or {}
                except json.JSONDecodeError:
                    payload = {}
        else:
            try:
                payload = yaml.safe_load(yaml_path.read_text()) or {}
            except Exception:
                try:
                    payload = json.loads(json_path.read_text()) or {}
                except Exception:  # pragma: no cover - configuration error path
                    payload = {}
    elif json_path.exists():
        try:
            payload = json.loads(json_path.read_text()) or {}
        except json.JSONDecodeError:
            payload = {}

    result: dict[str, dict[str, Any]] = {}
    order_index = 0
    isos = payload.get("isos") if isinstance(payload, Mapping) else {}
    if isinstance(isos, Mapping):
        for iso_label, iso_data in isos.items():
            if not isinstance(iso_data, Mapping):
                continue
            states_section = iso_data.get("states")
            if not isinstance(states_section, Mapping):
                continue

            iso_display = str(iso_label).strip()
            iso_token = normalize_iso_name(iso_display) or normalize_token(iso_display)
            if not iso_display and not iso_token:
                continue

            states: list[str] = []
            for state_code in states_section.keys():
                state_text = str(state_code).strip().upper()
                if state_text:
                    states.append(state_text)
            if not states:
                continue

            states.sort()
            entry = {"label": iso_display or iso_token, "states": states, "order": order_index}
            order_index += 1
            if iso_display:
                result[iso_display] = entry
                result[iso_display.upper()] = entry
            if iso_token:
                result[iso_token] = entry

    return result


def _format_demand_curve_label(relative_path: Path) -> str:
    """Return a human-readable label for a demand curve path."""

    parts = list(relative_path.parts)
    display_parts: list[str] = []
    for index, part in enumerate(parts):
        text = part
        if index == len(parts) - 1:
            text = Path(part).stem
        text = text.replace("_", " ").strip()
        if not text:
            continue
        display_parts.append(text.title())
    if display_parts:
        return " / ".join(display_parts)
    stem = relative_path.stem if hasattr(relative_path, "stem") else str(relative_path)
    return stem.replace("_", " ").title()


def _load_demand_curve_catalog() -> dict[str, dict[str, Any]]:
    """Load pre-saved demand curve profiles from the electricity input directory."""

    base_dir = _DEMAND_CURVE_BASE_DIR
    if not base_dir.exists():
        return {}

    search_dirs = []
    demand_subdir = base_dir / "demand_curves"
    if demand_subdir.exists():
        search_dirs.append(demand_subdir)
    search_dirs.append(base_dir)

    catalog: dict[str, dict[str, Any]] = {}
    for directory in search_dirs:
        if not directory.exists():
            continue
        iterator = directory.rglob("*.csv") if directory != base_dir else directory.glob("*.csv")
        for path in sorted(iterator):
            if path.is_dir():
                continue
            try:
                relative_path = path.relative_to(base_dir)
            except ValueError:
                relative_path = Path(path.name)
            key = relative_path.as_posix()
            if key in catalog:
                continue
            try:
                raw = pd.read_csv(path)
            except Exception:  # pragma: no cover - defensive
                LOGGER.debug("Skipping demand curve file %s; unable to load CSV.", path, exc_info=True)
                continue
            if not isinstance(raw, pd.DataFrame) or raw.empty:
                continue

            column_map = {str(col).strip().lower(): col for col in raw.columns}
            year_col = column_map.get("year")
            if year_col is None:
                continue

            demand_col: str | None = None
            for candidate in ("demand_mwh", "demand", "load_mwh", "load"):
                if candidate in column_map:
                    demand_col = column_map[candidate]
                    break
            if demand_col is None:
                continue

            normalized = raw[[year_col, demand_col]].copy()
            normalized.columns = ["year", "demand_mwh"]
            normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce")
            normalized["demand_mwh"] = pd.to_numeric(normalized["demand_mwh"], errors="coerce")
            normalized = normalized.dropna(subset=["year", "demand_mwh"])
            if normalized.empty:
                continue
            normalized["year"] = normalized["year"].astype(int)
            normalized["demand_mwh"] = normalized["demand_mwh"].astype(float)
            normalized = normalized.drop_duplicates(subset=["year"], keep="last")
            normalized = normalized.sort_values("year").reset_index(drop=True)

            catalog[key] = {
                "label": _format_demand_curve_label(relative_path),
                "path": path,
                "data": normalized,
            }

    return catalog


def _build_demand_output_frame(
    years: Sequence[int],
    regions: Sequence[Any] | Mapping[Any, Any] | None,
    demand_module: Mapping[str, Any] | None,
    *,
    fallback_regions: Sequence[Any] | None = None,
) -> pd.DataFrame | None:
    """Construct a DataFrame of demand by region-year using selected curves."""

    normalized_years: list[int] = []
    for year in years or []:
        try:
            normalized_years.append(int(year))
        except (TypeError, ValueError):
            continue
    normalized_years = sorted(dict.fromkeys(normalized_years))
    if not normalized_years:
        return None

    def _normalize_regions(values: Sequence[Any] | None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        if not values:
            return normalized
        for entry in values:
            if entry in (None, ""):
                continue
            resolved_value = canonical_region_value(entry)
            if isinstance(resolved_value, bool):
                resolved_text = str(int(resolved_value))
            elif isinstance(resolved_value, (int, float)):
                resolved_text = str(int(resolved_value))
            elif isinstance(resolved_value, str):
                resolved_text = resolved_value.strip()
            else:
                resolved_text = str(entry).strip()
            if not resolved_text:
                continue
            lowered = resolved_text.lower()
            if lowered in {"all", _ALL_REGIONS_LABEL.lower(), "default", "system"}:
                continue
            canonical = normalize_region(resolved_text)
            if not canonical:
                canonical = resolved_text.strip().upper()
            if canonical not in seen:
                normalized.append(canonical)
                seen.add(canonical)
        return normalized

    region_values: Sequence[Any] | None
    if isinstance(regions, Mapping):
        region_values = list(regions.keys())
    else:
        region_values = regions

    normalized_regions = _normalize_regions(region_values)
    if not normalized_regions and fallback_regions is not None:
        normalized_regions = _normalize_regions(fallback_regions)
    if not normalized_regions:
        normalized_regions = _normalize_regions(list(DEFAULT_REGION_METADATA))
    if not normalized_regions:
        return None

    selection_map: dict[str, str] = {}
    iso_scenario_lookup: dict[str, tuple[str, pd.DataFrame]] = {}
    enabled_flag = True
    if isinstance(demand_module, Mapping):
        enabled_flag = bool(demand_module.get("enabled", True))
        curve_map = demand_module.get("curve_by_region")
        if isinstance(curve_map, Mapping):
            for region_key, curve_key in curve_map.items():
                if not curve_key:
                    continue
                resolved_key = canonical_region_value(region_key)
                if isinstance(resolved_key, bool):
                    normalized_key = str(int(resolved_key))
                elif isinstance(resolved_key, (int, float)):
                    normalized_key = str(int(resolved_key))
                elif isinstance(resolved_key, str):
                    normalized_key = resolved_key.strip()
                else:
                    normalized_key = str(region_key).strip()
                if not normalized_key:
                    continue
                lowered = normalized_key.lower()
                if lowered in {"all", _ALL_REGIONS_LABEL.lower(), "default", "system"}:
                    continue
                selection_map[normalized_key] = str(curve_key)
                selection_map.setdefault(lowered, str(curve_key))

        iso_map = demand_module.get("load_forecasts")
        if isinstance(iso_map, Mapping):
            state_selection_map, iso_pairs = _normalize_state_forecast_selection(iso_map)
            processed_pairs: set[tuple[str, str]] = set()

            for iso_label, scenario_name in sorted(iso_pairs):
                iso_clean = str(iso_label).strip()
                scenario_clean = str(scenario_name).strip()
                if not iso_clean or not scenario_clean:
                    continue

                iso_token = normalize_iso_name(iso_clean) or normalize_token(iso_clean) or iso_clean
                pair_key = (iso_token, scenario_clean)
                if pair_key in processed_pairs:
                    continue

                scenario_frame = _load_iso_scenario_frame(
                    _cached_input_root(), iso_clean, scenario_clean
                )
                if scenario_frame.empty:
                    LOGGER.debug(
                        "Unable to load ISO scenario %s/%s", iso_clean, scenario_clean, exc_info=True
                    )
                    processed_pairs.add(pair_key)
                    continue

                normalized_frame = _scenario_frame_to_demand(scenario_frame)
                if normalized_frame.empty:
                    processed_pairs.add(pair_key)
                    continue

                frames: dict[str, pd.DataFrame] = {}
                for region_id, zone_frame in normalized_frame.groupby("region"):
                    working = zone_frame.loc[:, ["year", "demand_mwh"]]
                    working = working.dropna(subset=["year", "demand_mwh"])
                    if working.empty:
                        continue
                    working["year"] = working["year"].astype(int)
                    frames[str(region_id)] = working.loc[:, ["year", "demand_mwh"]]

                if frames:
                    scenario_key = _encode_forecast_selection(iso_clean, scenario_clean)
                    for region_id, frame in frames.items():
                        variants = {
                            str(region_id),
                            str(region_id).replace("-", "_"),
                            str(region_id).lower(),
                        }
                        for variant in variants:
                            if variant:
                                iso_scenario_lookup.setdefault(variant, (scenario_key, frame))

                processed_pairs.add(pair_key)

            if state_selection_map:
                try:
                    from engine.data_loaders.load_forecasts import (
                        load_demand_forecasts_selection,
                    )
                except Exception:  # pragma: no cover - optional dependency
                    LOGGER.debug("State demand loader import failed", exc_info=True)
                else:
                    state_frames = load_demand_forecasts_selection(state_selection_map)
                    aggregated = pd.DataFrame()

                    if isinstance(state_frames, Mapping):
                        aggregated = pd.DataFrame(state_frames)
                        if not aggregated.empty:
                            aggregated = aggregated.stack().reset_index()
                            aggregated.columns = ["state", "year", "demand_mwh"]
                    elif isinstance(state_frames, pd.DataFrame):
                        working = state_frames.copy()
                        if not working.empty:
                            column_lookup = {
                                str(column).strip().lower(): column for column in working.columns
                            }
                            state_col = column_lookup.get("state")
                            year_col = column_lookup.get("year")
                            load_col = (
                                column_lookup.get("demand_mwh")
                                or column_lookup.get("demand")
                                or column_lookup.get("load_mwh")
                                or column_lookup.get("load")
                                or column_lookup.get("load_gwh")
                            )
                            if state_col and year_col and load_col:
                                working = working.loc[:, [state_col, year_col, load_col]].copy()
                                working.columns = ["state", "year", "load_value"]
                                working["state"] = working["state"].astype(str).str.upper()
                                working["year"] = pd.to_numeric(
                                    working["year"], errors="coerce"
                                ).astype("Int64")
                                multiplier = 1_000.0 if "gwh" in load_col.lower() else 1.0
                                working["demand_mwh"] = pd.to_numeric(
                                    working["load_value"], errors="coerce"
                                ) * multiplier
                                working = working.dropna(
                                    subset=["state", "year", "demand_mwh"]
                                )
                                if not working.empty:
                                    working["year"] = working["year"].astype(int)
                                    aggregated = (
                                        working.groupby(["state", "year"], as_index=False)[
                                            "demand_mwh"
                                        ].sum()
                                    )

                    if not aggregated.empty:
                        aggregated["year"] = pd.to_numeric(
                            aggregated["year"], errors="coerce"
                        ).astype("Int64")
                        aggregated["demand_mwh"] = pd.to_numeric(
                            aggregated["demand_mwh"], errors="coerce"
                        )
                        aggregated = aggregated.dropna(
                            subset=["state", "year", "demand_mwh"]
                        )
                        aggregated["state"] = aggregated["state"].astype(str).str.upper()
                        aggregated["year"] = aggregated["year"].astype(int)

                        for state_code, state_rows in aggregated.groupby("state"):
                            config = state_selection_map.get(state_code)
                            if not config:
                                continue
                            scenario_key = _encode_forecast_selection(
                                config.get("iso", state_code),
                                config.get("scenario", ""),
                            )
                            if not scenario_key:
                                continue
                            state_frame_values = state_rows.loc[:, ["year", "demand_mwh"]].copy()
                            state_frame_values = state_frame_values.sort_values("year").reset_index(
                                drop=True
                            )
                            variants = {state_code, state_code.lower()}
                            for variant in variants:
                                if variant:
                                    iso_scenario_lookup[variant] = (
                                        scenario_key,
                                        state_frame_values,
                                    )

    if not enabled_flag:
        selection_map = {}

    catalog = _load_demand_curve_catalog()
    records: list[dict[str, Any]] = []

    for region in normalized_regions:
        resolved_region = canonical_region_value(region)
        if isinstance(resolved_region, bool):
            region_code = str(int(resolved_region))
        elif isinstance(resolved_region, (int, float)):
            region_code = str(int(resolved_region))
        elif isinstance(resolved_region, str):
            region_code = resolved_region.strip()
        else:
            region_code = str(region).strip()
        if not region_code:
            continue
        lookup_keys = [region_code, region_code.lower()]
        display_label = canonical_region_label(region_code)
        if display_label:
            lookup_keys.extend([display_label, display_label.lower()])
        if isinstance(region, str):
            lookup_keys.extend([region, region.lower(), region.strip().lower()])
        scenario_entry = None
        if iso_scenario_lookup:
            scenario_entry = (
                iso_scenario_lookup.get(region_code)
                or iso_scenario_lookup.get(region_code.replace("-", "_"))
                or iso_scenario_lookup.get(region_code.lower())
            )

        selected_curve: str | None = None
        entry = None
        values_by_year: dict[int, float] = {}
        curve_label = _FALLBACK_DEMAND_LABEL

        if scenario_entry is not None:
            scenario_key, scenario_frame = scenario_entry
            if isinstance(scenario_frame, pd.DataFrame) and not scenario_frame.empty:
                for _, row in scenario_frame.iterrows():
                    try:
                        year_val = int(row["year"])
                        demand_val = float(row["demand_mwh"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    values_by_year[year_val] = demand_val
            selected_curve = scenario_key
            _, label_candidate = _decode_forecast_selection(scenario_key)
            if label_candidate:
                curve_label = label_candidate.replace("_", " ")
            else:
                curve_label = scenario_key.split("/", 1)[-1].replace("_", " ")
        else:
            for key in lookup_keys:
                candidate = selection_map.get(key)
                if candidate:
                    selected_curve = candidate
                    break
            entry = catalog.get(selected_curve) if selected_curve else None
            if selected_curve == _DEMAND_CURVE_FLAT_KEY:
                for year in normalized_years:
                    values_by_year[year] = _FLAT_DEMAND_MWH
                curve_label = _FLAT_DEMAND_LABEL
            elif entry is not None:
                data = entry.get("data")
                if isinstance(data, pd.DataFrame) and not data.empty:
                    for _, row in data.iterrows():
                        try:
                            year_val = int(row["year"])
                            demand_val = float(row["demand_mwh"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        values_by_year[year_val] = demand_val
                curve_label = entry.get("label", _FALLBACK_DEMAND_LABEL)
            else:
                if selected_curve:
                    LOGGER.warning(
                        "Demand curve '%s' for region %s not found; defaulting to %.1f MWh.",
                        selected_curve,
                        region,
                        _FALLBACK_DEMAND_MWH,
                    )
                selected_curve = None

        curve_key = selected_curve if selected_curve else _DEMAND_CURVE_FALLBACK_KEY
        if not selected_curve:
            curve_label = _FALLBACK_DEMAND_LABEL

        missing_years: list[int] = []
        for year in normalized_years:
            demand_value = values_by_year.get(year)
            if demand_value is None:
                demand_value = _FALLBACK_DEMAND_MWH
                missing_years.append(int(year))
            records.append(
                {
                    "region": region_code,
                    "year": int(year),
                    "demand_mwh": float(demand_value),
                    "curve_key": curve_key,
                    "curve_label": curve_label,
                }
            )

        if missing_years:
            LOGGER.warning(
                "Demand curve '%s' provided no data for region %s in years %s; "
                "defaulted to %.1f MWh.",
                curve_key,
                region_code,
                ", ".join(str(year) for year in missing_years),
                _FALLBACK_DEMAND_MWH,
            )

    if not records:
        return None

    return pd.DataFrame(records)


__all__ = [
    "_ALL_REGIONS_LABEL",
    "_DEMAND_CURVE_BASE_DIR",
    "_DEMAND_CURVE_FLAT_KEY",
    "_DEMAND_CURVE_FALLBACK_KEY",
    "_FALLBACK_DEMAND_LABEL",
    "_FALLBACK_DEMAND_MWH",
    "_FLAT_DEMAND_LABEL",
    "_FLAT_DEMAND_MWH",
    "_ScenarioSelection",
    "_FrameProxy",
    "_canonical_forecast_selection",
    "_coerce_manifest_list",
    "_decode_forecast_selection",
    "_default_scenario_manifests",
    "_demand_frame_from_manifests",
    "_discovered_region_names",
    "_encode_forecast_selection",
    "_extract_record_value",
    "_format_demand_curve_label",
    "_iso_state_groups",
    "_known_state_codes",
    "_load_demand_curve_catalog",
    "_load_demand_frame",
    "_manifests_from_selection",
    "_forecast_bundles_from_manifests",
    "_forecast_bundles_from_selection",
    "_normalize_state_codes",
    "_normalize_state_forecast_selection",
    "_ordered_scenarios",
    "_select_scenario_manifests",
    "_states_from_config",
    "_summarize_forecasts",
    "select_forecast_bundles",
]
