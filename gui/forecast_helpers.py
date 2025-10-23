"""Helper utilities for discovering and caching forecast metadata."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from common.data_access.load_forecasts import repo_load_forecasts
from engine.normalization import normalize_iso_name, normalize_token
from engine.settings import input_root

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency wrapper
    import regions.load_forecasts as regions_forecasts
except ModuleNotFoundError:  # pragma: no cover - optional dependency wrapper
    regions_forecasts = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency wrapper
    from engine.data_loaders.load_forecasts import (
        available_iso_scenarios as _available_iso_scenarios,
    )
except Exception:  # pragma: no cover - compatibility guard
    _available_iso_scenarios = None  # type: ignore[assignment]


__all__ = [
    "_available_iso_scenarios_map",
    "_cached_forecast_frame",
    "_cached_input_root",
    "_cached_iso_scenario_map",
    "_call_loader_variants",
    "_clear_forecast_cache",
    "_coerce_base_path",
    "_discover_bundle_records",
    "_discover_iso_scenarios",
    "discover_iso_scenarios",
    "_discover_iso_zones",
    "_load_iso_scenario_frame",
    "_iter_strings",
    "_regions_available_iso_scenarios",
    "_regions_available_zones",
    "_regions_load_forecasts_frame",
    "_regions_scenario_frame",
    "_regions_scenario_index",
    "_regions_zones_for",
    "_resolve_forecast_base_path",
    "_scenario_frame_subset",
    "forecast_frame_error",
]


def _base_path_key(base_path: str | os.PathLike[str] | None) -> str:
    """Return a stable dictionary key for ``base_path`` values."""

    if base_path is None:
        return "__default__"

    try:
        return os.fspath(base_path)
    except TypeError:  # pragma: no cover - defensive guard
        return str(base_path)


_FORECAST_FRAME_ERRORS: dict[str, str] = {}

_REGION_ISO_HINTS: tuple[tuple[str, str], ...] = (
    ("ISO-NE", "ISO-NE"),
    ("ISO_NE", "ISO-NE"),
    ("NYISO", "NYISO"),
    ("PJM", "PJM"),
    ("MISO", "MISO"),
    ("SPP", "SPP"),
    ("ERCOT", "ERCOT"),
    ("CAISO", "CAISO"),
    ("FRCC", "SOUTHEAST"),
    ("SOCO", "SOUTHEAST"),
    ("TVA", "SOUTHEAST"),
    ("DUK", "SOUTHEAST"),
    ("SANTEE", "SOUTHEAST"),
    ("SCEG", "SOUTHEAST"),
    ("ENTERGY", "SOUTHEAST"),
    ("FPL", "SOUTHEAST"),
    ("FPC", "SOUTHEAST"),
    ("TECO", "SOUTHEAST"),
    ("JEA", "SOUTHEAST"),
    ("FMPA", "SOUTHEAST"),
    ("ONTARIO", "CANADA"),
    ("QUEBEC", "CANADA"),
    ("MARITIMES", "CANADA"),
    ("CANADA", "CANADA"),
)

_ISO_DISPLAY: Mapping[str, str] = {
    "iso_ne": "ISO-NE",
    "nyiso": "NYISO",
    "pjm": "PJM",
    "miso": "MISO",
    "spp": "SPP",
    "ercot": "ERCOT",
    "caiso": "CAISO",
    "southeast": "SOUTHEAST",
    "canada": "CANADA",
}


def _infer_iso_from_region(region: Any) -> str:
    token = str(region or "").strip().upper()
    if not token:
        return ""

    for prefix, iso_label in _REGION_ISO_HINTS:
        if token.startswith(prefix):
            return iso_label

    prefix = token.split("_", 1)[0]
    normalized = normalize_iso_name(prefix)
    if normalized:
        return _ISO_DISPLAY.get(normalized, normalized.upper())

    return prefix or ""


def _regions_load_forecasts_frame(*, base_path: str | None) -> pd.DataFrame:
    """Return the consolidated load forecast frame for ``base_path``."""

    root = Path(_coerce_base_path(base_path))

    try:
        frame = repo_load_forecasts(root)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Load forecast CSV not found: {root}") from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Failed to load load forecasts from {root}: {exc}") from exc

    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    if frame.empty:
        columns = [
            "iso",
            "region_id",
            "state",
            "zone",
            "scenario",
            "year",
            "load_gwh",
            "load_mwh",
            "state_or_province",
            "scenario_name",
        ]
        return pd.DataFrame(columns=columns)

    working = frame.copy()
    region_series = (
        working["region_id"] if "region_id" in working.columns else pd.Series("", index=working.index)
    )
    working["region_id"] = region_series.astype("string").str.strip().str.upper()

    scenario_name_series = (
        working["scenario_name"]
        if "scenario_name" in working.columns
        else pd.Series("", index=working.index)
    )
    working["scenario_name"] = scenario_name_series.astype("string").str.strip()

    scenario_series = (
        working["scenario"] if "scenario" in working.columns else pd.Series("", index=working.index)
    )
    working["scenario"] = scenario_series.astype("string").str.strip()

    state_series = (
        working["state_or_province"]
        if "state_or_province" in working.columns
        else pd.Series("", index=working.index)
    )
    working["state_or_province"] = state_series.astype("string").str.strip()
    working["state"] = working["state_or_province"].fillna("").astype(str).str.upper()
    working["zone"] = working["region_id"].astype(str)
    working["iso"] = working["region_id"].map(_infer_iso_from_region).astype("string")
    working["year"] = pd.to_numeric(working.get("year"), errors="coerce").astype("Int64")
    working["load_gwh"] = pd.to_numeric(working.get("load_gwh"), errors="coerce")
    working["load_mwh"] = pd.to_numeric(working.get("load_mwh"), errors="coerce")

    columns = [
        "iso",
        "region_id",
        "state",
        "zone",
        "scenario",
        "year",
        "load_gwh",
        "load_mwh",
        "state_or_province",
        "scenario_name",
    ]
    for column in columns:
        if column not in working.columns:
            if column in {"load_gwh", "load_mwh"}:
                working[column] = pd.Series(dtype="float64")
            elif column == "year":
                working[column] = pd.Series(dtype="Int64")
            else:
                working[column] = pd.Series(dtype="object")

    return working.loc[:, columns]


def _regions_scenario_index(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return a mapping of ISO → scenarios derived from ``frame``."""

    if frame.empty:
        return {}

    if "iso" not in frame.columns or "scenario" not in frame.columns:
        return {}

    mapping: dict[str, set[str]] = {}
    iso_series = frame["iso"].astype(str)
    scenario_series = frame["scenario"].astype(str)
    for iso_value, scenario_value in zip(iso_series, scenario_series):
        iso_label = iso_value.strip()
        scenario_label = scenario_value.strip()
        if not iso_label or not scenario_label:
            continue
        mapping.setdefault(iso_label, set()).add(scenario_label)
        normalized_iso = normalize_iso_name(iso_label)
        if normalized_iso and normalized_iso != iso_label:
            mapping.setdefault(normalized_iso, set()).add(scenario_label)

    return {key: sorted(values) for key, values in mapping.items()}


def _regions_scenario_frame(
    frame: pd.DataFrame, iso: str, scenario: str
) -> pd.DataFrame:
    """Return rows for ``iso``/``scenario`` from ``frame``."""

    if frame.empty:
        return pd.DataFrame(columns=frame.columns)

    iso_series = frame.get("iso")
    scenario_series = frame.get("scenario")
    if iso_series is None or scenario_series is None:
        return pd.DataFrame(columns=frame.columns)

    iso_label = str(iso).strip()
    scenario_label = str(scenario).strip()
    if not iso_label or not scenario_label:
        return pd.DataFrame(columns=frame.columns)

    iso_lower = iso_series.astype(str).str.lower()
    scenario_lower = scenario_series.astype(str).str.lower()
    mask = (iso_lower == iso_label.lower()) & (scenario_lower == scenario_label.lower())
    subset = frame.loc[mask]
    if subset.empty:
        iso_token = normalize_iso_name(iso_label)
        if iso_token:
            mask = (iso_lower == iso_token.lower()) & (
                scenario_lower == scenario_label.lower()
            )
            subset = frame.loc[mask]

    if subset.empty:
        return pd.DataFrame(columns=frame.columns)

    cleaned = subset.copy()
    if "region_id" not in cleaned.columns and "zone" in cleaned.columns:
        cleaned["region_id"] = cleaned["zone"].astype(str)
    elif "zone" not in cleaned.columns and "region_id" in cleaned.columns:
        cleaned["zone"] = cleaned["region_id"].astype(str)

    return cleaned


def _regions_zones_for(frame: pd.DataFrame, iso: str, scenario: str) -> list[str]:
    """Return the sorted zones discovered for ``iso``/``scenario``."""

    subset = _regions_scenario_frame(frame, iso, scenario)
    if subset.empty:
        return []

    zone_labels: set[str] = set()
    if "zone" in subset.columns:
        zone_labels.update(
            {
                str(value).strip()
                for value in subset["zone"].dropna().astype(str)
                if str(value).strip()
            }
        )
    if not zone_labels and "region_id" in subset.columns:
        zone_labels.update(
            {
                str(value).strip()
                for value in subset["region_id"].dropna().astype(str)
                if str(value).strip()
            }
        )

    return sorted(zone_labels)


def _regions_available_iso_scenarios(
    base_path: str | os.PathLike[str] | None = None,
) -> dict[str, list[str]]:
    try:
        frame = _regions_load_forecasts_frame(base_path=base_path)
    except RuntimeError:
        return {}
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug("Unable to obtain available ISO scenarios", exc_info=True)
        return {}

    frame_obj = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    return _regions_scenario_index(frame_obj)


def _regions_available_zones(
    base_path: str | os.PathLike[str] | None,
    iso: str,
    scenario: str,
) -> list[str]:
    try:
        frame = _regions_load_forecasts_frame(base_path=base_path)
    except RuntimeError:
        frame = pd.DataFrame()
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug(
            "Unable to load cached forecast zones for %s/%s", iso, scenario, exc_info=True
        )
        frame = pd.DataFrame()

    if isinstance(frame, pd.DataFrame):
        frame_obj = frame
    else:
        frame_obj = pd.DataFrame(frame)

    zones = _regions_zones_for(frame_obj, iso, scenario)
    if zones:
        return zones

    return _discover_iso_zones(base_path, iso, scenario)


def _available_iso_scenarios_map(base_path: str | None) -> Mapping[str, list[str]]:
    """Return a mapping of ISO → scenarios for ``base_path``."""

    try:
        mapping = _regions_available_iso_scenarios(base_path=base_path)
    except Exception:
        return {}

    return {str(key): list(map(str, values)) for key, values in mapping.items()}


def _discover_iso_scenarios(base_path: str | None, iso: str) -> list[str]:
    """Return the list of scenarios available for ``iso``."""

    mapping = _regions_available_iso_scenarios(base_path)
    iso_label = str(iso).strip()
    if not iso_label:
        return []

    scenarios = mapping.get(iso_label)
    if scenarios:
        return list(scenarios)

    normalized = normalize_iso_name(iso_label)
    if not normalized:
        return []

    for key, values in mapping.items():
        if normalize_iso_name(key) == normalized and values:
            return list(values)

    return []


def discover_iso_scenarios(base_path: str | None, iso: str) -> list[str]:
    """Public wrapper for :func:`_discover_iso_scenarios`."""

    return _discover_iso_scenarios(base_path, iso)


def _discover_iso_zones(base_path: str | None, iso: str, scenario: str) -> list[str]:
    """Return the list of zones for ``iso``/``scenario``."""

    iso_label = str(iso).strip()
    scenario_label = str(scenario).strip()
    if not iso_label or not scenario_label:
        return []

    iso_tokens = {iso_label.lower()}
    normalized_iso = normalize_iso_name(iso_label)
    if normalized_iso:
        iso_tokens.add(str(normalized_iso).lower())

    scenario_tokens = {scenario_label.lower()}
    normalized_scenario = normalize_token(scenario_label)
    if normalized_scenario:
        scenario_tokens.add(str(normalized_scenario).lower())

    bundles = _discover_bundle_records(base_path)
    if not bundles:
        return []

    def _bundle_value(candidate: Any, attribute: str) -> Any:
        value = getattr(candidate, attribute, None)
        if value is None and isinstance(candidate, Mapping):
            try:
                value = candidate[attribute]
            except (KeyError, TypeError):
                value = None
        return value

    zones: set[str] = set()
    for bundle in bundles:
        bundle_iso_value = _bundle_value(bundle, "iso")
        bundle_iso = str(bundle_iso_value or "").strip()
        if not bundle_iso:
            continue

        bundle_iso_tokens = {bundle_iso.lower()}
        bundle_iso_normalized = normalize_iso_name(bundle_iso)
        if bundle_iso_normalized:
            bundle_iso_tokens.add(str(bundle_iso_normalized).lower())

        if iso_tokens.isdisjoint(bundle_iso_tokens):
            continue

        bundle_scenarios = []
        for attribute in ("scenario", "manifest_name"):
            value = _bundle_value(bundle, attribute)
            if value is not None:
                text = str(value).strip()
                if text:
                    bundle_scenarios.append(text)

        if not bundle_scenarios:
            continue

        scenario_match = False
        for candidate in bundle_scenarios:
            candidate_tokens = {candidate.lower()}
            normalized_candidate = normalize_token(candidate)
            if normalized_candidate:
                candidate_tokens.add(str(normalized_candidate).lower())
            if scenario_tokens.intersection(candidate_tokens):
                scenario_match = True
                break

        if not scenario_match:
            continue

        zone_candidates: Iterable[str] = []
        regions_attr = getattr(bundle, "regions", None)
        if callable(regions_attr):
            try:
                zone_candidates = list(regions_attr())  # type: ignore[call-arg]
            except TypeError:
                zone_candidates = []
        elif isinstance(regions_attr, Iterable) and not isinstance(
            regions_attr, (str, bytes, bytearray)
        ):
            zone_candidates = regions_attr

        if not zone_candidates:
            if isinstance(bundle, Mapping):
                zones_value = bundle.get("zones")
                if isinstance(zones_value, Iterable) and not isinstance(
                    zones_value, (str, bytes, bytearray)
                ):
                    zone_candidates = zones_value

        if not zone_candidates:
            frames = getattr(bundle, "frames", None)
            if isinstance(frames, Mapping):
                zone_candidates = frames.keys()

        if not zone_candidates:
            source_files = getattr(bundle, "source_files", None)
            if isinstance(source_files, Mapping):
                zone_candidates = source_files.keys()

        for zone in zone_candidates:
            zone_label = str(zone).strip()
            if zone_label:
                zones.add(zone_label)

    return sorted(zones)


def _load_iso_scenario_frame(base_path: str | None, iso: str, scenario: str) -> pd.DataFrame:
    """Return a normalized frame for ``iso``/``scenario``."""

    try:
        frame = _regions_load_forecasts_frame(base_path=base_path)
    except RuntimeError:
        frame = pd.DataFrame()
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug(
            "Unable to load ISO scenario %s/%s", iso, scenario, exc_info=True
        )
        frame = pd.DataFrame()

    frame_obj = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    working = _regions_scenario_frame(frame_obj, iso, scenario)
    if working.empty:
        return pd.DataFrame(
            columns=[
                "iso",
                "region_id",
                "state",
                "zone",
                "scenario",
                "year",
                "load_gwh",
                "load_mwh",
                "state_or_province",
                "scenario_name",
            ]
        )

    original_columns = set(working.columns)

    if "region_id" not in original_columns and "zone" in working.columns:
        working["region_id"] = working["zone"].astype(str)
    if "zone" not in original_columns and "region_id" in working.columns:
        working["zone"] = working["region_id"].astype(str)
    if "state" not in original_columns:
        state_series = working.get("state_or_province", "")
        working["state"] = pd.Series(state_series).astype(str)

    defaults: dict[str, pd.Series] = {
        "iso": pd.Series(dtype="object"),
        "region_id": pd.Series(dtype="object"),
        "state": pd.Series(dtype="object"),
        "zone": pd.Series(dtype="object"),
        "scenario": pd.Series(dtype="object"),
        "year": pd.Series(dtype="int64"),
        "load_gwh": pd.Series(dtype="float64"),
        "load_mwh": pd.Series(dtype="float64"),
        "state_or_province": pd.Series(dtype="object"),
        "scenario_name": pd.Series(dtype="object"),
    }
    for column, series in defaults.items():
        if column not in working.columns:
            working[column] = series

    iso_series = working["iso"].fillna(iso).astype(str)
    working["iso"] = iso_series.mask(iso_series.str.strip() == "", str(iso))
    scenario_series = working["scenario"].fillna(scenario).astype(str)
    working["scenario"] = scenario_series.mask(
        scenario_series.str.strip() == "", str(scenario)
    )

    working["state"] = working["state"].astype(str).str.strip().str.upper()
    working["state_or_province"] = (
        working["state_or_province"].astype("string").str.strip().fillna("")
    )
    working["load_mwh"] = pd.to_numeric(working["load_mwh"], errors="coerce")

    columns = [
        "iso",
        "region_id",
        "state",
        "zone",
        "scenario",
        "year",
        "load_gwh",
        "load_mwh",
        "state_or_province",
        "scenario_name",
    ]
    return working.loc[:, columns]


def _coerce_base_path(base_path: str | os.PathLike[str] | None) -> str:
    """Return ``base_path`` as a string, defaulting to the configured input root."""

    if base_path is None:
        return _resolve_forecast_base_path()
    return str(base_path)


def _call_loader_variants(
    loader: Callable[..., Any],
    variants: Sequence[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    default: Any,
) -> Any:
    """Invoke ``loader`` with candidate ``variants`` returning ``default`` on failure."""

    for args, kwargs in variants:
        try:
            return loader(*args, **kwargs)
        except TypeError:
            continue
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug(
                "Loader %s failed for args=%s kwargs=%s",
                getattr(loader, "__name__", loader),
                args,
                kwargs,
                exc_info=True,
            )
            return default
    return default


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


def _discover_bundle_records(base_path: str | None) -> list[Any]:
    """Return raw bundle records discovered for ``base_path``."""

    loader = _available_iso_scenarios
    if loader is None:
        return []

    root = _coerce_base_path(base_path)
    try:
        manifests = loader(base_path=root)
    except TypeError:
        manifests = loader(root)
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug("available_iso_scenarios loader failed", exc_info=True)
        manifests = []

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for manifest in manifests or []:
        iso_value = getattr(manifest, "iso", None)
        scenario_value = getattr(manifest, "scenario", None)
        zone_value = getattr(manifest, "zone", None)
        path_value = getattr(manifest, "path", None)
        if not iso_value or not scenario_value:
            continue
        key = (str(iso_value), str(scenario_value))
        entry = records.setdefault(
            key,
            {
                "iso": str(iso_value),
                "scenario": str(scenario_value),
                "zones": set(),
                "path": str(path_value) if path_value is not None else None,
            },
        )
        if zone_value:
            entry["zones"].add(str(zone_value))

    normalized: list[dict[str, Any]] = []
    for entry in records.values():
        zones = sorted(entry.pop("zones")) if entry.get("zones") else []
        normalized.append({**entry, "zones": zones})

    return normalized


def _scenario_frame_subset(
    frame: pd.DataFrame,
    iso: str,
    scenario: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=frame.columns)

    iso_series = frame.get("iso")
    scenario_series = frame.get("scenario")
    if iso_series is None or scenario_series is None:
        return pd.DataFrame(columns=frame.columns)

    iso_lower = str(iso).lower()
    scenario_lower = str(scenario).lower()
    mask = (
        iso_series.astype(str).str.lower() == iso_lower
    ) & (scenario_series.astype(str).str.lower() == scenario_lower)
    subset = frame.loc[mask]
    if subset.empty:
        iso_token = normalize_iso_name(str(iso))
        if iso_token and iso_token.lower() != iso_lower:
            mask = (
                iso_series.astype(str).str.lower() == iso_token.lower()
            ) & (scenario_series.astype(str).str.lower() == scenario_lower)
            subset = frame.loc[mask]
    if subset.empty:
        return subset

    cleaned = subset.copy()
    if "region_id" not in cleaned.columns and "zone" in cleaned.columns:
        cleaned["region_id"] = cleaned["zone"].astype(str)
    elif "zone" not in cleaned.columns and "region_id" in cleaned.columns:
        cleaned["zone"] = cleaned["region_id"].astype(str)
    return cleaned


@lru_cache(maxsize=1)
def _cached_input_root() -> str:
    """Return the configured input root as a cached string path."""

    return str(input_root())


@lru_cache(maxsize=1)
def _cached_forecast_frame(base_path: str | None) -> pd.DataFrame:
    """Return cached façade forecast table for ``base_path``."""

    columns = [
        "iso",
        "region_id",
        "state",
        "zone",
        "scenario",
        "year",
        "load_gwh",
        "load_mwh",
        "state_or_province",
        "scenario_name",
    ]
    key = _base_path_key(base_path)
    try:
        frame = _regions_load_forecasts_frame(base_path=base_path)
    except RuntimeError as exc:
        _FORECAST_FRAME_ERRORS[key] = str(exc)
        return pd.DataFrame(columns=columns)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.debug("Unable to load façade forecast table", exc_info=True)
        message = str(exc).strip() or "Unable to load load forecasts; see logs for details."
        _FORECAST_FRAME_ERRORS[key] = message
        return pd.DataFrame(columns=columns)

    _FORECAST_FRAME_ERRORS.pop(key, None)
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    return pd.DataFrame(frame, columns=columns)


@lru_cache(maxsize=1)
def _cached_iso_scenario_map(base_path: str | None) -> dict[str, list[str]]:
    """Return cached ISO → scenarios mapping for ``base_path``."""

    discovered = _regions_available_iso_scenarios(base_path=base_path)
    mapping: dict[str, list[str]] = {}
    for iso_key, scenarios in discovered.items():
        entries = [str(value) for value in scenarios]
        mapping[str(iso_key)] = sorted(entries)
    return mapping


def _clear_forecast_cache() -> None:
    """Clear façade-backed forecast caches."""

    _FORECAST_FRAME_ERRORS.clear()
    _cached_forecast_frame.cache_clear()
    _cached_iso_scenario_map.cache_clear()

    try:
        repo_load_forecasts.cache_clear()
    except AttributeError:  # pragma: no cover - defensive guard
        pass

    loader_cache_clear = getattr(_available_iso_scenarios, "cache_clear", None)
    if callable(loader_cache_clear):  # pragma: no branch - attribute guard
        loader_cache_clear()


def forecast_frame_error(base_path: str | os.PathLike[str] | None) -> str | None:
    """Return the last load forecast error recorded for ``base_path``."""

    return _FORECAST_FRAME_ERRORS.get(_base_path_key(base_path))


def _resolve_forecast_base_path() -> str:
    """Return the current input root and clear caches when it changes."""

    cached_root = _cached_input_root()

    input_root.cache_clear()
    _cached_input_root.cache_clear()

    current_root = _cached_input_root()
    if current_root != cached_root:
        _clear_forecast_cache()

    return current_root

