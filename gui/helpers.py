"""Convenience helpers for the Streamlit GUI forecast discovery flow."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from gui.engine_module import ensure_engine_package

ensure_engine_package()

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    st = None  # type: ignore[assignment]

from engine.settings import configure_load_forecast_path, input_root
from engine.normalization import normalize_iso_name

from engine.data_loaders import load_demand_forecasts_selection
try:
    from engine.data_loaders import available_iso_scenarios as _available_iso_scenarios
except (ImportError, AttributeError):  # pragma: no cover - compatibility guard
    _available_iso_scenarios = None  # type: ignore[assignment]

try:
    from engine.io import load_forecasts_strict as _strict_loader
except ImportError:  # pragma: no cover - optional dependency guard
    _strict_loader = None  # type: ignore[assignment]
else:
    strict_read = getattr(_strict_loader, "read_iso_scenario", None)
    _StrictValidationError = getattr(_strict_loader, "ValidationError", Exception)


if _strict_loader is None:
    strict_read = None  # type: ignore[assignment]
    _StrictValidationError = Exception
from engine.regions.shares import state_weights_for_region

LOG = logging.getLogger(__name__)


def _root() -> Path:
    cache_clear = getattr(input_root, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()

    if os.getenv("GRANITELEDGER_INPUT_ROOT"):
        try:
            configure_load_forecast_path(None)
        except Exception:  # pragma: no cover - defensive guard
            LOG.debug(
                "GUI/helpers: unable to clear configured load forecast root",
                exc_info=True,
            )

    root = input_root()
    LOG.info("GUI/helpers: load_forecasts root=%s", root)
    return root


def _format_validation_error(error: Exception, *, context: str | None = None) -> str:
    """Return a human friendly message for strict loader validation failures."""

    location = getattr(error, "file", None) or getattr(error, "path", None)
    if location is None:
        location = context or "<unknown>"
    else:
        location = str(location)

    descriptors: list[str] = []
    row = getattr(error, "row", None)
    if row is not None:
        descriptors.append(f"row {row}")
    column = getattr(error, "column", None)
    if column:
        descriptors.append(f"column {column}")

    detail = f" ({', '.join(descriptors)})" if descriptors else ""
    reason = getattr(error, "reason", None)
    reason_suffix = f": {reason}" if reason else ""

    return f"{location}{detail}{reason_suffix}"


def _discover_with_strict(base_path: Path) -> list[dict[str, Any]]:
    """Return discovery metadata using the strict forecast loader."""

    if _strict_loader is None:
        return []

    try:
        table = _strict_loader.build_table(base_path=base_path)
    except _StrictValidationError as exc:  # type: ignore[misc]
        message = _format_validation_error(exc)
        LOG.error("GUI/helpers: strict discovery failed: %s", message)
        raise RuntimeError(f"Strict load forecast validation failed: {message}") from exc

    if table is None or getattr(table, "empty", False):
        return []

    try:
        grouped = table.groupby(["iso", "scenario"], sort=True)
    except Exception:  # pragma: no cover - defensive guard
        LOG.debug("GUI/helpers: strict discovery unable to group table", exc_info=True)
        return []

    bundles: list[dict[str, Any]] = []
    for (iso_value, scenario_value), frame in grouped:
        iso_label = str(iso_value)
        scenario_label = str(scenario_value)

        scenario_frame = frame
        if strict_read is not None:
            try:
                scenario_frame = strict_read(  # type: ignore[misc]
                    iso_label,
                    scenario_label,
                    base_path=base_path,
                )
            except _StrictValidationError as exc:  # type: ignore[misc]
                message = _format_validation_error(exc, context=f"{iso_label}/{scenario_label}")
                LOG.error("GUI/helpers: strict read failed: %s", message)
                raise RuntimeError(f"Strict load forecast validation failed: {message}") from exc
            except TypeError:
                try:
                    scenario_frame = strict_read(  # type: ignore[misc]
                        iso_label,
                        scenario_label,
                        root=base_path,
                    )
                except _StrictValidationError as exc:  # type: ignore[misc]
                    message = _format_validation_error(exc, context=f"{iso_label}/{scenario_label}")
                    LOG.error("GUI/helpers: strict read failed: %s", message)
                    raise RuntimeError(
                        f"Strict load forecast validation failed: {message}"
                    ) from exc
                except TypeError:  # pragma: no cover - unexpected signature
                    scenario_frame = frame

        if not hasattr(scenario_frame, "get") and hasattr(frame, "get"):
            scenario_frame = frame

        zones_series = None
        if hasattr(scenario_frame, "get"):
            try:
                zones_series = scenario_frame.get("zone")  # type: ignore[assignment]
            except Exception:  # pragma: no cover - defensive guard
                zones_series = None
        if zones_series is None and hasattr(scenario_frame, "__getitem__"):
            try:
                zones_series = scenario_frame["zone"]  # type: ignore[index]
            except Exception:
                zones_series = None

        region_series = None
        if hasattr(scenario_frame, "get"):
            try:
                region_series = scenario_frame.get("region_id")  # type: ignore[assignment]
            except Exception:  # pragma: no cover - defensive guard
                region_series = None
        if region_series is None and hasattr(scenario_frame, "__getitem__"):
            try:
                region_series = scenario_frame["region_id"]  # type: ignore[index]
            except Exception:
                region_series = None

        regions: set[str] = set()
        if region_series is not None:
            try:
                iterator = getattr(region_series, "dropna", lambda: region_series)()
            except TypeError:
                iterator = region_series
            for entry in getattr(iterator, "unique", lambda: iterator)():  # type: ignore[misc]
                text = str(entry).strip()
                if text:
                    regions.add(text)
        elif zones_series is not None:
            try:
                iterator = getattr(zones_series, "dropna", lambda: zones_series)()
            except TypeError:
                iterator = zones_series
            for entry in getattr(iterator, "unique", lambda: iterator)():  # type: ignore[misc]
                text = str(entry).strip()
                if text:
                    regions.add(text)

        csv_count = 0
        if zones_series is not None:
            try:
                csv_count = int(getattr(zones_series, "nunique", lambda: len(zones_series))())
            except TypeError:  # pragma: no cover - fallback for iterables
                zones_list = list(zones_series)
                csv_count = len(zones_list)
                zones_series = zones_list
        elif regions:
            csv_count = len(regions)

        bundles.append(
            {
                "iso": normalize_iso_name(iso_label),
                "manifest": scenario_label,
                "regions": sorted(regions),
                "dir": str(base_path / iso_label / scenario_label),
                "csv_count": csv_count,
            }
        )

    return bundles


def _scan_forecast_directories(base_path: Path) -> list[dict[str, Any]]:
    """Return discovery metadata by scanning the filesystem."""

    bundles: list[dict[str, Any]] = []
    try:
        iso_dirs = sorted(
            [entry for entry in base_path.iterdir() if entry.is_dir()],
            key=lambda item: item.name.lower(),
        )
    except OSError:
        return bundles

    for iso_dir in iso_dirs:
        iso_label = normalize_iso_name(iso_dir.name)
        try:
            scenario_dirs = sorted(
                [entry for entry in iso_dir.iterdir() if entry.is_dir()],
                key=lambda item: item.name.lower(),
            )
        except OSError:
            continue

        for scenario_dir in scenario_dirs:
            csv_files = sorted(
                [path for path in scenario_dir.glob("*.csv") if path.is_file()],
                key=lambda item: item.name.lower(),
            )
            if not csv_files:
                continue
            regions = sorted({path.stem.strip() for path in csv_files if path.stem})
            bundles.append(
                {
                    "iso": iso_label,
                    "manifest": scenario_dir.name,
                    "regions": [region for region in regions if region],
                    "dir": str(scenario_dir),
                    "csv_count": len(csv_files),
                }
            )

    return bundles


def list_iso_scenarios(iso: str, raw: bool = False) -> list[str]:
    """Return available scenario directories for ``iso``."""

    iso_key = normalize_iso_name(iso)
    base_path = _root()
    for entry in base_path.iterdir():
        if not entry.is_dir():
            continue
        if normalize_iso_name(entry.name) != iso_key:
            continue
        scenarios = sorted(
            {child.name for child in entry.iterdir() if child.is_dir()},
            key=str.lower,
        )
        if raw:
            return scenarios
        return scenarios
    return []


    bundles: list[dict[str, Any]] = []
    try:
        iso_dirs = sorted(
            [entry for entry in base_path.iterdir() if entry.is_dir()],
            key=lambda item: item.name.lower(),
        )
    except OSError:
        return bundles

    for iso_dir in iso_dirs:
        iso_label = normalize_iso_name(iso_dir.name)
        try:
            scenario_dirs = sorted(
                [entry for entry in iso_dir.iterdir() if entry.is_dir()],
                key=lambda item: item.name.lower(),
            )
        except OSError:
            continue

        for scenario_dir in scenario_dirs:
            csv_files = sorted(
                [path for path in scenario_dir.glob("*.csv") if path.is_file()],
                key=lambda item: item.name.lower(),
            )
            if not csv_files:
                continue
            regions = sorted({path.stem.strip() for path in csv_files if path.stem})
            bundles.append(
                {
                    "iso": iso_label,
                    "manifest": scenario_dir.name,
                    "regions": [region for region in regions if region],
                    "dir": str(scenario_dir),
                    "csv_count": len(csv_files),
                }
            )

    return bundles


def list_iso_scenarios(iso: str, raw: bool = False) -> list[str]:
    """Return available scenario directories for ``iso``."""

    iso_key = normalize_iso_name(iso)
    base_path = _root()

    lookup_key = iso_key.upper().replace("-", "_")
    if _available_iso_scenarios is not None:
        try:
            combos = _available_iso_scenarios(str(base_path))
        except TypeError:  # pragma: no cover - legacy signature support
            combos = _available_iso_scenarios(base_path)  # type: ignore[misc]
        scenarios = sorted(
            {
                scenario
                for entry in combos
                if isinstance(entry, Mapping)
                and str(entry.get("iso", "")).upper() == lookup_key
                and (scenario := str(entry.get("scenario", "")).strip())
            },
            key=str.lower,
        )
        if scenarios:
            return scenarios if raw else scenarios

    for entry in base_path.iterdir():
        if not entry.is_dir():
            continue
        if normalize_iso_name(entry.name) != iso_key:
            continue
        scenarios = sorted(
            {child.name for child in entry.iterdir() if child.is_dir()},
            key=str.lower,
        )
        return scenarios if raw else scenarios
    return []


def discover_all() -> list[dict[str, str]]:
    """Discover all load forecast scenarios using the configured input root."""

    base = _root()

    if not base.exists():
        LOG.warning("Load forecast root not found: %s", base)
        return []

    bundles = []
    for iso_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        iso = normalize_iso_name(iso_dir.name)
        for scen_dir in sorted(p for p in iso_dir.iterdir() if p.is_dir()):
            csvs = [f for f in scen_dir.glob("*.csv") if f.is_file()]
            if not csvs:
                continue
            bundles.append({
                "iso": iso,
                "manifest": scen_dir.name,
                "regions": [f.stem for f in csvs],
                "dir": str(scen_dir),
                "csv_count": len(csvs),
            })

    LOG.info("Discovered %d load forecast bundles under %s", len(bundles), base)
    return bundles


def _custom_forecast_frame(custom: Mapping[str, Any] | None) -> pd.DataFrame:
    """Return a DataFrame of custom demand overrides."""

    if not isinstance(custom, Mapping) or not custom:
        return pd.DataFrame(
            columns=["state", "iso", "region_id", "scenario", "year", "load_gwh"]
        )

    frames: list[pd.DataFrame] = []
    for state_key, payload in custom.items():
        if not isinstance(payload, Mapping):
            continue
        records = payload.get("records")
        if isinstance(records, pd.DataFrame):
            working = records.copy()
        elif isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
            try:
                working = pd.DataFrame.from_records(records)
            except Exception:
                continue
        else:
            continue

        if working.empty or "demand_mwh" not in working.columns:
            continue

        columns = {"region", "year", "demand_mwh"}
        if not columns.issubset(working.columns):
            continue

        candidate = working.loc[:, ["region", "year", "demand_mwh"]].copy()
        candidate["region"] = candidate["region"].astype(str)
        candidate["year"] = pd.to_numeric(candidate["year"], errors="coerce")
        candidate["demand_mwh"] = pd.to_numeric(candidate["demand_mwh"], errors="coerce")
        candidate = candidate.dropna(subset=["region", "year", "demand_mwh"])
        if candidate.empty:
            continue

        candidate["region_id"] = candidate["region"].astype(str)
        candidate["year"] = candidate["year"].astype(int)
        candidate["load_gwh"] = candidate["demand_mwh"].astype(float) / 1000.0

        state_value = payload.get("state") or state_key
        scenario_source = payload.get("source")
        scenario_label = str(scenario_source).strip() if scenario_source else "Uploaded"

        candidate["state"] = str(state_value).strip().upper()
        candidate["iso"] = "CUSTOM"
        candidate["scenario"] = scenario_label if scenario_label else "Uploaded"

        frames.append(
            candidate.loc[:, ["state", "iso", "region_id", "scenario", "year", "load_gwh"]]
        )

    if not frames:
        return pd.DataFrame(
            columns=["state", "iso", "region_id", "scenario", "year", "load_gwh"]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.groupby(
        ["state", "iso", "region_id", "scenario", "year"], as_index=False
    )["load_gwh"].sum()
    return combined


def _decode_forecast_selection(value: Any) -> tuple[str | None, str | None]:
    """Return ISO/scenario pair extracted from ``value``.

    The GUI stores forecast selections as strings in the form ``ISO::scenario``
    (with historic variants that used ``/`` or ``|`` as delimiters).  If the
    value cannot be parsed, ``(None, None)`` is returned.
    """

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


def _normalize_state_selection(
    selection: Mapping[str, Any] | None,
) -> dict[str, dict[str, str]]:
    """Coerce GUI forecast selections into the loader's state mapping.

    ``selection`` originates from the demand module configuration and may be a
    nested mapping of state→{"iso", "scenario"} pairs or a mapping of state
    codes to encoded ``ISO::scenario`` strings.  The helper normalizes ISO
    labels and filters out entries that lack an ISO or scenario token.
    """

    state_map: dict[str, dict[str, str]] = {}
    if not isinstance(selection, Mapping):
        return state_map

    for raw_state, payload in selection.items():
        if payload in (None, ""):
            continue

        state_code = str(raw_state).strip().upper()
        if len(state_code) != 2:
            continue

        iso_value: str | None = None
        scenario_value: str | None = None
        if isinstance(payload, Mapping):
            iso_candidate = payload.get("iso")
            scenario_candidate = payload.get("scenario")
            if iso_candidate not in (None, ""):
                iso_value = str(iso_candidate).strip() or None
            if scenario_candidate not in (None, ""):
                scenario_value = str(scenario_candidate).strip() or None
        else:
            iso_value, scenario_value = _decode_forecast_selection(payload)

        if not scenario_value:
            continue

        iso_text = (iso_value or "").strip()
        iso_token = normalize_iso_name(iso_text) if iso_text else ""
        iso_label = iso_token or iso_text
        if not iso_label:
            continue

        state_map[state_code] = {
            "iso": iso_label,
            "scenario": str(scenario_value).strip(),
        }

    return state_map


def build_demand(
    years: Sequence[int] | None,
    selection: Mapping[str, Any] | None,
    *,
    custom: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return demand forecasts for the selected states and optional year filter."""

    state_selection = _normalize_state_selection(selection)

    custom_df = _custom_forecast_frame(custom) if custom else None

    if not state_selection:
        if custom_df is None or custom_df.empty:
            raise RuntimeError("State selections are required to load demand forecasts.")
        demand_df = custom_df
    else:
        demand_df = load_demand_forecasts_selection(state_selection, root=_root())
        if demand_df.empty:
            if custom_df is None or custom_df.empty:
                raise RuntimeError(
                    "Demand frame empty after selection; inspect CSV headers and regions."
                )
            demand_df = custom_df
        elif custom_df is not None and not custom_df.empty:
            demand_df = pd.concat([demand_df, custom_df], ignore_index=True, sort=False)

    if "year" not in demand_df.columns and "timestamp" in demand_df.columns:
        year_series = pd.to_datetime(demand_df["timestamp"], errors="coerce")
        demand_df = demand_df.assign(year=year_series.dt.year)

    if "year" in demand_df.columns:
        demand_df["year"] = pd.to_numeric(demand_df["year"], errors="coerce")
        demand_df = demand_df.dropna(subset=["year"]).reset_index(drop=True)
        demand_df["year"] = demand_df["year"].astype(int)

    if years and "year" in demand_df.columns:
        try:
            target_years = {int(year) for year in years}
        except (TypeError, ValueError):
            target_years = set()
        if target_years:
            demand_df = demand_df[demand_df["year"].isin(target_years)].reset_index(drop=True)

    return demand_df



def build_run_payload(
    years: Sequence[int] | None,
    selection: Mapping[str, Any] | None,
    *,
    custom: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    demand = build_demand(years, selection, custom=custom)
    payload = {
        "years": sorted(set(demand["year"].tolist())),
        "demand": demand,
        "meta": {
            "input_root": str(_root()),
            "bundles": discover_all(),
        },
    }
    if custom:
        payload["custom_load_forecasts"] = custom
    LOG.info("GUI/helpers: payload years=%s rows=%d", payload["years"], len(demand))
    return payload


def resolve_state_selection(states: list[str]) -> dict[str, float]:
    """Return aggregated region weights for a sequence of state codes."""

    region_weights: dict[str, float] = {}
    for entry in states:
        if not entry:
            continue
        state_code = str(entry).strip().upper()
        if not state_code:
            continue
        try:
            weights = state_weights_for_region(state_code)
        except Exception as exc:
            LOG.warning(
                "Unable to resolve regions for state %s: %s", state_code, exc, exc_info=True
            )
            if st is not None:
                st.warning(f"Failed to resolve regions for state {state_code}: {exc}")
            continue
        for region, weight in weights.items():
            try:
                numeric = float(weight)
            except (TypeError, ValueError):
                continue
            if not region:
                continue
            region_weights[region] = region_weights.get(region, 0.0) + numeric

    if not region_weights:
        return {}

    filtered = {
        region: weight
        for region, weight in region_weights.items()
        if abs(float(weight)) >= 0.005
    }
    return dict(sorted(filtered.items()))


__all__ = [
    "build_demand",
    "build_run_payload",
    "discover_all",
    "resolve_state_selection",
    "list_iso_scenarios",
]
