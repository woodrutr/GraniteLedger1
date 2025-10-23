"""High level helpers for working with electricity load forecasts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from common.regions_schema import ISO_DISPLAY_NAMES
from common.schemas.load_forecast import parse_load_forecast_csv
from engine import settings as engine_settings
from engine.io.load_forecast import ISO_DIR_CANON
from engine.normalization import normalize_iso_name, normalize_region_id, normalize_token
from regions.registry import REGIONS

LOG = logging.getLogger(__name__)


_ZONE_COLUMNS = [
    "region_id",
    "state_or_province",
    "scenario_name",
    "year",
    "load_mwh",
    "_region_lower",
    "_scenario_lower",
]


def _empty_zone_frame() -> pd.DataFrame:
    """Return an empty zone frame with the canonical column order."""

    return pd.DataFrame(columns=_ZONE_COLUMNS)

def _region_registry() -> Mapping[str, Any]:
    """Return the registry mapping used for zone lookups."""

    return REGIONS


__all__ = [
    "available_iso_scenarios",
    "discover_zones",
    "load_forecasts",
    "load_iso_scenario",
    "load_iso_scenario_table",
    "load_table",
    "scenario_index",
    "zones_for",
    "load_demand_forecasts_selection",
    "load_forecast_by_state",
    "aggregate_to_state",
    "parse_folder_token",
    "load_zone_forecast",
]


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


def _default_input_root() -> Path:
    """Return the repository default electricity forecast directory."""

    return engine_settings.input_root()


def _input_root() -> Path:
    """Return the dynamic input root (indirection kept for monkeypatching)."""

    return _default_input_root()


def _load_consolidated_forecasts(root: Path | str) -> pd.DataFrame:
    """Return the legacy consolidated load forecast table."""

    from engine.io.load_forecast import load_load_forecasts

    return load_load_forecasts(root)


def _strict_module():
    """Return the strict load forecast module for validation helpers."""

    from engine.io import load_forecasts_strict as strict_module

    return strict_module


def load_forecasts(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Return the consolidated load forecast table for the requested path."""

    if csv_path is None:
        base = _input_root()
        try:
            base_resolved = base.resolve()
        except OSError:
            base_resolved = base

        if base_resolved.is_file():
            target = base_resolved
        else:
            candidates = [
                base_resolved / "load_forecasts.csv",
                base_resolved / "electricity" / "load_forecasts" / "load_forecasts.csv",
            ]
            existing = next((candidate for candidate in candidates if candidate.exists()), None)
            target = existing.resolve() if existing is not None else candidates[-1]
    else:
        target = Path(csv_path)
        try:
            target = target.resolve()
        except OSError:
            target = target

    if target.is_dir():
        manifest_path = target / "load_forecasts.csv"
        manifest_frame = _load_manifest_frame(manifest_path) if manifest_path.exists() else pd.DataFrame()
        if manifest_frame.empty:
            from engine.io.load_forecast import load_load_forecasts

            raw = load_load_forecasts(target)
        else:
            raw = manifest_frame
        if raw is None or raw.empty:
            return _empty_zone_frame()
        working = raw.copy()
        if "scenario_name" not in working.columns and "scenario" in working.columns:
            working["scenario_name"] = working["scenario"]
        if "load_mwh" not in working.columns and "load_gwh" in working.columns:
            working["load_mwh"] = working["load_gwh"] * 1000.0

        base_columns = [
            "region_id",
            "state_or_province",
            "iso",
            "scenario_name",
            "scenario",
            "year",
            "load_mwh",
            "load_gwh",
        ]
        available = [column for column in base_columns if column in working.columns]
        return working.loc[:, available]

    if not target.exists():
        raise FileNotFoundError(f"Load forecast CSV not found: {target}")

    frame = parse_load_forecast_csv(target)
    working = frame.copy()
    working["load_mwh"] = working["load_gwh"] * 1000.0
    columns = ["region_id", "scenario_name", "scenario", "year", "load_mwh", "load_gwh"]
    return working.loc[:, columns]


def _load_manifest_frame(csv_path: Path) -> pd.DataFrame:
    """Return a DataFrame approximating :func:`load_load_forecasts` output."""

    from engine.io.load_forecast import _infer_iso_from_region

    try:
        raw = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    normalized = {str(column).strip().lower(): column for column in raw.columns}
    region_col = next(
        (normalized.get(name) for name in ("region_id", "region", "iso_zone")),
        None,
    )
    year_col = normalized.get("year")
    if year_col is None and "timestamp" in normalized:
        timestamp_col = normalized["timestamp"]
        raw["year"] = pd.to_datetime(raw[timestamp_col], errors="coerce").dt.year
        year_col = "year"

    load_gwh_col = next(
        (normalized.get(name) for name in ("load_gwh", "demand_gwh", "load")),
        None,
    )
    load_mwh_col = next(
        (normalized.get(name) for name in ("load_mwh", "demand_mwh")),
        None,
    )

    if region_col is None or year_col is None:
        return pd.DataFrame()

    demand_series: pd.Series
    if load_gwh_col is not None:
        demand_series = pd.to_numeric(raw[load_gwh_col], errors="coerce")
    elif load_mwh_col is not None:
        demand_series = pd.to_numeric(raw[load_mwh_col], errors="coerce") / 1000.0
    else:
        demand_series = pd.Series(dtype=float)

    if demand_series.empty:
        return pd.DataFrame()

    scenario_name_col = normalized.get("scenario_name") or normalized.get("scenario")
    scenario_series = (
        raw[scenario_name_col].astype(str).str.strip()
        if scenario_name_col is not None
        else pd.Series(["DEFAULT"] * len(raw))
    )
    scenario_token = scenario_series.str.lower()

    state_col = normalized.get("state")
    state_series = (
        raw[state_col].astype(str).str.strip().str.upper()
        if state_col is not None
        else pd.Series([pd.NA] * len(raw))
    )

    working = pd.DataFrame(
        {
            "region_id": raw[region_col].astype(str).str.strip(),
            "year": pd.to_numeric(raw[year_col], errors="coerce"),
            "scenario_name": scenario_series,
            "scenario": scenario_token,
            "load_gwh": demand_series,
            "state": state_series,
        }
    )

    working = working.dropna(subset=["region_id", "year", "load_gwh"], how="any")
    if working.empty:
        return pd.DataFrame()

    def _safe_region(value: str | Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return normalize_region_id(text)
        except ValueError:
            return text.replace(" ", "_").replace("-", "_").upper()

    working["region_id"] = working["region_id"].map(_safe_region)
    working["year"] = working["year"].astype(int)
    working["load_gwh"] = working["load_gwh"].astype(float)

    iso_col = normalized.get("iso")
    if iso_col is not None:
        iso_series = raw[iso_col].astype(str).str.strip().str.upper()
    else:
        iso_series = working["region_id"].map(_infer_iso_from_region).astype(str)
    working["iso"] = iso_series

    return working


def _match_case_insensitive(parent: Path, target: str) -> Path | None:
    """Return a child directory of ``parent`` that matches ``target`` flexibly."""

    if not parent.is_dir():
        return None

    target_lower = target.lower()
    target_token = normalize_token(target)

    for entry in parent.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.lower() == target_lower:
            return entry
        if target_token and normalize_token(entry.name) == target_token:
            return entry
    return None


def _resolve_iso_directory(root: Path, iso: str) -> Path | None:
    """Return the directory that stores forecasts for ``iso``."""

    iso_token = iso.strip()
    if not iso_token:
        return None

    if root.name.lower() == iso_token.lower():
        return root

    match = _match_case_insensitive(root, iso_token)
    if match is not None:
        return match

    iso_norm = normalize_iso_name(iso_token)
    if iso_norm and iso_norm != iso_token:
        match = _match_case_insensitive(root, iso_norm)
        if match is not None:
            return match

    return None


def _resolve_iso_scenario_path(base_path: Path, iso: str, scenario: str) -> Path:
    """Return the directory containing CSVs for ``iso``/``scenario``."""

    iso_token = iso.strip()
    scenario_token = scenario.strip()
    if not iso_token:
        raise ValueError("iso is required")
    if not scenario_token:
        raise ValueError("scenario is required")

    # If base_path already points at the scenario directory we are done.
    if (
        base_path.is_dir()
        and base_path.name.lower() == scenario_token.lower()
        and base_path.parent.name.lower() == iso_token.lower()
    ):
        return base_path

    # Allow callers to pass the ISO directory directly.
    if base_path.is_dir() and base_path.name.lower() == iso_token.lower():
        match = _match_case_insensitive(base_path, scenario_token)
        if match is not None:
            return match
        scenario_norm = normalize_token(scenario_token)
        if scenario_norm:
            match = _match_case_insensitive(base_path, scenario_norm)
            if match is not None:
                return match

    search_roots = [
        base_path,
        base_path / "electricity" / "load_forecasts",
        base_path / "load_forecasts",
    ]

    scenario_norm = normalize_token(scenario_token)

    for root in search_roots:
        if not root.exists():
            continue
        iso_dir = _resolve_iso_directory(root, iso_token)
        if iso_dir is None:
            continue
        match = _match_case_insensitive(iso_dir, scenario_token)
        if match is not None:
            return match
        if scenario_norm:
            match = _match_case_insensitive(iso_dir, scenario_norm)
            if match is not None:
                return match

    default_root = base_path / "electricity" / "load_forecasts"
    raise ValueError(
        "Load forecast path does not exist: "
        f"'{default_root / iso_token / scenario_token}'."
    )


# ---------------------------------------------------------------------------
# Folder naming utilities
# ---------------------------------------------------------------------------


def _pretty_source_token(token: str) -> str:
    """Return a human-friendly label for ``token``."""

    cleaned = str(token or "").strip()
    if not cleaned:
        return ""

    iso_token = normalize_iso_name(cleaned)
    display = ISO_DISPLAY_NAMES.get(iso_token)
    if display:
        return display

    if cleaned.isupper():
        return cleaned

    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", cleaned)
    spaced = re.sub(r"[\-_]+", " ", spaced)
    spaced = " ".join(spaced.split())
    return spaced.title() if spaced else cleaned.title()


def parse_folder_token(folder: str) -> tuple[str, int, str]:
    """Return ``(source_label, vintage_year, scenario)`` for ``folder``."""

    name = folder.strip()
    tokens = [token for token in re.split(r"[_\-]+", name) if token]
    if len(tokens) < 3:
        raise ValueError("expected format <source>_<vintage>_<scenario>")

    scenario_token = tokens[-1]
    vintage_token = tokens[-2]
    if not vintage_token.isdigit():
        raise ValueError("expected numeric vintage token before the scenario name")

    source_tokens = tokens[:-2]
    if not source_tokens:
        raise ValueError("missing source token in bundle directory name")

    source_label = " ".join(filter(None, (_pretty_source_token(token) for token in source_tokens)))
    scenario_label = _pretty_source_token(scenario_token)
    return source_label or scenario_token, int(vintage_token), scenario_label or scenario_token


# ---------------------------------------------------------------------------
# Scenario manifest discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioManifest:
    """Descriptor for a single ISO/zone scenario CSV on disk."""

    iso: str
    scenario: str
    zone: str
    path: Path


def _iter_zone_files(iso_dir: Path) -> Iterator[ScenarioManifest]:
    iso_norm = normalize_iso_name(iso_dir.name) or iso_dir.name.lower()
    for scenario_dir in sorted(p for p in iso_dir.iterdir() if p.is_dir()):
        try:
            _, _, scenario_label = parse_folder_token(scenario_dir.name)
        except ValueError:
            scenario_label = scenario_dir.name
        scenario_token = scenario_label.strip().lower()
        for csv_path in sorted(
            p
            for p in scenario_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".csv"
        ):
            try:
                zone_label = normalize_region_id(csv_path.stem)
            except ValueError:
                token = normalize_token(csv_path.stem)
                zone_label = token if token else csv_path.stem.strip().lower()
            yield ScenarioManifest(
                iso=iso_norm,
                scenario=scenario_token,
                zone=zone_label,
                path=csv_path,
            )


def _available_from_manifest(root: Path) -> list[ScenarioManifest]:
    csv_path = root / "load_forecasts.csv"
    if not csv_path.exists():
        return []

    try:
        frame = _load_consolidated_forecasts(root)
    except Exception:  # pragma: no cover - defensive guard
        LOG.debug("available_iso_scenarios: failed to load consolidated table", exc_info=True)
        return []

    if frame.empty:
        return []

    manifests: list[ScenarioManifest] = []
    for (iso, scenario, zone), subset in frame.groupby(["iso", "scenario", "region_id"], sort=True):
        try:
            region_id = normalize_region_id(zone)
        except ValueError:
            region_id = str(zone).strip().upper()
        manifests.append(
            ScenarioManifest(
                iso=normalize_iso_name(iso) or iso.lower(),
                scenario=str(scenario).strip().lower(),
                zone=region_id,
                path=csv_path,
            )
        )
    return manifests


def available_iso_scenarios(iso: str | None = None, base_path: str | Path | None = None) -> list[ScenarioManifest]:
    """Return available ISO scenario zone manifests."""

    root = Path(base_path) if base_path is not None else _input_root()
    if not root.exists():
        return []

    iso_directories: list[Path] = []
    if iso is None:
        iso_directories = [p for p in root.iterdir() if p.is_dir()]
    else:
        match = _resolve_iso_directory(root, iso)
        if match is not None:
            iso_directories = [match]

    manifests: list[ScenarioManifest] = []
    for iso_dir in iso_directories:
        manifests.extend(_iter_zone_files(iso_dir))

    if manifests:
        return manifests

    # Fall back to the consolidated manifest when legacy folders are absent.
    manifest_entries = _available_from_manifest(root)
    if iso is None:
        return manifest_entries

    iso_norm = normalize_iso_name(iso)
    iso_lower = iso.strip().lower()
    return [
        entry
        for entry in manifest_entries
        if entry.iso == iso_norm or entry.iso == iso_lower
    ]


# ---------------------------------------------------------------------------
# Zone level loading helpers
# ---------------------------------------------------------------------------


def _load_all_zones(root: str | Path) -> pd.DataFrame:
    """Return zone-level forecasts discovered under ``root``.

    When a consolidated ``load_forecasts.csv`` exists at ``root`` the loader
    normalizes the manifest directly.  Otherwise the legacy directory walker is
    used to assemble the table.
    """

    root_path = Path(root)
    csv_path = root_path / "load_forecasts.csv" if root_path.is_dir() else root_path

    if csv_path.is_file():
        frame = pd.read_csv(csv_path)
        if frame.empty:
            return _empty_zone_frame()

        working = frame.copy()
        working.columns = working.columns.str.strip().str.lower()
        if "state" in working.columns and "state_or_province" not in working.columns:
            working = working.rename(columns={"state": "state_or_province"})

        required = {"region_id", "scenario_name", "year", "load_gwh"}
        missing = required - set(working.columns)
        if missing:
            raise ValueError(
                "Missing required columns in consolidated load forecast: "
                f"{', '.join(sorted(missing))}"
            )

        working["region_id"] = (
            working["region_id"].astype(str).str.strip().str.upper()
        )
        working["scenario_name"] = (
            working["scenario_name"].astype(str).str.strip()
        )
        if "state_or_province" in working.columns:
            state_series = (
                working["state_or_province"].astype(str).str.strip().str.upper()
            )
            state_series = state_series.where(state_series != "", pd.NA)
        else:
            state_series = pd.Series(pd.NA, index=working.index, dtype="object")

        working["state_or_province"] = state_series
        working["year"] = pd.to_numeric(working["year"], errors="coerce").astype(
            "Int64"
        )
        working["load_gwh"] = pd.to_numeric(working["load_gwh"], errors="coerce")

        working = working.dropna(
            subset=["region_id", "scenario_name", "year", "load_gwh"], how="any"
        )
        if working.empty:
            return _empty_zone_frame()

        working["load_mwh"] = working["load_gwh"] * 1000.0
        working["load_mwh"] = working["load_mwh"].astype(float)
        working["_region_lower"] = working["region_id"].str.lower()
        working["_scenario_lower"] = working["scenario_name"].str.lower()

        result = working.loc[:, _ZONE_COLUMNS]
        return result.sort_values(["region_id", "scenario_name", "year"]).reset_index(
            drop=True
        )

    try:
        legacy = _load_consolidated_forecasts(root_path)
    except FileNotFoundError:
        return _empty_zone_frame()
    except Exception:
        LOG.debug("Failed to load legacy forecasts from %%s", root_path, exc_info=True)
        return _empty_zone_frame()

    if not isinstance(legacy, pd.DataFrame) or legacy.empty:
        return _empty_zone_frame()

    working = legacy.copy()
    working.columns = [str(column).strip().lower() for column in working.columns]

    if "region_id" not in working.columns:
        return _empty_zone_frame()

    working["region_id"] = working["region_id"].astype(str).str.strip().str.upper()

    if "scenario_name" in working.columns:
        scenario_series = working["scenario_name"].astype(str).str.strip()
    else:
        scenario_series = working.get("scenario", pd.Series(index=working.index, dtype="object"))
        scenario_series = scenario_series.astype(str).str.strip()
    working["scenario_name"] = scenario_series

    if "state_or_province" in working.columns:
        state_series = working["state_or_province"].astype(str).str.strip().str.upper()
    elif "state" in working.columns:
        state_series = working["state"].astype(str).str.strip().str.upper()
    else:
        state_series = pd.Series(pd.NA, index=working.index, dtype="object")
    state_series = state_series.where(state_series != "", pd.NA)
    working["state_or_province"] = state_series

    if "year" not in working.columns:
        return _empty_zone_frame()

    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype("Int64")

    if "load_gwh" in working.columns:
        demand_gwh = pd.to_numeric(working["load_gwh"], errors="coerce")
    elif "load_mwh" in working.columns:
        demand_gwh = pd.to_numeric(working["load_mwh"], errors="coerce") / 1000.0
    else:
        demand_gwh = pd.Series([0.0] * len(working), index=working.index, dtype=float)

    working["load_gwh"] = demand_gwh
    working = working.dropna(
        subset=["region_id", "scenario_name", "year", "load_gwh"], how="any"
    )
    if working.empty:
        return _empty_zone_frame()

    working["load_mwh"] = (working["load_gwh"].astype(float)) * 1000.0
    working["_region_lower"] = working["region_id"].str.lower()
    working["_scenario_lower"] = working["scenario_name"].str.lower()

    result = working.loc[:, _ZONE_COLUMNS]
    return result.sort_values(["region_id", "scenario_name", "year"]).reset_index(
        drop=True
    )


def _load_zone_forecast(
    root: str | Path,
    region_id: str,
    scenario_name: str,
) -> pd.DataFrame:
    """Return records for ``region_id``/``scenario_name`` from ``root``."""

    frame = _load_all_zones(root)
    if frame.empty:
        return _empty_zone_frame()

    try:
        region_token = normalize_region_id(region_id)
    except ValueError:
        region_token = str(region_id or "").strip().upper()

    scenario_token = str(scenario_name or "").strip().lower()
    if not region_token or not scenario_token:
        return _empty_zone_frame()

    mask = (
        frame["_region_lower"] == region_token.lower()
    ) & (frame["_scenario_lower"] == scenario_token)

    if not mask.any():
        return _empty_zone_frame()

    subset = frame.loc[mask].copy()
    return subset.sort_values("year").reset_index(drop=True)


def load_zone_forecast(
    iso: str,
    zone: str,
    scenario: str,
    *,
    base_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return a demand frame for ``zone`` within ``iso``/``scenario``."""

    root = Path(base_path) if base_path is not None else _input_root()
    scenario_path = _resolve_iso_scenario_path(root, iso, scenario)
    try:
        zone_token = normalize_region_id(zone)
    except ValueError:
        zone_token = str(zone).strip().upper()

    csv_path = None
    for candidate in scenario_path.iterdir():
        if not candidate.is_file() or candidate.suffix.lower() != ".csv":
            continue
        try:
            region_id = normalize_region_id(candidate.stem)
        except ValueError:
            region_id = candidate.stem.strip().upper()
        if region_id == zone_token:
            csv_path = candidate
            break

    if csv_path is None:
        LOG.warning(
            "Zone '%s' missing in ISO '%s' selection '%s'", zone_token, iso, scenario
        )
        return pd.DataFrame(columns=["year", "demand_mwh"])

    frame = pd.read_csv(csv_path)
    frame.columns = [col.strip() for col in frame.columns]

    year_col = next((c for c in frame.columns if c.lower() == "year"), None)
    if year_col is None:
        raise ValueError(f"{csv_path}: missing 'Year' column")

    demand_col = next((c for c in frame.columns if c.lower() in {"demand_mwh", "load_mwh", "demand", "load"}), None)
    if demand_col is None:
        demand_col = next((c for c in frame.columns if c.lower().endswith("gwh")), None)

    if demand_col is None:
        raise ValueError(f"{csv_path}: missing load column")

    years = pd.to_numeric(frame[year_col], errors="coerce")
    demand = pd.to_numeric(frame[demand_col], errors="coerce")
    if "gwh" in demand_col.lower() and "mwh" not in demand_col.lower():
        demand = demand * 1000.0

    cleaned = pd.DataFrame({"year": years, "demand_mwh": demand})
    cleaned = cleaned.dropna(subset=["year", "demand_mwh"], how="any")
    cleaned["year"] = cleaned["year"].astype(int)
    cleaned["demand_mwh"] = cleaned["demand_mwh"].astype(float)
    return cleaned.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ISO scenario loading
# ---------------------------------------------------------------------------


def _normalize_scenario_label(label: str) -> str:
    return normalize_token(label) or label.strip().lower()


def load_iso_scenario_table(
    iso: str,
    scenario: str,
    *,
    base_path: str | Path | None = None,
    input_root: str | Path | None = None,
) -> pd.DataFrame:
    """Return the canonical table for an ISO scenario."""

    if base_path is not None and input_root is not None:
        raise TypeError("Specify only one of base_path or input_root")

    root = (
        Path(base_path)
        if base_path is not None
        else Path(input_root)
        if input_root is not None
        else _input_root()
    )

    try:
        scenario_path = _resolve_iso_scenario_path(root, iso, scenario)
    except ValueError:
        frame = pd.DataFrame()
        last_error: Exception | None = None
        for candidate in (
            Path(root) / "electricity" / "load_forecasts",
            Path(root),
        ):
            try:
                frame = _load_consolidated_forecasts(candidate)
            except Exception as exc:  # pragma: no cover - defensive fallthrough
                last_error = exc
                continue
            if not frame.empty:
                break
        if frame.empty:
            if last_error is not None:
                raise last_error
            raise
        iso_token = normalize_token(iso) or iso.strip().lower()
        scenario_token = _normalize_scenario_label(scenario)
        iso_norms = frame["iso"].astype(str).map(
            lambda value: normalize_token(value) or value.strip().lower()
        )
        iso_mask = iso_norms == iso_token
        if not iso_mask.any():
            raise
        iso_frame = frame.loc[iso_mask].copy()
        scenario_norms = iso_frame["scenario"].astype(str).str.lower()
        match_mask = scenario_norms == scenario_token
        if not match_mask.any() and scenario_token:
            match_mask = scenario_norms.str.contains(scenario_token, regex=False)
        if not match_mask.any() and scenario_token:
            target_tokens = {
                token for token in scenario_token.replace("-", "_").split("_") if token
            }
            if target_tokens:
                overlaps = scenario_norms.apply(
                    lambda value: len(
                        target_tokens
                        & {
                            token
                            for token in value.replace("-", "_").split("_")
                            if token
                        }
                    )
                )
                if overlaps.max() > 0:
                    match_mask = overlaps == overlaps.max()
        subset = iso_frame.loc[match_mask]
        if subset.empty:
            raise
        table = subset.rename(columns={"region_id": "zone"})
        table["zone"] = table["zone"].map(
            lambda value: normalize_region_id(value).replace("-", "_")
            if isinstance(value, str)
            else value
        )
        return table.loc[:, ["iso", "zone", "scenario", "year", "load_gwh"]].reset_index(drop=True)

    csv_files = sorted(
        p for p in scenario_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"
    )
    if not csv_files:
        raise ValueError(f"No load forecast CSV files found in '{scenario_path}'.")

    iso_label = ISO_DIR_CANON.get(normalize_iso_name(iso), iso.strip().upper())
    scenario_token = scenario_path.name

    rows: list[dict[str, Any]] = []
    for csv_path in csv_files:
        try:
            region = normalize_region_id(csv_path.stem, iso=iso_label)
        except ValueError:
            try:
                region = normalize_region_id(csv_path.stem)
            except ValueError:
                token = normalize_token(csv_path.stem)
                region = token.upper() if token else csv_path.stem.strip().upper()

        frame = pd.read_csv(csv_path)
        frame.columns = [col.strip() for col in frame.columns]

        year_col = next((c for c in frame.columns if c.lower() == "year"), None)
        if year_col is None:
            raise ValueError(f"{csv_path}: missing 'Year' column")

        load_col = None
        for candidate in frame.columns:
            lowered = candidate.lower()
            if any(hint in lowered for hint in ("load", "demand", "gwh", "mwh")):
                load_col = candidate
                break
        if load_col is None:
            for candidate in frame.columns:
                try:
                    candidate_region = normalize_region_id(candidate, iso=iso_label)
                except ValueError:
                    candidate_region = ""
                if candidate_region == region:
                    load_col = candidate
                    break
                if normalize_token(candidate) == normalize_token(region):
                    load_col = candidate
                    break
        if load_col is None:
            raise ValueError(f"{csv_path}: missing load column")

        years = pd.to_numeric(frame[year_col], errors="coerce")
        loads = pd.to_numeric(frame[load_col], errors="coerce")
        if "mwh" in load_col.lower() and "gwh" not in load_col.lower():
            loads = loads / 1000.0

        for year, load in zip(years, loads):
            if pd.isna(year) or pd.isna(load):
                continue
            rows.append(
                {
                    "iso": iso_label,
                    "zone": region,
                    "scenario": scenario_token,
                    "year": int(year),
                    "load_gwh": float(load),
                }
            )

    table = pd.DataFrame(rows, columns=["iso", "zone", "scenario", "year", "load_gwh"])
    return table.sort_values(["zone", "year"]).reset_index(drop=True)


def load_iso_scenario(*args, **kwargs) -> pd.DataFrame:
    """Backward compatible wrapper around :func:`load_iso_scenario_table`."""

    if not args and not kwargs:
        raise TypeError("load_iso_scenario requires arguments")

    if args and isinstance(args[0], (str, Path)) and not kwargs.get("base_path"):
        base_path = args[0]
        iso = args[1] if len(args) > 1 else kwargs.get("iso")
        scenario = args[2] if len(args) > 2 else kwargs.get("scenario")
        return load_iso_scenario_table(iso, scenario, base_path=base_path)

    return load_iso_scenario_table(*args, **kwargs)


def _load_region_state_map(
    regions_states_csv: str | Path,
    stats_zones_json: str | Path,
) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, set[str]] = {}

    csv_path = Path(regions_states_csv)
    if csv_path.exists():
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            frame = pd.DataFrame()
        if not frame.empty:
            normalized = {str(col).strip().lower(): col for col in frame.columns}
            region_col = normalized.get("region_id") or normalized.get("region")
            state_col = normalized.get("state")
            if region_col is not None and state_col is not None:
                for _, row in frame.iterrows():
                    region_value = row.get(region_col)
                    state_value = row.get(state_col)
                    if not isinstance(region_value, str) or not isinstance(state_value, str):
                        continue
                    region_key = _safe_region_token(region_value)
                    state_code = state_value.strip().upper()
                    if not state_code:
                        continue
                    mapping.setdefault(region_key, set()).add(state_code)

    def _populate_from_payload(payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return

        iso_sections: Iterable[tuple[str | None, Any]]
        if "isos" in payload and isinstance(payload["isos"], Mapping):
            iso_sections = payload["isos"].items()
        else:
            iso_sections = [(None, payload)]

        for _, iso_data in iso_sections:
            if isinstance(iso_data, Mapping) and "states" in iso_data:
                state_records = iso_data["states"]
            else:
                state_records = iso_data

            if not isinstance(state_records, Mapping):
                continue

            for state, zones in state_records.items():
                nested_zones = zones
                if not isinstance(nested_zones, Iterable):
                    continue
                state_code = str(state).strip().upper()
                if not state_code:
                    continue
                for zone in nested_zones:
                    if not isinstance(zone, str):
                        continue
                    region_key = _safe_region_token(zone)
                    if not region_key:
                        continue
                    mapping.setdefault(region_key, set()).add(state_code)

    json_path = Path(stats_zones_json)
    candidates = [json_path]
    repo_root = Path(__file__).resolve().parents[2]
    fallback_json = repo_root / "input" / "regions" / "iso_state_zones.json"
    if fallback_json not in candidates:
        candidates.append(fallback_json)

    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        _populate_from_payload(payload)
        if mapping:
            break

    return {
        region_id: tuple(sorted(states))
        for region_id, states in mapping.items()
        if states
    }


def _load_state_region_shares(
    regions_states_csv: str | Path,
    stats_zones_json: str | Path,
) -> dict[str, dict[str, float]]:
    """Return mapping of state codes to region share weights."""

    state_regions: dict[str, dict[str, float]] = {}

    share_frame: pd.DataFrame | None = None
    try:  # pragma: no cover - optional dependency path
        from engine.regions.shares import load_zone_to_state_share
    except Exception:  # pragma: no cover - optional dependency path
        share_frame = None
    else:
        try:
            share_frame = load_zone_to_state_share()
        except Exception:  # pragma: no cover - configuration error path
            share_frame = None

    if share_frame is not None and not share_frame.empty:
        try:
            iterator = share_frame.itertuples(index=False)
        except AttributeError:  # pragma: no cover - defensive guard
            iterator = []
        for entry in iterator:
            try:
                region_value = getattr(entry, "region_id")
                state_value = getattr(entry, "state")
                share_value = getattr(entry, "share")
            except AttributeError:  # pragma: no cover - defensive guard
                continue

            if not isinstance(region_value, str) or not isinstance(state_value, str):
                continue

            region_key = _safe_region_token(region_value)
            state_code = state_value.strip().upper()
            if not region_key or not state_code:
                continue

            try:
                share_float = float(share_value)
            except (TypeError, ValueError):
                continue

            if share_float <= 0:
                continue

            state_regions.setdefault(state_code, {})[region_key] = share_float

    region_state_map = _load_region_state_map(regions_states_csv, stats_zones_json)
    for region_id, states in region_state_map.items():
        region_key = _safe_region_token(region_id)
        if not region_key:
            continue
        state_iterable = states if isinstance(states, (list, tuple, set)) else (states,)
        state_list = [str(state).strip().upper() for state in state_iterable if str(state).strip()]
        if not state_list:
            continue
        default_share = 1.0 / len(state_list)
        for state_code in state_list:
            region_shares = state_regions.setdefault(state_code, {})
            region_shares.setdefault(region_key, default_share)

    normalized: dict[str, dict[str, float]] = {}
    for state_code, region_shares in state_regions.items():
        if not region_shares:
            continue
        total = sum(float(value) for value in region_shares.values())
        if total <= 0:
            count = len(region_shares)
            if count == 0:
                continue
            share_value = 1.0 / count
            normalized[state_code] = {
                region: share_value for region in sorted(region_shares)
            }
        else:
            normalized[state_code] = {
                region: float(value) / total for region, value in sorted(region_shares.items())
            }

    return normalized


def _safe_region_token(value: Any) -> str:
    """Normalize ``value`` to a canonical region token when possible."""

    text = str(value).strip()
    if not text:
        return ""
    try:
        return normalize_region_id(text)
    except ValueError:
        return text.replace("-", "_").upper()


def _aggregate_manifest_demand(root_path: Path) -> pd.DataFrame:
    """Return aggregate region demand from a consolidated manifest."""

    csv_path = root_path if root_path.is_file() else root_path / "load_forecasts.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["year", "region", "demand_mwh"])

    try:
        frame = parse_load_forecast_csv(csv_path)
    except Exception:
        return pd.DataFrame(columns=["year", "region", "demand_mwh"])

    if frame.empty:
        return pd.DataFrame(columns=["year", "region", "demand_mwh"])

    working = frame.loc[:, ["region_id", "year", "load_gwh"]].copy()
    working["demand_mwh"] = working["load_gwh"] * 1000.0
    working = working.dropna(subset=["region_id", "year", "demand_mwh"], how="any")

    if working.empty:
        return pd.DataFrame(columns=["year", "region", "demand_mwh"])

    working["year"] = working["year"].astype(int)
    working["demand_mwh"] = working["demand_mwh"].astype(float)

    grouped = (
        working.groupby(["region_id", "year"], as_index=False)["demand_mwh"].sum()
    )
    grouped["region"] = grouped["region_id"].astype(str)
    return grouped.sort_values(["region_id", "year"]).reset_index(drop=True)


def load_iso_scenario_bundle(*args, **kwargs) -> pd.DataFrame:
    """Backward compatible alias for :func:`load_iso_scenario_table`."""

    return load_iso_scenario_table(*args, **kwargs)


def load_demand_forecasts_selection(
    selection: Mapping[str, Mapping[str, str]] | None = None,
    *,
    root: str | Path = "input/electricity/load_forecasts",
    base_path: str | Path | None = None,
    regions_states_csv: str | Path = "input/electricity/cem_inputs/regions_states.csv",
    stats_zones_json: str | Path = "input/electricity/cem_inputs/stats_zones.json",
) -> pd.DataFrame:
    """Return demand forecasts for the requested ``selection``.

    When ``selection`` is ``None`` the loader returns aggregate region/year totals
    for the requested ``base_path`` (or ``root``).  This covers smoke-test
    scenarios that only need a coarse demand frame.  When a selection mapping is
    supplied the historic behaviour of filtering by state/ISO/scenario is
    preserved.
    """

    root_path = Path(base_path) if base_path is not None else Path(root)
    frame = _load_consolidated_forecasts(root_path)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        csv_path = root_path if root_path.is_file() else root_path / "load_forecasts.csv"
        manifest_frame = _load_manifest_frame(csv_path)
        frame = manifest_frame if not manifest_frame.empty else frame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame()

    aggregated_mode = selection is None and base_path is not None

    if aggregated_mode:
        if frame.empty:
            return _aggregate_manifest_demand(root_path)
        grouped = frame.groupby(["region_id", "year"], as_index=False)["load_gwh"].sum()
        grouped["demand_mwh"] = grouped["load_gwh"] * 1000.0
        grouped["region"] = grouped["region_id"].astype(str)
        grouped = grouped.loc[:, ["year", "region_id", "region", "demand_mwh"]]
        return grouped.sort_values(["region_id", "year"]).reset_index(drop=True)

    if selection is None or not isinstance(selection, Mapping) or not selection:
        raise ValueError("Selection must contain at least one state entry.")

    if frame.empty:
        return pd.DataFrame(columns=["state", "iso", "region_id", "scenario", "year", "load_gwh"])

    state_region_shares = _load_state_region_shares(regions_states_csv, stats_zones_json)
    if "state" in frame.columns:
        frame["state"] = frame["state"].astype(str).str.strip().str.upper()
    else:
        frame["state"] = pd.NA

    normalized_selection: dict[str, Dict[str, str]] = {}
    for state, config in selection.items():
        if not isinstance(config, Mapping):
            continue
        iso_value = str(config.get("iso", "")).strip()
        scenario_value = str(config.get("scenario", "")).strip()
        state_code = str(state).strip().upper()
        if not state_code or not iso_value or not scenario_value:
            continue
        normalized_selection[state_code] = {
            "iso": normalize_iso_name(iso_value) or iso_value.lower(),
            "scenario": scenario_value.lower(),
        }

    if not normalized_selection:
        raise ValueError("No valid state selections provided.")

    frame["iso_token"] = frame["iso"].astype(str).str.lower().str.replace("-", "_", regex=False)
    frame["scenario_token"] = frame["scenario"].astype(str).str.lower()
    frame["_region_token"] = frame["region_id"].map(_safe_region_token)

    frames: list[pd.DataFrame] = []
    for state_code, config in normalized_selection.items():
        iso_label = config["iso"]
        scenario_label = config["scenario"]
        state_shares = state_region_shares.get(state_code)

        subset = frame[
            (frame["iso_token"] == iso_label)
            & (frame["scenario_token"] == scenario_label)
        ].copy()

        if state_shares:
            allowed_regions = {key for key in state_shares.keys() if key}
            if allowed_regions:
                weighted_subset = subset[subset["_region_token"].isin(allowed_regions)].copy()
                if not weighted_subset.empty:
                    weighted_subset["_state_share"] = weighted_subset["_region_token"].map(
                        state_shares
                    )
                    weighted_subset = weighted_subset.dropna(subset=["_state_share"])
                    if not weighted_subset.empty:
                        weighted_subset["state"] = state_code
                        weighted_subset["load_gwh"] = pd.to_numeric(
                            weighted_subset["load_gwh"], errors="coerce"
                        )
                        weighted_subset = weighted_subset.dropna(subset=["load_gwh"])
                        if not weighted_subset.empty:
                            weighted_subset["load_gwh"] = (
                                weighted_subset["load_gwh"].astype(float)
                                * weighted_subset["_state_share"].astype(float)
                            )
                            frames.append(
                                weighted_subset.loc[
                                    :,
                                    ["state", "iso", "region_id", "scenario", "year", "load_gwh"],
                                ]
                            )
                            continue

        subset = subset[subset["state"] == state_code].copy()
        if subset.empty:
            continue
        subset["state"] = state_code
        frames.append(subset.loc[:, ["state", "iso", "region_id", "scenario", "year", "load_gwh"]])

    if not frames:
        raise ValueError("No matching state selections found.")

    result = pd.concat(frames, ignore_index=True)
    columns = ["state", "iso", "region_id", "scenario", "year", "load_gwh"]
    result = result.loc[:, columns]
    result = result.sort_values(columns).reset_index(drop=True)
    return result


def load_forecast_by_state(*args, **kwargs) -> pd.DataFrame:
    """Backward compatible alias for :func:`load_demand_forecasts_selection`."""

    return load_demand_forecasts_selection(*args, **kwargs)


def aggregate_to_state(df: pd.DataFrame) -> pd.DataFrame:
    """Sum zone-level ``load_gwh`` to state-year totals."""

    if df.empty:
        return pd.DataFrame(columns=["state", "year", "load_gwh"])

    grouped = df.groupby(["state", "year"], as_index=False)["load_gwh"].sum()
    return grouped.sort_values(["state", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strict table loading & helpers
# ---------------------------------------------------------------------------


def load_table(*args, **kwargs) -> pd.DataFrame:
    """Return the canonical strict load forecast table with error logging."""

    strict = _strict_module()
    try:
        frame = strict.build_table(*args, **kwargs)
    except strict.ValidationError as exc:
        reason = exc.reason or "validation failed"
        message = f"Strict load forecast validation failed for {exc.file}: {reason}"
        LOG.error(message)
        raise

    categorical_columns = ["iso", "scenario", "zone", "region_id"]
    for column in categorical_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("category")
    return frame


def validate_forecasts(base_path: str | Path) -> None:
    """Raise ``ValidationError`` if any forecast CSV fails strict checks."""

    strict = _strict_module()
    strict.build_table(base_path=base_path, use_cache=False)


def scenario_index(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return mapping of ISO identifiers to available scenario labels."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    if "iso" not in frame.columns or "scenario" not in frame.columns:
        return {}

    working = pd.DataFrame(
        {
            "iso": frame["iso"].astype(str),
            "scenario": frame["scenario"].astype(str),
        }
    )

    index: dict[str, list[str]] = {}
    for iso_value, subset in working.groupby("iso", sort=True):
        scenarios = sorted({scenario for scenario in subset["scenario"] if scenario})
        if scenarios:
            index[str(iso_value)] = scenarios
    return index


def zones_for(frame: pd.DataFrame, iso: str, scenario: str) -> list[str]:
    """Return sorted zone names for ``iso`` and ``scenario``."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    if "iso" not in frame.columns or "scenario" not in frame.columns:
        return []

    iso_token = str(iso).strip().lower()
    scenario_token = str(scenario).strip().lower()
    if not iso_token or not scenario_token:
        return []

    if "zone" in frame.columns:
        zone_column = "zone"
    elif "region_id" in frame.columns:
        zone_column = "region_id"
    else:
        return []

    iso_series = frame["iso"].astype(str).str.lower()
    scenario_series = frame["scenario"].astype(str).str.lower()
    mask = (iso_series == iso_token) & (scenario_series == scenario_token)
    if not mask.any():
        return []

    zones = frame.loc[mask, zone_column].astype(str)
    return sorted(zone for zone in zones.unique() if zone)


def discover_zones(*_args, **_kwargs) -> list[str]:  # pragma: no cover - legacy stub
    """Legacy helper retained for compatibility with historic imports."""

    LOG.warning("discover_zones is deprecated; use available_iso_scenarios instead")
    return []

