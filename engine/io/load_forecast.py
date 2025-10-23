"""Canonical helpers for reading load forecast bundles.

This module exposes :func:`load_load_forecasts`, a single entry point that
reads either the consolidated ``load_forecasts.csv`` table or the legacy
``<iso>/<scenario>/*.csv`` directory structure.  The loader always returns a
canonical, tidy :class:`pandas.DataFrame` the rest of the engine can consume.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from engine.normalization import normalize_iso_name, normalize_region_id

__all__ = ["ISO_DIR_CANON", "load_load_forecasts"]


# ---------------------------------------------------------------------------
# Canonical ISO label helpers
# ---------------------------------------------------------------------------

# Folder-name -> canonical ISO label used across the app/engine
ISO_DIR_CANON: dict[str, str] = {
    "iso_ne": "ISO-NE",
    "isone": "ISO-NE",
    "iso-ne": "ISO-NE",
    "ne": "ISO-NE",
    "nyiso": "NYISO",
    "pjm": "PJM",
    "miso": "MISO",
    "southeast": "SOUTHEAST",
    "spp": "SPP",
    "ercot": "ERCOT",
    "canada": "CANADA",
    "caiso": "CAISO",
}

_REGION_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

# Tolerate load columns named ``Load_GWh``, ``demand_mwh`` and friends.
_DEMAND_COL_HINTS: tuple[str, ...] = ("gwh", "load", "demand")

# Region prefixes → canonical ISO label hints used when loading the consolidated
# ``load_forecasts.csv`` file.  The prefixes intentionally cover both historic
# ISO region identifiers (``ISO-NE_CT``) and the Southeast system zones that do
# not share a common prefix (``SOCO_SYS`` etc.).
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


def _canon_region_id(value: str) -> str:
    """Normalise a CSV filename stem into a dispatch zone identifier."""

    return value.replace(" ", "_").replace("-", "_").upper()


def _extract_region_token(value: str | None) -> str:
    """Return the first plausible region token from ``value``."""

    text = str(value or "").strip()
    if not text:
        return ""
    match = _REGION_TOKEN.search(text)
    if match:
        return match.group(0)
    return text.split()[0]


# Tolerate load columns named ``Load_GWh``, ``Demand_MWh`` and friends.
_DEMAND_COL_HINTS: tuple[str, ...] = ("gwh", "load", "demand")


@lru_cache(maxsize=1)
def _zone_iso_hints() -> tuple[dict[str, str], set[str]]:
    """Return mapping of region id → ISO plus tokens that imply "SOUTHEAST"."""

    mapping: dict[str, str] = {}
    southeast_tokens: set[str] = set()

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [repo_root / "input" / "regions" / "iso_state_zones.json"]

    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:  # pragma: no cover - best effort hints
            continue
        if not isinstance(payload, dict):
            continue
        iso_sections = payload.get("isos") if isinstance(payload.get("isos"), dict) else payload
        if not isinstance(iso_sections, dict):
            continue
        for iso_label, iso_data in iso_sections.items():
            if not isinstance(iso_data, dict):
                continue
            states = iso_data.get("states")
            if not isinstance(states, dict):
                continue
            for zones in states.values():
                if not isinstance(zones, Iterable):
                    continue
                for zone in zones:
                    if not isinstance(zone, str):
                        continue
                    token = _canon_region_id(zone)
                    if not token:
                        continue
                    mapping.setdefault(token, str(iso_label))
                    if str(iso_label).strip().upper() == "SOUTHEAST" and not token.startswith("PJM"):
                        southeast_tokens.add(token)
        if mapping:
            break

    return mapping, southeast_tokens


# ---------------------------------------------------------------------------
# Utility normalisation helpers
# ---------------------------------------------------------------------------

def _canon_region_id(stem: str) -> str:
    """Normalise a CSV filename stem into a dispatch zone identifier."""

    return stem.replace(" ", "_").replace("-", "_").upper()


def _extract_region_token(value: str | None) -> str:
    """Return the first plausible region token from ``value``."""

    text = str(value or "").strip()
    if not text:
        return ""
    match = _REGION_TOKEN.search(text)
    if match:
        return match.group(0)
    return text.split()[0]


def _normalize_region(value: str | None) -> str:
    token = _extract_region_token(value)
    if not token:
        return ""
    try:
        return normalize_region_id(token)
    except ValueError:
        return _canon_region_id(token)


def _normalize_iso_label(value: str | None) -> str:
    token = normalize_iso_name(value)
    if token:
        return ISO_DIR_CANON.get(token, token)
    text = str(value or "").strip()
    if not text:
        return ""
    return ISO_DIR_CANON.get(text.lower(), text.upper())


def _infer_iso_from_region(region: str) -> str:
    """Return a best-effort ISO label inferred from ``region``."""

    token = _canon_region_id(region)
    if not token:
        return ""

    for prefix, iso_label in _REGION_ISO_HINTS:
        if token.startswith(prefix):
            return iso_label

    mapping, southeast_tokens = _zone_iso_hints()
    if token in mapping:
        hinted = mapping[token]
        if hinted:
            return _normalize_iso_label(hinted)
    if token in southeast_tokens:
        return "SOUTHEAST"

    prefix = token.split("_", 1)[0]
    if prefix:
        return _normalize_iso_label(prefix)
    return token


def _pick_demand_col(columns: Iterable[str]) -> str | None:
    """Return the first column name that looks like a demand/load column."""

    for column in columns:
        lowered = column.lower()
        if any(hint in lowered for hint in _DEMAND_COL_HINTS):
            return column
    return None


def _load_from_flat_table(csv_path: Path) -> pd.DataFrame:
    """Return load forecasts parsed from ``load_forecasts.csv``."""

    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Failed to read {csv_path}: {exc}") from exc

    if frame.empty:
        raise RuntimeError(f"{csv_path} contains no load forecast data")

    rename_map: dict[str, str] = {}
    lower_lookup = {str(column).strip().lower(): column for column in frame.columns}
    if "scenario_name" in lower_lookup:
        rename_map[lower_lookup["scenario_name"]] = "scenario"
    if "scenario" in lower_lookup:
        rename_map.setdefault(lower_lookup["scenario"], "scenario")
    if "region_id" in lower_lookup:
        rename_map[lower_lookup["region_id"]] = "region_id"
    if "region" in lower_lookup:
        rename_map.setdefault(lower_lookup["region"], "region_id")
    if "zone" in lower_lookup:
        rename_map.setdefault(lower_lookup["zone"], "region_id")
    if "iso_zone" in lower_lookup:
        rename_map.setdefault(lower_lookup["iso_zone"], "region_id")
    if "state_or_pr" in lower_lookup:
        rename_map[lower_lookup["state_or_pr"]] = "state"
    if "state" in lower_lookup:
        rename_map.setdefault(lower_lookup["state"], "state")
    if "iso" in lower_lookup:
        rename_map[lower_lookup["iso"]] = "iso"
    if "year" in lower_lookup:
        rename_map[lower_lookup["year"]] = "year"
    if "timestamp" in lower_lookup:
        rename_map.setdefault(lower_lookup["timestamp"], "timestamp")
    if "load_gwh" in lower_lookup:
        rename_map[lower_lookup["load_gwh"]] = "load_gwh"
    if "demand_gwh" in lower_lookup:
        rename_map.setdefault(lower_lookup["demand_gwh"], "load_gwh")
    if "load" in lower_lookup and "load_gwh" not in rename_map:
        rename_map[lower_lookup["load"]] = "load_gwh"
    if "demand" in lower_lookup and "load_gwh" not in rename_map:
        rename_map[lower_lookup["demand"]] = "load_gwh"
    if "demand_mwh" in lower_lookup:
        rename_map[lower_lookup["demand_mwh"]] = "load_mwh"
    if "load_mwh" in lower_lookup:
        rename_map[lower_lookup["load_mwh"]] = "load_mwh"

    if rename_map:
        frame = frame.rename(columns=rename_map)

    if "year" not in frame.columns and "timestamp" in frame.columns:
        timestamp_series = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["year"] = timestamp_series.dt.year
    if "timestamp" in frame.columns:
        frame = frame.drop(columns=["timestamp"])

    required_columns = {"region_id", "scenario", "year"}
    missing = required_columns - {str(col) for col in frame.columns}
    if missing:
        raise ValueError(
            f"{csv_path} is missing required column(s): {', '.join(sorted(missing))}"
        )

    if "load_gwh" not in frame.columns and "load_mwh" not in frame.columns:
        raise ValueError(
            f"{csv_path} must include either a 'load_gwh' or 'load_mwh' column"
        )

    working = frame.copy()
    if "load_mwh" in working.columns:
        working["load_mwh"] = pd.to_numeric(working["load_mwh"], errors="coerce")
    if "load_gwh" not in working.columns and "load_mwh" in working.columns:
        working["load_gwh"] = working["load_mwh"] / 1000.0

    working["region_id"] = working["region_id"].map(_normalize_region)
    working["scenario"] = working["scenario"].astype(str).str.strip()
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype("Int64")
    working["load_gwh"] = pd.to_numeric(working["load_gwh"], errors="coerce")

    working = working.dropna(
        subset=["region_id", "scenario", "year", "load_gwh"], how="any"
    )
    if working.empty:
        raise ValueError(f"{csv_path} contains no valid load forecast records")

    if "iso" in working.columns:
        working["iso"] = working["iso"].map(_normalize_iso_label)
    else:
        working["iso"] = working["region_id"].map(_infer_iso_from_region)

    working["iso"] = working["iso"].astype(str).str.strip().str.upper()
    working["region_id"] = working["region_id"].astype(str).str.strip().str.upper()
    group_keys = ["iso", "region_id", "scenario", "year"]
    numeric_columns = [column for column in ("load_gwh", "load_mwh") if column in working.columns]

    aggregations: dict[str, str] = {column: "sum" for column in numeric_columns}
    for column in working.columns:
        if column in group_keys or column in numeric_columns:
            continue
        aggregations[column] = "first"

    aggregated = (
        working.groupby(group_keys, as_index=False)
        .agg(aggregations)
        .sort_values(group_keys)
        .reset_index(drop=True)
    )

    columns = group_keys + numeric_columns
    extra_columns = [column for column in aggregated.columns if column not in columns]
    return aggregated.loc[:, columns + extra_columns]

def _load_from_directory(root: Path) -> pd.DataFrame:
    """Return load forecasts by crawling the legacy directory structure."""

    frames: list[pd.DataFrame] = []

    for iso_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        iso_key = iso_dir.name.strip().lower()
        iso_label = ISO_DIR_CANON.get(iso_key, iso_key.upper())

        for scenario_dir in sorted(p for p in iso_dir.iterdir() if p.is_dir()):
            scenario_label = scenario_dir.name
            csv_files = sorted(
                p
                for p in scenario_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            )
            if not csv_files:
                continue

            for csv_path in csv_files:
                region_id = _canon_region_id(csv_path.stem)

                try:
                    frame = pd.read_csv(csv_path)
                except Exception as exc:  # pragma: no cover - defensive
                    raise RuntimeError(f"Failed to read {csv_path}: {exc}") from exc

                frame.columns = [str(col).strip() for col in frame.columns]

                year_col = next((c for c in frame.columns if c.lower() == "year"), None)
                if year_col is None:
                    raise ValueError(f"{csv_path}: missing 'Year' column")

                demand_col = _pick_demand_col(frame.columns)
                if demand_col is None:
                    raise ValueError(
                        f"{csv_path}: missing load column (cols={frame.columns.tolist()})"
                    )

                year_series = pd.to_numeric(frame[year_col], errors="coerce")
                load_series = pd.to_numeric(frame[demand_col], errors="coerce")
                demand_lower = demand_col.strip().lower()
                if "mwh" in demand_lower and "gwh" not in demand_lower:
                    load_series = load_series / 1000.0

                trimmed = pd.DataFrame(
                    {
                        "iso": iso_label,
                        "region_id": region_id,
                        "scenario": scenario_label,
                        "year": year_series,
                        "load_gwh": load_series,
                    }
                )

                frames.append(trimmed[["iso", "region_id", "scenario", "year", "load_gwh"]])

    if not frames:
        raise RuntimeError(
            f"No zone CSVs found under {root} (expected <iso>/<scenario>/*.csv)"
        )

    out = pd.concat(frames, ignore_index=True)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["load_gwh"] = pd.to_numeric(out["load_gwh"], errors="coerce")
    out["region_id"] = out["region_id"].astype(str).str.strip().str.upper()
    out["iso"] = out["iso"].astype(str).str.strip().str.upper()
    out["scenario"] = out["scenario"].astype(str).str.strip()

    out = out.dropna(subset=["iso", "region_id", "scenario", "year", "load_gwh"], how="any")
    out = out.sort_values(["iso", "region_id", "scenario", "year"]).reset_index(drop=True)
    return out


def load_load_forecasts(
    root_dir: str | Path = "input/electricity/load_forecasts",
) -> pd.DataFrame:
    """Load all ISO/scenario forecast CSVs found under ``root_dir``."""

    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Load forecast directory not found: {root}")

    if root.is_file():
        return _load_from_flat_table(root)

    csv_path = root / "load_forecasts.csv"
    if csv_path.exists():
        return _load_from_flat_table(csv_path)

    return _load_from_directory(root)


__all__ = ["ISO_DIR_CANON", "load_load_forecasts"]
