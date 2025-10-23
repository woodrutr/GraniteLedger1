"""Utilities to construct allowance policy objects from configuration mappings."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
import warnings
from typing import Any, cast

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(object, None)

from policy.allowance_annual import ConfigError, RGGIPolicyAnnual
from regions.registry import STATE_INDEX


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before accessing policy loader helpers."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for config.policy_loader; install it with `pip install pandas`."
        )

_FILL_DIRECTIVES = {"forward", "ffill", "pad"}


def _normalize_policy_zones(cfg: MutableMapping[str, Any]) -> None:
    """Expand deprecated ``covered_states`` entries into zone-keyed mappings."""

    if "covered_states" not in cfg:
        return

    covered_states = cfg.get("covered_states")
    warnings.warn(
        "'covered_states' is deprecated; specify canonical 'covered_zones' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    def _coerce_state_entries(states: Any) -> Iterable[str]:
        if states is None:
            return []
        if isinstance(states, Mapping):
            return [state for state, flag in states.items() if bool(flag)]
        if isinstance(states, (str, bytes)):
            return [states]
        if isinstance(states, Iterable):
            return list(states)
        raise TypeError("covered_states must be provided as an iterable of state codes")

    def _coerce_zone_mapping(zones: Any) -> dict[str, bool]:
        if zones is None:
            return {}
        if isinstance(zones, Mapping):
            return {
                str(zone).strip(): bool(flag)
                for zone, flag in zones.items()
                if str(zone).strip()
            }
        if isinstance(zones, (str, bytes)):
            zone_label = str(zones).strip()
            return {zone_label: True} if zone_label else {}
        if isinstance(zones, Iterable):
            mapping: dict[str, bool] = {}
            for zone in zones:
                zone_label = str(zone).strip()
                if zone_label:
                    mapping[zone_label] = True
            return mapping
        raise TypeError(
            "covered_zones must be provided as a mapping or iterable of region IDs"
        )

    existing_zones = _coerce_zone_mapping(cfg.get("covered_zones"))
    seen_states: set[str] = set()
    missing_states: list[str] = []

    for entry in _coerce_state_entries(covered_states):
        state_code = str(entry).strip().upper()
        if not state_code or state_code in seen_states:
            continue
        seen_states.add(state_code)
        zones = STATE_INDEX.get(state_code)
        if not zones:
            missing_states.append(state_code)
            continue
        for zone in zones:
            zone_label = str(zone).strip()
            if zone_label:
                existing_zones[zone_label] = True

    if missing_states:
        missing_list = ", ".join(sorted(missing_states))
        raise ValueError(
            f"covered_states references unknown or unmapped states: {missing_list}"
        )

    cfg["covered_zones"] = {zone: existing_zones[zone] for zone in sorted(existing_zones)}


def _coerce_bool_flag(value: Any, key: str, *, default: bool = True) -> bool:
    """Interpret ``value`` as a boolean flag with a fallback ``default``."""

    if value in (None, ""):
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1", "on"}:
            return True
        if normalized in {"false", "f", "no", "n", "0", "off"}:
            return False
    raise TypeError(f"{key} must be a boolean value")


def _coerce_year_list(years: Any) -> list[int]:
    """Convert an iterable of years into a sorted list of unique integers."""

    if years is None:
        return []
    if isinstance(years, Mapping):  # ambiguous structure
        raise TypeError("years must be provided as an iterable of integers, not a mapping")
    if isinstance(years, (str, bytes)):
        raise TypeError("years must be provided as an iterable of integers")

    if not isinstance(years, Iterable):
        raise TypeError("years must be provided as an iterable of integers")

    normalized: list[int] = []
    seen: set[int] = set()
    for value in years:
        try:
            year = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Invalid year value {value!r}; years must be integers") from exc
        if year not in seen:
            normalized.append(year)
            seen.add(year)
    normalized.sort()
    return normalized


def _extract_year_map(entry: Any) -> tuple[Any, bool]:
    """Return the raw year-value mapping and whether forward-fill is requested."""

    fill_forward = False
    year_map = entry
    if isinstance(entry, Mapping):
        fill_token = entry.get("fill")
        if isinstance(fill_token, str) and fill_token.lower() in _FILL_DIRECTIVES:
            fill_forward = True
        elif fill_token and str(fill_token).lower() in _FILL_DIRECTIVES:
            fill_forward = True
        if entry.get("fill_forward"):
            fill_forward = True

        if "values" in entry:
            year_map = entry["values"]
        elif "data" in entry:
            year_map = entry["data"]
        elif "year_map" in entry:
            year_map = entry["year_map"]
        else:  # strip directive keys
            year_map = {k: v for k, v in entry.items() if k not in {"fill", "fill_forward"}}
    return year_map, fill_forward


def _normalize_year_value_pairs(year_map: Any, key: str) -> dict[int, Any]:
    """Convert supported year-value structures into an integer keyed dictionary."""

    _ensure_pandas()

    if isinstance(year_map, Mapping):
        iterator = year_map.items()
    elif isinstance(year_map, Iterable) and not isinstance(year_map, (str, bytes)):
        iterator = []
        for item in year_map:
            if isinstance(item, Mapping):
                year = item.get("year")
                value = item.get("value", item.get("amount"))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                year, value = item
            else:
                continue
            iterator.append((year, value))
    else:
        raise TypeError(f"{key} must map years to values")

    normalized: dict[int, Any] = {}
    for year, value in iterator:
        try:
            year_int = int(year)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{key} has non-integer year {year!r}") from exc
        if year_int in normalized:
            raise ValueError(f"{key} defines duplicate year {year_int}")
        if pd.isna(value):
            raise ValueError(f"{key} has missing value for year {year_int}")
        normalized[year_int] = value

    if not normalized:
        raise ValueError(f"{key} must define at least one year")

    return normalized


def _ensure_contiguous(index: Iterable[int], key: str) -> None:
    """Ensure that the provided index covers every year in its span."""

    years = sorted(int(year) for year in index)
    if not years:
        return
    full_span = set(range(years[0], years[-1] + 1))
    missing = sorted(full_span.difference(years))
    if missing:
        raise ValueError(f"{key} has gaps for years: {missing}")


def _validate_year_alignment(series_map: Mapping[str, pd.Series]) -> list[int]:
    """Ensure each series shares the same ordered year index."""

    iterator = iter(series_map.items())
    first_key, first_series = next(iterator)
    expected_years = [int(year) for year in first_series.index]

    for key, series in iterator:
        years = [int(year) for year in series.index]
        if years != expected_years:
            expected_set = set(expected_years)
            year_set = set(years)
            missing = sorted(expected_set - year_set)
            extra = sorted(year_set - expected_set)
            detail_parts: list[str] = []
            if missing:
                detail_parts.append(f"missing years {missing}")
            if extra:
                detail_parts.append(f"extra years {extra}")
            detail = "; ".join(detail_parts) if detail_parts else "misaligned year order"
            raise ValueError(f"{key} {detail}; expected years {expected_years}")
    return expected_years


def _coerce_numeric_series(series: pd.Series, key: str) -> pd.Series:
    """Cast a series to float values, raising a descriptive error on failure."""

    _ensure_pandas()

    try:
        numeric = series.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} values must be numeric") from exc
    return numeric


def series_from_year_map(cfg: Mapping[str, Any], key: str) -> pd.Series:
    """Extract a pandas Series keyed by year from ``cfg`` for ``key``."""

    _ensure_pandas()

    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")
    if key not in cfg:
        raise KeyError(f"Configuration missing required key: {key}")

    entry = cfg[key]
    year_map, fill_forward = _extract_year_map(entry)
    if fill_forward and key != "floor":
        raise ValueError(f"Fill-forward is only supported for 'floor', not {key!r}")

    normalized = _normalize_year_value_pairs(year_map, key)
    series = pd.Series(normalized).sort_index()
    series.name = key

    target_years = []
    cfg_years = _coerce_year_list(cfg.get("years")) if "years" in cfg else []
    if cfg_years:
        target_years = cfg_years
    elif fill_forward and not series.empty:
        start, end = int(series.index.min()), int(series.index.max())
        target_years = list(range(start, end + 1))

    if target_years:
        reindexed = series.reindex(target_years)
        if fill_forward:
            reindexed = reindexed.ffill()
        series = reindexed

    if series.isna().any():
        missing_years = [int(idx) for idx, value in series.items() if pd.isna(value)]
        raise ValueError(f"{key} missing values for years: {missing_years}")

    if not target_years:
        _ensure_contiguous(series.index, key)

    series.attrs["fill_forward"] = fill_forward
    return series


def load_annual_policy(cfg: Mapping[str, Any]) -> RGGIPolicyAnnual:
    """Construct an :class:`RGGIPolicyAnnual` from a configuration mapping."""

    _ensure_pandas()

    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")

    if isinstance(cfg, MutableMapping):
        _normalize_policy_zones(cfg)
        cfg_dict = dict(cfg)
    else:
        cfg_dict = dict(cfg)
        _normalize_policy_zones(cfg_dict)

    enabled_raw = cfg_dict.get("enabled")
    if enabled_raw is None and "policy_enabled" in cfg_dict:
        enabled_raw = cfg_dict.get("policy_enabled")
    enabled = _coerce_bool_flag(enabled_raw, "enabled", default=True)
    ccr1_enabled = _coerce_bool_flag(cfg_dict.get("ccr1_enabled"), "ccr1_enabled", default=True)
    ccr2_enabled = _coerce_bool_flag(cfg_dict.get("ccr2_enabled"), "ccr2_enabled", default=True)

    control_period_years_value = cfg_dict.get("control_period_years")
    control_period_years = None
    if control_period_years_value not in (None, "", []):
        try:
            control_period_years = int(control_period_years_value)
        except (TypeError, ValueError) as exc:
            raise TypeError("control_period_years must be a positive integer") from exc
        if control_period_years <= 0:
            raise ValueError("control_period_years must be a positive integer")

    resolution_raw = cfg_dict.get("resolution") or cfg_dict.get("type")
    resolution = "annual"
    if resolution_raw not in (None, ""):
        if isinstance(resolution_raw, str):
            resolution_candidate = resolution_raw.strip().lower()
        else:
            resolution_candidate = str(resolution_raw).strip().lower()
        if resolution_candidate in {"annual", "daily"}:
            resolution = resolution_candidate
        else:
            raise ValueError("resolution must be either 'annual' or 'daily'")

    if enabled:
        try:
            cap_series = _coerce_numeric_series(series_from_year_map(cfg_dict, "cap"), "cap")
        except KeyError as exc:
            raise ConfigError("enabled carbon policy requires 'cap' data") from exc
        years = [int(year) for year in cap_series.index]

        cfg_with_years = dict(cfg_dict)
        cfg_with_years["years"] = years

        def _required_series(key: str) -> pd.Series:
            try:
                return _coerce_numeric_series(series_from_year_map(cfg_with_years, key), key)
            except KeyError as exc:  # pragma: no cover - exercised in ConfigError tests
                raise ConfigError(f"enabled carbon policy requires '{key}' data") from exc

        floor_series = _required_series("floor")
        ccr1_trigger = _required_series("ccr1_trigger")
        ccr1_qty = _required_series("ccr1_qty")
        ccr2_trigger = _required_series("ccr2_trigger")
        ccr2_qty = _required_series("ccr2_qty")

        cp_key = "cp_id" if "cp_id" in cfg_dict else "control_period"
        if cp_key not in cfg_dict:
            raise ConfigError("enabled carbon policy requires 'cp_id' data")
        cp_series = series_from_year_map(cfg_with_years, cp_key).astype(str)
        cp_series.name = "cp_id"

        _validate_year_alignment(
            {
                "cap": cap_series,
                "floor": floor_series,
                "ccr1_trigger": ccr1_trigger,
                "ccr1_qty": ccr1_qty,
                "ccr2_trigger": ccr2_trigger,
                "ccr2_qty": ccr2_qty,
                "cp_id": cp_series,
            }
        )
    else:
        target_years = _coerce_year_list(cfg_dict.get("years")) if "years" in cfg_dict else []
        cfg_with_years = dict(cfg_dict)
        if target_years:
            cfg_with_years["years"] = target_years

        def _optional_numeric_series(key: str) -> pd.Series:
            if key not in cfg_dict:
                return pd.Series(dtype=float)
            series = series_from_year_map(cfg_with_years, key)
            return _coerce_numeric_series(series, key)

        cap_series = _optional_numeric_series("cap")
        floor_series = _optional_numeric_series("floor")
        ccr1_trigger = _optional_numeric_series("ccr1_trigger")
        ccr1_qty = _optional_numeric_series("ccr1_qty")
        ccr2_trigger = _optional_numeric_series("ccr2_trigger")
        ccr2_qty = _optional_numeric_series("ccr2_qty")

        cp_series = pd.Series(dtype=str, name="cp_id")
        if "cp_id" in cfg_dict or "control_period" in cfg_dict:
            cp_key = "cp_id" if "cp_id" in cfg_dict else "control_period"
            cp_series = series_from_year_map(cfg_with_years, cp_key).astype(str)
            cp_series.name = "cp_id"

    bank0 = float(cfg_dict.get("bank0", 0.0))
    full_compliance_raw = cfg_dict.get("full_compliance_years", set())
    if full_compliance_raw is None:
        full_compliance_years: set[int] = set()
    else:
        if isinstance(full_compliance_raw, (str, bytes)):
            raise TypeError("full_compliance_years must be an iterable of years")
        try:
            full_compliance_years = {int(year) for year in full_compliance_raw}
        except (TypeError, ValueError) as exc:
            raise TypeError("full_compliance_years must be an iterable of years") from exc

    annual_surrender_frac = float(cfg_dict.get("annual_surrender_frac", 0.5))
    carry_pct = float(cfg_dict.get("carry_pct", 1.0))

    return RGGIPolicyAnnual(
        cap=cap_series,
        floor=floor_series,
        ccr1_trigger=ccr1_trigger,
        ccr1_qty=ccr1_qty,
        ccr2_trigger=ccr2_trigger,
        ccr2_qty=ccr2_qty,
        cp_id=cp_series,
        bank0=float(bank0),
        full_compliance_years=full_compliance_years,
        annual_surrender_frac=float(annual_surrender_frac),
        carry_pct=float(carry_pct),
        enabled=enabled,
        ccr1_enabled=ccr1_enabled,
        ccr2_enabled=ccr2_enabled,
        control_period_length=control_period_years,
        resolution=resolution,
    )


__all__ = ["series_from_year_map", "load_annual_policy"]
