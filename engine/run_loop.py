"""Annual fixed-point integration between dispatch and allowance market."""
from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Dict, Iterable, Mapping, Sequence, cast

try:  # pragma: no cover - optional dependency for canonical region mapping
    from gui.region_metadata import canonical_region_value, region_metadata
except ModuleNotFoundError:  # pragma: no cover - compatibility when GUI is unavailable
    def canonical_region_value(value: object) -> object:  # type: ignore[return-type]
        return value

    def region_metadata(value: object):  # type: ignore[return-type]
        return None

try:  # pragma: no cover - optional dependency guard
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(object, None)


from engine.frames.bundle import ModelInputBundle
from engine.prices.normalize import coerce_price_mapping
from engine.prices.types import CarbonPriceVector
from engine.normalization import normalize_region_id


def _cp_from_entry(entry: object, year: int) -> CarbonPriceVector:
    """Build canonical CarbonPriceVector from a policy entry mapping/object."""

    if isinstance(entry, CarbonPriceVector):
        return entry

    entry_mapping: Mapping[str, Any] | None = entry if isinstance(entry, Mapping) else None

    def _from_entry(names: tuple[str, ...]) -> Any:
        for name in names:
            if entry_mapping is not None and name in entry_mapping:
                value = entry_mapping[name]
            else:
                value = getattr(entry, name, None)
            if value not in (None, ""):
                return value
        return None

    price_mapping = {
        "all": _from_entry(("all", "allowance_price_last", "allowance_price", "p_co2_all")),
        "effective": _from_entry(
            (
                "effective",
                "effective_price_last",
                "carbon_price",
                "effective_carbon_price",
                "p_co2_eff",
            )
        ),
        "exempt": _from_entry(("exempt", "exogenous_price_last", "p_co2_exc")),
        "last": _from_entry(("last", "price_last", "p_co2")),
    }

    default_last_raw = _from_entry(("last", "price_last", "p_co2"))
    try:
        default_last = float(default_last_raw) if default_last_raw is not None else None
    except (TypeError, ValueError):
        default_last = None

    return coerce_price_mapping(price_mapping, default_last=default_last, year=year)

ANNUAL_OUTPUT_COLUMNS = [
    "year",
    "cp_last",
    "allowance_price",
    "cp_all",
    "cp_exempt",
    "cp_effective",
    "iterations",
    "emissions_tons",
    "allowances_minted",
    "allowances_available",
    "bank_start",
    "bank",
    "surrender",
    "obligation",
    "finalized",
    "shortage_flag",
    "ccr1_trigger",
    "ccr1_issued",
    "ccr2_trigger",
    "ccr2_issued",
    "floor",
]


def _synthesized_demand(years: Sequence[int], region: str = "system") -> pd.DataFrame:
    """Return a minimal demand table covering ``years`` for fallback runs."""

    if not years:
        years = [0]

    records = [
        {"year": int(year), "region": str(region), "demand_mwh": 1.0}
        for year in years
    ]
    return pd.DataFrame(records, columns=["year", "region", "demand_mwh"])


def _synthesized_units(region: str = "system") -> pd.DataFrame:
    """Return a simple generating fleet for fallback runs."""

    canonical_region = str(region)
    return pd.DataFrame(
        [
            {
                "unit_id": f"{canonical_region}_stub",
                "region": canonical_region,
                "fuel": "gas",
                "cap_mw": 100.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 9.0,
                "vom_per_mwh": 0.0,
                "fuel_price_per_mmbtu": 3.0,
                "ef_ton_per_mwh": 0.5,
            }
        ]
    )

from dispatch.interface import DispatchResult
from dispatch.lp_network import solve_from_frames as solve_network_from_frames
from dispatch.lp_single import solve as solve_single
from common.regions_schema import REGION_MAP as REGIONS
from engine.normalization import normalize_region_id
from engine.outputs import EngineOutputs
from engine.regions.shares import load_state_to_regions, load_zone_to_state_share
from engine.allowance import enforce_bank_trajectory
from engine.emissions import apply_declining_cap, summarize_emissions
from engine.audits import run_audits
from engine.constants import FLOW_TOL, PRICE_TOL
from policy.allowance_annual import (
    ConfigError,
    RGGIPolicyAnnual,
    allowance_initial_state,
    clear_year as allowance_clear_year,
    finalize_period_if_needed as allowance_finalize_period,
)
from policy.allowance_supply import AllowanceSupply

try:  # pragma: no cover - prefer refactored IO namespace
    from granite_io.frames_api import Frames  # type: ignore
except Exception:  # pragma: no cover - fallback when refactor package unavailable
    from io_loader import Frames


LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass
class FixedPointRuntimeState:
    """Mutable state tracked while iterating the annual fixed-point solver."""

    bank: float
    allowance_state: object
    previous_price: float


def _env_flag(name: str) -> bool:
    """Return True when environment variable ``name`` evaluates to truthy."""

    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no", "off"}


REGION_SHARES_STRICT = _env_flag("REGION_SHARES_STRICT")


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before running the engine."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for engine.run_loop; install it with `pip install pandas`."
        )


def _num(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _sanitize_region_weights(weights: Mapping[str, Any] | None) -> dict[str, float]:
    """Return canonicalized region weight mappings."""

    if not weights:
        return {}

    sanitized: dict[str, float] = {}
    for region, value in weights.items():
        if region in (None, ""):
            continue
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        key = str(region).strip()
        if not key:
            continue
        normalized = normalize_region_id(key)
        canonical = normalized or key
        sanitized[canonical] = sanitized.get(canonical, 0.0) + numeric
    return sanitized


def _coerce_states(states: Sequence[str] | Mapping[str, Any] | None) -> tuple[str, ...]:
    """Normalise optional state selections into canonical identifiers."""

    if states is None:
        return ()
    if isinstance(states, Mapping):
        iterable: Iterable[object] = states.keys()
    elif isinstance(states, (str, bytes)):
        iterable = [states]
    else:
        iterable = states

    normalised: list[str] = []
    seen: set[str] = set()
    for entry in iterable:
        if entry is None:
            continue
        token = str(entry).strip().upper()
        if not token or token in seen:
            continue
        seen.add(token)
        normalised.append(token)
    return tuple(normalised)


def _validate_state_share_coverage(states: tuple[str, ...]) -> None:
    """Ensure configured state shares cover mapped regions when strict mode is on."""

    if not REGION_SHARES_STRICT or not states:
        return

    state_regions = load_state_to_regions()
    share_df = load_zone_to_state_share()

    share_lookup: dict[str, dict[str, float]] = {}
    if not share_df.empty:
        working = share_df.copy()
        working["state"] = working["state"].astype(str).str.upper()
        working["region_id"] = working["region_id"].astype(str)
        grouped = working.groupby(["state", "region_id"])["share"].sum()
        for (state_key, region_id), value in grouped.items():
            share_lookup.setdefault(state_key, {})[region_id] = float(value)

    shortfalls: list[str] = []
    for state in states:
        regions = list(state_regions.get(state, ()))
        if not regions:
            shortfalls.append(f"{state} (no regions configured in state_to_regions.json)")
            continue

        coverage = 0.0
        missing_regions: list[str] = []
        shares_for_state = share_lookup.get(state, {})
        for region in regions:
            weight = float(shares_for_state.get(region, 0.0))
            coverage += weight
            if weight <= 1e-9:
                missing_regions.append(region)

        if coverage >= 0.95:
            continue

        details: list[str] = [f"coverage={coverage:.3f}"]
        if missing_regions:
            details.append("missing regions: " + ", ".join(sorted(missing_regions)))
        else:
            details.append("configured regions: " + ", ".join(sorted(regions)))
        shortfalls.append(f"{state} (" + "; ".join(details) + ")")

    if shortfalls:
        joined = "; ".join(shortfalls)
        raise RuntimeError(
            "REGION_SHARES_STRICT is enabled but zone-to-state shares are incomplete: "
            f"{joined}. Update zone_to_state_share.csv to cover the mapped regions."
        )


def _validate_frame_regions(frames: Frames, extra: Iterable[str] | None = None) -> None:
    """Validate that frame region identifiers exist in the registry."""

    allowed = set(REGIONS.keys())
    encountered: set[str] = set()

    def _collect(values: Iterable[Any]) -> None:
        for entry in values:
            if entry in (None, ""):
                continue
            text = str(entry).strip()
            if not text:
                continue
            normalized = normalize_region_id(text)
            canonical = normalized or text
            if canonical:
                encountered.add(canonical)

    try:
        demand_df = frames.demand()
    except Exception:
        demand_df = None
    if demand_df is not None and not demand_df.empty:
        _collect(demand_df["region"].unique())

    try:
        units_df = frames.units()
    except Exception:
        units_df = None
    if units_df is not None and not units_df.empty:
        _collect(units_df["region"].unique())

    try:
        transmission_df = frames.transmission()
    except Exception:
        transmission_df = None
    if transmission_df is not None and not transmission_df.empty:
        from_series = transmission_df.get("from_region")
        if from_series is not None:
            _collect(from_series.unique())
        to_series = transmission_df.get("to_region")
        if to_series is not None:
            _collect(to_series.unique())

    try:
        coverage_df = frames.coverage()
    except Exception:
        coverage_df = None
    if coverage_df is not None and not coverage_df.empty:
        _collect(coverage_df["region"].unique())

    if extra:
        _collect(extra)

    unknown = sorted(region for region in encountered if region not in allowed)
    if unknown:
        raise ValueError(
            "Frames reference regions not registered in regions.registry: "
            + ", ".join(unknown)
        )


def effective_carbon_price(
    allowance_price: float, exogenous_price: float, deep: bool
) -> float:
    """Return the effective marginal carbon price based on configuration."""

    try:
        allowance_component = float(allowance_price)
    except (TypeError, ValueError):
        allowance_component = 0.0
    try:
        exogenous_component = float(exogenous_price)
    except (TypeError, ValueError):
        exogenous_component = 0.0

    if deep:
        return allowance_component + exogenous_component
    return max(allowance_component, exogenous_component)


def _trace_price_components(
    label: str,
    *,
    year: int | object,
    allowance: object,
    exogenous: object,
    effective: object,
    deep: bool,
) -> None:
    LOGGER.debug(
        "price_components %s year=%s allowance=%s exogenous=%s effective=%s mode=%s",
        label,
        _normalize_progress_year(year),
        _num(allowance),
        _num(exogenous),
        _num(effective),
        "deep" if deep else "shallow",
    )


def _all_regions_marked_covered(result: object) -> bool:
    """Return ``True`` when dispatch outputs indicate universal coverage."""

    coverage_map = getattr(result, "region_coverage", None)
    if isinstance(coverage_map, Mapping):
        seen_flag = False
        for value in coverage_map.values():
            if value is None:
                continue
            seen_flag = True
            if not bool(value):
                return False
        if seen_flag:
            return True

    coverage_detail = getattr(result, "generation_by_coverage", None)
    if isinstance(coverage_detail, Mapping):
        try:
            non_covered = float(coverage_detail.get("non_covered", 0.0) or 0.0)
        except (TypeError, ValueError):
            non_covered = 0.0
        if math.isnan(non_covered):
            non_covered = 0.0
        try:
            covered = float(coverage_detail.get("covered", 0.0) or 0.0)
        except (TypeError, ValueError):
            covered = 0.0
        if math.isnan(covered):
            covered = 0.0
        tolerance = 1e-9
        if non_covered > tolerance:
            return False
        if covered > tolerance:
            return True

    return False


def _coerce_years(policy: Any, years: Iterable[int] | None) -> list[Any]:
    """Return policy index labels corresponding to ``years``.

    Policy series may be indexed by compliance-period labels rather
    than raw calendar years. This helper maps requested years to those
    labels using the policy's ``cap`` index and optional
    ``compliance_year_for`` method.
    """

    if years is None:
        series_index = getattr(policy.cap, 'index', [])
        return list(series_index) if series_index is not None else []

    requested = list(years)
    index_obj = getattr(policy.cap, 'index', None)
    series_index = list(index_obj) if index_obj is not None else []
    index_set = set(series_index)

    mapper = getattr(policy, 'compliance_year_for', None)
    selected: list[Any] = []

    for entry in requested:
        if entry in index_set:
            selected.append(entry)
            continue
        target_year = None
        if callable(mapper):
            try:
                target_year = mapper(entry)
            except Exception:  # pragma: no cover - defensive guard
                target_year = None
        if target_year is None:
            try:
                target_year = int(entry)
            except (TypeError, ValueError):
                target_year = None
        if target_year is None:
            continue
        for label in series_index:
            if callable(mapper):
                try:
                    label_year = mapper(label)
                except Exception:  # pragma: no cover - defensive guard
                    continue
            else:
                try:
                    label_year = int(label)
                except (TypeError, ValueError):
                    continue
            if label_year == target_year:
                selected.append(label)

    if not selected:
        return []

    seen: set[Any] = set()
    ordered: list[Any] = []
    for label in selected:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _normalize_progress_year(value: object) -> object:
    """Return a JSON-serialisable representation for progress callbacks."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _canonical_region_key(value: object) -> str:
    """Return a canonical region identifier suitable for aggregation."""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return stripped
        if stripped in REGIONS:
            return stripped
        if stripped.islower() and "_" not in stripped and "-" not in stripped:
            return stripped

    resolved = canonical_region_value(value)
    if isinstance(resolved, int):
        metadata = region_metadata(resolved)
        if metadata is not None:
            return str(metadata.code or metadata.label or resolved)
        return str(resolved)
    if isinstance(resolved, str):
        metadata = region_metadata(resolved)
        if metadata is not None:
            return str(metadata.code or metadata.label or resolved)
        normalized = resolved.strip()
        return normalized or str(value)
    return str(value)


def _extract_scenario_names(frames: Frames) -> list[str]:
    """Return scenario names recorded in the ``Frames`` metadata if available."""

    meta = getattr(frames, "_meta", {})
    if not isinstance(meta, Mapping):
        return []

    ordered: list[str] = []

    def _append_candidates(values: Iterable[object]) -> None:
        for candidate in values:
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text and text not in ordered:
                ordered.append(text)

    for key in ("scenario_names", "scenario_name", "scenarios", "scenario"):
        if key not in meta:
            continue
        value = meta[key]
        if isinstance(value, str):
            _append_candidates([value])
        elif isinstance(value, Mapping):
            _append_candidates(value.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            _append_candidates(value)
        else:
            _append_candidates([value])
        if ordered:
            break

    return ordered


def _compute_period_weights(policy: Any, periods: Sequence[Any]) -> dict[object, float]:
    """Return fractional weights for ``periods`` grouped by compliance year."""

    mapper = getattr(policy, "compliance_year_for", None)
    counts: defaultdict[int, int] = defaultdict(int)
    period_to_year: dict[Any, int] = {}

    for period in periods:
        calendar_year: int | None = None
        if callable(mapper):
            try:
                mapped = mapper(period)
            except Exception:  # pragma: no cover - defensive guard
                mapped = None
            if mapped is not None:
                try:
                    calendar_year = int(mapped)
                except (TypeError, ValueError):
                    calendar_year = None
        if calendar_year is None:
            try:
                calendar_year = int(period)
            except (TypeError, ValueError):
                calendar_year = None
        if calendar_year is None:
            continue
        period_to_year[period] = calendar_year
        counts[calendar_year] += 1

    weights: dict[object, float] = {}
    for period in periods:
        calendar_year = period_to_year.get(period)
        if calendar_year is None:
            weight = 1.0
        else:
            count = counts.get(calendar_year, 1)
            weight = 1.0 / count if count > 0 else 1.0
        weights[period] = weight
        normalized = _normalize_progress_year(period)
        weights[normalized] = weight
        if calendar_year is not None:
            weights[calendar_year] = weight

    return weights


def _modeled_years(policy: Any, periods: Iterable[Any]) -> list[int]:
    """Return sorted calendar years represented by ``periods``."""

    mapper = getattr(policy, "compliance_year_for", None)
    years: set[int] = set()

    for period in periods:
        calendar_year: int | None = None
        if callable(mapper):
            try:
                mapped = mapper(period)
            except Exception:  # pragma: no cover - defensive guard
                mapped = None
            if mapped is not None:
                try:
                    calendar_year = int(mapped)
                except (TypeError, ValueError):
                    calendar_year = None
        if calendar_year is None:
            try:
                calendar_year = int(period)
            except (TypeError, ValueError):
                calendar_year = None
        if calendar_year is not None:
            years.add(calendar_year)

    return sorted(years)


def _normalize_run_years(years: Iterable[int] | None) -> list[int]:
    """Return an inclusive range spanning the requested simulation years."""

    if years is None:
        return []

    normalized: list[int] = []
    for entry in years:
        try:
            normalized.append(int(entry))
        except (TypeError, ValueError):
            continue

    if not normalized:
        return []

    normalized.sort()
    start = normalized[0]
    end = normalized[-1]
    if end < start:
        start, end = end, start

    return list(range(int(start), int(end) + 1))


def _initial_price_for_year(price_initial: float | Mapping[int, float], year: int, fallback: float) -> float:
    """Return the starting allowance price for ``year`` using configured hints."""

    if isinstance(price_initial, Mapping):
        if year in price_initial:
            return float(price_initial[year])
        if price_initial:
            return float(next(iter(price_initial.values())))
        return float(fallback)
    return float(price_initial)


def _extract_emissions(dispatch_output: object) -> float:
    """Return the emissions ton value from a dispatch output."""

    attr = getattr(dispatch_output, 'emissions_tons', None)
    if attr is not None:
        return float(attr)
    if isinstance(dispatch_output, Mapping) and 'emissions_tons' in dispatch_output:
        return float(dispatch_output['emissions_tons'])
    return float(dispatch_output)


def _policy_value(series: Any, year: int, default: float = 0.0) -> float:
    """Return the policy value for ``year`` falling back to ``default``."""

    if series is None:
        return float(default)
    getter = getattr(series, 'get', None)
    if callable(getter):
        raw = getter(year, default)
    else:
        try:
            raw = series.loc[year]  # type: ignore[index]
        except Exception:  # pragma: no cover - defensive guard
            raw = default
    if raw is None:
        return float(default)
    if pd is not None:
        try:
            if pd.isna(raw):  # type: ignore[arg-type]
                return float(default)
        except AttributeError:  # pragma: no cover - defensive guard
            pass
    try:
        return float(raw)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return float(default)


def _policy_record_for_year(policy: Any, year: int) -> dict[str, float | bool]:
    """Return a mapping of scalar policy values for ``year``."""

    record = {
        'cap': _policy_value(getattr(policy, 'cap', None), year),
        'floor': _policy_value(getattr(policy, 'floor', None), year),
        'ccr1_trigger': _policy_value(getattr(policy, 'ccr1_trigger', None), year),
        'ccr1_qty': _policy_value(getattr(policy, 'ccr1_qty', None), year),
        'ccr2_trigger': _policy_value(getattr(policy, 'ccr2_trigger', None), year),
        'ccr2_qty': _policy_value(getattr(policy, 'ccr2_qty', None), year),
        'enabled': bool(getattr(policy, 'enabled', True)),
        'ccr1_enabled': bool(getattr(policy, 'ccr1_enabled', True)),
        'ccr2_enabled': bool(getattr(policy, 'ccr2_enabled', True)),
    }

    cp_series = getattr(policy, 'cp_id', None)
    cp_value: str | None = None
    if cp_series is not None:
        getter = getattr(cp_series, 'get', None)
        if callable(getter):
            cp_value = getter(year, None)
        else:
            try:
                cp_value = cp_series.loc[year]  # type: ignore[index]
            except Exception:  # pragma: no cover - defensive
                cp_value = None
    record['cp_id'] = str(cp_value) if cp_value is not None else 'NoPolicy'

    surrender_frac = getattr(policy, 'annual_surrender_frac', 0.5)
    try:
        record['annual_surrender_frac'] = float(surrender_frac)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        record['annual_surrender_frac'] = 0.5

    carry_pct = getattr(policy, 'carry_pct', 1.0)
    try:
        record['carry_pct'] = float(carry_pct)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        record['carry_pct'] = 1.0

    record['bank_enabled'] = bool(getattr(policy, 'banking_enabled', True))

    full_compliance_years = set(getattr(policy, 'full_compliance_years', set()))
    if year in full_compliance_years:
        record['full_compliance'] = True
    else:
        try:
            numeric_year = int(year)
        except (TypeError, ValueError):
            numeric_year = None
        if numeric_year is not None and numeric_year in full_compliance_years:
            record['full_compliance'] = True
        else:
            mapper = getattr(policy, 'compliance_year_for', None)
            if callable(mapper):
                try:
                    compliance_year = mapper(year)
                except Exception:  # pragma: no cover - defensive guard
                    compliance_year = None
                record['full_compliance'] = bool(
                    compliance_year is not None and compliance_year in full_compliance_years
                )
            else:
                record['full_compliance'] = False
    return record


def _build_allowance_supply(
    policy: Any,
    year: int,
    *,
    enable_floor: bool,
    enable_ccr: bool,
) -> tuple[AllowanceSupply, dict[str, float | bool]]:
    """Construct :class:`AllowanceSupply` for ``year`` along with the raw record."""

    record = _policy_record_for_year(policy, year)
    ccr1_enabled = bool(record.get('ccr1_enabled', True))
    ccr2_enabled = bool(record.get('ccr2_enabled', True))
    ccr1_qty = float(record.get('ccr1_qty', 0.0)) if ccr1_enabled else 0.0
    ccr2_qty = float(record.get('ccr2_qty', 0.0)) if ccr2_enabled else 0.0
    supply = AllowanceSupply(
        cap=float(record.get('cap', 0.0)),
        floor=float(record.get('floor', 0.0)),
        ccr1_trigger=float(record.get('ccr1_trigger', 0.0)),
        ccr1_qty=ccr1_qty,
        ccr2_trigger=float(record.get('ccr2_trigger', 0.0)),
        ccr2_qty=ccr2_qty,
        enabled=bool(record.get('enabled', True)),
        enable_floor=enable_floor,
        enable_ccr=enable_ccr and (ccr1_qty > 0.0 or ccr2_qty > 0.0),
    )
    record['ccr1_qty'] = ccr1_qty
    record['ccr2_qty'] = ccr2_qty
    return supply, record


def _solve_allowance_market_year(
    dispatch_solver: Callable[[int, float, float], object],
    year: int,
    supply: AllowanceSupply,
    bank_prev: float,
    outstanding_prev: float,
    *,
    policy_enabled: bool,
    high_price: float,
    tol: float,
    max_iter: int,
    annual_surrender_frac: float,
    carry_pct: float,
    banking_enabled: bool,
    carbon_price: float = 0.0,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, object]:
    """Solve for the allowance clearing price for ``year`` using bisection.

    When provided, ``progress_cb`` receives updates for each iteration using the
    signature ``progress_cb(stage, payload)`` where ``stage`` is the literal
    string ``"iteration"`` and ``payload`` includes the ``year``, current
    iteration number, and clearing price estimates.
    """

    tol = max(float(tol), 0.0)
    max_iter_int = max(int(max_iter), 0)
    high_price = float(high_price)

    banking_enabled = bool(banking_enabled)
    deep_flag = bool(getattr(dispatch_solver, "deep_carbon_pricing", False))

    def _report_progress(
        stage: str,
        iteration: int,
        price: float | None,
        *,
        status: str,
        shortage: bool | None = None,
        converged: bool | None = None,
    ) -> None:
        if progress_cb is None:
            return
        payload: dict[str, object] = {
            "year": _normalize_progress_year(year),
            "iteration": int(iteration),
            "status": status,
            "max_iter": int(max_iter_int),
            "tolerance": float(tol),
        }
        if price is not None:
            payload["price"] = float(price)
        if shortage is not None:
            payload["shortage"] = bool(shortage)
        if converged is not None:
            payload["converged"] = bool(converged)
        progress_cb(stage, payload)

    def _issued_quantities(price: float, allowances: float) -> tuple[float, float]:
        if not supply.enable_ccr or not supply.enabled:
            return 0.0, 0.0
        remaining = max(allowances - float(supply.cap), 0.0)
        if remaining <= 0.0:
            return 0.0, 0.0
        issued1 = 0.0
        issued2 = 0.0
        if price >= supply.ccr1_trigger and supply.ccr1_qty > 0.0:
            issued1 = min(float(supply.ccr1_qty), remaining)
            remaining -= issued1
        if price >= supply.ccr2_trigger and supply.ccr2_qty > 0.0:
            issued2 = min(float(supply.ccr2_qty), remaining)
        return issued1, issued2

    bank_prev = max(0.0, float(bank_prev))
    if not banking_enabled:
        bank_prev = 0.0
    outstanding_prev = max(0.0, float(outstanding_prev))
    carry_pct = max(0.0, float(carry_pct)) if banking_enabled else 0.0
    surrender_frac = max(0.0, min(1.0, float(annual_surrender_frac)))

    def _finalize(summary: dict[str, object]) -> dict[str, object]:
        if banking_enabled:
            return summary
        adjusted = dict(summary)
        adjusted['bank_prev'] = 0.0
        adjusted['bank_unadjusted'] = 0.0
        adjusted['bank_new'] = 0.0
        allowances_total = adjusted.get('allowances_total')
        if allowances_total is None:
            allowances_total = adjusted.get('available_allowances', 0.0)
        adjusted['allowances_total'] = float(allowances_total)
        finalize_section = dict(adjusted.get('finalize', {}))
        finalize_section['bank_final'] = 0.0
        adjusted['finalize'] = finalize_section
        return adjusted

    def _log_year_summary(summary: Mapping[str, object]) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return

        def _as_float(value: object, default: float = 0.0) -> float:
            try:
                return float(value) if value is not None else float(default)
            except (TypeError, ValueError):
                return float(default)

        def _as_int(value: object, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        year_value = _as_int(summary.get('year'), year)
        price_estimate = _as_float(summary.get('cp_last'))
        reserve_budget = _as_float(getattr(supply, 'cap', 0.0) if supply.enabled else 0.0)
        reserve_available = _as_float(summary.get('available_allowances'), reserve_budget)
        reserve_withheld = max(0.0, reserve_budget - reserve_available)
        reserve_released = max(0.0, reserve_available - reserve_budget)
        ccr1_issued = _as_float(summary.get('ccr1_issued'))
        ccr2_issued = _as_float(summary.get('ccr2_issued'))
        ccr_active = (ccr1_issued > 0.0) or (ccr2_issued > 0.0)
        ecr_active = reserve_withheld > 0.0
        bank_prev_value = _as_float(summary.get('bank_prev'))
        bank_unadjusted = _as_float(summary.get('bank_unadjusted'))
        bank_new_value = _as_float(summary.get('bank_new'))
        surrendered_value = _as_float(summary.get('surrendered'))
        obligation_value = _as_float(summary.get('obligation_new'))
        emissions_value = _as_float(summary.get('emissions'))
        shortage_flag = bool(summary.get('shortage_flag', False))

        payload = {
            'year': year_value,
            'price_estimate': price_estimate,
            'reserve_budget': reserve_budget,
            'reserve_available': reserve_available,
            'reserve_withheld': reserve_withheld,
            'reserve_released': reserve_released,
            'ecr_active': ecr_active,
            'ccr_active': ccr_active,
            'ccr1_issued': ccr1_issued,
            'ccr2_issued': ccr2_issued,
            'bank_prev': bank_prev_value,
            'bank_unadjusted': bank_unadjusted,
            'bank_new': bank_new_value,
            'surrendered': surrendered_value,
            'obligation': obligation_value,
            'emissions': emissions_value,
            'shortage': shortage_flag,
        }

        LOGGER.debug('allowance_market_year %s', json.dumps(payload, sort_keys=True))

    if not policy_enabled or not supply.enabled:
        clearing_price = 0.0
        dispatch_result = dispatch_solver(
            year, clearing_price, carbon_price=carbon_price, emissions_cap_tons=supply.cap
        )
        emissions = _extract_emissions(dispatch_result)
        allowances = max(supply.available_allowances(clearing_price), emissions)
        ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, allowances)
        minted_allowances = float(max(allowances, emissions))
        total_allowances = minted_allowances  # bank is zero when policy disabled
        _report_progress(
            "iteration",
            0,
            clearing_price,
            status="policy-disabled",
            shortage=False,
            converged=True,
        )
        summary = {
            'year': year,
            'cp_last': float(clearing_price),
            'allowances_cap': float(supply.cap),
            'available_allowances': minted_allowances,
            'allowances_total': total_allowances,
            'bank_prev': 0.0,
            'bank_unadjusted': 0.0,
            'bank_new': 0.0,
            'surrendered': 0.0,
            'obligation_new': 0.0,
            'shortage_flag': False,
            'iterations': 0,
            'emissions': float(emissions),
            'ccr1_issued': float(ccr1_issued),
            'ccr2_issued': float(ccr2_issued),
            'finalize': {
                'finalized': False,
                'bank_final': 0.0,
                'remaining_obligation': 0.0,
                'surrendered_additional': 0.0,
            },
            '_dispatch_result': dispatch_result,
        }
        _log_year_summary(summary)
        return summary

    min_price = supply.floor if supply.enable_floor and supply.enabled else 0.0
    low = max(0.0, float(min_price))
    high = max(low, high_price if high_price > 0.0 else low)

    dispatch_low = dispatch_solver(year, low, carbon_price=carbon_price, emissions_cap_tons=supply.cap)
    emissions_low = _extract_emissions(dispatch_low)
    allowances_low = supply.available_allowances(low)

    total_allowances_low = bank_prev + allowances_low
    if total_allowances_low >= emissions_low:
        clearing_price = supply.enforce_floor(low)
        if clearing_price != low:
            dispatch_low = dispatch_solver(
                year, clearing_price, carbon_price=carbon_price, emissions_cap_tons=supply.cap
            )
            emissions_low = _extract_emissions(dispatch_low)
            allowances_low = supply.available_allowances(clearing_price)
            total_allowances_low = bank_prev + allowances_low
        surrendered = min(surrender_frac * emissions_low, total_allowances_low)
        # Bank should reflect unused allowances after emissions, not after surrender
        # The unsurrendered portion is tracked in the obligation for later settlement
        bank_unadjusted = max(total_allowances_low - emissions_low, 0.0)
        obligation = max(outstanding_prev + emissions_low - surrendered, 0.0)
        ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, allowances_low)
        bank_carry = max(bank_unadjusted * carry_pct, 0.0)
        _report_progress(
            "iteration",
            0,
            clearing_price,
            status="surplus",
            shortage=False,
            converged=True,
        )
        result = {
            'year': year,
            'cp_last': float(clearing_price),
            'allowances_cap': float(supply.cap),
            'available_allowances': float(allowances_low),
            'allowances_total': float(total_allowances_low),
            'bank_prev': float(bank_prev),
            'bank_unadjusted': float(bank_unadjusted),
            'bank_new': float(bank_carry),
            'surrendered': float(surrendered),
            'obligation_new': float(obligation),
            'shortage_flag': False,
            'iterations': 0,
            'emissions': float(emissions_low),
            'ccr1_issued': float(ccr1_issued),
            'ccr2_issued': float(ccr2_issued),
            'finalize': {
                'finalized': False,
                'bank_final': float(bank_carry),
                'remaining_obligation': float(obligation),
                'surrendered_additional': 0.0,
            },
            '_dispatch_result': dispatch_low,
        }
        finalized = _finalize(result)
        _log_year_summary(finalized)
        return finalized

    dispatch_high = dispatch_solver(year, high, carbon_price=carbon_price, emissions_cap_tons=supply.cap)
    emissions_high = _extract_emissions(dispatch_high)
    allowances_high = supply.available_allowances(high)

    if bank_prev + allowances_high < emissions_high - tol:
        clearing_price = supply.enforce_floor(high)
        if clearing_price != high:
            dispatch_high = dispatch_solver(
                year, clearing_price, carbon_price=carbon_price, emissions_cap_tons=supply.cap
            )
            emissions_high = _extract_emissions(dispatch_high)
            allowances_high = supply.available_allowances(clearing_price)
        total_allowances_high = bank_prev + allowances_high
        surrendered = min(surrender_frac * emissions_high, total_allowances_high)
        # Bank should reflect unused allowances after emissions, not after surrender
        # The unsurrendered portion is tracked in the obligation for later settlement
        bank_unadjusted = max(total_allowances_high - emissions_high, 0.0)
        obligation = max(outstanding_prev + emissions_high - surrendered, 0.0)
        ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, allowances_high)
        bank_carry = max(bank_unadjusted * carry_pct, 0.0)
        _report_progress(
            "iteration",
            max_iter_int,
            clearing_price,
            status="shortage",
            shortage=True,
            converged=True,
        )
        result = {
            'year': year,
            'cp_last': float(clearing_price),
            'allowances_cap': float(supply.cap),
            'available_allowances': float(allowances_high),
            'allowances_total': float(total_allowances_high),
            'bank_prev': float(bank_prev),
            'bank_unadjusted': float(bank_unadjusted),
            'bank_new': float(bank_carry),
            'surrendered': float(surrendered),
            'obligation_new': float(obligation),
            'shortage_flag': True,
            'iterations': max_iter_int,
            'emissions': float(emissions_high),
            'ccr1_issued': float(ccr1_issued),
            'ccr2_issued': float(ccr2_issued),
            'finalize': {
                'finalized': False,
                'bank_final': float(bank_carry),
                'remaining_obligation': float(obligation),
                'surrendered_additional': 0.0,
            },
            '_dispatch_result': dispatch_high,
        }
        finalized = _finalize(result)
        _log_year_summary(finalized)
        return finalized

    best_price = high
    best_allowances = allowances_high
    best_emissions = emissions_high
    best_dispatch = dispatch_high
    iteration_count = 0

    low_bound = low
    high_bound = high

    for iteration in range(1, max_iter_int + 1):
        mid = 0.5 * (low_bound + high_bound)
        dispatch_mid = dispatch_solver(year, mid, carbon_price=carbon_price, emissions_cap_tons=supply.cap)
        emissions_mid = _extract_emissions(dispatch_mid)
        allowances_mid = supply.available_allowances(mid)
        total_allowances_mid = bank_prev + allowances_mid
        iteration_count = iteration
        shortage_mid = emissions_mid > total_allowances_mid + tol
        _trace_price_components(
            "iteration",
            year=year,
            allowance=mid,
            exogenous=carbon_price,
            effective=effective_carbon_price(mid, carbon_price, deep_flag),
            deep=deep_flag,
        )
        _report_progress(
            "iteration",
            iteration,
            mid,
            status="bisection",
            shortage=shortage_mid,
        )
        if total_allowances_mid + tol >= emissions_mid:
            best_price = mid
            best_allowances = allowances_mid
            best_emissions = emissions_mid
            best_dispatch = dispatch_mid
            high_bound = mid
            if abs(total_allowances_mid - emissions_mid) <= tol:
                break
        else:
            low_bound = mid
        if abs(high_bound - low_bound) <= max(tol, PRICE_TOL):
            break

    clearing_price = supply.enforce_floor(best_price)
    if clearing_price != best_price:
        best_dispatch = dispatch_solver(
            year, clearing_price, carbon_price=carbon_price, emissions_cap_tons=supply.cap
        )
        best_emissions = _extract_emissions(best_dispatch)
        best_allowances = supply.available_allowances(clearing_price)
    total_allowances = bank_prev + best_allowances

    surrendered = min(surrender_frac * best_emissions, total_allowances)
    # Bank should reflect unused allowances after emissions, not after surrender
    bank_unadjusted = max(total_allowances - best_emissions, 0.0)
    obligation = max(outstanding_prev + best_emissions - surrendered, 0.0)
    shortage_flag = best_emissions > (total_allowances + tol)
    ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, best_allowances)
    bank_carry = max(bank_unadjusted * carry_pct, 0.0)

    _report_progress(
        "iteration",
        iteration_count,
        clearing_price,
        status="final",
        shortage=shortage_flag,
        converged=(max_iter_int <= 0) or (iteration_count < max_iter_int),
    )

    result = {
        'year': year,
        'cp_last': float(clearing_price),
        'allowances_cap': float(supply.cap),
        'available_allowances': float(best_allowances),
        'allowances_total': float(total_allowances),
        'bank_prev': float(bank_prev),
        'bank_unadjusted': float(bank_unadjusted),
        'bank_new': float(bank_carry),
        'surrendered': float(surrendered),
        'obligation_new': float(obligation),
        'shortage_flag': bool(shortage_flag),
        'iterations': iteration_count,
        'emissions': float(best_emissions),
        'ccr1_issued': float(ccr1_issued),
        'ccr2_issued': float(ccr2_issued),
        'finalize': {
            'finalized': False,
            'bank_final': float(bank_carry),
            'remaining_obligation': float(obligation),
            'surrendered_additional': 0.0,
        },
        '_dispatch_result': best_dispatch,
    }
    finalized = _finalize(result)
    _trace_price_components(
        "market",
        year=year,
        allowance=clearing_price,
        exogenous=carbon_price,
        effective=effective_carbon_price(clearing_price, carbon_price, deep_flag),
        deep=deep_flag,
    )
    _log_year_summary(finalized)
    return finalized

class AnnualFixedPointRunner:
    """Orchestrate the fixed-point iteration between dispatch and allowance cost."""

    def __init__(
        self,
        policy: RGGIPolicyAnnual,
        dispatch_model: Callable[[int, float], object],
        *,
        years: Iterable[int] | None,
        price_initial: float | Mapping[int, float],
        tol: float,
        max_iter: int,
        relaxation: float,
    ) -> None:
        if not callable(dispatch_model):
            raise TypeError(
                "dispatch_model must be callable with signature (year, price) -> emissions"
            )

        self.policy = policy
        self.dispatch_model = dispatch_model
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.relaxation = float(relaxation)
        self.price_initial = price_initial
        self.years = tuple(_coerce_years(policy, years))
        self.banking_enabled = bool(getattr(policy, "banking_enabled", True))

        starting_bank = float(getattr(policy, "bank0", 0.0)) if self.banking_enabled else 0.0
        previous_price = (
            float(price_initial) if not isinstance(price_initial, Mapping) else 0.0
        )
        self.runtime = FixedPointRuntimeState(
            bank=max(0.0, starting_bank),
            allowance_state=allowance_initial_state(),
            previous_price=float(previous_price),
        )

    def run(self) -> dict[int, dict]:
        if not getattr(self.policy, "enabled", True):
            return self._run_policy_disabled()

        results: dict[int, dict] = {}
        for year in self.years:
            year_result = self._run_year(year)
            results[year] = year_result
        return results

    def _run_policy_disabled(self) -> dict[int, dict]:
        disabled_results: dict[int, dict] = {}
        for year in self.years:
            emissions = _extract_emissions(self.dispatch_model(year, 0.0))
            disabled_results[year] = {
                "year": year,
                "bank_prev": 0.0,
                "available_allowances": emissions,
                "cp_last": 0.0,
                "cp_all": 0.0,
                "cp_exempt": 0.0,
                "cp_effective": 0.0,
                "ccr1_issued": 0.0,
                "ccr2_issued": 0.0,
                "surrendered": 0.0,
                "bank_new": 0.0,
                "obligation_new": 0.0,
                "shortage_flag": False,
                "iterations": 0,
                "emissions": emissions,
                "finalize": {"finalized": False, "bank_final": 0.0},
            }
        return disabled_results

    def _run_year(self, year: Any) -> dict:
        bank_start = self.runtime.bank if self.banking_enabled else 0.0
        price_guess = _initial_price_for_year(
            self.price_initial, year, self.runtime.previous_price
        )
        iteration_count = 0
        emissions = 0.0
        market_record: dict[str, float | bool | str] | None = None
        accepted_state: object | None = None
        base_state_for_year = self.runtime.allowance_state

        last_iteration_payload: dict[str, object] | None = None

        for iteration in range(1, self.max_iter + 1):
            iteration_count = iteration
            dispatch_output = self.dispatch_model(year, price_guess)
            emissions = _extract_emissions(dispatch_output)
            trial_record, trial_state = allowance_clear_year(
                self.policy,
                base_state_for_year,
                year,
                emissions_tons=emissions,
                bank_prev=bank_start,
                expected_price_guess=price_guess,
            )
            market_record = trial_record
            cleared_price = float(trial_record["cp_last"])
            guess_before_update = float(price_guess)
            residual = float(cleared_price - guess_before_update)
            payload = {
                "year": int(year),
                "iteration": int(iteration),
                "guess": float(guess_before_update),
                "cleared": float(cleared_price),
                "residual": float(residual),
                "tolerance": float(self.tol),
                "relaxation": float(self.relaxation),
            }
            if not math.isfinite(cleared_price):
                payload["status"] = "invalid"
                LOGGER.error(
                    "allowance_fixed_point_iteration %s", json.dumps(payload, sort_keys=True)
                )
                raise RuntimeError(
                    f"Allowance price solver produced a non-finite value for year {year}: "
                    f"{json.dumps(payload, sort_keys=True)}"
                )
            if abs(residual) <= self.tol:
                payload["status"] = "accepted"
                LOGGER.debug(
                    "allowance_fixed_point_iteration %s", json.dumps(payload, sort_keys=True)
                )
                price_guess = cleared_price
                accepted_state = trial_state
                last_iteration_payload = payload
                break
            next_guess = guess_before_update + self.relaxation * residual
            payload["next_guess"] = float(next_guess)
            payload["status"] = "relaxed"
            LOGGER.debug(
                "allowance_fixed_point_iteration %s", json.dumps(payload, sort_keys=True)
            )
            last_iteration_payload = payload
            if not math.isfinite(next_guess) or abs(next_guess) > 1e6:
                payload["status"] = "diverged"
                LOGGER.error(
                    "allowance_fixed_point_iteration %s", json.dumps(payload, sort_keys=True)
                )
                raise RuntimeError(
                    f"Allowance price solver diverged for year {year}: "
                    f"{json.dumps(payload, sort_keys=True)}"
                )
            price_guess = next_guess
        else:  # pragma: no cover - defensive guard
            context_payload = dict(last_iteration_payload or {})
            context_payload.setdefault("year", int(year))
            context_payload.setdefault("iteration", int(iteration_count))
            context_payload.setdefault("tolerance", float(self.tol))
            context_payload["status"] = "failed"
            LOGGER.error(
                "allowance_fixed_point_iteration %s", json.dumps(context_payload, sort_keys=True)
            )
            raise RuntimeError(
                f"Allowance price failed to converge for year {year}: "
                f"{json.dumps(context_payload, sort_keys=True)}"
            )

        assert market_record is not None  # for type checker
        assert accepted_state is not None  # for type checker
        self.runtime.allowance_state = accepted_state
        market_result = dict(market_record)
        market_result["iterations"] = iteration_count
        market_result["emissions"] = emissions

        bank_value = float(market_result.get("bank_new", 0.0))
        if not self.banking_enabled:
            bank_value = 0.0
        finalize_summary, next_state = allowance_finalize_period(
            self.policy, self.runtime.allowance_state, year
        )
        market_result["finalize"] = finalize_summary
        self.runtime.allowance_state = next_state
        if finalize_summary.get("finalized"):
            bank_value = float(finalize_summary.get("bank_final", bank_value))
            if not self.banking_enabled:
                bank_value = 0.0

        self.runtime.bank = bank_value
        self.runtime.previous_price = float(price_guess)

        return market_result


def run_annual_fixed_point(
    policy: RGGIPolicyAnnual,
    dispatch_model: Callable[[int, float], object],
    *,
    years: Iterable[int] | None = None,
    price_initial: float | Mapping[int, float] = 0.0,
    tol: float = 1e-3,
    max_iter: int = 25,
    relaxation: float = 0.5,
) -> dict[int, dict]:
    """Iterate annually to find a fixed-point between dispatch and allowance cost."""

    _ensure_pandas()

    runner = AnnualFixedPointRunner(
        policy,
        dispatch_model,
        years=years,
        price_initial=price_initial,
        tol=tol,
        max_iter=max_iter,
        relaxation=relaxation,
    )
    return runner.run()


def _dispatch_from_frames(
    frames: Frames | Mapping[str, pd.DataFrame],
    *,
    use_network: bool = False,
    period_weights: Mapping[Any, float] | None = None,
    carbon_price_schedule: Mapping[int, float]
    | Mapping[str, Any]
    | float
    | None = None,
    deep_carbon_pricing: bool = False,
) -> Callable[[Any, float, float], object]:

    """Build a dispatch callback that solves using the frame container."""

    _ensure_pandas()

    frames_obj = Frames.coerce(frames, carbon_price_schedule=carbon_price_schedule)

    region_catalog: set[str] = set()
    try:
        demand_table = frames_obj.demand()
    except Exception:
        demand_table = None
    if isinstance(demand_table, pd.DataFrame) and not demand_table.empty:
        if "region" in demand_table.columns:
            region_catalog.update(
                str(value)
                for value in demand_table["region"].dropna().astype(str)
            )
    try:
        unit_table = frames_obj.units()
    except Exception:
        unit_table = None
    if isinstance(unit_table, pd.DataFrame) and not unit_table.empty:
        if "region" in unit_table.columns:
            region_catalog.update(
                str(value)
                for value in unit_table["region"].dropna().astype(str)
            )
    try:
        transmission_table = frames_obj.transmission()
    except Exception:
        transmission_table = None

    demand_rows = int(getattr(demand_table, "shape", (0,))[0]) if isinstance(demand_table, pd.DataFrame) else 0
    unit_rows = int(getattr(unit_table, "shape", (0,))[0]) if isinstance(unit_table, pd.DataFrame) else 0
    interface_rows = (
        int(getattr(transmission_table, "shape", (0,))[0])
        if isinstance(transmission_table, pd.DataFrame)
        else 0
    )

    LOGGER.info(
        "run_loop: frame counts demand_rows=%s generation_rows=%s interface_rows=%s",
        demand_rows,
        unit_rows,
        interface_rows,
    )

    missing_components: list[str] = []
    if demand_rows == 0:
        missing_components.append("demand")
    if unit_rows == 0:
        missing_components.append("generation")
    if interface_rows == 0 and use_network:
        missing_components.append("interfaces")

    if missing_components:
        if demand_rows == 0 and unit_rows == 0 and interface_rows == 0:
            LOGGER.warning(
                "run_loop: all core frames empty (%s); solver bypass permitted",
                ", ".join(missing_components),
            )
        else:
            message = (
                "run_loop: incomplete frames detected; missing components: "
                + ", ".join(missing_components)
            )
            LOGGER.error(message)
            raise RuntimeError(message)

    weights: dict[object, float] = {}
    if period_weights:
        for period, raw_weight in period_weights.items():
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                continue
            if weight <= 0.0:
                continue
            weights[period] = weight
            normalized = _normalize_progress_year(period)
            weights[normalized] = weight

    schedule_lookup: dict[int | None, float] = {}

    def _ingest_schedule(payload: Mapping[Any, Any] | float | None) -> None:
        if payload is None:
            return
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                try:
                    year = int(key) if key is not None else None
                except (TypeError, ValueError):
                    continue
                try:
                    schedule_lookup[year] = float(value)
                except (TypeError, ValueError):
                    continue
            return
        try:
            schedule_lookup[None] = float(payload)
        except (TypeError, ValueError):
            return

    if hasattr(frames_obj, "carbon_price_schedule"):
        _ingest_schedule(getattr(frames_obj, "carbon_price_schedule", None))
    _ingest_schedule(carbon_price_schedule)

    if LOGGER.isEnabledFor(logging.DEBUG):
        schedule_keys = sorted(key for key in schedule_lookup if key is not None)
        LOGGER.debug(
            "dispatch_schedule_lookup keys=%s default=%s",
            schedule_keys,
            schedule_lookup.get(None),
        )

    if not schedule_lookup or all(abs(value) <= 1e-9 for value in schedule_lookup.values()):
        LOGGER.warning(
            "run_loop: carbon_price_schedule empty or zero-valued; check GUI manifest configuration"
        )

    if isinstance(demand_table, pd.DataFrame) and not demand_table.empty:
        demand_years = {
            int(year)
            for year in pd.to_numeric(demand_table.get("year"), errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        }
    else:
        demand_years = set()
    schedule_years = {key for key in schedule_lookup if key is not None}
    if demand_years and schedule_years and not schedule_years.issuperset(demand_years):
        missing = sorted(demand_years - schedule_years)
        LOGGER.warning(
            "run_loop: carbon price schedule missing demand years: missing=%s schedule_years=%s",
            missing,
            sorted(schedule_years),
        )

    def _price_for(period: Any) -> float:
        if not schedule_lookup:
            return 0.0
        try:
            year_key = int(period)
        except (TypeError, ValueError):
            year_key = None
        if year_key is not None and year_key in schedule_lookup:
            return float(schedule_lookup[year_key])
        normalized = _normalize_progress_year(period)
        if isinstance(normalized, int) and normalized in schedule_lookup:
            return float(schedule_lookup[normalized])
        if None in schedule_lookup:
            return float(schedule_lookup[None])
        return 0.0

    def _weight_for(period: Any) -> float:
        weight = weights.get(period)
        if weight is not None:
            return weight
        normalized = _normalize_progress_year(period)
        return float(weights.get(normalized, 1.0))

    def _scaled_frames(period: Any, weight: float) -> Frames:
        if abs(weight - 1.0) <= 1e-12:
            return frames_obj
        try:
            normalized_year = int(period)
        except (TypeError, ValueError):
            normalized_year = period
        demand_df = frames_obj.demand()
        if 'year' not in demand_df.columns:
            return frames_obj
        mask = demand_df['year'] == normalized_year
        if not mask.any():
            return frames_obj
        adjusted = demand_df.copy()
        adjusted.loc[mask, 'demand_mwh'] = (
            adjusted.loc[mask, 'demand_mwh'].astype(float) / weight
        )
        return frames_obj.with_frame('demand', adjusted)

    def _scale_result(result: object, weight: float) -> object:
        if abs(weight - 1.0) <= 1e-12:
            return result
        if isinstance(result, DispatchResult):
            scale = float(weight)
            gen_by_fuel = {key: float(value) * scale for key, value in result.gen_by_fuel.items()}
            emissions_by_region = {
                _canonical_region_key(key): float(value) * scale
                for key, value in result.emissions_by_region.items()
            }
            flows = {key: float(value) * scale for key, value in result.flows.items()}
            demand_by_region = {
                _canonical_region_key(key): float(value) * scale
                for key, value in getattr(result, "demand_by_region", {}).items()
            }
            generation_by_region_detail: Dict[str, Dict[str, float]] = {}
            generation_by_region_total: Dict[str, float] = {}
            detail_source = getattr(result, "generation_detail_by_region", None)
            if isinstance(detail_source, Mapping) and detail_source:
                for region, fuels in detail_source.items():
                    region_key = _canonical_region_key(region)
                    fuel_map: Dict[str, float] = {}
                    total = 0.0
                    for fuel, value in fuels.items():
                        try:
                            scaled_value = float(value) * scale
                        except (TypeError, ValueError):
                            continue
                        fuel_map[str(fuel)] = scaled_value
                        total += scaled_value
                    generation_by_region_detail[region_key] = fuel_map
                    generation_by_region_total[region_key] = total
            else:
                for region, total_value in getattr(result, "generation_by_region", {}).items():
                    region_key = _canonical_region_key(region)
                    try:
                        scaled_total = float(total_value) * scale
                    except (TypeError, ValueError):
                        scaled_total = 0.0
                    generation_by_region_detail[region_key] = {"total": scaled_total}
                    generation_by_region_total[region_key] = scaled_total
            generation_by_coverage = {
                key: float(value) * scale for key, value in result.generation_by_coverage.items()
            }
            emissions_by_fuel = {
                key: float(value) * scale for key, value in result.emissions_by_fuel.items()
            }
            generation_by_unit = {
                key: float(value) * scale for key, value in result.generation_by_unit.items()
            }
            variable_cost_by_fuel = {
                key: float(value) * scale for key, value in result.variable_cost_by_fuel.items()
            }
            allowance_cost_by_fuel = {
                key: float(value) * scale for key, value in result.allowance_cost_by_fuel.items()
            }
            carbon_price_cost_by_fuel = {
                key: float(value) * scale for key, value in result.carbon_price_cost_by_fuel.items()
            }
            total_cost_by_fuel = {
                key: float(value) * scale for key, value in result.total_cost_by_fuel.items()
            }
            capacity_by_region: Dict[str, Dict[str, Dict[str, float]]] = {}
            for region, fuels in getattr(result, "capacity_by_region", {}).items():
                region_key = _canonical_region_key(region)
                if not isinstance(fuels, Mapping):
                    continue
                region_caps: Dict[str, Dict[str, float]] = {}
                for fuel, values in fuels.items():
                    if isinstance(values, Mapping):
                        region_caps[str(fuel)] = {
                            "capacity_mwh": float(values.get("capacity_mwh", 0.0)),
                            "capacity_mw": float(values.get("capacity_mw", 0.0)),
                        }
                    else:
                        try:
                            candidate = float(values)
                        except (TypeError, ValueError):
                            continue
                        region_caps[str(fuel)] = {
                            "capacity_mwh": candidate,
                            "capacity_mw": 0.0,
                        }
                if region_caps:
                    capacity_by_region[region_key] = region_caps

            costs_by_region: Dict[str, Dict[str, Dict[str, float]]] = {}
            for region, fuels in getattr(result, "costs_by_region", {}).items():
                region_key = _canonical_region_key(region)
                if not isinstance(fuels, Mapping):
                    continue
                region_costs: Dict[str, Dict[str, float]] = {}
                for fuel, values in fuels.items():
                    if not isinstance(values, Mapping):
                        continue
                    region_costs[str(fuel)] = {
                        key: float(values.get(key, 0.0)) * scale
                        for key in (
                            "variable_cost",
                            "allowance_cost",
                            "carbon_price_cost",
                            "total_cost",
                        )
                    }
                if region_costs:
                    costs_by_region[region_key] = region_costs
            for region_key in region_catalog:
                emissions_by_region.setdefault(region_key, 0.0)
                demand_by_region.setdefault(region_key, 0.0)
                generation_by_region_detail.setdefault(region_key, {})
                capacity_by_region.setdefault(region_key, {})
                costs_by_region.setdefault(region_key, {})

            capacity_builds = []
            for entry in result.capacity_builds:
                scaled_entry = dict(entry)
                if "generation_mwh" in scaled_entry:
                    scaled_entry["generation_mwh"] = float(scaled_entry["generation_mwh"]) * scale
                if "opex_cost" in scaled_entry:
                    scaled_entry["opex_cost"] = float(scaled_entry["opex_cost"]) * scale
                if "emissions_tons" in scaled_entry:
                    scaled_entry["emissions_tons"] = float(scaled_entry["emissions_tons"]) * scale
                capacity_builds.append(scaled_entry)

            allowance_value = float(getattr(result, "allowance_cost", 0.0))
            carbon_value = float(getattr(result, "carbon_price", 0.0))
            effective_value = float(
                getattr(result, "effective_carbon_price", max(allowance_value, carbon_value))
            )
            schedule_snapshot = getattr(result, "carbon_price_schedule", {})
            if isinstance(schedule_snapshot, Mapping):
                schedule_value = dict(schedule_snapshot)
            else:
                schedule_value = dict(schedule_lookup)

            constraint_duals = {
                key: {inner_key: float(inner_value) for inner_key, inner_value in bucket.items()}
                for key, bucket in getattr(result, "constraint_duals", {}).items()
            }

            return DispatchResult(
                gen_by_fuel=gen_by_fuel,
                region_prices=dict(result.region_prices),
                emissions_tons=float(result.emissions_tons) * scale,
                emissions_by_region=emissions_by_region,
                flows=flows,
                emissions_by_fuel=emissions_by_fuel,
                capacity_mwh_by_fuel=dict(result.capacity_mwh_by_fuel),
                capacity_mw_by_fuel=dict(result.capacity_mw_by_fuel),
                generation_by_unit=generation_by_unit,
                capacity_mwh_by_unit=dict(result.capacity_mwh_by_unit),
                capacity_mw_by_unit=dict(result.capacity_mw_by_unit),
                variable_cost_by_fuel=variable_cost_by_fuel,
                allowance_cost_by_fuel=allowance_cost_by_fuel,
                carbon_price_cost_by_fuel=carbon_price_cost_by_fuel,
                total_cost_by_fuel=total_cost_by_fuel,
                demand_by_region=demand_by_region,
                generation_by_region=generation_by_region_total,
                generation_detail_by_region=generation_by_region_detail,
                generation_by_coverage=generation_by_coverage,
                capacity_by_region=capacity_by_region,
                costs_by_region=costs_by_region,
                imports_to_covered=float(result.imports_to_covered) * scale,
                exports_from_covered=float(result.exports_from_covered) * scale,
                region_coverage=dict(result.region_coverage),
                capacity_builds=capacity_builds,
                allowance_cost=allowance_value,
                carbon_price=carbon_value,
                effective_carbon_price=effective_value,
                carbon_price_schedule=schedule_value,
                constraint_duals=constraint_duals,
                total_cost=float(result.total_cost) * scale,
            )
        return result



    def _normalize_extra_price(value: float | None) -> float:
        if value in (None, ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    deep_flag = bool(deep_carbon_pricing)
    if not deep_flag:
        deep_flag = bool(getattr(frames_obj, "deep_carbon_pricing_enabled", False))

    def dispatch(year: Any, allowance_cost: float, carbon_price: float | None = None, emissions_cap_tons: float | None = None):
        weight = _weight_for(year)
        frames_for_year = _scaled_frames(year, weight)
        schedule_price = _price_for(year)
        allowance_component = float(allowance_cost)

        if carbon_price in (None, ""):
            exogenous_component = schedule_price
        else:
            exogenous_component = _normalize_extra_price(carbon_price)

        dispatch_allowance_cost = float(allowance_component)
        dispatch_carbon_price = max(0.0, float(exogenous_component))

        try:
            year_int = int(year)
        except (TypeError, ValueError):
            year_int = year

        effective_value = effective_carbon_price(
            dispatch_allowance_cost,
            dispatch_carbon_price,
            deep_flag,
        )
        LOGGER.debug(
            "dispatch_call year=%s weight=%.6f allowance_cost=%.6f exogenous=%.6f deep=%s cap=%s",
            year_int,
            float(weight),
            float(dispatch_allowance_cost),
            float(dispatch_carbon_price),
            deep_flag,
            emissions_cap_tons,
        )
        _trace_price_components(
            "dispatch",
            year=year_int,
            allowance=dispatch_allowance_cost,
            exogenous=dispatch_carbon_price,
            effective=effective_value,
            deep=deep_flag,
        )


        if use_network:
            raw_result = solve_network_from_frames(
                frames_for_year,
                year,
                dispatch_allowance_cost,
                carbon_price=dispatch_carbon_price,
                emissions_cap_tons=emissions_cap_tons,
            )
        else:
            coverage_by_region: dict[str, bool] = {}
            try:
                normalized_year = int(year_int)
            except Exception:
                normalized_year = year
            try:
                coverage_by_region = frames_for_year.coverage_for_year(int(normalized_year))
            except Exception:
                coverage_by_region = {}

            if coverage_by_region:
                default_region = next(iter(REGIONS))
                relevant_regions: set[str] = set()

                try:
                    units_df = frames_for_year.units()
                except Exception:
                    units_df = None
                if units_df is not None and "region" in units_df.columns:
                    unit_regions = (
                        units_df["region"].fillna(default_region).astype(str).unique()
                    )
                    relevant_regions.update(str(region) for region in unit_regions)

                try:
                    demand_df = frames_for_year.demand()
                except Exception:
                    demand_df = None
                if demand_df is not None and "region" in demand_df.columns:
                    demand_regions = (
                        demand_df["region"].fillna(default_region).astype(str).unique()
                    )
                    relevant_regions.update(str(region) for region in demand_regions)

                coverage_flags = {
                    region: bool(coverage_by_region.get(region, True))
                    for region in relevant_regions
                }
                if len({flag for flag in coverage_flags.values()}) > 1:
                    raise ConfigError(
                        "Single-region dispatch requires uniform carbon coverage. "
                        "Adjust the carbon coverage selection in the GUI so all assets "
                        "share the same coverage status."
                    )

            raw_result = solve_single(
                year,
                dispatch_allowance_cost,
                frames=frames_for_year,
                carbon_price=dispatch_carbon_price,
                emissions_cap_tons=emissions_cap_tons,
            )

        scaled_result = _scale_result(raw_result, weight)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "dispatch_result year=%s allowance=%.6f carbon_price=%.6f effective=%.6f prices=%s",
                year_int,
                float(getattr(scaled_result, "allowance_cost", dispatch_allowance_cost)),
                float(getattr(scaled_result, "carbon_price", dispatch_carbon_price)),
                float(getattr(scaled_result, "effective_carbon_price", effective_value)),
                dict(getattr(scaled_result, "region_prices", {})),
            )
        return scaled_result

    setattr(dispatch, "deep_carbon_pricing", bool(deep_flag))
    setattr(dispatch, "carbon_price_schedule", dict(schedule_lookup))

    class _DispatchSolver:
        """Callable dispatch wrapper with helpers for load profile injection."""

        def __init__(self) -> None:
            self._dispatch = dispatch
            self.deep_carbon_pricing = bool(deep_flag)
            self.carbon_price_schedule = dict(schedule_lookup)
            self.loads: dict[str, dict[int, float]] = {}

        def set_load_profile(self, df: pd.DataFrame) -> None:
            """Inject custom ISO/zone/year load data into the dispatch frames."""

            if pd is None:  # pragma: no cover - defensive guard when pandas missing
                raise ImportError(
                    "pandas is required to update load profiles for dispatch; install it with `pip install pandas`."
                )

            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)

            if df.empty:
                raise ValueError("load profile DataFrame must contain at least one row")

            required_columns = {"iso", "zone", "year", "load_mwh"}
            missing = [column for column in required_columns if column not in df.columns]
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(
                    f"load profile DataFrame is missing required columns: {missing_list}"
                )

            records: list[dict[str, object]] = []
            load_map: dict[str, dict[int, float]] = {}

            for row in df.itertuples(index=False):
                iso_value = getattr(row, "iso", None)
                zone_value = getattr(row, "zone", None)
                year_value = getattr(row, "year", None)
                demand_value = getattr(row, "load_mwh", None)

                iso_text = str(iso_value).strip() if iso_value not in (None, "") else ""
                zone_text = str(zone_value).strip() if zone_value not in (None, "") else ""
                if not iso_text or not zone_text:
                    continue

                try:
                    year_int = int(year_value)
                except (TypeError, ValueError):
                    continue

                try:
                    demand_float = float(demand_value)
                except (TypeError, ValueError):
                    continue

                key = f"{iso_text}_{zone_text}"
                load_map.setdefault(key, {})[year_int] = demand_float

                region_label = zone_text
                try:
                    region_id = normalize_region_id(region_label, iso=iso_text)
                except ValueError:
                    region_id = region_label or key

                records.append(
                    {
                        "year": year_int,
                        "region": region_id,
                        "demand_mwh": demand_float,
                    }
                )

            if not records:
                raise ValueError("no valid rows found in load profile DataFrame")

            demand_df = pd.DataFrame(records)
            demand_df = (
                demand_df.groupby(["year", "region"], sort=True)["demand_mwh"]
                .sum()
                .reset_index()
            )

            nonlocal frames_obj, demand_table, region_catalog, demand_years
            frames_obj = frames_obj.with_frame("demand", demand_df)
            demand_table = demand_df
            demand_years = {
                int(year)
                for year in pd.to_numeric(demand_df["year"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            }
            region_catalog.update(demand_df["region"].astype(str).tolist())

            schedule_years = {key for key in schedule_lookup if key is not None}
            if demand_years and schedule_years and not schedule_years.issuperset(demand_years):
                missing_years = sorted(demand_years - schedule_years)
                LOGGER.warning(
                    "run_loop: carbon price schedule missing demand years after load profile injection: %s",
                    missing_years,
                )

            self.loads = load_map

        def __call__(
            self,
            year: Any,
            allowance_cost: float,
            carbon_price: float | None = None,
            emissions_cap_tons: float | None = None,
        ):
            return self._dispatch(year, allowance_cost, carbon_price=carbon_price, emissions_cap_tons=emissions_cap_tons)

    return _DispatchSolver()


def run_fixed_point_from_frames(
    frames: Frames | Mapping[str, pd.DataFrame],
    *,
    years: Iterable[int] | None = None,
    price_initial: float | Mapping[int, float] = 0.0,
    tol: float = 1e-3,
    max_iter: int = 25,
    relaxation: float = 0.5,
    use_network: bool = False,
    carbon_price_schedule: Mapping[int, float]
    | Mapping[str, Any]
    | float
    | None = None,
    carbon_price_value: float = 0.0,
    deep_carbon_pricing: bool = False,
) -> dict[int, dict]:
    """Run the annual fixed-point integration using in-memory frames."""

    _ensure_pandas()

    schedule_lookup_source = _prepare_carbon_price_schedule(
        carbon_price_schedule,
        carbon_price_value,
    )

    frames_obj = Frames.coerce(frames, carbon_price_schedule=schedule_lookup_source)
    policy_spec = frames_obj.policy()
    policy = policy_spec.to_policy()
    years_sequence = _coerce_years(policy, years)
    period_weights = _compute_period_weights(policy, years_sequence)
    dispatch_kwargs = dict(
        use_network=use_network,
        period_weights=period_weights,
        carbon_price_schedule=dict(schedule_lookup_source),
        deep_carbon_pricing=deep_carbon_pricing,
    )
    if deep_carbon_pricing:
        dispatch_kwargs["deep_carbon_pricing"] = bool(deep_carbon_pricing)
    dispatch_solver = _dispatch_from_frames(frames_obj, **dispatch_kwargs)

    def dispatch_model(year: int, allowance_cost: float) -> float:
        return _extract_emissions(
            dispatch_solver(year, allowance_cost)
        )

    return run_annual_fixed_point(
        policy,
        dispatch_model,
        years=years,
        price_initial=price_initial,
        tol=tol,
        max_iter=max_iter,
        relaxation=relaxation,
    )


@dataclass
class AnnualAggregation:
    periods: list[Any] = field(default_factory=list)
    price_last: float = 0.0
    allowance_price_last: float = 0.0
    exogenous_price_last: float = 0.0
    effective_price_last: float = 0.0
    iterations_max: int = 0
    emissions_sum: float = 0.0
    available_allowances_sum: float = 0.0
    cap_sum: float = 0.0
    bank_prev_first: float | None = None
    bank_new_last: float = 0.0
    obligation_last: float = 0.0
    shortage_any: bool = False
    finalize_last: dict[str, object] = field(default_factory=dict)
    finalized: bool = False
    bank_final: float = 0.0
    surrendered_sum: float = 0.0
    surrendered_extra: float = 0.0
    ccr1_trigger_last: float | None = None
    ccr2_trigger_last: float | None = None
    ccr1_issued_sum: float = 0.0
    ccr2_issued_sum: float = 0.0
    floor_last: float = 0.0
    emissions_by_region: DefaultDict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    price_by_region: dict[str, float] = field(default_factory=dict)
    flows: DefaultDict[tuple[str, str], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    demand_by_region: DefaultDict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    generation_by_region_detail: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    capacity_by_region_mwh: dict[str, dict[str, float]] = field(default_factory=dict)
    capacity_by_region_mw: dict[str, dict[str, float]] = field(default_factory=dict)
    costs_by_region: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    peak_demand_by_region: DefaultDict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    unserved_capacity_by_region: DefaultDict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    def get(self, key: str, default: object | None = None) -> object | None:
        """Dictionary-style getter used by aggregation routines."""
        if hasattr(self, key):
            value = getattr(self, key)
            return value if value is not None else default
        return default

    def setdefault(self, key: str, default: object) -> object:
        """Mimic ``dict.setdefault`` for dynamically attached aggregates."""
        if not hasattr(self, key) or getattr(self, key) is None:
            setattr(self, key, default)
            return default
        return getattr(self, key)

def _build_engine_outputs(
    years: Sequence[Any],
    raw_results: Mapping[Any, Mapping[str, object]],
    dispatch_solver: Callable[..., object],
    policy: RGGIPolicyAnnual,
    *,
    limiting_factors: Sequence[str] | None = None,
    demand_summary: Mapping[int, float] | None = None,
    states: Sequence[str] | Mapping[str, Any] | None = None,
) -> EngineOutputs:
    """Convert fixed-point results into structured engine outputs."""

    _ensure_pandas()

    aggregated: dict[int, AnnualAggregation] = {}
    registry_order: list[str] = []
    for region in REGIONS:
        canonical = _canonical_region_key(region)
        if not canonical or canonical in registry_order:
            continue
        registry_order.append(canonical)
    all_regions: set[str] = set(registry_order)
    policy_enabled = bool(getattr(policy, "enabled", True))
    banking_enabled = bool(getattr(policy, "banking_enabled", True)) if policy_enabled else False
    deep_enabled = bool(getattr(dispatch_solver, "deep_carbon_pricing", False))
    schedule_snapshot = getattr(dispatch_solver, "carbon_price_schedule", {})
    carbon_price_applied: dict[int, float] = {}

    def _float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _optional_float(value: object) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        else:
            return result

    for period in years:
        summary_raw = raw_results.get(period)
        if isinstance(summary_raw, Mapping):
            summary = dict(summary_raw)
        else:
            summary = {}
        price = float(summary.get("cp_last", 0.0))
        exogenous_component = float(summary.get("cp_exempt", 0.0))
        dispatch_result = summary.pop("_dispatch_result", None)
        if dispatch_result is None:
            dispatch_result = dispatch_solver(
                period, price, carbon_price=exogenous_component
            )
        if (not policy_enabled) and _all_regions_marked_covered(dispatch_result):
            allowance_component = _float(summary.get("cp_all"), 0.0)
            exogenous_component = _float(summary.get("cp_exempt"), 0.0)
            if (
                math.isclose(allowance_component, 0.0, abs_tol=1e-9)
                and not math.isclose(exogenous_component, 0.0, abs_tol=1e-9)
            ):
                allowance_component = exogenous_component
                summary["cp_all"] = allowance_component
                summary["cp_exempt"] = 0.0
                deep_flag = bool(summary.get("deep_carbon_pricing", deep_enabled))
                summary["cp_effective"] = effective_carbon_price(
                    allowance_component,
                    0.0,
                    deep_flag,
                )
                exogenous_component = 0.0
        emissions_total = float(summary.get("emissions", _extract_emissions(dispatch_result)))

        compliance_year = getattr(policy, "compliance_year_for", None)
        if callable(compliance_year):
            try:
                calendar_year = int(compliance_year(period))
            except Exception:  # pragma: no cover - defensive guard
                calendar_year = int(period)
        else:
            try:
                calendar_year = int(period)
            except (TypeError, ValueError):
                calendar_year = hash(period)

        entry = aggregated.setdefault(calendar_year, AnnualAggregation())
        setattr(entry, "deep_pricing", bool(summary.get("deep_carbon_pricing", deep_enabled)))

        entry.periods.append(period)
        entry.price_last = price
        entry.allowance_price_last = _float(summary.get("cp_all"), price)
        entry.exogenous_price_last = _float(summary.get("cp_exempt"), 0.0)
        entry.effective_price_last = _float(summary.get("cp_effective"), price)
        floor_candidate = summary.get("floor")
        if floor_candidate is not None:
            entry.floor_last = _float(floor_candidate, entry.floor_last)
        _trace_price_components(
            "aggregate",
            year=calendar_year,
            allowance=entry.allowance_price_last,
            exogenous=entry.exogenous_price_last,
            effective=entry.effective_price_last,
            deep=bool(getattr(entry, "deep_pricing", deep_enabled)),
        )
        iterations_value = summary.get("iterations", 0)

        entry.iterations_max = max(entry.iterations_max, _int(iterations_value, 0))
        entry.emissions_sum += emissions_total
        entry.available_allowances_sum += _float(summary.get("available_allowances"), 0.0)
        cap_value = summary.get("allowances_cap")
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "annual_aggregation_cap year=%s cap=%s total=%s",
                calendar_year,
                cap_value,
                summary.get("allowances_total"),
            )
        entry.cap_sum += _float(cap_value, _float(summary.get("allowances_total"), 0.0))
        if entry.bank_prev_first is None:
            entry.bank_prev_first = _float(summary.get("bank_prev"), 0.0)
        entry.bank_new_last = _float(summary.get("bank_new"), 0.0)
        entry.obligation_last = _float(summary.get("obligation_new"), 0.0)
        entry.shortage_any = entry.shortage_any or bool(summary.get("shortage_flag", False))
        entry.surrendered_sum += _float(summary.get("surrendered"), 0.0)

        ccr1_trigger = _optional_float(summary.get("ccr1_trigger"))
        if ccr1_trigger is not None:
            entry.ccr1_trigger_last = ccr1_trigger

        ccr2_trigger = _optional_float(summary.get("ccr2_trigger"))
        if ccr2_trigger is not None:
            entry.ccr2_trigger_last = ccr2_trigger

        entry.ccr1_issued_sum += _float(summary.get("ccr1_issued"), 0.0)
        entry.ccr2_issued_sum += _float(summary.get("ccr2_issued"), 0.0)

        carbon_price_applied[calendar_year] = entry.exogenous_price_last

        finalize_raw = summary.get("finalize", {})
        finalize_data = dict(finalize_raw) if isinstance(finalize_raw, Mapping) else {}
        entry.finalize_last = finalize_data
        if finalize_data:
            entry.finalized = bool(finalize_data.get("finalized", entry.finalized))
            entry.bank_final = _float(
                finalize_data.get("bank_final"), entry.bank_new_last
            )
            entry.obligation_last = _float(
                finalize_data.get("remaining_obligation"), entry.obligation_last
            )
            entry.shortage_any = entry.shortage_any or bool(
                finalize_data.get("shortage_flag", False)
            )
            entry.surrendered_extra = _float(
                finalize_data.get("surrendered_additional"), entry.surrendered_extra
            )
        elif not entry.finalized:
            entry.bank_final = entry.bank_new_last

        emissions_by_region = getattr(dispatch_result, "emissions_by_region", None)
        if isinstance(emissions_by_region, Mapping):
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "dispatch_result emissions_by_region period=%s %s",
                    period,
                    dict(emissions_by_region),
                )
            added_region = False
            for region, value in emissions_by_region.items():
                key = _canonical_region_key(region)
                all_regions.add(key)
                entry.emissions_by_region[key] += _float(value, 0.0)
                added_region = True
            if not added_region:
                all_regions.add("system")
                entry.emissions_by_region["system"] += emissions_total
        else:
            all_regions.add("system")
            entry.emissions_by_region["system"] += emissions_total

        region_prices = getattr(dispatch_result, "region_prices", {})
        if isinstance(region_prices, Mapping):
            for region, value in region_prices.items():
                region_key = _canonical_region_key(region)
                all_regions.add(region_key)
                entry.price_by_region[region_key] = _float(value, 0.0)

        flows = getattr(dispatch_result, "flows", {})
        if isinstance(flows, Mapping):
            for key, value in flows.items():
                if isinstance(key, tuple) and len(key) == 2:
                    key_norm = (
                        _canonical_region_key(key[0]),
                        _canonical_region_key(key[1]),
                    )
                    all_regions.update(key_norm)
                    entry.flows[key_norm] += _float(value, 0.0)

        demand_by_region_entry = entry.setdefault(
            "demand_by_region", defaultdict(float)
        )
        for region, value in getattr(dispatch_result, "demand_by_region", {}).items():
            region_key = _canonical_region_key(region)
            demand_by_region_entry[region_key] += _float(value, 0.0)
            all_regions.add(region_key)

        peak_demand_entry = entry.setdefault(
            "peak_demand_by_region", defaultdict(float)
        )
        for region, value in getattr(dispatch_result, "peak_demand_by_region", {}).items():
            region_key = _canonical_region_key(region)
            peak_value = _float(value, 0.0)
            peak_demand_entry[region_key] = max(
                peak_demand_entry.get(region_key, 0.0), peak_value
            )
            all_regions.add(region_key)

        unserved_capacity_entry = entry.setdefault(
            "unserved_capacity_by_region", defaultdict(float)
        )
        for region, value in getattr(dispatch_result, "unserved_capacity_by_region", {}).items():
            region_key = _canonical_region_key(region)
            shortfall_value = _float(value, 0.0)
            unserved_capacity_entry[region_key] = max(
                unserved_capacity_entry.get(region_key, 0.0), shortfall_value
            )
            all_regions.add(region_key)

        generation_region_entry = entry.setdefault("generation_by_region_detail", {})
        for region, fuels in getattr(dispatch_result, "generation_by_region", {}).items():
            region_key = _canonical_region_key(region)
            all_regions.add(region_key)
            region_bucket = generation_region_entry.setdefault(region_key, {})
            if isinstance(fuels, Mapping):
                for fuel, value in fuels.items():
                    fuel_key = str(fuel)
                    region_bucket[fuel_key] = region_bucket.get(fuel_key, 0.0) + _float(value, 0.0)

        capacity_region_mwh_entry = entry.setdefault("capacity_by_region_mwh", {})
        capacity_region_mw_entry = entry.setdefault("capacity_by_region_mw", {})
        for region, fuels in getattr(dispatch_result, "capacity_by_region", {}).items():
            region_key = _canonical_region_key(region)
            all_regions.add(region_key)
            region_mwh = capacity_region_mwh_entry.setdefault(region_key, {})
            region_mw = capacity_region_mw_entry.setdefault(region_key, {})
            if isinstance(fuels, Mapping):
                for fuel, values in fuels.items():
                    fuel_key = str(fuel)
                    if isinstance(values, Mapping):
                        capacity_mwh = _float(values.get("capacity_mwh", 0.0), 0.0)
                        capacity_mw = _float(values.get("capacity_mw", 0.0), 0.0)
                    else:
                        capacity_mwh = _float(values, 0.0)
                        capacity_mw = 0.0
                    region_mwh[fuel_key] = max(region_mwh.get(fuel_key, 0.0), capacity_mwh)
                    region_mw[fuel_key] = max(region_mw.get(fuel_key, 0.0), capacity_mw)

        costs_region_entry = entry.setdefault("costs_by_region", {})
        for region, fuels in getattr(dispatch_result, "costs_by_region", {}).items():
            region_key = _canonical_region_key(region)
            all_regions.add(region_key)
            region_costs = costs_region_entry.setdefault(region_key, {})
            if isinstance(fuels, Mapping):
                for fuel, values in fuels.items():
                    if not isinstance(values, Mapping):
                        continue
                    fuel_key = str(fuel)
                    cost_bucket = region_costs.setdefault(
                        fuel_key,
                        {
                            "variable_cost": 0.0,
                            "allowance_cost": 0.0,
                            "carbon_price_cost": 0.0,
                            "total_cost": 0.0,
                        },
                    )
                    cost_bucket["variable_cost"] += _float(values.get("variable_cost", 0.0), 0.0)
                    cost_bucket["allowance_cost"] += _float(values.get("allowance_cost", 0.0), 0.0)
                    cost_bucket["carbon_price_cost"] += _float(
                        values.get("carbon_price_cost", 0.0), 0.0
                    )
                    cost_bucket["total_cost"] += _float(values.get("total_cost", 0.0), 0.0)

        generation_by_fuel = entry.setdefault("generation_by_fuel", defaultdict(float))
        for fuel, value in getattr(dispatch_result, "gen_by_fuel", {}).items():
            generation_by_fuel[str(fuel)] += float(value)

        emissions_by_fuel_entry = entry.setdefault("emissions_by_fuel", defaultdict(float))
        for fuel, value in getattr(dispatch_result, "emissions_by_fuel", {}).items():
            emissions_by_fuel_entry[str(fuel)] += float(value)

        capacity_mwh_entry = entry.setdefault("capacity_by_fuel_mwh", {})
        for fuel, value in getattr(dispatch_result, "capacity_mwh_by_fuel", {}).items():
            fuel_key = str(fuel)
            current = float(capacity_mwh_entry.get(fuel_key, 0.0))
            candidate = float(value)
            capacity_mwh_entry[fuel_key] = max(current, candidate)

        capacity_mw_entry = entry.setdefault("capacity_by_fuel_mw", {})
        for fuel, value in getattr(dispatch_result, "capacity_mw_by_fuel", {}).items():
            fuel_key = str(fuel)
            current = float(capacity_mw_entry.get(fuel_key, 0.0))
            candidate = float(value)
            capacity_mw_entry[fuel_key] = max(current, candidate)

        generation_by_unit_entry = entry.setdefault("generation_by_unit", defaultdict(float))
        for unit, value in getattr(dispatch_result, "generation_by_unit", {}).items():
            generation_by_unit_entry[str(unit)] += float(value)

        capacity_unit_mwh_entry = entry.setdefault("capacity_by_unit_mwh", {})
        for unit, value in getattr(dispatch_result, "capacity_mwh_by_unit", {}).items():
            unit_key = str(unit)
            current = float(capacity_unit_mwh_entry.get(unit_key, 0.0))
            candidate = float(value)
            capacity_unit_mwh_entry[unit_key] = max(current, candidate)

        capacity_unit_mw_entry = entry.setdefault("capacity_by_unit_mw", {})
        for unit, value in getattr(dispatch_result, "capacity_mw_by_unit", {}).items():
            unit_key = str(unit)
            current = float(capacity_unit_mw_entry.get(unit_key, 0.0))
            candidate = float(value)
            capacity_unit_mw_entry[unit_key] = max(current, candidate)

        variable_cost_entry = entry.setdefault("variable_cost_by_fuel", defaultdict(float))
        for fuel, value in getattr(dispatch_result, "variable_cost_by_fuel", {}).items():
            variable_cost_entry[str(fuel)] += float(value)

        allowance_cost_entry = entry.setdefault("allowance_cost_by_fuel", defaultdict(float))
        for fuel, value in getattr(dispatch_result, "allowance_cost_by_fuel", {}).items():
            allowance_cost_entry[str(fuel)] += float(value)

        carbon_cost_entry = entry.setdefault("carbon_price_cost_by_fuel", defaultdict(float))
        for fuel, value in getattr(dispatch_result, "carbon_price_cost_by_fuel", {}).items():
            carbon_cost_entry[str(fuel)] += float(value)

        total_cost_entry = entry.setdefault("total_cost_by_fuel", defaultdict(float))
        for fuel, value in getattr(dispatch_result, "total_cost_by_fuel", {}).items():
            total_cost_entry[str(fuel)] += float(value)

    annual_rows: list[dict[str, object]] = []
    emissions_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    generation_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    emissions_fuel_rows: list[dict[str, object]] = []
    stranded_rows: list[dict[str, object]] = []
    regional_demand_rows: list[dict[str, object]] = []
    peak_demand_rows: list[dict[str, object]] = []
    regional_generation_rows: list[dict[str, object]] = []
    regional_capacity_rows: list[dict[str, object]] = []
    regional_cost_rows: list[dict[str, object]] = []
    generation_technology_rows: list[dict[str, object]] = []
    capacity_technology_rows: list[dict[str, object]] = []
    emissions_map: dict[str, dict[int, float]] = {}

    if registry_order and all_regions:
        seen_regions: set[str] = set()
        ordered: list[str] = []
        registry_set = set(registry_order)
        for region in registry_order:
            if region not in all_regions or region in seen_regions:
                continue
            ordered.append(region)
            seen_regions.add(region)
        for region in sorted(all_regions - registry_set):
            if region in seen_regions:
                continue
            ordered.append(region)
            seen_regions.add(region)
        region_order = ordered
    elif all_regions:
        region_order = sorted(all_regions)
    else:
        region_order = None

    for year in sorted(aggregated):
        entry = aggregated[year]
        minted_base = float(entry.cap_sum)
        ccr_additions = float(entry.ccr1_issued_sum) + float(entry.ccr2_issued_sum)
        minted_available = float(entry.available_allowances_sum)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "annual_minted_components year=%s cap_sum=%s ccr=%s available=%s",
                year,
                minted_base,
                ccr_additions,
                minted_available,
            )
        base_plus_ccr = minted_base + ccr_additions
        if base_plus_ccr <= 0.0:
            minted = max(minted_available, 0.0)
        else:
            minted = max(base_plus_ccr, minted_available)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "annual_minted_final year=%s minted=%s bank_prev=%s",
                year,
                minted,
                entry.bank_prev_first,
            )
        bank_prev = float(entry.bank_prev_first or 0.0)
        allowances_total = bank_prev + minted

        surrendered_total = float(entry.get("surrendered_sum", 0.0)) + float(entry.get("surrendered_extra", 0.0))
        bank_final = float(entry.get("bank_final", entry.get("bank_new_last", 0.0)))
        obligation_final = float(entry.get("obligation_last", 0.0))
        cp = _cp_from_entry(entry, year)
        price_value = float(cp.last or 0.0)
        allowance_value = float(cp.all)
        exogenous_value = float(cp.exempt)
        effective_value = float(cp.effective)
        iterations_value = int(entry.get("iterations_max", 0))
        shortage_flag = bool(entry.get("shortage_any", False))
        finalized = bool(entry.get("finalized", False))

        ccr1_trigger = float(entry.get("ccr1_trigger_last", 0.0))
        ccr2_trigger = float(entry.get("ccr2_trigger_last", 0.0))
        ccr1_issued = float(entry.get("ccr1_issued_sum", 0.0))
        ccr2_issued = float(entry.get("ccr2_issued_sum", 0.0))

        _trace_price_components(
            "aggregate",
            year=year,
            allowance=allowance_value,
            exogenous=exogenous_value,
            effective=effective_value,
            deep=bool(getattr(entry, "deep_pricing", deep_enabled)),
        )

        if LOGGER.isEnabledFor(logging.DEBUG):
            snapshot = {
                "price_last": entry.price_last,
                "allowance_price_last": entry.allowance_price_last,
                "exogenous_price_last": entry.exogenous_price_last,
                "effective_price_last": entry.effective_price_last,
                "iterations_max": entry.iterations_max,
            }
            LOGGER.debug("annual_aggregation year=%s snapshot=%s", year, snapshot)

        annual_rows.append(
            {
                "year": year,
                "cp_last": price_value,
                "allowance_price": allowance_value,
                "cp_all": allowance_value,
                "cp_exempt": exogenous_value,
                "cp_effective": effective_value,
                "iterations": iterations_value,
                "emissions_tons": float(entry.emissions_sum),
                "allowances_minted": minted,
                "allowances_available": allowances_total,
                "bank": bank_final,
                "bank_start": bank_prev,
                "surrender": surrendered_total,
                "obligation": obligation_final,
                "finalized": finalized,
                "shortage_flag": shortage_flag,
                "ccr1_trigger": ccr1_trigger,
                "ccr1_issued": ccr1_issued,
                "ccr2_trigger": ccr2_trigger,
                "ccr2_issued": ccr2_issued,
                "floor": float(getattr(entry, "floor_last", 0.0)),
            }
        )


        emissions_by_region_entry = entry.emissions_by_region
        if not emissions_by_region_entry:
            emissions_by_region_entry["system"] = float(entry.emissions_sum)
            all_regions.add("system")
        else:
            all_regions.update(str(key) for key in emissions_by_region_entry)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "emissions_by_region year=%s %s",
                year,
                dict(emissions_by_region_entry),
            )
        region_keys: list[str]
        if registry_order:
            seen: set[str] = set()
            region_keys = []
            for candidate in list(registry_order) + list(emissions_by_region_entry):
                canonical = _canonical_region_key(candidate)
                if canonical in seen:
                    continue
                region_keys.append(canonical)
                seen.add(canonical)
        else:
            region_keys = sorted(
                _canonical_region_key(region) for region in emissions_by_region_entry
            )

        for region in region_keys:
            value = float(emissions_by_region_entry.get(region, 0.0))
            emissions_rows.append({"year": year, "region": region, "emissions_tons": value})
            region_record = emissions_map.setdefault(region, {})
            region_record[int(year)] = value

        if entry.price_by_region:
            all_regions.update(str(key) for key in entry.price_by_region)

        price_region_keys = region_keys if registry_order else list(entry.price_by_region)
        if not price_region_keys:
            price_region_keys = region_keys
        for region in price_region_keys:
            value = entry.price_by_region.get(region, 0.0)
            price_rows.append({"year": year, "region": region, "price": float(value)})

        # Ensure flows table includes at least a zero self-flow for every modeled region.
        flow_regions_seen: set[str] = set()
        if not entry.flows:
            default_regions = region_keys or sorted(all_regions) or ["system"]
            for region in default_regions:
                flow_rows.append(
                    {
                        "year": year,
                        "from_region": region,
                        "to_region": region,
                        "flow_mwh": 0.0,
                    }
                )
                flow_regions_seen.add(region)

        for (region_a, region_b), value in entry.flows.items():
            region_a_key = _canonical_region_key(region_a)
            region_b_key = _canonical_region_key(region_b)
            flow_value = float(value)
            if flow_value >= 0.0:
                flow_rows.append(
                    {
                        "year": year,
                        "from_region": region_a_key,
                        "to_region": region_b_key,
                        "flow_mwh": flow_value,
                    }
                )
                flow_regions_seen.update({region_a_key, region_b_key})
            else:
                flow_rows.append(
                    {
                        "year": year,
                        "from_region": region_b_key,
                        "to_region": region_a_key,
                        "flow_mwh": -flow_value,
                    }
                )
                flow_regions_seen.update({region_a_key, region_b_key})

        if region_keys:
            missing_flow_regions = [
                region for region in region_keys if region not in flow_regions_seen
            ]
        else:
            missing_flow_regions = [
                region
                for region in (sorted(all_regions) if all_regions else ["system"])
                if region not in flow_regions_seen
            ]
        for region in missing_flow_regions:
            flow_rows.append(
                {
                    "year": year,
                    "from_region": region,
                    "to_region": region,
                    "flow_mwh": 0.0,
                }
            )
            flow_regions_seen.add(region)

        generation_fuel_map: Mapping[str, float] = entry.get("generation_by_fuel", {})
        for fuel, value in sorted(generation_fuel_map.items()):
            generation_rows.append(
                {"year": year, "fuel": fuel, "generation_mwh": float(value)}
            )
            generation_technology_rows.append(
                {
                    "year": year,
                    "technology": str(fuel),
                    "generation_mwh": float(value),
                }
            )

        capacity_mwh_map: Mapping[str, float] = entry.get("capacity_by_fuel_mwh", {})
        capacity_mw_map: Mapping[str, float] = entry.get("capacity_by_fuel_mw", {})
        capacity_fuels = sorted(set(capacity_mwh_map) | set(capacity_mw_map))
        for fuel in capacity_fuels:
            capacity_rows.append(
                {
                    "year": year,
                    "fuel": fuel,
                    "capacity_mwh": float(capacity_mwh_map.get(fuel, 0.0)),
                    "capacity_mw": float(capacity_mw_map.get(fuel, 0.0)),
                }
            )
            capacity_technology_rows.append(
                {
                    "year": year,
                    "technology": str(fuel),
                    "capacity_mwh": float(capacity_mwh_map.get(fuel, 0.0)),
                    "capacity_mw": float(capacity_mw_map.get(fuel, 0.0)),
                }
            )

        emissions_fuel_map: Mapping[str, float] = entry.get("emissions_by_fuel", {})
        for fuel, value in sorted(emissions_fuel_map.items()):
            emissions_fuel_rows.append(
                {"year": year, "fuel": fuel, "emissions_tons": float(value)}
            )

        variable_map: Mapping[str, float] = entry.get("variable_cost_by_fuel", {})
        allowance_map: Mapping[str, float] = entry.get("allowance_cost_by_fuel", {})
        carbon_price_map: Mapping[str, float] = entry.get("carbon_price_cost_by_fuel", {})
        total_cost_map: Mapping[str, float] = entry.get("total_cost_by_fuel", {})
        cost_fuels = sorted(
            set(variable_map)
            | set(allowance_map)
            | set(carbon_price_map)
            | set(total_cost_map)
        )
        for fuel in cost_fuels:
            cost_rows.append(
                {
                    "year": year,
                    "fuel": fuel,
                    "variable_cost": float(variable_map.get(fuel, 0.0)),
                    "allowance_cost": float(allowance_map.get(fuel, 0.0)),
                    "carbon_price_cost": float(carbon_price_map.get(fuel, 0.0)),
                    "total_cost": float(total_cost_map.get(fuel, 0.0)),
                }
            )

        demand_region_map = getattr(entry, "demand_by_region", {})
        if isinstance(demand_region_map, Mapping):
            for region, value in demand_region_map.items():
                region_key = _canonical_region_key(region)
                regional_demand_rows.append(
                    {
                        "year": year,
                        "region": region_key,
                        "demand_mwh": _float(value, 0.0),
                    }
                )

        peak_region_map = getattr(entry, "peak_demand_by_region", {})
        shortfall_region_map = getattr(entry, "unserved_capacity_by_region", {})
        if isinstance(peak_region_map, Mapping):
            for region, requirement in peak_region_map.items():
                region_key = _canonical_region_key(region)
                shortfall_value = 0.0
                if isinstance(shortfall_region_map, Mapping):
                    shortfall_value = _float(
                        shortfall_region_map.get(region_key, 0.0), 0.0
                    )
                peak_demand_rows.append(
                    {
                        "year": year,
                        "region": region_key,
                        "peak_demand_mw": _float(requirement, 0.0),
                        "shortfall_mw": shortfall_value,
                    }
                )

        generation_region_map = getattr(entry, "generation_by_region_detail", {})
        if isinstance(generation_region_map, Mapping):
            for region, fuels in generation_region_map.items():
                if not isinstance(fuels, Mapping):
                    continue
                region_key = _canonical_region_key(region)
                for fuel, value in fuels.items():
                    regional_generation_rows.append(
                        {
                            "year": year,
                            "region": region_key,
                            "fuel": str(fuel),
                            "generation_mwh": _float(value, 0.0),
                        }
                    )

        capacity_region_mwh_map = getattr(entry, "capacity_by_region_mwh", {})
        capacity_region_mw_map = getattr(entry, "capacity_by_region_mw", {})
        if isinstance(capacity_region_mwh_map, Mapping) or isinstance(capacity_region_mw_map, Mapping):
            region_keys = set(getattr(capacity_region_mwh_map, "keys", lambda: [])()) | set(
                getattr(capacity_region_mw_map, "keys", lambda: [])()
            )
            for region in region_keys:
                region_key = _canonical_region_key(region)
                mwh_map = capacity_region_mwh_map.get(region, {})
                mw_map = capacity_region_mw_map.get(region, {})
                if not isinstance(mwh_map, Mapping):
                    mwh_map = {}
                if not isinstance(mw_map, Mapping):
                    mw_map = {}
                fuels = set(getattr(mwh_map, "keys", lambda: [])()) | set(
                    getattr(mw_map, "keys", lambda: [])()
                )
                for fuel in fuels:
                    fuel_key = str(fuel)
                    regional_capacity_rows.append(
                        {
                            "year": year,
                            "region": region_key,
                            "fuel": fuel_key,
                            "capacity_mwh": _float(mwh_map.get(fuel, 0.0), 0.0),
                            "capacity_mw": _float(mw_map.get(fuel, 0.0), 0.0),
                        }
                    )

        costs_region_map = getattr(entry, "costs_by_region", {})
        if isinstance(costs_region_map, Mapping):
            for region, fuels in costs_region_map.items():
                if not isinstance(fuels, Mapping):
                    continue
                region_key = _canonical_region_key(region)
                for fuel, values in fuels.items():
                    if not isinstance(values, Mapping):
                        continue
                    regional_cost_rows.append(
                        {
                            "year": year,
                            "region": region_key,
                            "fuel": str(fuel),
                            "variable_cost": _float(values.get("variable_cost", 0.0), 0.0),
                            "allowance_cost": _float(values.get("allowance_cost", 0.0), 0.0),
                            "carbon_price_cost": _float(
                                values.get("carbon_price_cost", 0.0), 0.0
                            ),
                            "total_cost": _float(values.get("total_cost", 0.0), 0.0),
                        }
                    )

        capacity_unit_mwh_map: Mapping[str, float] = entry.get("capacity_by_unit_mwh", {})
        capacity_unit_mw_map: Mapping[str, float] = entry.get("capacity_by_unit_mw", {})
        generation_unit_map: Mapping[str, float] = entry.get("generation_by_unit", {})
        for unit, cap_value in capacity_unit_mwh_map.items():
            cap_float = float(cap_value)
            if cap_float <= FLOW_TOL:
                continue
            generated = float(generation_unit_map.get(unit, 0.0))
            if abs(generated) <= FLOW_TOL:
                stranded_rows.append(
                    {
                        "year": year,
                        "unit": unit,
                        "capacity_mwh": cap_float,
                        "capacity_mw": float(capacity_unit_mw_map.get(unit, 0.0)),
                    }
                )


    annual_df = pd.DataFrame(annual_rows, columns=ANNUAL_OUTPUT_COLUMNS)
    if not annual_df.empty:
        annual_df = annual_df.sort_values("year").reset_index(drop=True)
        if (
            schedule_snapshot
            and "cp_exempt" in annual_df.columns
            and annual_df["cp_exempt"].sum() == 0.0
        ):
            LOGGER.warning(
                "cp_exempt is zero-sum; verify carbon_price_schedule mapping and GUI inputs."
            )
        if policy_enabled:
            annual_df = apply_declining_cap(annual_df)
        if policy_enabled and banking_enabled:
            annual_df = enforce_bank_trajectory(annual_df)

    emissions_total: dict[int, float] = {}
    emissions_df = pd.DataFrame(emissions_rows, columns=["year", "region", "emissions_tons"])
    if not emissions_df.empty:
        emissions_df = emissions_df.sort_values(["year", "region"]).reset_index(drop=True)

        # Normalize year, region, and emissions columns
        emissions_df["year"] = pd.to_numeric(emissions_df["year"], errors="coerce")
        emissions_df = emissions_df.dropna(subset=["year"])
        emissions_df["year"] = emissions_df["year"].astype(int)
        emissions_df["region"] = emissions_df["region"].astype(str)
        emissions_df["emissions_tons"] = pd.to_numeric(
            emissions_df["emissions_tons"], errors="coerce"
        ).fillna(0.0)

        # Build region→year→emissions mapping
        filtered_map: dict[str, dict[int, float]] = {}
        for region, group in emissions_df.groupby("region", sort=False):
            region_key = str(region)
            region_years: dict[int, float] = {}
            for row in group.itertuples(index=False):
                try:
                    region_years[int(getattr(row, "year"))] = float(
                        getattr(row, "emissions_tons")
                    )
                except (TypeError, ValueError):
                    continue
            if region_years:
                filtered_map[region_key] = region_years
        totals, derived_map = summarize_emissions(emissions_df)
        if derived_map:
            emissions_map = {region: dict(values) for region, values in derived_map.items()}
        else:
            emissions_map = filtered_map
        emissions_total = totals
    else:
        LOGGER.warning(
            "Regional emissions aggregation produced no rows for years: %s",
            sorted(aggregated),
        )
        emissions_map = {}
        emissions_total = {}


    price_df = pd.DataFrame(price_rows, columns=["year", "region", "price"])
    if not price_df.empty:
        price_df = price_df.sort_values(["year", "region"]).reset_index(drop=True)

    flows_columns = ["year", "from_region", "to_region", "flow_mwh"]
    flows_df = pd.DataFrame(flow_rows, columns=flows_columns)
    if not flows_df.empty:
        flows_df = flows_df.sort_values(flows_columns[:-1]).reset_index(drop=True)

    generation_df = pd.DataFrame(
        generation_rows, columns=["year", "fuel", "generation_mwh"]
    )
    if not generation_df.empty:
        generation_df = generation_df.sort_values(["year", "fuel"]).reset_index(drop=True)

    generation_technology_df = pd.DataFrame(
        generation_technology_rows,
        columns=["year", "technology", "generation_mwh"],
    )
    if not generation_technology_df.empty:
        generation_technology_df = generation_technology_df.sort_values(
            ["year", "technology"]
        ).reset_index(drop=True)

    capacity_df = pd.DataFrame(
        capacity_rows, columns=["year", "fuel", "capacity_mwh", "capacity_mw"]
    )
    if not capacity_df.empty:
        capacity_df = capacity_df.sort_values(["year", "fuel"]).reset_index(drop=True)

    capacity_technology_df = pd.DataFrame(
        capacity_technology_rows,
        columns=["year", "technology", "capacity_mwh", "capacity_mw"],
    )
    if not capacity_technology_df.empty:
        capacity_technology_df = capacity_technology_df.sort_values(
            ["year", "technology"]
        ).reset_index(drop=True)

    cost_df = pd.DataFrame(
        cost_rows,
        columns=[
            "year",
            "fuel",
            "variable_cost",
            "allowance_cost",
            "carbon_price_cost",
            "total_cost",
        ],
    )
    if not cost_df.empty:
        cost_df = cost_df.sort_values(["year", "fuel"]).reset_index(drop=True)

    regional_demand_df = pd.DataFrame(
        regional_demand_rows, columns=["year", "region", "demand_mwh"]
    )
    if not regional_demand_df.empty:
        regional_demand_df = regional_demand_df.sort_values(["year", "region"]).reset_index(
            drop=True
        )

    peak_demand_df = pd.DataFrame(
        peak_demand_rows,
        columns=["year", "region", "peak_demand_mw", "shortfall_mw"],
    )
    if not peak_demand_df.empty:
        peak_demand_df = peak_demand_df.sort_values(["year", "region"]).reset_index(
            drop=True
        )
        peak_demand_df["available_capacity_mw"] = (
            peak_demand_df["peak_demand_mw"] - peak_demand_df["shortfall_mw"]
        ).clip(lower=0.0)

    regional_generation_df = pd.DataFrame(
        regional_generation_rows,
        columns=["year", "region", "fuel", "generation_mwh"],
    )
    if not regional_generation_df.empty:
        regional_generation_df = regional_generation_df.sort_values(
            ["year", "region", "fuel"]
        ).reset_index(drop=True)

    regional_capacity_df = pd.DataFrame(
        regional_capacity_rows,
        columns=["year", "region", "fuel", "capacity_mwh", "capacity_mw"],
    )
    if not regional_capacity_df.empty:
        regional_capacity_df = regional_capacity_df.sort_values(
            ["year", "region", "fuel"]
        ).reset_index(drop=True)

    regional_cost_df = pd.DataFrame(
        regional_cost_rows,
        columns=[
            "year",
            "region",
            "fuel",
            "variable_cost",
            "allowance_cost",
            "carbon_price_cost",
            "total_cost",
        ],
    )
    if not regional_cost_df.empty:
        regional_cost_df = regional_cost_df.sort_values(
            ["year", "region", "fuel"]
        ).reset_index(drop=True)

    emissions_fuel_df = pd.DataFrame(
        emissions_fuel_rows, columns=["year", "fuel", "emissions_tons"]
    )
    if not emissions_fuel_df.empty:
        emissions_fuel_df = emissions_fuel_df.sort_values(["year", "fuel"]).reset_index(
            drop=True
        )

    stranded_df = pd.DataFrame(
        stranded_rows, columns=["year", "unit", "capacity_mwh", "capacity_mw"]
    )
    if not stranded_df.empty:
        stranded_df = stranded_df.sort_values(["year", "unit"]).reset_index(drop=True)

    normalized_demand: dict[int, float] = {}
    if demand_summary:
        for year, value in demand_summary.items():
            try:
                normalized_demand[int(year)] = float(value)
            except (TypeError, ValueError):
                continue

    audits: dict[str, object] = {}
    if normalized_demand:
        audits = run_audits(
            annual_df=annual_df,
            emissions_by_region_df=emissions_df,
            emissions_by_fuel_df=emissions_fuel_df,
            generation_by_fuel_df=generation_df,
            capacity_by_fuel_df=capacity_df,
            cost_by_fuel_df=cost_df,
            stranded_units_df=stranded_df,
            demand_by_year=normalized_demand,
        )

    schedule_dict = dict(schedule_snapshot) if isinstance(schedule_snapshot, Mapping) else {}

    outputs = EngineOutputs(
        annual=annual_df,
        emissions_by_region=emissions_df,
        price_by_region=price_df,
        flows=flows_df,
        limiting_factors=list(limiting_factors or []),
        emissions_total=emissions_total,
        emissions_by_region_map=emissions_map,
        generation_by_fuel=generation_df,
        generation_by_technology=generation_technology_df,
        capacity_by_fuel=capacity_df,
        capacity_by_technology=capacity_technology_df,
        cost_by_fuel=cost_df,
        demand_by_region=regional_demand_df,
        peak_demand_by_region=peak_demand_df,
        generation_by_region=regional_generation_df,
        capacity_by_region=regional_capacity_df,
        cost_by_region=regional_cost_df,
        emissions_by_fuel=emissions_fuel_df,
        stranded_units=stranded_df,
        carbon_price_schedule=schedule_dict,
        carbon_price_applied=carbon_price_applied,
        audits=audits,
        states=states if states is not None else (),
    )

    if LOGGER.isEnabledFor(logging.DEBUG) and carbon_price_applied:
        LOGGER.debug("carbon_price_applied %s", carbon_price_applied)

    if LOGGER.isEnabledFor(logging.INFO):
        summary_table = outputs.emissions_summary_table()
        if summary_table.empty:
            if emissions_df.empty:
                LOGGER.info(
                    "Regional emissions summary: no data available for this run (engine aggregation returned an empty table)."
                )
            else:
                LOGGER.info(
                    "Regional emissions summary: emissions totals evaluate to zero for this run."
                )
        else:
            LOGGER.info(
                "Regional emissions summary:\n%s",
                summary_table.to_string(index=False),
            )

    return outputs


def _schedule_has_price(schedule: Mapping[int | None, float]) -> bool:
    """Return ``True`` when ``schedule`` encodes a non-zero price signal."""

    for key, value in schedule.items():
        if key is not None:
            return True
        try:
            if not math.isclose(float(value), 0.0, abs_tol=1e-12):
                return True
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            continue
    return False


def _coerce_price_schedule(
    schedule: Mapping[int, float] | Mapping[str, Any] | float | None,
) -> dict[int | None, float]:
    """Normalise ``schedule`` into a mapping keyed by integer years.

    When ``schedule`` contains an entry for ``None`` the value is preserved and
    treated as the default price to apply when a specific year is not present.
    """

    if schedule is None:
        return {}

    if isinstance(schedule, Mapping):
        normalised: dict[int | None, float] = {}
        had_candidate = False
        for key, value in schedule.items():
            if key is None and value in (None, ""):
                continue
            had_candidate = True
            if key is None:
                year: int | None = None
            else:
                try:
                    year = int(key)
                except (TypeError, ValueError):
                    continue
            try:
                price = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            normalised[year] = price

        if had_candidate and not normalised:
            raise ValueError(
                "Carbon price schedule contains no valid year/price pairs; "
                "ensure years are numeric and prices are finite values."
            )
        return normalised

    try:
        value = float(schedule)
    except (TypeError, ValueError):
        return {}

    return {None: value}  # type: ignore[index]


def _collect_schedule_targets(
    *,
    run_years: Iterable[int] | None,
    modeled_years: Iterable[int] | None,
    demand_years: Iterable[int] | None,
) -> list[int]:
    """Return sorted unique years that should receive price schedule values."""

    targets: list[int] = []

    for sequence in (run_years, modeled_years, demand_years):
        if not sequence:
            continue
        for entry in sequence:
            try:
                targets.append(int(entry))
            except (TypeError, ValueError):
                continue

    if not targets:
        return []

    return sorted(dict.fromkeys(targets))


def _validate_price_schedule_alignment(
    schedule_lookup: Mapping[int | None, float],
    expanded_schedule: Mapping[int, float],
    target_years: Iterable[int],
) -> None:
    """Validate that ``schedule_lookup`` covers ``target_years`` sensibly."""

    if not schedule_lookup:
        return

    schedule_active = _schedule_has_price(schedule_lookup)
    if not schedule_active:
        return

    target_set = {int(year) for year in target_years}
    if schedule_lookup and target_set and not expanded_schedule:
        raise ValueError(
            "Carbon price schedule could not be aligned with simulation years; "
            "verify demand and dispatch year inputs."
        )

    explicit_years = {
        int(key)
        for key in schedule_lookup
        if key is not None
    }

    if explicit_years and not target_set:
        raise ValueError(
            "Carbon price schedule specifies explicit years but no simulation years "
            "were identified; provide dispatch or demand years to align the schedule."
        )

    if target_set:
        missing = sorted(year for year in target_set if year not in expanded_schedule)
        if missing:
            raise ValueError(
                "Carbon price schedule is missing values for the following years: "
                + ", ".join(str(year) for year in missing)
            )

    if target_set and explicit_years and not (explicit_years & target_set):
        raise ValueError(
            "Carbon price schedule years do not overlap with the modeled dispatch years; "
            "double-check the configured carbon price timeline."
        )


def _prepare_carbon_price_schedule(
    schedule: Mapping[int, float] | Mapping[str, Any] | float | None,
    value: float | None,
) -> dict[int | None, float]:
    """Return a normalised schedule including a default carbon price value."""

    normalised = dict(_coerce_price_schedule(schedule))
    if value in (None, ""):
        default = 0.0
    else:
        try:
            default = float(value)
        except (TypeError, ValueError):
            default = 0.0
    normalised[None] = default
    return normalised


def _expand_price_schedule(
    schedule: Mapping[int | None, float],
    years: Iterable[int],
) -> dict[int, float]:
    """Forward-fill ``schedule`` across ``years`` returning a dense mapping."""

    expanded: dict[int, float] = {}

    normalized_years: list[int] = []
    for entry in years:
        try:
            normalized_years.append(int(entry))
        except (TypeError, ValueError):
            continue

    if not normalized_years:
        return expanded

    schedule_items: list[tuple[int, float]] = []
    default_price = schedule.get(None)

    for key, value in schedule.items():
        if key is None:
            continue
        try:
            year_key = int(key)
        except (TypeError, ValueError):
            continue
        try:
            price_value = float(value)
        except (TypeError, ValueError):
            continue
        schedule_items.append((year_key, price_value))

    schedule_items.sort(key=lambda item: item[0])

    if default_price is None and schedule_items:
        default_price = schedule_items[0][1]
    if default_price is None:
        default_price = 0.0

    current_price = float(default_price)
    total_schedule = len(schedule_items)
    index = 0

    for year in sorted(dict.fromkeys(normalized_years)):
        while index < total_schedule and schedule_items[index][0] <= year:
            current_price = float(schedule_items[index][1])
            index += 1
        expanded[year] = float(current_price)

    return expanded


def run_end_to_end(
    bundle: ModelInputBundle,
    *,
    years: Iterable[int] | None = None,
    carbon_price_schedule: Mapping[int, float] | Mapping[str, Any] | float | None = None,
    **kwargs: Any,
) -> EngineOutputs:
    """Execute the engine using a :class:`ModelInputBundle` abstraction."""

    schedule = carbon_price_schedule
    if schedule is None:
        schedule = bundle.vectors.as_price_schedule("carbon_price")
    run_years = years if years is not None else bundle.years
    return run_end_to_end_from_frames(
        bundle.frames,
        years=run_years,
        carbon_price_schedule=schedule,
        **kwargs,
    )


def run_end_to_end_from_frames(
    frames: Frames | Mapping[str, pd.DataFrame],
    *,
    years: Iterable[int] | None = None,
    price_initial: float | Mapping[int, float] = 0.0,
    tol: float = 1e-3,
    max_iter: int = 25,
    relaxation: float = 0.5,
    enable_floor: bool = True,
    enable_ccr: bool = True,
    price_cap: float = 1000.0,
    use_network: bool = False,
    carbon_price_schedule: Mapping[int, float] | Mapping[str, Any] | float | None = None,
    carbon_price_value: float = 0.0,
    deep_carbon_pricing: bool = False,
    progress_cb: ProgressCallback | None = None,
    stage_cb: ProgressCallback | None = None,
    states: Sequence[str] | Mapping[str, Any] | None = None,
) -> EngineOutputs:
    """Run the integrated dispatch and allowance engine returning structured outputs.

    When ``progress_cb`` is provided the callable receives updates for the
    overall run as well as each simulated year using the ``(stage, payload)``
    convention described in :func:`_solve_allowance_market_year`.
    ``stage_cb`` receives the same updates for high-level milestones and
    preparation steps, enabling richer progress reporting in interactive UIs.
    """

    _ensure_pandas()

    normalized_states = _coerce_states(states)
    _validate_state_share_coverage(normalized_states)

    def _emit_stage(stage: str, payload: Mapping[str, object]) -> None:
        if stage_cb is None and progress_cb is None:
            return
        payload_dict = dict(payload)
        if stage_cb is not None:
            try:
                stage_cb(stage, payload_dict)
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("stage callback failed for %s", stage)
        if progress_cb is not None:
            try:
                progress_cb(stage, dict(payload_dict))
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("progress callback failed for %s", stage)

    try:
        schedule_lookup_source = _prepare_carbon_price_schedule(
            carbon_price_schedule,
            carbon_price_value,
        )
    
        run_years = _normalize_run_years(years)
        price_schedule_defined_years = sorted(
            int(key)
            for key in schedule_lookup_source
            if key is not None
        )
    
        frames_obj = Frames.coerce(frames, carbon_price_schedule=schedule_lookup_source)
        meta_mapping = getattr(frames_obj, "_meta", {})
        if isinstance(meta_mapping, Mapping):
            region_weights_meta = _sanitize_region_weights(
                meta_mapping.get("region_weights")
                if isinstance(meta_mapping.get("region_weights"), Mapping)
                else None
            )
        else:
            region_weights_meta = {}
        if not region_weights_meta and isinstance(frames, Mapping):
            region_weights_meta = _sanitize_region_weights(
                frames.get("region_weights")
                if isinstance(frames.get("region_weights"), Mapping)
                else None
            )
        if region_weights_meta:
            updated_meta = dict(meta_mapping) if isinstance(meta_mapping, Mapping) else {}
            updated_meta["region_weights"] = region_weights_meta
            setattr(frames_obj, "_meta", updated_meta)
        _validate_frame_regions(frames_obj, region_weights_meta.keys())
        _emit_stage("compiling_assumptions", {"status": "start"})
        policy_spec = frames_obj.policy()
        policy = policy_spec.to_policy()
    
        scenario_names = _extract_scenario_names(frames_obj)
    
        fallback_years: list[int] = []
        if run_years:
            fallback_years = list(run_years)
        if not fallback_years:
            try:
                fallback_years = [int(year) for year in policy_spec.cap.index]
            except Exception:
                fallback_years = []
        if not fallback_years:
            fallback_years = list(price_schedule_defined_years)
    
        demand_summary: dict[int, float] = {}
        _emit_stage("loading_load_forecasts", {"status": "start"})
        demand_fallback = False
        try:
            demand_df = frames_obj.demand()
            demand_fallback = False
        except Exception as exc:
            LOGGER.warning(
                "run_loop: demand frame unavailable; synthesizing fallback demand (%s)",
                exc,
            )
            demand_df = _synthesized_demand(fallback_years)
            frames_obj = frames_obj.with_frame("demand", demand_df)
            demand_fallback = True
        demand_regions: list[str] = []
        if not demand_df.empty and "region" in demand_df.columns:
            demand_regions = sorted(
                {
                    str(region).strip()
                    for region in demand_df["region"].dropna().astype(str)
                    if str(region).strip()
                }
            )
        if not demand_df.empty:
            grouped = demand_df.groupby("year")["demand_mwh"].sum()
            demand_summary = {int(year): float(value) for year, value in grouped.items()}
            regions_with_positive_demand = demand_df.loc[
                demand_df["demand_mwh"] > 0.0, "region"
            ].nunique()
            regions_with_demand = int(regions_with_positive_demand)
            demand_total = float(demand_df["demand_mwh"].astype(float).sum())
        else:
            regions_with_demand = 0

        demand_total = 0.0
        demand_years: list[int] = []
        try:
            if "demand_mwh" in demand_df:
                demand_total = float(demand_df["demand_mwh"].astype(float).sum())
        except Exception:  # pragma: no cover - defensive conversion
            demand_total = 0.0
        try:
            if "year" in demand_df:
                demand_years = [
                    int(year)
                    for year in sorted(
                        {int(value) for value in demand_df["year"].dropna().tolist()}
                    )
                ]
        except Exception:  # pragma: no cover - defensive conversion
            demand_years = sorted(demand_summary.keys()) if demand_summary else []
            demand_total = 0.0

        _emit_stage(
            "loading_load_forecasts",
            {
                "status": "complete",
                "regions_with_demand": regions_with_demand,
                "total_demand_mwh": float(demand_total),
                "years": demand_years,
                "demand_summary_mwh": dict(demand_summary),
                "source": "synthesized" if demand_fallback else "frames",
                "status": "complete" if not demand_fallback else "fallback",
                "years": sorted(demand_summary.keys()),
                "regions": demand_regions,
                "demand_by_year_mwh": dict(sorted(demand_summary.items()))
                if demand_summary
                else {},
                "total_demand_mwh": float(demand_total),
            },
        )

        _emit_stage("initializing_fleet", {"status": "start"})
        units_fallback = False
        try:
            units_df = frames_obj.units()
            units_fallback = False
        except Exception as exc:
            LOGGER.warning(
                "run_loop: units frame unavailable; synthesizing fallback fleet (%s)",
                exc,
            )
            units_df = _synthesized_units()
            frames_obj = frames_obj.with_frame("units", units_df)
            units_fallback = True

        if units_df.empty:
            units_with_capacity = 0
            total_available_capacity = 0.0
            unit_regions: list[str] = []
        else:
            if {"cap_mw", "availability"}.issubset(units_df.columns):
                available_capacity = (
                    units_df["cap_mw"].astype(float)
                    * units_df["availability"].astype(float)
                )
            else:
                available_capacity = pd.Series(dtype=float)

            if available_capacity.empty:
                units_with_capacity = 0
                total_available_capacity = 0.0
            else:
                units_with_capacity = int((available_capacity > 0.0).sum())
                total_available_capacity = float(available_capacity.sum())

            if "region" in units_df.columns:
                unit_regions = sorted(
                    {
                        str(region).strip()
                        for region in units_df["region"].dropna().astype(str)
                        if str(region).strip()
                    }
                )
            else:
                unit_regions = []

        total_capacity_mw = 0.0
        try:
            if not units_df.empty and {"cap_mw", "availability"}.issubset(units_df.columns):
                total_capacity_mw = float(
                    units_df["cap_mw"].astype(float)
                    .mul(units_df["availability"].astype(float))
                    .sum()
                )
        except Exception:  # pragma: no cover - defensive conversion
            total_capacity_mw = 0.0

        id_column = "unique_id" if "unique_id" in units_df.columns else "unit_id"
        if id_column in units_df:
            try:
                unit_count = int(units_df[id_column].nunique())
            except Exception:  # pragma: no cover - defensive conversion
                unit_count = int(len(units_df))
        else:
            unit_count = int(len(units_df))

        try:
            region_count = int(units_df["region"].nunique()) if "region" in units_df else 0
        except Exception:  # pragma: no cover - defensive conversion
            region_count = 0
            if not units_df.empty:
                total_available_capacity = float(available_capacity.sum())
                unit_regions = (
                    sorted(
                        {
                            str(region).strip()
                            for region in units_df["region"].dropna().astype(str)
                            if str(region).strip()
                        }
                    )
                    if "region" in units_df
                    else []
                )

        _emit_stage(
            "initializing_fleet",
            {
                "status": "complete",
                "units_with_capacity": units_with_capacity,
                "unit_count": unit_count,
                "regions": region_count,
                "total_capacity_mw": float(total_capacity_mw),
                "source": "synthesized" if units_fallback else "frames",
                "status": "complete" if not units_fallback else "fallback",
                "regions": unit_regions,
                "unit_count": int(len(units_df.index)) if hasattr(units_df, "index") else 0,
                "units_with_capacity": int(units_with_capacity),
                "available_capacity_mw": float(total_available_capacity),
            },
        )

        _emit_stage("building_interfaces", {"status": "start", "use_network": bool(use_network)})
        try:
            transmission_df = frames_obj.transmission()
        except Exception as exc:
            raise RuntimeError(f"E_POLICY_BAD: transmission frame invalid ({exc})") from exc
    
        if transmission_df.empty:
            interfaces_defined = 0
        else:
            interfaces_defined = int((transmission_df["limit_mw"] > 0.0).sum())
    
        pre_solve_counts = {
            "regions_with_demand": regions_with_demand,
            "units": units_with_capacity,
            "interfaces": interfaces_defined,
        }
    
        period_candidates: Iterable[int] | None = run_years if run_years else years
        years_sequence = _coerce_years(policy, period_candidates)
        modeled_years = _modeled_years(policy, years_sequence)
    
        if run_years and modeled_years:
            # use requested range directly when both are available
            years_sequence = list(run_years)
        elif run_years and not modeled_years:
            years_sequence = list(run_years)
            modeled_years = list(run_years)
        elif not run_years and modeled_years:
            start_modeled = modeled_years[0]
            end_modeled = modeled_years[-1]
            run_years = list(range(start_modeled, end_modeled + 1))
            years_sequence = list(run_years)
        else:
            fallback_years: list[int] = []
            for year in price_schedule_defined_years:
                fallback_years.append(int(year))
            if fallback_years:
                start = fallback_years[0]
                end = fallback_years[-1]
                run_years = list(range(start, end + 1))
                years_sequence = list(run_years)
                modeled_years = list(run_years)
            else:
                years_sequence = list(years_sequence)
    
        if not modeled_years:
            modeled_years = list(run_years)
    
        if not run_years and years_sequence:
            try:
                run_years = [int(year) for year in years_sequence]
            except (TypeError, ValueError):
                run_years = []
    
        years_sequence = list(years_sequence)
        selected_years_log = [_normalize_progress_year(period) for period in years_sequence]
    
        if LOGGER.isEnabledFor(logging.INFO):
            LOGGER.info(
                "simulation_selection years=%s scenarios=%s",
                selected_years_log,
                scenario_names or [],
            )
            LOGGER.info(
                "pre_solve_counts regions_with_demand=%d units=%d interfaces=%d use_network=%s",
                pre_solve_counts["regions_with_demand"],
                pre_solve_counts["units"],
                pre_solve_counts["interfaces"],
                bool(use_network),
            )
    
        if pre_solve_counts["regions_with_demand"] == 0:
            raise RuntimeError(
                "E_DEMAND_EMPTY: no demand regions with positive load were provided"
            )
        if pre_solve_counts["units"] == 0:
            raise RuntimeError(
                "E_SUPPLY_EMPTY: no generating units with positive available capacity were provided"
            )
        if not years_sequence:
            raise RuntimeError("E_POLICY_BAD: no simulation years selected")
        if (
            use_network
            and pre_solve_counts["interfaces"] == 0
            and pre_solve_counts["regions_with_demand"] > 1
        ):
            raise RuntimeError(
                "E_NETWORK_REQD_EMPTY: network dispatch requested but no transmission interfaces were supplied"
            )
    
        period_weights = _compute_period_weights(policy, years_sequence)
        default_carbon_price = float(schedule_lookup_source.get(None, 0.0))
    
        schedule_targets = _collect_schedule_targets(
            run_years=run_years,
            modeled_years=modeled_years,
            demand_years=demand_summary.keys() if demand_summary else None,
        )
        expanded_schedule = _expand_price_schedule(
            schedule_lookup_source,
            schedule_targets or (run_years if run_years else modeled_years),
        )
        _validate_price_schedule_alignment(
            schedule_lookup_source,
            expanded_schedule,
            schedule_targets,
        )
        schedule_lookup_source = expanded_schedule
        frames_obj = Frames.coerce(frames_obj, carbon_price_schedule=schedule_lookup_source)
    
        limiting_factors: list[str] = []
    
        if run_years and price_schedule_defined_years:
            max_defined_year = max(price_schedule_defined_years)
            if max_defined_year < run_years[-1]:
                limiting_factors.append(f"Carbon price schedule truncated at {max_defined_year}")
    
        if enable_floor and run_years:
            floor_series = getattr(policy, 'floor', None)
            floor_active = False
            if floor_series is not None:
                for year in run_years[1:]:
                    if _policy_value(floor_series, year, 0.0) > 0.0:
                        floor_active = True
                        break
            if not floor_active:
                limiting_factors.append("Floor schedule inactive")
    
        if enable_ccr and run_years:
            last_ccr_year: int | None = None
            for year in run_years:
                qty1 = _policy_value(getattr(policy, 'ccr1_qty', None), year, 0.0)
                qty2 = _policy_value(getattr(policy, 'ccr2_qty', None), year, 0.0)
                if qty1 > 0.0 or qty2 > 0.0:
                    last_ccr_year = year
            if last_ccr_year is None:
                limiting_factors.append(
                    f"CCR not triggered beyond {run_years[0] - 1 if run_years else 0}"
                )
            elif last_ccr_year < run_years[-1]:
                limiting_factors.append(f"CCR not triggered beyond {last_ccr_year}")
        dispatch_kwargs = dict(
            use_network=use_network,
            period_weights=period_weights,
            carbon_price_schedule=dict(schedule_lookup_source),
            deep_carbon_pricing=deep_carbon_pricing,
        )
        if deep_carbon_pricing:
            dispatch_kwargs["deep_carbon_pricing"] = bool(deep_carbon_pricing)
        dispatch_solver = _dispatch_from_frames(frames_obj, **dispatch_kwargs)
        years_sequence = list(years_sequence)
        total_years = len(years_sequence)
    
        _emit_stage(
            "run_start",
            {
                "total_years": total_years,
                "years": list(years_sequence),
                "max_iter": int(max_iter),
                "tolerance": float(tol),
            },
        )
    
        results: dict[Any, dict[str, object]] = {}
        policy_enabled_global = bool(getattr(policy, 'enabled', True))
        banking_enabled_global = bool(getattr(policy, 'banking_enabled', True))
        bank_prev = float(policy.bank0) if (policy_enabled_global and banking_enabled_global) else 0.0
        bank_prev = max(0.0, bank_prev)
    
        cp_track: dict[str, dict[str, float | list[int] | None]] = {}
    
        price_schedule = dict(schedule_lookup_source)
        bank_exhausted_year: int | None = None
    
        def _price_for_year(period: Any) -> float:
            try:
                year_int = int(period)
            except (TypeError, ValueError):
                year_int = None
            if year_int is not None and year_int in price_schedule:
                return float(price_schedule[year_int])
            return float(default_carbon_price)
    
        def _emit_year_debug(
            year_value: Any,
            summary: Mapping[str, object],
            record_snapshot: Mapping[str, object] | None,
        ) -> None:
            if not LOGGER.isEnabledFor(logging.DEBUG):
                return
    
            def _as_float(value: object, default: float = 0.0) -> float:
                try:
                    return float(value) if value is not None else float(default)
                except (TypeError, ValueError):
                    return float(default)
    
            def _as_int(value: object, default: int) -> int:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return int(default)
    
            record_snapshot = record_snapshot or {}
    
            reserve_cap = _as_float(record_snapshot.get('cap'))
            reserve_floor = _as_float(record_snapshot.get('floor'))
            ecr_trigger = _as_float(record_snapshot.get('ecr_trigger'))
            ccr1_trigger = _as_float(record_snapshot.get('ccr1_trigger'))
            ccr2_trigger = _as_float(record_snapshot.get('ccr2_trigger'))
            reserve_budget = _as_float(
                summary.get('allowances_total'), _as_float(summary.get('available_allowances'))
            )
            reserve_available = _as_float(summary.get('available_allowances'))
            reserve_withheld = max(0.0, reserve_cap - reserve_available)
            reserve_released = max(0.0, reserve_available - reserve_cap)
            payload = {
                'year': _as_int(summary.get('year'), _as_int(year_value, 0)),
                'price_raw': _as_float(summary.get('cp_last')),
                'price_allowance': _as_float(summary.get('cp_all')),
                'price_exogenous': _as_float(summary.get('cp_exempt')),
                'price_effective': _as_float(summary.get('cp_effective')),
                'reserve_cap': reserve_cap,
                'reserve_floor': reserve_floor,
                'ecr_trigger': ecr_trigger,
                'ccr1_trigger': ccr1_trigger,
                'ccr2_trigger': ccr2_trigger,
                'reserve_budget': reserve_budget,
                'reserve_withheld': reserve_withheld,
                'reserve_released': reserve_released,
                'ccr1_release': _as_float(summary.get('ccr1_issued')),
                'ccr2_release': _as_float(summary.get('ccr2_issued')),
                'bank_in': _as_float(summary.get('bank_prev')),
                'bank_out': _as_float(summary.get('bank_new')),
                'bank_outstanding': _as_float(summary.get('obligation_new')),
                'available_allowances': reserve_available,
                'emissions': _as_float(summary.get('emissions')),
                'shortage_flag': bool(summary.get('shortage_flag', False)),
            }
    
            LOGGER.debug('allowance_year_metrics %s', json.dumps(payload, sort_keys=True))
    
        for idx, year in enumerate(years_sequence):
            _emit_stage(
                "year_start",
                {
                    "year": _normalize_progress_year(year),
                    "index": idx,
                    "total_years": total_years,
                    "max_iter": int(max_iter),
                },
            )
    
            carbon_price_value = _price_for_year(year)
    
            if not policy_enabled_global:
                dispatch_result = dispatch_solver(
                    year, 0.0, carbon_price=carbon_price_value
                )
                emissions = _extract_emissions(dispatch_result)
                effective_price = effective_carbon_price(
                    0.0, float(carbon_price_value), deep_carbon_pricing
                )
                summary_disabled: dict[str, object] = {
                    "year": year,
                    "cp_last": float(carbon_price_value),
                    "cp_all": 0.0,
                    "cp_exempt": float(carbon_price_value),
                    "cp_effective": effective_carbon_price(
                        0.0, float(carbon_price_value), deep_carbon_pricing
                    ),
                    "available_allowances": float(emissions),
                    "allowances_total": float(emissions),
                    "bank_prev": 0.0,
                    "bank_unadjusted": 0.0,
                    "bank_new": 0.0,
                    "surrendered": 0.0,
                    "obligation_new": 0.0,
                    "shortage_flag": False,
                    "iterations": 0,
                    "emissions": float(emissions),
                    "ccr1_issued": 0.0,
                    "ccr2_issued": 0.0,
                    "finalize": {
                        "finalized": False,
                        "bank_final": 0.0,
                        "remaining_obligation": 0.0,
                        "surrendered_additional": 0.0,
                    },
                    "_dispatch_result": dispatch_result,
                }
    
                results[year] = summary_disabled
                _emit_year_debug(year, summary_disabled, {})
                if progress_cb is not None:
                    progress_cb(
                        "year_complete",
                        {
                            "year": int(year),
                            "index": idx,
                            "total_years": total_years,
                            "shortage": False,
                            "price": float(carbon_price_value),
                            "iterations": 0,
                        },
                    )
                continue
    
            supply, record = _build_allowance_supply(
                policy,
                year,
                enable_floor=enable_floor,
                enable_ccr=enable_ccr,
            )
    
            policy_enabled_year = bool(supply.enabled) and policy_enabled_global
            cp_id = str(record.get('cp_id', 'NoPolicy'))
            surrender_frac = float(
                record.get('annual_surrender_frac', getattr(policy, 'annual_surrender_frac', 0.5))
            )
            carry_pct = float(record.get('carry_pct', getattr(policy, 'carry_pct', 1.0)))
            banking_enabled_year = banking_enabled_global and bool(
                record.get('bank_enabled', banking_enabled_global)
            )
            if not policy_enabled_year:
                banking_enabled_year = False
            if not banking_enabled_year:
                carry_pct = 0.0
    
            bank_prev_effective = bank_prev if (banking_enabled_year and policy_enabled_year) else 0.0
    
            if (
                banking_enabled_global
                and policy_enabled_year
                and bank_exhausted_year is None
                and idx < len(years_sequence) - 1
                and bank_prev_effective <= 1e-9
            ):
                try:
                    bank_exhausted_year = int(year)
                except (TypeError, ValueError):
                    bank_exhausted_year = None
    
            state: dict[str, float | list[int] | None] | None = None
            outstanding_prev = 0.0
            if policy_enabled_year:
                state = cp_track.setdefault(
                    cp_id,
                    {
                        'emissions': 0.0,
                        'surrendered': 0.0,
                        'cap': 0.0,
                        'ccr1': 0.0,
                        'ccr2': 0.0,
                        'bank_start': bank_prev_effective,
                        'outstanding': 0.0,
                        'years': [],
                    },
                )
    
                if state.get('bank_start') is None:
                    state['bank_start'] = bank_prev_effective
    
                years_list = state.setdefault('years', [])
                if isinstance(years_list, list) and year not in years_list:
                    years_list.append(year)
    
                outstanding_prev = float(state.get('outstanding', 0.0))
    
            summary = _solve_allowance_market_year(
                dispatch_solver,
                year,
                supply,
                bank_prev_effective,
                outstanding_prev,
                policy_enabled=policy_enabled_year,
                high_price=price_cap,
                tol=tol,
                max_iter=max_iter,
                annual_surrender_frac=surrender_frac,
                carry_pct=carry_pct,
                banking_enabled=banking_enabled_year,
                carbon_price=carbon_price_value,
                progress_cb=progress_cb,
            )
            summary["allowances_cap"] = float(record.get("cap", supply.cap))

            try:
                clearing_price = float(summary.get('cp_last', 0.0))
            except (TypeError, ValueError):
                clearing_price = 0.0
            exogenous_price = float(carbon_price_value)
            allowance_price = max(clearing_price - exogenous_price, 0.0)
            effective_price = effective_carbon_price(
                allowance_price, exogenous_price, deep_carbon_pricing
            )
            summary["cp_all"] = allowance_price
            summary["cp_exempt"] = exogenous_price
            summary["cp_effective"] = effective_price
    
            emissions = float(summary.get('emissions', 0.0))
            surrendered = float(summary.get('surrendered', 0.0))
            bank_unadjusted = float(summary.get('bank_unadjusted', summary.get('bank_new', 0.0)))
            obligation = float(summary.get('obligation_new', 0.0))
            ccr1_issued = float(summary.get('ccr1_issued', 0.0))
            ccr2_issued = float(summary.get('ccr2_issued', 0.0))
    
            if state is not None:
                state['emissions'] = float(state.get('emissions', 0.0)) + emissions
                state['surrendered'] = float(state.get('surrendered', 0.0)) + surrendered
                state['cap'] = float(state.get('cap', 0.0)) + float(record.get('cap', 0.0))
                state['ccr1'] = float(state.get('ccr1', 0.0)) + ccr1_issued
                state['ccr2'] = float(state.get('ccr2', 0.0)) + ccr2_issued
                state['outstanding'] = obligation
                state['bank_last_unadjusted'] = bank_unadjusted
                state['bank_last_carried'] = float(summary.get('bank_new', 0.0))
    
            finalize_summary = dict(summary.get('finalize', {}))
    
            if not policy_enabled_year:
                finalize_summary.setdefault('finalized', False)
                finalize_summary.setdefault('bank_final', float(summary.get('bank_new', 0.0)))
                finalize_summary.setdefault('remaining_obligation', 0.0)
                finalize_summary.setdefault('surrendered_additional', 0.0)
                summary['finalize'] = finalize_summary
                results[year] = summary
                bank_prev = float(summary.get('bank_new', 0.0)) if banking_enabled_year else 0.0
                if state is not None:
                    state['outstanding'] = 0.0
                _emit_year_debug(year, summary, record)
                payload: dict[str, object] = {
                    "year": _normalize_progress_year(year),
                    "index": idx,
                    "total_years": total_years,
                    "shortage": bool(summary.get('shortage_flag', False)),
                    "max_iter": int(max_iter),
                }
                price_value = summary.get('cp_last')
                try:
                    if price_value is not None:
                        payload['price'] = float(price_value)
                except (TypeError, ValueError):
                    pass
                iterations_value = summary.get('iterations')
                try:
                    payload['iterations'] = int(iterations_value)
                except (TypeError, ValueError):
                    payload['iterations'] = 0
                _emit_stage('year_complete', payload)
                continue
    
            is_final_year = bool(record.get('full_compliance', False))
            if not is_final_year:
                if idx + 1 < len(years_sequence):
                    next_year = years_sequence[idx + 1]
                    next_record = _policy_record_for_year(policy, next_year)
                    next_cp_id = str(next_record.get('cp_id', 'NoPolicy'))
                    if next_cp_id != cp_id:
                        is_final_year = True
                else:
                    is_final_year = True
    
            if is_final_year:
                outstanding_before = obligation
                surrender_additional = min(outstanding_before, bank_unadjusted)
                remaining_obligation = max(outstanding_before - surrender_additional, 0.0)
                bank_after_trueup = max(bank_unadjusted - surrender_additional, 0.0)
                bank_carry = max(bank_after_trueup * carry_pct, 0.0)
    
                summary['bank_unadjusted'] = bank_after_trueup
                summary['bank_new'] = bank_carry
                summary['obligation_new'] = remaining_obligation
    
                if state is not None:
                    state['surrendered'] = float(state.get('surrendered', 0.0)) + surrender_additional
                    state['bank_last_unadjusted'] = bank_after_trueup
                    state['bank_last_carried'] = bank_carry
                    state['outstanding'] = 0.0
    
                total_allowances = 0.0
                if state is not None:
                    total_allowances = float(state.get('bank_start', 0.0)) + float(state.get('cap', 0.0))
                    total_allowances += float(state.get('ccr1', 0.0)) + float(state.get('ccr2', 0.0))
    
                finalize_summary = {
                    'finalized': True,
                    'cp_id': cp_id,
                    'bank_final': float(bank_carry),
                    'remaining_obligation': float(remaining_obligation),
                    'surrendered_additional': float(surrender_additional),
                    'shortage_flag': bool(remaining_obligation > 1e-9),
                    'cp_emissions': float(state.get('emissions', 0.0)) if state is not None else float(emissions),
                    'cp_surrendered': float(state.get('surrendered', 0.0)) if state is not None else float(surrendered),
                    'cp_cap': float(state.get('cap', 0.0)) if state is not None else float(record.get('cap', 0.0)),
                    'cp_ccr1': float(state.get('ccr1', 0.0)) if state is not None else float(ccr1_issued),
                    'cp_ccr2': float(state.get('ccr2', 0.0)) if state is not None else float(ccr2_issued),
                    'bank_start': float(state.get('bank_start', 0.0)) if state is not None else 0.0,
                    'cp_allowances_total': float(total_allowances),
                }
    
                bank_prev = bank_carry if banking_enabled_year else 0.0
            else:
                bank_carry = max(float(summary.get('bank_new', 0.0)), 0.0)
                finalize_summary = {
                    'finalized': False,
                    'bank_final': bank_carry,
                    'remaining_obligation': float(obligation),
                    'surrendered_additional': 0.0,
                }
                summary['bank_new'] = bank_carry
                summary['obligation_new'] = obligation
                bank_prev = bank_carry if banking_enabled_year else 0.0
    
            summary['finalize'] = finalize_summary
            results[year] = summary
            _emit_year_debug(year, summary, record)
    
            payload = {
                'year': _normalize_progress_year(year),
                'index': idx,
                'total_years': total_years,
                'shortage': bool(summary.get('shortage_flag', False)),
                'max_iter': int(max_iter),
            }
            price_value = summary.get('cp_last')
            try:
                if price_value is not None:
                    payload['price'] = float(price_value)
            except (TypeError, ValueError):
                pass
            iterations_value = summary.get('iterations')
            try:
                payload['iterations'] = int(iterations_value)
            except (TypeError, ValueError):
                payload['iterations'] = 0
            _emit_stage('year_complete', payload)
    
        if bank_exhausted_year is not None:
            limiting_factors.append(
                f"Bank exhausted in {bank_exhausted_year}, no further carryforward"
            )
    
        ordered_years = list(years_sequence)
        for period in ordered_years:
            results.setdefault(period, {})
        _emit_stage(
            'run_complete',
            {
                'total_years': total_years,
                'years': list(years_sequence),
            },
        )
    
        return _build_engine_outputs(
            ordered_years,
            results,
            dispatch_solver,
            policy,
            limiting_factors=limiting_factors,
            demand_summary=demand_summary,
            states=normalized_states,
        )
    except Exception as exc:
        _emit_stage("run_failed", {"error": str(exc)})
        raise
