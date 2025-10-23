
"""Outputs aggregation extraction.

Provides build_outputs(...) which preserves legacy behavior by delegating to
the existing aggregation logic inside engine.run_loop or EngineOutputs helpers.
No GUI imports.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, List

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from dispatch.interface import DispatchResult  # type: ignore
except Exception:  # pragma: no cover
    DispatchResult = object  # type: ignore

from dataclasses import dataclass

@dataclass(frozen=True)
class CarbonPriceVector:
    clearing_price: float
    floor_price: Optional[float] = None
    ccr1_triggered: bool = False
    ccr2_triggered: bool = False
    applied_floor: bool = False

try:
    from engine.outputs import EngineOutputs  # type: ignore
except Exception as e:  # pragma: no cover
    raise

YearResult = Tuple[int, CarbonPriceVector, DispatchResult]

def _try_legacy_builders(
    year_results: Sequence[YearResult],
    schedule_snapshot: Optional[Mapping[int, float]],
    normalized_demand: Optional[Mapping[int, float]],
    audits_enabled: bool,
) -> EngineOutputs:
    """Attempt to call legacy private builders to preserve behavior."""
    from engine import run_loop as _legacy  # defer to avoid cycles
    # Candidate builder names to probe in order
    candidates = [
        "_build_outputs_from_year_results",
        "_build_outputs_bundle",
        "_aggregate_outputs_from_year_results",
        "_build_outputs",
    ]
    last_err: Optional[Exception] = None
    for name in candidates:
        fn = getattr(_legacy, name, None)
        if fn is None:
            continue
        try:
            return fn(  # type: ignore[misc,call-arg]
                year_results=year_results,
                schedule_snapshot=schedule_snapshot,
                normalized_demand=normalized_demand,
                audits_enabled=audits_enabled,
            )
        except TypeError:
            # Try positional style
            try:
                return fn(year_results, schedule_snapshot, normalized_demand, audits_enabled)  # type: ignore[misc,call-arg]
            except Exception as e2:  # keep trying others
                last_err = e2
        except Exception as e:
            last_err = e
    # Try EngineOutputs class helpers if present
    for helper in ("from_year_results", "build_from_year_results", "aggregate") :
        meth = getattr(EngineOutputs, helper, None)
        if meth is None:
            continue
        try:
            return meth(  # type: ignore[misc,call-arg]
                year_results=year_results,
                schedule_snapshot=schedule_snapshot,
                normalized_demand=normalized_demand,
                audits_enabled=audits_enabled,
            )
        except TypeError:
            try:
                return meth(year_results, schedule_snapshot, normalized_demand, audits_enabled)  # type: ignore[misc,call-arg]
            except Exception as e3:
                last_err = e3
        except Exception as e4:
            last_err = e4
    if last_err:
        raise last_err
    raise RuntimeError("No legacy output builder found; wire this shim to engine.run_loop aggregation.")

def build_outputs(
    year_results: List[YearResult],
    *,
    schedule_snapshot: Optional[Mapping[int, float]] = None,
    normalized_demand: Optional[Mapping[int, float]] = None,
    audits_enabled: bool = True,
) -> EngineOutputs:
    """Aggregate per-year results into an EngineOutputs bundle.

    This extraction preserves column names and dtypes by delegating to the
    legacy engine aggregation. It avoids GUI imports and global state.
    """
    # Defensive: ensure deterministic year ordering
    try:
        year_results = sorted(year_results, key=lambda t: int(t[0]))
    except Exception:
        pass
    return _try_legacy_builders(
        year_results=year_results,
        schedule_snapshot=schedule_snapshot,
        normalized_demand=normalized_demand,
        audits_enabled=audits_enabled,
    )

__all__ = ["build_outputs", "CarbonPriceVector", "EngineOutputs"]
