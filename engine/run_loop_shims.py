
"""Compatibility shims for legacy engine.run_loop internals.

Provides deprecated wrappers that forward to the new split modules:
- engine.policy.clearance.solve_allowance_market_year
- engine.dispatch.run.run_dispatch_year
- engine.outputs.build.build_outputs

These keep old imports alive during migration.
"""
from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional, Tuple, List

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from dispatch.interface import DispatchResult  # type: ignore
except Exception:  # pragma: no cover
    DispatchResult = object  # type: ignore

# New modules
from engine.policy.clearance import solve_allowance_market_year as _new_clearance
from engine.dispatch.run import run_dispatch_year as _new_dispatch
from engine.outputs.build import build_outputs as _new_build
from engine.outputs import EngineOutputs  # re-export type

def _warn(name: str) -> None:
    warnings.warn(
        f"engine.run_loop.{name} is deprecated; use the split modules (policy.clearance, dispatch.run, outputs.build)",
        DeprecationWarning,
        stacklevel=3,
    )

# --- Deprecated internals (now thin wrappers) ---
def _solve_allowance_market_year(
    *,
    dispatch_cb,
    year: int,
    supply,
    bank_prev: float,
    outstanding_prev: float,
    policy_enabled: bool,
    high_price: float,
    tol: float,
    max_iter: int,
    annual_surrender_frac: float,
    carry_pct: float,
    banking_enabled: bool,
    carbon_price: float = 0.0,
    floor_enabled: bool = True,
    ccr_enabled: bool = True,
    ccr1_enabled: bool = True,
    ccr2_enabled: bool = True,
    control_period_len: Optional[int] = None,
):
    _warn("_solve_allowance_market_year")
    return _new_clearance(
        dispatch_cb=dispatch_cb,
        year=year,
        supply=supply,
        bank_prev=bank_prev,
        outstanding_prev=outstanding_prev,
        policy_enabled=policy_enabled,
        high_price=high_price,
        tol=tol,
        max_iter=max_iter,
        annual_surrender_frac=annual_surrender_frac,
        carry_pct=carry_pct,
        banking_enabled=banking_enabled,
        carbon_price=carbon_price,
        floor_enabled=floor_enabled,
        ccr_enabled=ccr_enabled,
        ccr1_enabled=ccr1_enabled,
        ccr2_enabled=ccr2_enabled,
        control_period_len=control_period_len,
    )

def _dispatch_from_frames(
    frames,
    year: int,
    allowance_cost: float,
    carbon_price: float,
    use_network: bool,
    period_weights=None,
    carbon_price_schedule=None,
    deep_carbon_pricing: bool = False,
) -> DispatchResult:
    _warn("_dispatch_from_frames")
    return _new_dispatch(
        frames=frames,
        year=year,
        allowance_cost=allowance_cost,
        carbon_price=carbon_price,
        use_network=use_network,
        period_weights=period_weights,
        carbon_price_schedule=carbon_price_schedule,
        deep_carbon_pricing=deep_carbon_pricing,
    )

# The aggregation helper name varies; provide a canonical one and aliases.
def _build_outputs_from_year_results(
    year_results,
    schedule_snapshot=None,
    normalized_demand=None,
    audits_enabled: bool = True,
) -> EngineOutputs:
    _warn("_build_outputs_from_year_results")
    return _new_build(
        year_results=year_results,
        schedule_snapshot=schedule_snapshot,
        normalized_demand=normalized_demand,
        audits_enabled=audits_enabled,
    )

# Aliases commonly used in the legacy code
_build_outputs_bundle = _build_outputs_from_year_results
_aggregate_outputs_from_year_results = _build_outputs_from_year_results
_build_outputs = _build_outputs_from_year_results
