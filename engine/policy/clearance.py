
"""Policy clearance extraction shim.

Provides a public, engine-local function `solve_allowance_market_year` with
the requested signature. Implementation delegates to the legacy
`engine.run_loop._solve_allowance_market_year` to preserve behavior
unchanged. Dataclasses are defined for forward use, but this shim returns
the legacy return exactly to avoid behavior drift. Future refactors can
replace the delegation with a direct implementation using `dispatch_cb`.

This module contains **no GUI imports**.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any

# Forward types (opaque to keep this shim independent of GUI)
try:
    from dispatch.interface import DispatchResult  # type: ignore
except Exception:  # pragma: no cover
    DispatchResult = object  # type: ignore

# Lightweight dataclasses requested in the brief.
@dataclass(frozen=True)
class CarbonPriceVector:
    clearing_price: float
    floor_price: Optional[float] = None
    ccr1_triggered: bool = False
    ccr2_triggered: bool = False
    applied_floor: bool = False

@dataclass(frozen=True)
class ClearanceStatus:
    converged: bool
    iterations: int
    message: Optional[str] = None

def solve_allowance_market_year(
    dispatch_cb: Callable[[int, float, Optional[float]], DispatchResult],
    year: int,
    supply: Any,  # policy.allowance_supply.AllowanceSupply
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
    floor_enabled: bool = True,
    ccr_enabled: bool = True,
    ccr1_enabled: bool = True,
    ccr2_enabled: bool = True,
    control_period_len: Optional[int] = None,
) -> Tuple[CarbonPriceVector, ClearanceStatus]:
    """Shim that delegates to the legacy clearance routine to preserve meaning.

    Parameters match the program manager's specification. To keep behavior
    unchanged, we call `engine.run_loop._solve_allowance_market_year`.
    If the legacy function signature does not accept `dispatch_cb`, we fall
    back to a call without it.

    Returns
    -------
    (CarbonPriceVector, ClearanceStatus)
        For now, returns the legacy tuple unchanged if it already matches
        this structure; otherwise, wraps minimal values when possible.
    """
    # Defer import to avoid circulars during test collection
    from engine import run_loop as _legacy

    base_kwargs = {
        "dispatch_cb": dispatch_cb,
        "year": year,
        "supply": supply,
        "bank_prev": bank_prev,
        "outstanding_prev": outstanding_prev,
        "policy_enabled": policy_enabled,
        "high_price": high_price,
        "tol": tol,
        "max_iter": max_iter,
        "annual_surrender_frac": annual_surrender_frac,
        "carry_pct": carry_pct,
        "banking_enabled": banking_enabled,
        "carbon_price": carbon_price,
        "floor_enabled": floor_enabled,
        "ccr_enabled": ccr_enabled,
        "ccr1_enabled": ccr1_enabled,
        "ccr2_enabled": ccr2_enabled,
        "control_period_len": control_period_len,
    }

    try:
        legacy_ret = _legacy._solve_allowance_market_year(  # type: ignore[attr-defined]
            **{k: v for k, v in base_kwargs.items() if v is not None or k not in {"control_period_len"}}
        )
    except TypeError:
        fallback_kwargs = base_kwargs.copy()
        dispatch_solver = fallback_kwargs.pop("dispatch_cb")
        fallback_kwargs["dispatch_solver"] = dispatch_solver
        for deprecated in ("floor_enabled", "ccr_enabled", "ccr1_enabled", "ccr2_enabled", "control_period_len"):
            fallback_kwargs.pop(deprecated, None)
        legacy_ret = _legacy._solve_allowance_market_year(  # type: ignore[attr-defined]
            **fallback_kwargs
        )

    # If the legacy function already returns the desired dataclasses, pass-through.
    if isinstance(legacy_ret, tuple) and len(legacy_ret) == 2:
        a, b = legacy_ret
        if isinstance(a, CarbonPriceVector) and isinstance(b, ClearanceStatus):
            return legacy_ret  # type: ignore[return-value]

    # Otherwise, try to minimally wrap common legacy shapes.
    # Common case: (price: float, converged: bool, iters: int, meta: dict)
    if isinstance(legacy_ret, tuple) and len(legacy_ret) >= 2:
        price_like = legacy_ret[0]
        status_like = legacy_ret[1]
        try:
            cp = float(price_like)  # type: ignore[arg-type]
            pv = CarbonPriceVector(clearing_price=cp)
            if isinstance(status_like, dict):
                st = ClearanceStatus(
                    converged=bool(status_like.get("converged", False)),
                    iterations=int(status_like.get("iterations", 0)),
                    message=status_like.get("message"),
                )
            else:
                # best-effort
                st = ClearanceStatus(converged=bool(status_like), iterations=int(legacy_ret[2]) if len(legacy_ret) > 2 else 0)
            return pv, st
        except Exception:
            pass

    # Last resort: construct dummy wrappers to avoid breaking callers.
    return CarbonPriceVector(clearing_price=float("nan")), ClearanceStatus(converged=False, iterations=0, message="delegated return shape unrecognized")
