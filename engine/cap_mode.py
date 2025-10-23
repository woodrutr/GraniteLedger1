"""Allowance market clearing under an endogenous cap price."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
import pandas as pd

from src.common.iteration_status import IterationStatus

from engine.constants import EMISS_TOL, FLOW_TOL, ITER_MAX_CAP_SOLVER, PRICE_TOL
from .compute_intensity import resolve_iteration_limit


LOGGER = logging.getLogger(__name__)

P_MAX = 2_000.0
MAX_ITERS = ITER_MAX_CAP_SOLVER


class _DispatchFrameLike(Protocol):
    base_costs: pd.Series
    emission_rates: pd.Series
    capacities: pd.Series


@dataclass(frozen=True)
class CapParams:
    """Container describing allowance supply schedules for cap mode."""

    budgets: Mapping[int, float]
    reserve: Mapping[int, float]
    ccr1_trigger: Mapping[int, float]
    ccr1_amount: Mapping[int, float]
    ccr2_trigger: Mapping[int, float]
    ccr2_amount: Mapping[int, float]
    banking: bool = True


@dataclass(frozen=True)
class CapRunResult:
    """Structured output from :func:`run_cap_mode`."""

    table: pd.DataFrame
    diagnostics: list[dict[str, Any]]

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        if not isinstance(self.table, pd.DataFrame):
            raise TypeError("table must be a pandas DataFrame")

    # Delegate convenient accessors to the underlying DataFrame so callers
    # can use ``result['cp_last']`` directly in tests and plotting utilities.
    def __getitem__(self, key: Any) -> Any:  # pragma: no cover - thin wrapper
        return self.table.__getitem__(key)

    def __iter__(self):  # pragma: no cover - thin wrapper
        return iter(self.table)

    def __len__(self) -> int:  # pragma: no cover - thin wrapper
        return len(self.table)

    def to_records(self) -> list[dict[str, Any]]:
        """Return the output table as a list of dictionaries."""

        return self.table.to_dict(orient="records")


def align_series(
    series: Mapping[int, float] | None, years: Iterable[int], fill: float = 0.0
) -> dict[int, float]:
    """Return ``series`` aligned to ``years`` carrying the last value forward."""

    years_list = [int(year) for year in years]
    if not series:
        return {year: float(fill) for year in years_list}

    normalized: dict[int, float] = {}
    last: float | None = None
    for year in sorted(years_list):
        if year in series:
            last = float(series[year])
        if last is None:
            last = float(fill)
        normalized[year] = float(last)
    return normalized


def dispatch_min_cost(
    frames_for_year: _DispatchFrameLike, mc: pd.Series, demand_mwh: float
) -> pd.Series:
    """Return the minimum-cost dispatch by technology for ``demand_mwh``."""

    if not isinstance(mc, pd.Series):
        raise TypeError("marginal costs must be provided as a pandas Series")

    try:
        capacities = frames_for_year.capacities
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError("frames_for_year must define a 'capacities' Series") from exc

    if not isinstance(capacities, pd.Series):
        capacities = pd.Series(capacities)

    demand = float(max(demand_mwh, 0.0))

    capacity_aligned = capacities.reindex(mc.index).fillna(0.0)
    ordered = mc.sort_values(kind="mergesort")

    dispatch = pd.Series(0.0, index=ordered.index, dtype=float)
    remaining = demand

    for tech in ordered.index:
        cap = float(capacity_aligned.get(tech, 0.0))
        if cap <= 0.0:
            continue
        allocation = cap if remaining >= cap else remaining
        if allocation > 0.0:
            dispatch.at[tech] = allocation
            remaining -= allocation
        if remaining <= FLOW_TOL:
            remaining = 0.0
            break

    if remaining > FLOW_TOL:
        LOGGER.debug(
            "Dispatch shortage encountered: demand=%s, supplied=%s",
            demand,
            demand - remaining,
        )

    return dispatch.reindex(mc.index, fill_value=0.0)


def _supply_y(y: int, p: float, par: CapParams) -> float:
    reserve = float(par.reserve.get(y, 0.0))
    price = max(float(p), reserve)
    rel1 = 0.0
    rel2 = 0.0
    ccr1_trigger = float(par.ccr1_trigger.get(y, float("inf")))
    ccr2_trigger = float(par.ccr2_trigger.get(y, float("inf")))
    if price >= ccr1_trigger:
        rel1 = float(par.ccr1_amount.get(y, 0.0))
    if price >= ccr2_trigger:
        rel2 = float(par.ccr2_amount.get(y, 0.0))
    return float(par.budgets.get(y, 0.0)) + rel1 + rel2


def _emissions_y(
    y: int,
    p: float,
    frames_for_year: _DispatchFrameLike,
    demand_mwh: float,
) -> float:
    base_costs = frames_for_year.base_costs
    emission_rates = frames_for_year.emission_rates
    if not isinstance(base_costs, pd.Series) or not isinstance(emission_rates, pd.Series):
        raise TypeError("frames_for_year must provide base_costs and emission_rates as Series")

    marginal_cost = base_costs + emission_rates * float(p)
    generation = dispatch_min_cost(frames_for_year, marginal_cost, demand_mwh)
    emissions = (emission_rates.reindex(generation.index, fill_value=0.0) * generation).sum()

    if p > 0.0:
        marginal_without_price = base_costs
        if np.allclose(marginal_cost.values, marginal_without_price.values):
            LOGGER.warning("Dispatch cost not using carbon price.")

    return float(emissions)


class CapInfeasibleError(RuntimeError):
    """Error raised when the cap cannot be met within the price search range."""


def solve_price_for_year(
    y: int,
    bank_prev: float,
    frames_for_year: _DispatchFrameLike,
    demand_mwh: float,
    par: CapParams,
) -> tuple[float, float, IterationStatus, float]:
    """Solve for the clearing price ``p`` in year ``y`` returning price, bank, and status."""

    reserve = float(par.reserve.get(y, 0.0))
    p_lo = reserve
    p_hi = P_MAX

    def excess(price: float) -> float:
        supply = _supply_y(y, price, par)
        emissions = _emissions_y(y, price, frames_for_year, demand_mwh)
        available = supply + (bank_prev if par.banking else 0.0)
        return available - emissions

    excess_lo = excess(p_lo)
    if excess_lo < 0.0:
        trial = max(p_lo, 1.0)
        while trial < P_MAX and excess(trial) < 0.0:
            trial = min(trial * 2.0, P_MAX)
        p_hi = trial
        if p_hi >= P_MAX and excess(p_hi) < 0.0:
            raise CapInfeasibleError("Cap infeasible at P_MAX.")
    else:
        p_hi = p_lo

    max_iterations = max(1, int(resolve_iteration_limit(ITER_MAX_CAP_SOLVER)))
    iterations = 0
    while iterations < max_iterations and abs(p_hi - p_lo) > PRICE_TOL:
        p_mid = 0.5 * (p_lo + p_hi)
        if excess(p_mid) >= 0.0:
            p_hi = p_mid
        else:
            p_lo = p_mid
        iterations += 1

    p_raw = p_hi
    clearing_price = max(p_raw, reserve)
    supply = _supply_y(y, clearing_price, par)
    emissions = _emissions_y(y, clearing_price, frames_for_year, demand_mwh)
    available = supply + (bank_prev if par.banking else 0.0)

    if emissions > available + EMISS_TOL:
        raise CapInfeasibleError(
            "Clearing emissions exceed available allowances; cap infeasible at solved price."
        )

    bank_new = float(max(available - emissions, 0.0)) if par.banking else 0.0
    converged = abs(p_hi - p_lo) <= PRICE_TOL
    message = None
    if not converged and iterations >= max_iterations:
        message = "Iteration limit reached before meeting price tolerance"
        LOGGER.warning(
            "Cap price solver reached iteration limit for year %s (limit=%s)",
            y,
            max_iterations,
        )

    status = IterationStatus(
        iterations=int(iterations),
        converged=converged,
        limit=max_iterations,
        message=message,
        metadata={"year": int(y)},
    )

    return float(clearing_price), float(bank_new), status, float(p_raw)


def run_cap_mode(
    run_years: Iterable[int],
    frames_by_year: Mapping[int, _DispatchFrameLike],
    demand_sched: Mapping[int, float],
    par: CapParams,
    bank0: float,
) -> CapRunResult:
    """Execute the allowance market clearing for ``run_years`` under cap mode."""

    ordered_years = [int(year) for year in run_years]
    bank = float(bank0 if par.banking else 0.0)
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for year in ordered_years:
        frames_for_year = frames_by_year[year]
        demand_value = float(demand_sched[year])
        bank_in = bank
        try:
            price, bank, status, price_raw = solve_price_for_year(
                year,
                bank,
                frames_for_year,
                demand_value,
                par,
            )
        except CapInfeasibleError:
            LOGGER.error("Cap infeasible at P_MAX for year %s", year)
            raise

        supply = _supply_y(year, price, par)
        emissions = _emissions_y(year, price, frames_for_year, demand_value)
        available = supply + (bank_in if par.banking else 0.0)
        shortage = emissions > available + EMISS_TOL

        records.append(
            {
                "year": year,
                "cp_last": round(price, 6),
                "cp_all": round(price, 6),
                "cp_exempt": 0.0,
                "cp_effective": round(price, 6),
                "allowance_price": round(price, 6),
                "iterations": int(status.iterations),
                "emissions_tons": round(emissions, 3),
                "allowances_total": round(supply, 3),
                "available_allowances": round(available, 3),
                "bank": round(bank, 3),
                "shortage_flag": bool(shortage),
            }
        )

        diagnostics.append(
            {
                "year": year,
                "p_raw": price_raw,
                "reserve": float(par.reserve.get(year, 0.0)),
                "ccr1_trig": float(par.ccr1_trigger.get(year, float("inf"))),
                "ccr2_trig": float(par.ccr2_trigger.get(year, float("inf"))),
                "ccr_release_total": float(supply - float(par.budgets.get(year, 0.0))),
                "budget": float(par.budgets.get(year, 0.0)),
                "demand_mwh": demand_value,
                "emissions": emissions,
                "bank_in": bank_in if par.banking else 0.0,
                "bank_out": bank if par.banking else 0.0,
                "shortage": bool(shortage),
                "iterations": int(status.iterations),
                "iteration_converged": bool(status.converged),
                "iteration_limit": int(status.limit or 0),
                "iteration_message": status.message,
            }
        )

        LOGGER.debug(
            "Year %s cleared: price=%s, supply=%s, emissions=%s, bank_in=%s, bank_out=%s",
            year,
            price,
            supply,
            emissions,
            bank_in,
            bank,
        )

    df = pd.DataFrame(records, columns=[
        "year",
        "cp_last",
        "cp_all",
        "cp_exempt",
        "cp_effective",
        "allowance_price",
        "iterations",
        "emissions_tons",
        "allowances_total",
        "available_allowances",
        "bank",
        "shortage_flag",
    ])
    if not df.empty:
        df = df.set_index("year", drop=False)

    return CapRunResult(df, diagnostics)


__all__ = [
    "CapParams",
    "CapRunResult",
    "CapInfeasibleError",
    "MAX_ITERS",
    "align_series",
    "dispatch_min_cost",
    "run_cap_mode",
    "solve_price_for_year",
]

