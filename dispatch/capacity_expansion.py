"""Utility helpers for auditing and triggering capacity expansion builds."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Callable, Dict, List

import pandas as pd

from src.common.iteration_status import IterationStatus

from engine.compute_intensity import resolve_iteration_limit
from engine.constants import FLOW_TOL, HOURS_PER_YEAR, ITER_MAX_CAP_EXPANSION


LOGGER = logging.getLogger(__name__)


@dataclass
class PlannedBuild:
    """Container tracking a single capacity expansion decision."""

    candidate_id: str
    unit_id: str
    unique_id: str
    capacity_mw: float
    reason: str
    npv_positive: bool


def _capital_recovery_factor(rate: float, lifetime: float) -> float:
    """Return the capital recovery factor for ``rate`` and ``lifetime``."""

    lifetime = max(float(lifetime), 1.0)
    if rate <= 0.0:
        return 1.0 / lifetime
    ratio = (1.0 + rate) ** lifetime
    return rate * ratio / (ratio - 1.0)


def _effective_cost(
    row: pd.Series,
    *,
    discount_rate: float,
    allowance_cost: float,
    carbon_price: float,
) -> float:
    """Return the levelized cost of energy for ``row`` including carbon effects."""

    availability = max(float(row.availability), 0.0)
    cap_mw = max(float(row.cap_mw), 0.0)
    if availability <= 0.0 or cap_mw <= 0.0:
        return float("inf")

    crf = _capital_recovery_factor(discount_rate, float(row.lifetime_years))
    annual_capex = float(row.capex_per_mw) * cap_mw * crf
    annual_fixed = float(row.fixed_om_per_mw) * cap_mw
    expected_mwh = cap_mw * availability * HOURS_PER_YEAR
    fixed_component = (annual_capex + annual_fixed) / max(expected_mwh, FLOW_TOL)

    carbon_cost = float(getattr(row, "carbon_cost_per_mwh", 0.0))
    variable_component = (
        float(row.vom_per_mwh)
        + float(row.hr_mmbtu_per_mwh) * float(row.fuel_price_per_mmbtu)
        + carbon_cost
    )
    emission_rate = float(getattr(row, "co2_short_ton_per_mwh", getattr(row, "ef_ton_per_mwh", 0.0)))
    carbon_component = emission_rate * (allowance_cost + carbon_price)

    return fixed_component + variable_component + carbon_component


def _build_unit_record(
    row: pd.Series, unit_id: str, unique_id: str, capacity_mw: float
) -> Dict[str, float]:
    """Return a dictionary describing a new dispatch unit from ``row``."""

    return {
        "unit_id": unit_id,
        "unique_id": unique_id,
        "region": str(row.region),
        "fuel": str(row.fuel),
        "cap_mw": float(capacity_mw),
        "availability": float(row.availability),
        "hr_mmbtu_per_mwh": float(row.hr_mmbtu_per_mwh),
        "vom_per_mwh": float(row.vom_per_mwh),
        "fuel_price_per_mmbtu": float(row.fuel_price_per_mmbtu),
        "carbon_cost_per_mwh": float(getattr(row, "carbon_cost_per_mwh", 0.0)),
        "ef_ton_per_mwh": float(
            getattr(row, "co2_short_ton_per_mwh", getattr(row, "ef_ton_per_mwh", 0.0))
        ),
    }


def _ensure_unique_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``df`` contains a normalized ``unique_id`` column."""

    fallback = df.get("unit_id")
    if fallback is None:
        fallback_series = pd.Series("", index=df.index, dtype=str)
    else:
        fallback_series = fallback.astype(str)

    if "unique_id" not in df.columns:
        df["unique_id"] = fallback_series
        return df

    unique_raw = df["unique_id"]
    normalized = unique_raw.astype(str).str.strip()
    invalid_tokens = {"", "nan", "none", "null"}
    mask = unique_raw.isna()
    mask |= normalized.str.lower().isin(invalid_tokens)
    df["unique_id"] = normalized.where(~mask, fallback_series).astype(str)
    return df


def _expensive_generation(
    generation: pd.Series, unit_costs: pd.Series, threshold: float
) -> float:
    """Return total generation from units with cost above ``threshold``."""

    mask = unit_costs > threshold + FLOW_TOL
    if mask.any():
        return float(generation[mask].sum())
    return 0.0


def _create_log_entry(
    record: PlannedBuild,
    row: pd.Series,
    generation_mwh: float,
) -> Dict[str, object]:
    """Return a structured log entry for ``record`` using ``row`` metadata."""

    capacity_mw = float(record.capacity_mw)
    capex_total = float(row.capex_per_mw) * capacity_mw
    fixed_om = float(row.fixed_om_per_mw) * capacity_mw
    variable_rate = (
        float(row.vom_per_mwh)
        + float(row.hr_mmbtu_per_mwh) * float(row.fuel_price_per_mmbtu)
        + float(getattr(row, "carbon_cost_per_mwh", 0.0))
    )
    variable_cost = variable_rate * float(generation_mwh)
    emissions = (
        float(getattr(row, "co2_short_ton_per_mwh", getattr(row, "ef_ton_per_mwh", 0.0)))
        * float(generation_mwh)
    )

    return {
        "candidate": str(record.candidate_id),
        "unit_id": str(record.unit_id),
        "unique_id": str(record.unique_id),
        "capacity_mw": capacity_mw,
        "generation_mwh": float(generation_mwh),
        "reason": record.reason,
        "npv_positive": bool(record.npv_positive),
        "capex_cost": capex_total,
        "opex_cost": fixed_om + variable_cost,
        "emissions_tons": emissions,
    }


def plan_capacity_expansion(
    base_units: pd.DataFrame,
    candidates: pd.DataFrame,
    base_summary: Dict[str, object],
    dispatch_solver: Callable[[pd.DataFrame], Dict[str, object]],
    *,
    allowance_cost: float,
    carbon_price: float,
    discount_rate: float,
) -> tuple[pd.DataFrame, Dict[str, object], List[Dict[str, object]], IterationStatus]:
    """Plan expansion decisions returning updated units, summary, log, and iteration status."""

    iteration_limit = max(1, int(resolve_iteration_limit(ITER_MAX_CAP_EXPANSION)))
    if candidates.empty:
        status = IterationStatus(
            iterations=0,
            converged=True,
            limit=iteration_limit,
            metadata={"builds": 0},
        )
        return base_units, base_summary, [], status

    current_units = base_units.copy(deep=True)
    current_units = _ensure_unique_id_column(current_units)
    summary = dict(base_summary)
    records: List[PlannedBuild] = []
    used: Dict[int, float] = {idx: 0.0 for idx in range(len(candidates))}

    candidates_sorted = _ensure_unique_id_column(candidates.reset_index(drop=True))
    order = list(candidates_sorted.index)

    # Extract zonal data with backwards compatibility
    region_prices = summary.get("region_prices", {})
    if not isinstance(region_prices, dict) or not region_prices:
        fallback_price = float(summary.get("price", 0.0) or 0.0)
        region_prices = {}
    else:
        fallback_price = 0.0
    
    unserved_by_region = summary.get("unserved_by_region", {})
    if not isinstance(unserved_by_region, dict) or not unserved_by_region:
        fallback_shortfall = float(summary.get("shortfall_mwh", 0.0) or 0.0)
        unserved_by_region = {}
    else:
        fallback_shortfall = 0.0

    # Shortage-driven builds -------------------------------------------------
    shortage_new_units: List[Dict[str, float]] = []
    
    if unserved_by_region:
        for zone, zone_shortfall in unserved_by_region.items():
            if zone_shortfall <= FLOW_TOL:
                continue
            
            remaining_shortfall = float(zone_shortfall)
            for idx in order:
                if remaining_shortfall <= FLOW_TOL:
                    break
                
                row = candidates_sorted.loc[idx]
                candidate_region = str(row.region)
                
                if candidate_region != zone:
                    continue
                
                max_builds = max(float(row.get("max_builds", 1.0)), 0.0)
                remaining = max_builds - used[idx]
                if remaining <= FLOW_TOL:
                    continue

                availability = max(float(row.availability), 0.0)
                cap_mw = max(float(row.cap_mw), 0.0)
                if availability <= 0.0 or cap_mw <= 0.0:
                    continue

                block_mwh = cap_mw * availability * HOURS_PER_YEAR
                if block_mwh <= FLOW_TOL:
                    continue

                builds_needed = math.ceil((remaining_shortfall - FLOW_TOL) / block_mwh)
                builds_to_use = int(min(builds_needed, math.floor(remaining + FLOW_TOL)))
                if builds_to_use <= 0:
                    continue

                zone_price = region_prices.get(zone, fallback_price)
                for build_no in range(builds_to_use):
                    base_unit_id = str(row.unit_id)
                    base_unique_id = str(getattr(row, "unique_id", base_unit_id))
                    suffix = int(used[idx] + build_no + 1)
                    unit_id = f"{base_unit_id}_build{suffix}"
                    unique_id = f"{base_unique_id}_build{suffix}"
                    shortage_new_units.append(
                        _build_unit_record(row, unit_id, unique_id, cap_mw)
                    )
                    records.append(
                        PlannedBuild(
                            candidate_id=str(getattr(row, "unique_id", row.unit_id)),
                            unit_id=unit_id,
                            unique_id=unique_id,
                            capacity_mw=cap_mw,
                            reason="supply_shortage",
                            npv_positive=_effective_cost(
                                row,
                                discount_rate=discount_rate,
                                allowance_cost=allowance_cost,
                                carbon_price=carbon_price,
                            )
                            < zone_price - FLOW_TOL,
                        )
                    )
                used[idx] += float(builds_to_use)
                remaining_shortfall = max(0.0, remaining_shortfall - builds_to_use * block_mwh)
    
    elif fallback_shortfall > FLOW_TOL:
        shortfall = fallback_shortfall
        for idx in order:
            if shortfall <= FLOW_TOL:
                break
            row = candidates_sorted.loc[idx]
            max_builds = max(float(row.get("max_builds", 1.0)), 0.0)
            remaining = max_builds - used[idx]
            if remaining <= FLOW_TOL:
                continue

            availability = max(float(row.availability), 0.0)
            cap_mw = max(float(row.cap_mw), 0.0)
            if availability <= 0.0 or cap_mw <= 0.0:
                continue

            block_mwh = cap_mw * availability * HOURS_PER_YEAR
            if block_mwh <= FLOW_TOL:
                continue

            builds_needed = math.ceil((shortfall - FLOW_TOL) / block_mwh)
            builds_to_use = int(min(builds_needed, math.floor(remaining + FLOW_TOL)))
            if builds_to_use <= 0:
                continue

            for build_no in range(builds_to_use):
                base_unit_id = str(row.unit_id)
                base_unique_id = str(getattr(row, "unique_id", base_unit_id))
                suffix = int(used[idx] + build_no + 1)
                unit_id = f"{base_unit_id}_build{suffix}"
                unique_id = f"{base_unique_id}_build{suffix}"
                shortage_new_units.append(
                    _build_unit_record(row, unit_id, unique_id, cap_mw)
                )
                records.append(
                    PlannedBuild(
                        candidate_id=str(getattr(row, "unique_id", row.unit_id)),
                        unit_id=unit_id,
                        unique_id=unique_id,
                        capacity_mw=cap_mw,
                        reason="supply_shortage",
                        npv_positive=_effective_cost(
                            row,
                            discount_rate=discount_rate,
                            allowance_cost=allowance_cost,
                            carbon_price=carbon_price,
                        )
                        < fallback_price - FLOW_TOL,
                    )
                )
            used[idx] += float(builds_to_use)
            shortfall = max(0.0, shortfall - builds_to_use * block_mwh)

    if shortage_new_units:
        current_units = pd.concat(
            [current_units, pd.DataFrame(shortage_new_units)],
            ignore_index=True,
        )
        summary = dispatch_solver(current_units)
        
        region_prices = summary.get("region_prices", {})
        if not isinstance(region_prices, dict) or not region_prices:
            fallback_price = float(summary.get("price", 0.0) or 0.0)
            region_prices = {}
        else:
            fallback_price = 0.0

    # Positive-NPV builds ----------------------------------------------------
    iteration_count = 0
    converged = True
    for iteration in range(iteration_limit):
        iteration_count += 1
        generation = summary.get("generation")
        unit_costs = summary.get("units")
        if not isinstance(generation, pd.Series) or not isinstance(unit_costs, pd.DataFrame):
            break
        unit_cost_series = unit_costs["marginal_cost"]
        built_any = False
        pending_npv_units: List[Dict[str, float]] = []

        for idx in order:
            row = candidates_sorted.loc[idx]
            candidate_region = str(row.region)
            
            zone_price = region_prices.get(candidate_region, fallback_price)
            if zone_price <= FLOW_TOL:
                continue
            
            max_builds = max(float(row.get("max_builds", 1.0)), 0.0)
            remaining = max_builds - used[idx]
            if remaining <= FLOW_TOL:
                continue

            effective_cost = _effective_cost(
                row,
                discount_rate=discount_rate,
                allowance_cost=allowance_cost,
                carbon_price=carbon_price,
            )
            if effective_cost >= zone_price - FLOW_TOL:
                continue

            expensive = _expensive_generation(generation, unit_cost_series, effective_cost)
            if expensive <= FLOW_TOL:
                continue

            availability = max(float(row.availability), 0.0)
            cap_mw = max(float(row.cap_mw), 0.0)
            if availability <= 0.0 or cap_mw <= 0.0:
                continue

            block_capacity = cap_mw * availability * HOURS_PER_YEAR
            max_replacable = remaining * block_capacity
            replace_mwh = min(expensive, max_replacable)
            if replace_mwh <= FLOW_TOL:
                continue

            capacity_needed = replace_mwh / (availability * HOURS_PER_YEAR)
            capacity_mw = min(cap_mw, capacity_needed)
            if capacity_mw <= FLOW_TOL:
                continue

            build_index = int(used[idx] + 1)
            base_unit_id = str(row.unit_id)
            base_unique_id = str(getattr(row, "unique_id", base_unit_id))
            unit_id = f"{base_unit_id}_build{build_index}"
            unique_id = f"{base_unique_id}_build{build_index}"
            pending_npv_units.append(
                _build_unit_record(row, unit_id, unique_id, capacity_mw)
            )
            records.append(
                PlannedBuild(
                    candidate_id=str(getattr(row, "unique_id", row.unit_id)),
                    unit_id=unit_id,
                    unique_id=unique_id,
                    capacity_mw=capacity_mw,
                    reason="npv_positive",
                    npv_positive=True,
                )
            )
            used[idx] += 1.0
            built_any = True
            break

        if pending_npv_units:
            current_units = pd.concat(
                [current_units, pd.DataFrame(pending_npv_units)],
                ignore_index=True,
            )
            summary = dispatch_solver(current_units)
            
            region_prices = summary.get("region_prices", {})
            if not isinstance(region_prices, dict) or not region_prices:
                fallback_price = float(summary.get("price", 0.0) or 0.0)
                region_prices = {}
            else:
                fallback_price = 0.0

        if not built_any:
            converged = True
            break

    else:
        LOGGER.warning(
            "plan_capacity_expansion reached iteration limit of %s", iteration_limit
        )
        converged = False

    final_generation = summary.get("generation")
    build_log: List[Dict[str, object]] = []
    if isinstance(final_generation, pd.Series):
        for record in records:
            try:
                row_idx = candidates_sorted.index[candidates_sorted["unit_id"] == record.candidate_id][0]
            except IndexError:
                continue
            row = candidates_sorted.loc[row_idx]
            generation_mwh = float(final_generation.get(record.unique_id, 0.0))
            build_log.append(_create_log_entry(record, row, generation_mwh))

    if iteration_count == 0 and not records:
        converged = True

    status = IterationStatus(
        iterations=iteration_count,
        converged=converged,
        limit=iteration_limit,
        message=None
        if converged
        else "Iteration limit reached before no further builds were identified",
        metadata={"builds": len(records)},
    )

    return current_units, summary, build_log, status


__all__ = ["plan_capacity_expansion"]

