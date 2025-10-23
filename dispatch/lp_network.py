"""Linear programming dispatch with a multi-region transmission network."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, TYPE_CHECKING, cast

try:  # pragma: no cover - optional dependency guard
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(Any, None)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

from config.dispatch import DISPATCH_SOLVER
from dispatch.solvers import LpBackend, LpSolveStatus, get_backend
from dispatch.capacity_expansion import plan_capacity_expansion
from engine.normalization import normalize_region
from engine.constants import FLOW_TOL, HOURS_PER_YEAR
from io_loader import Frames
from policy.generation_standard import GenerationStandardPolicy
from common.regions_schema import REGION_MAP

_DEFAULT_REGION = next(iter(REGION_MAP))

from .interface import DispatchResult
_TOL = 1e-9
_INTERFACE_FLOW_PENALTY = 1e-8


@dataclass(frozen=True)
class InterfaceSpec:
    """Specification for a transmission interface between two regions."""

    from_region: str
    to_region: str
    capacity_mw: float | None = None
    reverse_capacity_mw: float | None = None
    efficiency: float = 1.0
    added_cost_per_mwh: float = 0.0
    interface_id: str | None = None
    contracted_flow_mw_forward: float | None = None
    contracted_flow_mw_reverse: float | None = None
    interface_type: str | None = None
    notes: str | None = None
    profile_id: str | None = None
    in_service_year: int | None = None

PANDAS_REQUIRED_MESSAGE = "pandas is required to operate the network dispatch engine."

LOGGER = logging.getLogger(__name__)

# NREL maximum capacity factors by technology type
# Source: NREL ATB 2024, industry standard availability factors
NREL_MAX_CAPACITY_FACTORS = {
    'solar': 0.25,
    'wind': 0.35,
    'wind_offshore': 0.45,
    'coal': 0.85,
    'natural_gas_combined_cycle': 0.87,
    'natural_gas_combustion_turbine': 0.93,
    'nuclear': 0.90,
    'hydro': 0.52,
    'biomass': 0.80,
    'geothermal': 0.90,
    'oil': 0.08,
    # Common fuel name variations
    'gascombinedcycle': 0.87,
    'gassteam': 0.50,
    'gasturbine': 0.93,
    'gas': 0.75,
    'ng': 0.75,
    'other': 0.50,
}


def _get_availability_from_fuel(fuel: str) -> float:
    """Map fuel type to NREL maximum capacity factor."""
    fuel_normalized = str(fuel).lower().strip().replace(' ', '').replace('_', '')
    return NREL_MAX_CAPACITY_FACTORS.get(fuel_normalized, 0.85)  # Default to 85% if unknown


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before executing pandas-dependent logic."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for dispatch.lp_network; install it with `pip install pandas`."
        )

@dataclass(frozen=True)
class GeneratorSpec:
    """Specification of an individual generator in the dispatch problem."""

    name: str
    region: str
    fuel: str
    variable_cost: float
    capacity: float
    emission_rate: float
    covered: bool = True
    penalty_cost: float = 0.0

    def marginal_cost(self, allowance_cost: float, carbon_price: float = 0.0) -> float:
        """Return the effective marginal cost including carbon policy costs."""

        allowance = float(allowance_cost) if self.covered else 0.0
        price_component = float(carbon_price)
        carbon_cost = self.emission_rate * (allowance + price_component)
        return self.variable_cost + self.penalty_cost + carbon_cost


@dataclass
class _LpVariable:
    """Internal representation of a decision variable in the LP."""

    name: str
    lower: float
    upper: float | None
    cost: float
    kind: str
    generator: GeneratorSpec | None = None
    metadata: Dict[str, str] | None = None


def _coerce_interface_specs(
    interfaces: Sequence[InterfaceSpec]
    | Mapping[Tuple[str, str], float]
    | Iterable[Tuple[Tuple[str, str], float]]
    | None,
    *,
    interface_costs: Mapping[Tuple[str, str], float]
    | Iterable[Tuple[Tuple[str, str], float]]
    | None = None,
) -> List[InterfaceSpec]:
    """Return interface specifications with backward-compatible coercion."""

    if interfaces is None:
        coerced: List[InterfaceSpec] = []
    elif isinstance(interfaces, Mapping):
        coerced = _coerce_from_legacy_pairs(list(interfaces.items()))
    else:
        values = list(interfaces)
        if not values:
            coerced = []
        elif all(isinstance(value, InterfaceSpec) for value in values):
            coerced = [_sanitize_interface_spec(value) for value in values]
        else:
            coerced = _coerce_from_legacy_pairs(values)  # type: ignore[arg-type]

    cost_map = _normalize_interface_costs(interface_costs, coerced)
    if cost_map:
        coerced = [
            replace(spec, added_cost_per_mwh=cost_map[frozenset({spec.from_region, spec.to_region})])
            for spec in coerced
        ]

    return sorted(coerced, key=lambda spec: (spec.from_region, spec.to_region, spec.interface_id or ""))


def _coerce_from_legacy_pairs(
    items: Iterable[Tuple[Tuple[str, str], float]]
) -> List[InterfaceSpec]:
    """Convert legacy pair-based interface mappings into :class:`InterfaceSpec`."""

    aggregated: Dict[Tuple[str, str], Dict[str, object]] = {}
    for regions, limit in items:
        if len(regions) != 2:
            raise ValueError("Interface keys must contain exactly two regions.")

        region_a, region_b = (str(regions[0]), str(regions[1]))
        if region_a == region_b:
            raise ValueError("Interfaces must connect two distinct regions.")

        try:
            limit_value = float(limit)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError("Interface limits must be numeric values.") from exc
        if limit_value < 0.0:
            raise ValueError("Transfer capability must be non-negative.")

        limit_mw = limit_value / HOURS_PER_YEAR
        key = tuple(sorted((region_a, region_b)))
        entry = aggregated.setdefault(
            key,
            {
                "from_region": region_a,
                "to_region": region_b,
                "capacity_mw": None,
                "reverse_capacity_mw": None,
            },
        )

        if entry["from_region"] == region_a and entry["to_region"] == region_b:
            existing = entry["capacity_mw"]
            if existing is not None and abs(float(existing) - limit_mw) > _TOL:
                raise ValueError(
                    "Conflicting limits specified for interface between "
                    f"{region_a} and {region_b}.",
                )
            entry["capacity_mw"] = limit_mw
        elif entry["from_region"] == region_b and entry["to_region"] == region_a:
            existing = entry["reverse_capacity_mw"]
            if existing is not None and abs(float(existing) - limit_mw) > _TOL:
                raise ValueError(
                    "Conflicting limits specified for interface between "
                    f"{region_a} and {region_b}.",
                )
            entry["reverse_capacity_mw"] = limit_mw
        else:
            # First observation established canonical ordering; update reverse direction.
            entry["reverse_capacity_mw"] = limit_mw

    specs: List[InterfaceSpec] = []
    for entry in aggregated.values():
        forward = entry["capacity_mw"]
        reverse = entry["reverse_capacity_mw"]
        if forward is None and reverse is None:
            raise ValueError("Interfaces must specify at least one directional limit.")
        if forward is None:
            forward = reverse
        if reverse is None:
            reverse = forward
        specs.append(
            InterfaceSpec(
                from_region=str(entry["from_region"]),
                to_region=str(entry["to_region"]),
                capacity_mw=float(forward) if forward is not None else None,
                reverse_capacity_mw=float(reverse) if reverse is not None else None,
            )
        )

    return specs


def _sanitize_interface_spec(spec: InterfaceSpec) -> InterfaceSpec:
    """Return ``spec`` with normalized string fields and numeric defaults."""

    from_region = str(spec.from_region)
    to_region = str(spec.to_region)
    capacity = _optional_non_negative_float(spec.capacity_mw)
    reverse_capacity = _optional_non_negative_float(spec.reverse_capacity_mw)
    efficiency = _optional_non_negative_float(spec.efficiency, default=1.0)
    if efficiency is None or efficiency <= 0.0:
        efficiency = 1.0
    cost = _optional_non_negative_float(spec.added_cost_per_mwh, default=0.0)
    contracted_forward = _optional_non_negative_float(
        getattr(spec, "contracted_flow_mw_forward", None)
        if hasattr(spec, "contracted_flow_mw_forward")
        else getattr(spec, "contracted_flow_mw", None)
    )
    contracted_reverse = _optional_non_negative_float(
        getattr(spec, "contracted_flow_mw_reverse", None)
        if hasattr(spec, "contracted_flow_mw_reverse")
        else getattr(spec, "uncontracted_flow_mw", None)
    )

    interface_id = spec.interface_id
    if interface_id is not None:
        interface_id = str(interface_id)
        if not interface_id:
            interface_id = None

    interface_type = getattr(spec, "interface_type", None)
    if interface_type is not None:
        interface_type = str(interface_type)
        if not interface_type.strip():
            interface_type = None

    notes = spec.notes
    if notes is not None:
        notes = str(notes)
        if not notes.strip():
            notes = None

    profile_id = spec.profile_id
    if profile_id is not None:
        profile_id = str(profile_id)
        if not profile_id:
            profile_id = None

    in_service_year = spec.in_service_year
    if in_service_year is not None:
        try:
            in_service_year = int(in_service_year)
        except (TypeError, ValueError):
            in_service_year = None

    return InterfaceSpec(
        from_region=from_region,
        to_region=to_region,
        capacity_mw=capacity,
        reverse_capacity_mw=reverse_capacity,
        efficiency=efficiency,
        added_cost_per_mwh=cost if cost is not None else 0.0,
        interface_id=interface_id,
        contracted_flow_mw_forward=contracted_forward,
        contracted_flow_mw_reverse=contracted_reverse,
        interface_type=interface_type,
        notes=notes,
        profile_id=profile_id,
        in_service_year=in_service_year,
    )


def _optional_non_negative_float(
    value: float | int | None, *, default: float | None = None
) -> float | None:
    """Return ``value`` coerced to a non-negative float where possible."""

    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return max(0.0, numeric)


def _normalize_interface_costs(
    costs: Mapping[Tuple[str, str], float]
    | Iterable[Tuple[Tuple[str, str], float]]
    | None,
    interfaces: Sequence[InterfaceSpec],
) -> Dict[frozenset[str], float]:
    """Normalize optional wheeling costs keyed by interface region pairs."""

    keys = {frozenset({spec.from_region, spec.to_region}) for spec in interfaces}
    if not keys:
        return {}

    normalized: Dict[frozenset[str], float] = {key: 0.0 for key in keys}
    if not costs:
        return normalized

    items = costs.items() if isinstance(costs, Mapping) else costs
    for regions, value in items:
        if len(regions) != 2:
            raise ValueError("Interface cost keys must contain exactly two regions.")

        region_a, region_b = (str(regions[0]), str(regions[1]))
        if region_a == region_b:
            raise ValueError("Interface costs must reference two distinct regions.")

        key = frozenset({region_a, region_b})
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError("Interface costs must be numeric values.") from exc
        normalized[key] = max(0.0, numeric)

    return normalized


@lru_cache(maxsize=1)
def _backend_instance() -> LpBackend:
    """Return the configured LP backend instance."""

    try:
        return get_backend(DISPATCH_SOLVER)
    except ValueError as exc:
        raise ValueError(f"Unsupported dispatch solver backend: {DISPATCH_SOLVER!r}") from exc
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised when backend missing
        raise ModuleNotFoundError(
            f"Dispatch solver backend '{DISPATCH_SOLVER}' is unavailable: {exc}"
        ) from exc


def _build_failure_diagnostics(
    values: Sequence[float] | None,
    variables: Sequence[_LpVariable],
    equality_matrix: Sequence[Sequence[float]] | None,
    equality_rhs: Sequence[float] | None,
    equality_metadata: Sequence[Tuple[str, str]] | None,
    inequality_matrix: Sequence[Sequence[float]] | None,
    inequality_rhs: Sequence[float] | None,
    inequality_metadata: Sequence[Tuple[str, str]] | None,
) -> Dict[str, object]:
    """Assemble diagnostics describing the most violated constraints."""

    diagnostics: Dict[str, object] = {}
    if not values or len(values) != len(variables):
        diagnostics["detail"] = "Solver did not return a candidate solution."
        return diagnostics

    solution = list(values)

    if equality_matrix and equality_rhs and equality_metadata:
        worst = None
        for row, rhs, meta in zip(equality_matrix, equality_rhs, equality_metadata):
            total = sum(coeff * solution[idx] for idx, coeff in enumerate(row))
            residual = total - rhs
            if worst is None or abs(residual) > abs(worst["residual"]):
                worst = {
                    "type": meta[0],
                    "key": str(meta[1]),
                    "residual": float(residual),
                }
        if worst is not None:
            diagnostics["worst_equality"] = worst

    if inequality_matrix and inequality_rhs and inequality_metadata:
        worst_ineq = None
        for row, rhs, meta in zip(inequality_matrix, inequality_rhs, inequality_metadata):
            total = sum(coeff * solution[idx] for idx, coeff in enumerate(row))
            violation = total - rhs
            if violation > _TOL:
                if worst_ineq is None or violation > worst_ineq["violation"]:
                    worst_ineq = {
                        "type": meta[0],
                        "key": str(meta[1]),
                        "violation": float(violation),
                    }
        if worst_ineq is not None:
            diagnostics["worst_inequality"] = worst_ineq

    worst_lower = None
    worst_upper = None
    for value, variable in zip(solution, variables):
        lower = variable.lower
        upper = variable.upper
        if lower is not None and value < lower - _TOL:
            delta = lower - value
            if worst_lower is None or delta > worst_lower["violation"]:
                worst_lower = {
                    "variable": variable.name,
                    "violation": float(delta),
                }
        if upper is not None and value > upper + _TOL:
            delta = value - upper
            if worst_upper is None or delta > worst_upper["violation"]:
                worst_upper = {
                    "variable": variable.name,
                    "violation": float(delta),
                }

    if worst_lower is not None:
        diagnostics["worst_lower_bound"] = worst_lower
    if worst_upper is not None:
        diagnostics["worst_upper_bound"] = worst_upper

    return diagnostics



def solve(
    load_by_region: Mapping[str, float],
    generators: Sequence[GeneratorSpec],
    interfaces: Sequence[InterfaceSpec]
    | Mapping[Tuple[str, str], float]
    | Iterable[Tuple[Tuple[str, str], float]]
    | None,
    allowance_cost: float,
    carbon_price: float = 0.0,
    region_coverage: Mapping[str, bool] | None = None,
    *,
    year: int | None = None,
    generation_standard: GenerationStandardPolicy | None = None,
    interface_costs: Mapping[Tuple[str, str], float]
    | Iterable[Tuple[Tuple[str, str], float]]
    | None = None,
    unserved_energy_penalty: float = 1e6,
    peak_load_by_region: Mapping[str, float] | None = None,
    unserved_capacity_penalty: float = 1e6,
    emissions_cap_tons: float | None = None,
) -> DispatchResult:
    """Solve the economic dispatch problem with transmission interfaces.
    
    If emissions_cap_tons is provided, adds a hard constraint that total emissions
    must not exceed the cap. Solver will fail (return infeasible) if cap cannot be met.
    """

    if not generators and not load_by_region:
        raise ValueError(
            "At least one generator or load is required to solve the dispatch problem."
        )

    interface_specs = _coerce_interface_specs(
        interfaces, interface_costs=interface_costs
    )

    unserved_energy_penalty = max(0.0, float(unserved_energy_penalty))
    unserved_capacity_penalty = max(0.0, float(unserved_capacity_penalty))

    peak_load_map: Dict[str, float] = {}
    if peak_load_by_region:
        for region, value in peak_load_by_region.items():
            region_key = str(region)
            try:
                normalized = normalize_region(region)
                if normalized:
                    region_key = normalized
            except Exception:
                region_key = str(region)
            try:
                demand_value = float(value)
            except (TypeError, ValueError):
                continue
            if demand_value < 0.0:
                demand_value = 0.0
            peak_load_map[region_key] = demand_value

    region_coverage_map = {
        str(region): bool(flag)
        for region, flag in (region_coverage.items() if region_coverage else [])
    }

    regions = set(str(region) for region in load_by_region)
    regions.update(str(region) for region in peak_load_map)
    for generator in generators:
        regions.add(generator.region)
    for spec in interface_specs:
        regions.add(spec.from_region)
        regions.add(spec.to_region)

    if not regions:
        raise ValueError("No regions supplied for dispatch problem.")

    region_list = sorted(regions)
    region_index = {region: idx for idx, region in enumerate(region_list)}
    peak_requirement_by_region: Dict[str, float] = {
        region: float(peak_load_map.get(region, 0.0)) for region in region_list
    }
    for spec in interface_specs:
        efficiency = float(spec.efficiency if spec.efficiency is not None else 1.0)
        if efficiency <= 0.0:
            efficiency = 1.0
        forward_contract = float(spec.contracted_flow_mw_forward or 0.0)
        if forward_contract > 0.0:
            peak_requirement_by_region[spec.from_region] = (
                peak_requirement_by_region.get(spec.from_region, 0.0) + forward_contract
            )
            adjusted = peak_requirement_by_region.get(spec.to_region, 0.0) - forward_contract * efficiency
            peak_requirement_by_region[spec.to_region] = max(0.0, adjusted)
        reverse_contract = float(spec.contracted_flow_mw_reverse or 0.0)
        if reverse_contract > 0.0:
            peak_requirement_by_region[spec.to_region] = (
                peak_requirement_by_region.get(spec.to_region, 0.0) + reverse_contract
            )
            adjusted = peak_requirement_by_region.get(spec.from_region, 0.0) - reverse_contract * efficiency
            peak_requirement_by_region[spec.from_region] = max(0.0, adjusted)
    capacity_available_mw: Dict[str, float] = {region: 0.0 for region in region_list}

    if generation_standard is not None and year is None:
        raise ValueError("year must be provided when applying generation standard constraints")

    backend = _backend_instance()

    generators_by_region: Dict[str, list[int]] = {region: [] for region in region_list}
    tech_generators: Dict[Tuple[str, str], list[int]] = {}
    tech_capacity_totals: Dict[Tuple[str, str], float] = {}

    variables: List[_LpVariable] = []
    generator_indices: List[int] = []
    interface_variables: List[Tuple[InterfaceSpec, int | None, int | None]] = []
    load_shed_indices: Dict[int, str] = {}
    peak_shortage_indices: Dict[str, int] = {}
    inferred_region_coverage: Dict[str, bool] = {}
    zero_cost_flow_penalties: Dict[int, float] = {}

    equality_rows: List[List[float]] = [[] for _ in region_list]
    equality_rhs: List[float] = [float(load_by_region.get(region, 0.0)) for region in region_list]
    equality_metadata: List[Tuple[str, str]] = [
        ("load_balance", region) for region in region_list
    ]

    def _append_to_balance(coefficients: Sequence[float]) -> None:
        for row, value in zip(equality_rows, coefficients):
            row.append(float(value))

    for generator in generators:
        if generator.region not in region_index:
            raise ValueError(f"Region {generator.region} is not defined in the load data.")

        load_coeffs = [0.0] * len(region_list)
        load_coeffs[region_index[generator.region]] = 1.0
        _append_to_balance(load_coeffs)

        # Get availability from fuel type mapping to NREL max capacity factors
        availability = _get_availability_from_fuel(generator.fuel)
        capacity_limit = max(0.0, float(generator.capacity * availability))
        variable = _LpVariable(
            name=f"gen:{generator.name}",
            lower=0.0,
            upper=capacity_limit,
            cost=generator.marginal_cost(allowance_cost, carbon_price),
            kind="generator",
            generator=generator,
            metadata={"region": generator.region, "fuel": generator.fuel},
        )
        variables.append(variable)
        column_index = len(variables) - 1
        generator_indices.append(column_index)

        capacity_available_mw[generator.region] = capacity_available_mw.get(
            generator.region, 0.0
        ) + capacity_limit / HOURS_PER_YEAR

        generators_by_region[generator.region].append(column_index)
        fuel_key = str(generator.fuel).strip().lower()
        tech_key = (generator.region, fuel_key)
        tech_generators.setdefault(tech_key, []).append(column_index)
        tech_capacity_totals[tech_key] = tech_capacity_totals.get(tech_key, 0.0) + capacity_limit

        current_flag = inferred_region_coverage.get(generator.region)
        inferred_region_coverage[generator.region] = (
            generator.covered if current_flag is None else current_flag and generator.covered
        )

    for spec in interface_specs:
        region_a = spec.from_region
        region_b = spec.to_region

        forward_capacity_mw = spec.capacity_mw
        reverse_capacity_mw = spec.reverse_capacity_mw
        if forward_capacity_mw is None and reverse_capacity_mw is None:
            continue
        if forward_capacity_mw is None:
            forward_capacity_mw = reverse_capacity_mw
        if reverse_capacity_mw is None:
            reverse_capacity_mw = forward_capacity_mw

        forward_limit = (
            None
            if forward_capacity_mw is None
            else float(forward_capacity_mw) * HOURS_PER_YEAR
        )
        reverse_limit = (
            None
            if reverse_capacity_mw is None
            else float(reverse_capacity_mw) * HOURS_PER_YEAR
        )

        efficiency = float(spec.efficiency if spec.efficiency is not None else 1.0)
        if efficiency <= 0.0:
            efficiency = 1.0

        cost_rate = float(spec.added_cost_per_mwh)
        penalty_cost = _INTERFACE_FLOW_PENALTY if cost_rate <= 0.0 else 0.0

        metadata: Dict[str, str] = {"from": region_a, "to": region_b}
        if spec.interface_type is not None:
            metadata["type"] = str(spec.interface_type)
        if spec.interface_id is not None:
            metadata["interface_id"] = str(spec.interface_id)

        forward_coeffs = [0.0] * len(region_list)
        forward_coeffs[region_index[region_a]] = -1.0
        forward_coeffs[region_index[region_b]] = efficiency
        _append_to_balance(forward_coeffs)
        forward_contract_mw = float(spec.contracted_flow_mw_forward or 0.0)
        forward_lower = 0.0
        if forward_contract_mw > 0.0:
            forward_lower = forward_contract_mw * HOURS_PER_YEAR
            if forward_limit is not None and forward_lower > forward_limit + _TOL:
                raise ValueError(
                    "Contracted flow exceeds forward limit for interface "
                    f"{region_a}->{region_b}"
                )
        forward_variable = _LpVariable(
            name=f"flow:{region_a}->{region_b}",
            lower=forward_lower,
            upper=forward_limit,
            cost=cost_rate if cost_rate > 0.0 else penalty_cost,
            kind="flow",
            metadata=metadata,
        )
        variables.append(forward_variable)
        forward_idx = len(variables) - 1
        if penalty_cost:
            zero_cost_flow_penalties[forward_idx] = penalty_cost

        reverse_metadata: Dict[str, str] = {"from": region_b, "to": region_a}
        if spec.interface_type is not None:
            reverse_metadata["type"] = str(spec.interface_type)
        if spec.interface_id is not None:
            reverse_metadata["interface_id"] = str(spec.interface_id)

        reverse_coeffs = [0.0] * len(region_list)
        reverse_coeffs[region_index[region_a]] = efficiency
        reverse_coeffs[region_index[region_b]] = -1.0
        _append_to_balance(reverse_coeffs)
        reverse_contract_mw = float(spec.contracted_flow_mw_reverse or 0.0)
        reverse_lower = 0.0
        if reverse_contract_mw > 0.0:
            reverse_lower = reverse_contract_mw * HOURS_PER_YEAR
            if reverse_limit is not None and reverse_lower > reverse_limit + _TOL:
                raise ValueError(
                    "Contracted flow exceeds reverse limit for interface "
                    f"{region_b}->{region_a}"
                )
        reverse_variable = _LpVariable(
            name=f"flow:{region_b}->{region_a}",
            lower=reverse_lower,
            upper=reverse_limit,
            cost=cost_rate if cost_rate > 0.0 else penalty_cost,
            kind="flow",
            metadata=reverse_metadata,
        )
        variables.append(reverse_variable)
        reverse_idx = len(variables) - 1
        if penalty_cost:
            zero_cost_flow_penalties[reverse_idx] = penalty_cost

        interface_variables.append((spec, forward_idx, reverse_idx))

    curtailment_indices: Dict[int, str] = {}
    
    for region in region_list:
        demand = max(float(load_by_region.get(region, 0.0)), 0.0)
        
        # Unserved energy variable (when generation < demand)
        coeffs = [0.0] * len(region_list)
        coeffs[region_index[region]] = 1.0
        _append_to_balance(coeffs)
        load_variable = _LpVariable(
            name=f"unserved:{region}",
            lower=0.0,
            upper=demand,
            cost=float(unserved_energy_penalty),
            kind="unserved_energy",
            metadata={"region": region},
        )
        variables.append(load_variable)
        load_shed_indices[len(variables) - 1] = region
        
        # Curtailment/excess energy variable (when generation > demand + export capacity)
        # This allows clean generators to run below their maximum when surplus exists
        coeffs = [0.0] * len(region_list)
        coeffs[region_index[region]] = -1.0
        _append_to_balance(coeffs)
        curtailment_variable = _LpVariable(
            name=f"curtail:{region}",
            lower=0.0,
            upper=None,  # No upper limit on curtailment
            cost=0.0,  # Zero cost - curtailment is free
            kind="curtailment",
            metadata={"region": region},
        )
        variables.append(curtailment_variable)
        curtailment_indices[len(variables) - 1] = region

    inequality_rows: List[List[float]] = []
    inequality_rhs: List[float] = []
    inequality_metadata: List[Tuple[str, str]] = []

    for region in region_list:
        requirement = peak_requirement_by_region.get(region, 0.0)
        available = capacity_available_mw.get(region, 0.0)
        if requirement <= 0.0 and available <= 0.0 and region not in peak_load_map:
            continue

        _append_to_balance([0.0] * len(region_list))
        for row in inequality_rows:
            row.append(0.0)

        shortage_variable = _LpVariable(
            name=f"peak_shortage:{region}",
            lower=0.0,
            upper=None,
            cost=float(unserved_capacity_penalty),
            kind="unserved_capacity",
            metadata={"region": region},
        )
        variables.append(shortage_variable)
        shortage_idx = len(variables) - 1
        peak_shortage_indices[region] = shortage_idx

        row = [0.0] * len(variables)
        row[shortage_idx] = -1.0
        rhs_value = float(available - requirement)
        inequality_rows.append(row)
        inequality_rhs.append(rhs_value)
        inequality_metadata.append(("peak_demand", region))

    if generation_standard is not None:
        requirements = generation_standard.requirements_for_year(int(year))
        for requirement in requirements:
            region = str(requirement.region)
            if region not in region_index:
                continue

            tech_key = (region, requirement.technology_key)
            tech_indices = tech_generators.get(tech_key, [])
            available_capacity = tech_capacity_totals.get(tech_key, 0.0)

            if requirement.capacity_mw > 0.0:
                required_capacity = float(requirement.capacity_mw) * HOURS_PER_YEAR
                if available_capacity + _TOL < required_capacity:
                    available_mw = available_capacity / HOURS_PER_YEAR
                    raise ValueError(
                        "Generation standard for technology "
                        f"'{requirement.technology}' in region {region} requires "
                        f"{requirement.capacity_mw} MW but only {available_mw:.3f} MW is available"
                    )

            share = float(requirement.generation_share)
            if share <= 0.0:
                continue

            region_indices = generators_by_region.get(region, [])
            if not region_indices:
                raise ValueError(
                    "Generation standard requires generation in region "
                    f"{region}, but no generators are available"
                )
            if not tech_indices:
                raise ValueError(
                    "Generation standard for technology "
                    f"'{requirement.technology}' in region {region} cannot be enforced "
                    "because no matching generators are available"
                )

            row = [0.0] * len(variables)
            tech_index_set = set(tech_indices)
            for idx in tech_indices:
                row[idx] -= 1.0 - share
            for idx in region_indices:
                if idx in tech_index_set:
                    continue
                row[idx] += share

            inequality_rows.append(row)
            inequality_rhs.append(0.0)
            inequality_metadata.append(
                (
                    "generation_standard",
                    f"{region}:{requirement.technology_key}",
                )
            )

    # Add hard emissions cap constraint if specified
    if emissions_cap_tons is not None:
        emissions_row = [0.0] * len(variables)
        for gen_idx in generator_indices:
            var = variables[gen_idx]
            if var.generator is not None:
                # Coefficient is emission_rate (tons CO2 per MWh)
                emissions_row[gen_idx] = float(var.generator.emission_rate)
        
        inequality_rows.append(emissions_row)
        inequality_rhs.append(float(emissions_cap_tons))
        inequality_metadata.append(("emissions_cap", "total"))

    cost_vector = [variable.cost for variable in variables]
    bounds = [(variable.lower, variable.upper) for variable in variables]
    names = [variable.name for variable in variables]

    equality_matrix = equality_rows if equality_rows else None
    inequality_matrix = inequality_rows if inequality_rows else None
    equality_rhs_vector = equality_rhs if equality_rows else None
    inequality_rhs_vector = inequality_rhs if inequality_rows else None

    values, objective, status = backend.solve(
        cost_vector,
        inequality_matrix,
        inequality_rhs_vector,
        equality_matrix,
        equality_rhs_vector,
        bounds,
        names,
    )

    solution = [float(value) for value in values]
    if not status.success:
        diagnostics = _build_failure_diagnostics(
            solution,
            variables,
            equality_matrix,
            equality_rhs_vector,
            equality_metadata,
            inequality_matrix,
            inequality_rhs_vector,
            inequality_metadata,
        )
        LOGGER.error(
            "dispatch_network solver failure status=%s message=%s diagnostics=%s",
            status.status,
            status.message,
            diagnostics,
        )
        raise RuntimeError(
            "Dispatch linear program failed: "
            f"{status.message} (status={status.status}). Diagnostics: {diagnostics}"
        )

    penalty_total = 0.0
    if zero_cost_flow_penalties:
        penalty_total = sum(
            penalty * solution[idx]
            for idx, penalty in zero_cost_flow_penalties.items()
        )

    objective = float(objective) - penalty_total
    if objective < 0.0 and abs(objective) < FLOW_TOL:
        objective = 0.0

    constraint_duals: Dict[str, Dict[str, float]] = {}
    equality_duals = list(status.dual_eq) if status.dual_eq else []
    for idx, meta in enumerate(equality_metadata):
        category, key = meta
        bucket = constraint_duals.setdefault(category, {})
        value = equality_duals[idx] if idx < len(equality_duals) else 0.0
        bucket[str(key)] = float(value)
    inequality_duals = list(status.dual_ineq) if status.dual_ineq else []
    for idx, meta in enumerate(inequality_metadata):
        category, key = meta
        bucket = constraint_duals.setdefault(category, {})
        value = inequality_duals[idx] if idx < len(inequality_duals) else 0.0
        bucket[str(key)] = float(value)

    region_prices = {
        region: float(constraint_duals.get("load_balance", {}).get(region, 0.0))
        for region in region_list
    }

    gen_by_fuel: Dict[str, float] = {}
    emissions_tons = 0.0
    emissions_by_region_totals: Dict[str, float] = {region: 0.0 for region in region_list}
    generation_detail_by_region: Dict[str, Dict[str, float]] = {
        region: {} for region in region_list
    }
    generation_by_coverage: Dict[str, float] = {"covered": 0.0, "non_covered": 0.0}
    emissions_by_fuel_totals: Dict[str, float] = {}
    capacity_mwh_by_fuel: Dict[str, float] = {}
    capacity_mw_by_fuel: Dict[str, float] = {}
    generation_by_unit: Dict[str, float] = {}
    capacity_mwh_by_unit: Dict[str, float] = {}
    capacity_mw_by_unit: Dict[str, float] = {}
    variable_cost_by_fuel: Dict[str, float] = {}
    allowance_cost_by_fuel: Dict[str, float] = {}
    carbon_price_cost_by_fuel: Dict[str, float] = {}
    total_cost_by_fuel: Dict[str, float] = {}
    capacity_region_mwh: Dict[str, Dict[str, float]] = {region: {} for region in region_list}
    capacity_region_mw: Dict[str, Dict[str, float]] = {region: {} for region in region_list}
    variable_cost_region: Dict[str, Dict[str, float]] = {region: {} for region in region_list}
    allowance_cost_region: Dict[str, Dict[str, float]] = {region: {} for region in region_list}
    carbon_price_cost_region: Dict[str, Dict[str, float]] = {region: {} for region in region_list}
    total_cost_region: Dict[str, Dict[str, float]] = {region: {} for region in region_list}

    for idx in generator_indices:
        variable = variables[idx]
        generator = variable.generator
        assert generator is not None
        output = float(solution[idx])
        generation_by_unit[generator.name] = output
        gen_by_fuel.setdefault(generator.fuel, 0.0)
        gen_by_fuel[generator.fuel] += output
        emissions_tons += generator.emission_rate * output
        emissions_by_region_totals[generator.region] += generator.emission_rate * output
        region_generation = generation_detail_by_region.setdefault(generator.region, {})
        region_generation[generator.fuel] = region_generation.get(generator.fuel, 0.0) + output
        coverage_key = "covered" if generator.covered else "non_covered"
        generation_by_coverage[coverage_key] += output

        capacity_mwh = float(generator.capacity)
        capacity_mw = capacity_mwh / HOURS_PER_YEAR
        capacity_mwh_by_unit[generator.name] = capacity_mwh
        capacity_mw_by_unit[generator.name] = capacity_mw
        capacity_mwh_by_fuel[generator.fuel] = capacity_mwh_by_fuel.get(generator.fuel, 0.0) + capacity_mwh
        capacity_mw_by_fuel[generator.fuel] = capacity_mw_by_fuel.get(generator.fuel, 0.0) + capacity_mw
        region_capacity_mwh = capacity_region_mwh.setdefault(generator.region, {})
        region_capacity_mw = capacity_region_mw.setdefault(generator.region, {})
        region_capacity_mwh[generator.fuel] = (
            region_capacity_mwh.get(generator.fuel, 0.0) + capacity_mwh
        )
        region_capacity_mw[generator.fuel] = (
            region_capacity_mw.get(generator.fuel, 0.0) + capacity_mw
        )

        emissions_value = generator.emission_rate * output
        emissions_by_fuel_totals[generator.fuel] = (
            emissions_by_fuel_totals.get(generator.fuel, 0.0) + emissions_value
        )

        variable_rate = float(generator.variable_cost)
        allowance_rate = generator.emission_rate * (float(allowance_cost) if generator.covered else 0.0)
        carbon_price_rate = generator.emission_rate * float(carbon_price)
        total_rate = variable_rate + allowance_rate + carbon_price_rate

        variable_cost_by_fuel[generator.fuel] = (
            variable_cost_by_fuel.get(generator.fuel, 0.0) + variable_rate * output
        )
        allowance_cost_by_fuel[generator.fuel] = (
            allowance_cost_by_fuel.get(generator.fuel, 0.0) + allowance_rate * output
        )
        carbon_price_cost_by_fuel[generator.fuel] = (
            carbon_price_cost_by_fuel.get(generator.fuel, 0.0) + carbon_price_rate * output
        )
        total_cost_by_fuel[generator.fuel] = (
            total_cost_by_fuel.get(generator.fuel, 0.0) + total_rate * output
        )
        region_variable_cost = variable_cost_region.setdefault(generator.region, {})
        region_allowance_cost = allowance_cost_region.setdefault(generator.region, {})
        region_carbon_cost = carbon_price_cost_region.setdefault(generator.region, {})
        region_total_cost = total_cost_region.setdefault(generator.region, {})
        region_variable_cost[generator.fuel] = (
            region_variable_cost.get(generator.fuel, 0.0) + variable_rate * output
        )
        region_allowance_cost[generator.fuel] = (
            region_allowance_cost.get(generator.fuel, 0.0) + allowance_rate * output
        )
        region_carbon_cost[generator.fuel] = (
            region_carbon_cost.get(generator.fuel, 0.0) + carbon_price_rate * output
        )
        region_total_cost[generator.fuel] = (
            region_total_cost.get(generator.fuel, 0.0) + total_rate * output
        )

    unserved_energy_by_region: Dict[str, float] = {r: 0.0 for r in region_list}
    for idx, region in load_shed_indices.items():
        unserved_energy_by_region[region] += float(solution[idx])
    unserved_energy_total = sum(unserved_energy_by_region.values())

    curtailment_by_region: Dict[str, float] = {r: 0.0 for r in region_list}
    for idx, region in curtailment_indices.items():
        curtailment_by_region[region] += float(solution[idx])
    curtailment_total = sum(curtailment_by_region.values())

    unserved_capacity_by_region: Dict[str, float] = {r: 0.0 for r in region_list}
    for region, idx in peak_shortage_indices.items():
        unserved_capacity_by_region[region] = float(max(solution[idx], 0.0))
    unserved_capacity_total = sum(unserved_capacity_by_region.values())

    costs_by_region: Dict[str, Dict[str, Dict[str, float]]] = {}
    for region in region_list:
        fuels = set(variable_cost_region.get(region, {}))
        fuels |= set(allowance_cost_region.get(region, {}))
        fuels |= set(carbon_price_cost_region.get(region, {}))
        fuels |= set(total_cost_region.get(region, {}))
        if not fuels:
            continue
        region_costs: Dict[str, Dict[str, float]] = {}
        for fuel in fuels:
            region_costs[fuel] = {
                "variable_cost": float(
                    variable_cost_region.get(region, {}).get(fuel, 0.0)
                ),
                "allowance_cost": float(
                    allowance_cost_region.get(region, {}).get(fuel, 0.0)
                ),
                "carbon_price_cost": float(
                    carbon_price_cost_region.get(region, {}).get(fuel, 0.0)
                ),
                "total_cost": float(total_cost_region.get(region, {}).get(fuel, 0.0)),
            }
        costs_by_region[region] = region_costs

    capacity_by_region: Dict[str, Dict[str, Dict[str, float]]] = {}
    for region, cap_map in capacity_region_mwh.items():
        fuels = set(cap_map) | set(capacity_region_mw.get(region, {}))
        if not fuels:
            continue
        region_caps: Dict[str, Dict[str, float]] = {}
        for fuel in fuels:
            region_caps[fuel] = {
                "capacity_mwh": float(capacity_region_mwh.get(region, {}).get(fuel, 0.0)),
                "capacity_mw": float(capacity_region_mw.get(region, {}).get(fuel, 0.0)),
            }
        capacity_by_region[region] = region_caps

    demand_by_region = {
        region: float(load_by_region.get(region, 0.0)) for region in region_list
    }

    emissions_by_region = {
        region: float(total) for region, total in emissions_by_region_totals.items()
    }

    for region in region_list:
        region_key = str(region)
        emissions_by_region.setdefault(region_key, 0.0)
        generation_detail_by_region.setdefault(region_key, {})
        capacity_by_region.setdefault(region_key, {})
        costs_by_region.setdefault(region_key, {})

    flows: Dict[Tuple[str, str], float] = {}
    wheeling_cost_by_interface: Dict[Tuple[str, str], float] = {}
    for spec, forward_idx, reverse_idx in interface_variables:
        forward_val = float(solution[forward_idx]) if forward_idx is not None else 0.0
        reverse_val = float(solution[reverse_idx]) if reverse_idx is not None else 0.0
        net_flow = forward_val - reverse_val
        key = (spec.from_region, spec.to_region)
        flows[key] = flows.get(key, 0.0) + net_flow

        cost_rate = float(spec.added_cost_per_mwh)
        wheeling_cost_by_interface[key] = wheeling_cost_by_interface.get(key, 0.0) + (
            cost_rate * (forward_val + reverse_val)
        )

    generation_total_by_region_calc: Dict[str, float] = {}
    for region, fuels in generation_detail_by_region.items():
        total = 0.0
        for value in fuels.values():
            try:
                total += float(value)
            except (TypeError, ValueError):
                continue
        generation_total_by_region_calc[region] = total

    region_coverage_result = {
        region: bool(
            region_coverage_map.get(region, inferred_region_coverage.get(region, True))
        )
        for region in region_list
    }
    imports_to_covered = 0.0
    exports_from_covered = 0.0
    for region, covered in region_coverage_result.items():
        if not covered:
            continue
        load = float(load_by_region.get(region, 0.0))
        generation = generation_total_by_region_calc.get(region, 0.0)
        unmet = unserved_energy_by_region.get(region, 0.0)
        net_import = load - generation - unmet
        if net_import > _TOL:
            imports_to_covered += net_import
        elif net_import < -_TOL:
            exports_from_covered += -net_import

    effective_price = max(float(allowance_cost), float(carbon_price))

    return DispatchResult(
        gen_by_fuel=gen_by_fuel,
        region_prices=region_prices,
        emissions_tons=emissions_tons,
        emissions_by_region=emissions_by_region,
        flows=flows,
        emissions_by_fuel=emissions_by_fuel_totals,
        capacity_mwh_by_fuel=capacity_mwh_by_fuel,
        capacity_mw_by_fuel=capacity_mw_by_fuel,
        generation_by_unit=generation_by_unit,
        capacity_mwh_by_unit=capacity_mwh_by_unit,
        capacity_mw_by_unit=capacity_mw_by_unit,
        variable_cost_by_fuel=variable_cost_by_fuel,
        allowance_cost_by_fuel=allowance_cost_by_fuel,
        carbon_price_cost_by_fuel=carbon_price_cost_by_fuel,
        total_cost_by_fuel=total_cost_by_fuel,
        demand_by_region=demand_by_region,
        peak_demand_by_region={
            region: float(peak_requirement_by_region.get(region, 0.0))
            for region in region_list
        },
        generation_by_region=generation_total_by_region_calc,
        generation_detail_by_region=generation_detail_by_region,
        generation_by_coverage=generation_by_coverage,
        capacity_by_region=capacity_by_region,
        costs_by_region=costs_by_region,
        imports_to_covered=imports_to_covered,
        exports_from_covered=exports_from_covered,
        region_coverage=region_coverage_result,
        allowance_cost=float(allowance_cost),
        carbon_price=float(carbon_price),
        effective_carbon_price=effective_price,
        constraint_duals=constraint_duals,
        total_cost=float(objective),
        wheeling_cost_by_interface={
            pair: float(value) for pair, value in wheeling_cost_by_interface.items()
        },
        unserved_energy_by_region={
            region: float(value) for region, value in unserved_energy_by_region.items()
        },
        unserved_energy_total=float(unserved_energy_total),
        unserved_energy_penalty=float(unserved_energy_penalty),
        curtailment_by_region={
            region: float(value) for region, value in curtailment_by_region.items()
        },
        curtailment_total=float(curtailment_total),
        unserved_capacity_by_region={
            region: float(value) for region, value in unserved_capacity_by_region.items()
        },
        unserved_capacity_total=float(unserved_capacity_total),
        unserved_capacity_penalty=float(unserved_capacity_penalty),
    )


def solve_from_frames(
    frames: Frames | Mapping[str, pd.DataFrame],
    year: int,
    allowance_cost: float,
    carbon_price: float = 0.0,
    *,
    generation_standard: GenerationStandardPolicy | None = None,
    capacity_expansion: bool = False,
    discount_rate: float = 0.07,
    emissions_cap_tons: float | None = None,
) -> DispatchResult:
    """Solve the dispatch problem using frame-based inputs with optional capacity expansion.
    
    If emissions_cap_tons is provided, enforces a hard emissions cap constraint.
    """

    _ensure_pandas()

    frames_obj = Frames.coerce(frames)

    demand_df = frames_obj.demand()
    if "region" not in demand_df.columns:
        raise ValueError(
            "demand frame must include a 'region' column for regional emissions reporting"
        )

    if demand_df.empty:
        raise ValueError("demand frame is empty; cannot solve network dispatch")
    if "demand_mwh" not in demand_df.columns:
        raise ValueError("demand frame must include a 'demand_mwh' column")
    demand_year_df = demand_df[demand_df["year"] == year]
    if demand_year_df.empty:
        raise ValueError(f"no demand observations available for year {year}")
    demand_year_df = demand_year_df.copy()
    demand_year_df["region"] = demand_year_df["region"].map(
        lambda value: normalize_region(value) or str(value)
    )
    demand_year_df["mwh"] = demand_year_df["demand_mwh"].astype(float)
    demand_total_mwh = float(demand_year_df["mwh"].sum())
    assert demand_total_mwh > 0.0
    print("DEM", year, demand_total_mwh)
    demand_totals = demand_year_df.groupby("region")["mwh"].sum().astype(float)
    positive_regions = [region for region, total in demand_totals.items() if total > 0.0]
    if not positive_regions:
        raise ValueError(
            "demand must include non-zero load for at least one region when using the network solver"
        )


    load_mapping = {
        normalize_region(region) or str(region): float(value)
        for region, value in frames_obj.demand_for_year(year).items()
    }

    peak_load_mapping: Dict[str, float] = {}
    if frames_obj.has_frame("peak_demand"):
        try:
            peak_for_year = frames_obj.peak_demand_for_year(year)
        except KeyError as exc:
            raise ValueError(f"no peak demand observations available for year {year}") from exc
        peak_load_mapping = {
            normalize_region(region) or str(region): max(float(value), 0.0)
            for region, value in peak_for_year.items()
        }

    units = frames_obj.units()
    if "region" not in units.columns:
        raise ValueError("units frame must include a 'region' column for regional emissions reporting")
    if units["region"].isna().all():
        raise ValueError("units frame must specify at least one region for emissions reporting")

    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "dispatch_network inputs year=%s demand_regions=%s unit_regions=%s",
            year,
            sorted(str(region) for region in demand_df["region"].astype(str).unique()),
            sorted(str(region) for region in units["region"].astype(str).unique()),
        )
    fuels = frames_obj.fuels()
    coverage_by_fuel = {
        str(row.fuel): bool(row.covered) for row in fuels.itertuples(index=False)
    }
    coverage_by_region = frames_obj.coverage_for_year(year)

    generators: List[GeneratorSpec] = []
    for row in units.itertuples(index=False):
        region_raw = row.region
        region = (
            normalize_region(region_raw)
            if region_raw is not None and not pd.isna(region_raw)
            else _DEFAULT_REGION
        )
        if not region:
            region = _DEFAULT_REGION
        fuel = str(row.fuel)
        if coverage_by_region:
            covered = bool(coverage_by_region.get(region, True))
        else:
            covered = coverage_by_fuel.get(fuel, True)
        capacity = float(row.cap_mw) * float(row.availability) * HOURS_PER_YEAR
        carbon_cost = float(getattr(row, "carbon_cost_per_mwh", 0.0))
        variable_cost = (
            float(row.vom_per_mwh)
            + float(row.hr_mmbtu_per_mwh) * float(row.fuel_price_per_mmbtu)
            + carbon_cost
        )
        emission_rate = float(
            getattr(row, "co2_short_ton_per_mwh", getattr(row, "ef_ton_per_mwh", 0.0))
        )
        penalty_raw = getattr(row, "penalty_cost_per_mwh", None)
        if penalty_raw is None:
            penalty_raw = getattr(row, "penalty_per_mwh", None)
        penalty_cost = float(penalty_raw) if penalty_raw is not None else 0.0
        if penalty_cost != penalty_cost:  # NaN guard
            penalty_cost = 0.0

        unit_name = str(getattr(row, "unique_id", getattr(row, "unit_id", "")))
        generators.append(
            GeneratorSpec(
                name=unit_name,
                region=region,
                fuel=fuel,
                variable_cost=variable_cost,
                capacity=capacity,
                emission_rate=emission_rate,
                covered=bool(covered),
                penalty_cost=penalty_cost,
            )
        )

    def _as_optional_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        return numeric

    def _as_optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        return numeric

    interface_specs: List[InterfaceSpec] = []
    transmission_frame = frames_obj.transmission()
    for row in transmission_frame.itertuples(index=False):
        from_region_raw = getattr(row, "from_region", None)
        to_region_raw = getattr(row, "to_region", None)
        from_region = normalize_region(from_region_raw) or str(from_region_raw)
        to_region = normalize_region(to_region_raw) or str(to_region_raw)

        capacity = _as_optional_float(getattr(row, "capacity_mw", None))
        reverse_capacity = _as_optional_float(getattr(row, "reverse_capacity_mw", None))
        if capacity is None and reverse_capacity is None:
            limit_value = _as_optional_float(getattr(row, "limit_mw", None))
            if limit_value is not None:
                capacity = limit_value
            reverse_limit = _as_optional_float(getattr(row, "reverse_limit_mw", None))
            if reverse_capacity is None and reverse_limit is not None:
                reverse_capacity = reverse_limit

        efficiency = _as_optional_float(getattr(row, "efficiency", None))
        if efficiency is None or efficiency <= 0.0:
            efficiency = 1.0

        added_cost = _as_optional_float(getattr(row, "added_cost_per_mwh", None))
        if added_cost is None:
            added_cost = _as_optional_float(getattr(row, "cost_per_mwh", None)) or 0.0
        added_cost = max(0.0, added_cost)

        contracted_forward = _as_optional_float(getattr(row, "contracted_flow_mw_forward", None))
        if contracted_forward is None:
            contracted_forward = _as_optional_float(getattr(row, "contracted_flow_mw", None))
        contracted_reverse = _as_optional_float(getattr(row, "contracted_flow_mw_reverse", None))
        if contracted_reverse is None:
            contracted_reverse = _as_optional_float(getattr(row, "uncontracted_flow_mw", None))

        interface_type = getattr(row, "interface_type", None)
        if interface_type is not None:
            interface_type = str(interface_type)
            if not interface_type.strip():
                interface_type = None

        interface_id = getattr(row, "interface_id", None)
        if interface_id is not None:
            interface_id = str(interface_id)
            if not interface_id:
                interface_id = None

        notes = getattr(row, "notes", None)
        if notes is not None:
            notes = str(notes)
            if not notes.strip():
                notes = None

        profile_id = getattr(row, "profile_id", None)
        if profile_id is not None:
            profile_id = str(profile_id)
            if not profile_id:
                profile_id = None

        in_service_year = _as_optional_int(getattr(row, "in_service_year", None))

        interface_specs.append(
            InterfaceSpec(
                interface_id=interface_id,
                from_region=str(from_region),
                to_region=str(to_region),
                capacity_mw=capacity,
                reverse_capacity_mw=reverse_capacity,
                efficiency=efficiency,
                added_cost_per_mwh=added_cost,
                contracted_flow_mw_forward=contracted_forward,
                contracted_flow_mw_reverse=contracted_reverse,
                interface_type=interface_type,
                notes=notes,
                profile_id=profile_id,
                in_service_year=in_service_year,
            )
        )


    result = solve(
        load_by_region=load_mapping,
        generators=generators,
        interfaces=interface_specs,
        allowance_cost=allowance_cost,
        carbon_price=carbon_price,
        region_coverage=coverage_by_region,
        year=year,
        generation_standard=generation_standard,
        interface_costs=None,
        peak_load_by_region=peak_load_mapping,
        emissions_cap_tons=emissions_cap_tons,
    )

    # Capacity expansion logic
    if capacity_expansion and frames_obj.has_frame("expansion_candidates"):
        expansion_options = frames_obj.frame("expansion_candidates")
        
        if not expansion_options.empty:
            # Create dispatch summary with BOTH zonal AND legacy fields for backwards compatibility
            dispatch_summary = {
                "generation": pd.Series(result.generation_by_unit),
                "units": units,
                "region_prices": dict(result.region_prices),  # Zonal prices
                "price": max(result.region_prices.values()) if result.region_prices else 0.0,  # Legacy: system max price
                "unserved_by_region": dict(result.unserved_energy_by_region),  # Zonal shortfall
                "shortfall_mwh": result.unserved_energy_total,  # Legacy: total system shortfall
                "emissions_tons": result.emissions_tons,
            }
            
            def _dispatch_with_updated_units(updated_units: pd.DataFrame) -> dict:
                """Re-solve dispatch with updated unit fleet."""
                updated_frames = frames_obj.with_frame("units", updated_units)
                
                # Rebuild generators list from updated units
                new_generators: List[GeneratorSpec] = []
                for row in updated_units.itertuples(index=False):
                    region_raw = row.region
                    region = (
                        normalize_region(region_raw)
                        if region_raw is not None and not pd.isna(region_raw)
                        else _DEFAULT_REGION
                    )
                    if not region:
                        region = _DEFAULT_REGION
                    fuel = str(row.fuel)
                    if coverage_by_region:
                        covered = bool(coverage_by_region.get(region, True))
                    else:
                        covered = coverage_by_fuel.get(fuel, True)
                    capacity = float(row.cap_mw) * float(row.availability) * HOURS_PER_YEAR
                    carbon_cost = float(getattr(row, "carbon_cost_per_mwh", 0.0))
                    variable_cost = (
                        float(row.vom_per_mwh)
                        + float(row.hr_mmbtu_per_mwh) * float(row.fuel_price_per_mmbtu)
                        + carbon_cost
                    )
                    emission_rate = float(
                        getattr(row, "co2_short_ton_per_mwh", getattr(row, "ef_ton_per_mwh", 0.0))
                    )
                    penalty_raw = getattr(row, "penalty_cost_per_mwh", None)
                    if penalty_raw is None:
                        penalty_raw = getattr(row, "penalty_per_mwh", None)
                    penalty_cost = float(penalty_raw) if penalty_raw is not None else 0.0
                    if penalty_cost != penalty_cost:  # NaN guard
                        penalty_cost = 0.0
                    
                    unit_name = str(getattr(row, "unique_id", getattr(row, "unit_id", "")))
                    new_generators.append(
                        GeneratorSpec(
                            name=unit_name,
                            region=region,
                            fuel=fuel,
                            variable_cost=variable_cost,
                            capacity=capacity,
                            emission_rate=emission_rate,
                            covered=bool(covered),
                            penalty_cost=penalty_cost,
                        )
                    )
                
                # Re-solve with updated generators
                new_result = solve(
                    load_by_region=load_mapping,
                    generators=new_generators,
                    interfaces=interface_specs,
                    allowance_cost=allowance_cost,
                    carbon_price=carbon_price,
                    region_coverage=coverage_by_region,
                    year=year,
                    generation_standard=generation_standard,
                    interface_costs=None,
                    peak_load_by_region=peak_load_mapping,
                    emissions_cap_tons=emissions_cap_tons,
                )
                
                return {
                    "generation": pd.Series(new_result.generation_by_unit),
                    "units": updated_units,
                    "region_prices": dict(new_result.region_prices),  # Zonal prices
                    "price": max(new_result.region_prices.values()) if new_result.region_prices else 0.0,  # Legacy: system max price
                    "unserved_by_region": dict(new_result.unserved_energy_by_region),  # Zonal shortfall
                    "shortfall_mwh": new_result.unserved_energy_total,  # Legacy: total system shortfall
                    "emissions_tons": new_result.emissions_tons,
                }
            
            # Run capacity expansion
            updated_units, updated_summary, build_log, expansion_status = plan_capacity_expansion(
                units,
                expansion_options,
                dispatch_summary,
                _dispatch_with_updated_units,
                allowance_cost=allowance_cost,
                carbon_price=carbon_price,
                discount_rate=discount_rate,
            )
            
            # If capacity was added, re-solve with the expanded fleet
            if len(build_log) > 0:
                LOGGER.info(f"Capacity expansion built {len(build_log)} new units for year {year}")
                result = solve(
                    load_by_region=load_mapping,
                    generators=[
                        GeneratorSpec(
                            name=str(getattr(row, "unique_id", getattr(row, "unit_id", ""))),
                            region=normalize_region(row.region) if row.region is not None and not pd.isna(row.region) else _DEFAULT_REGION,
                            fuel=str(row.fuel),
                            variable_cost=(
                                float(row.vom_per_mwh)
                                + float(row.hr_mmbtu_per_mwh) * float(row.fuel_price_per_mmbtu)
                                + float(getattr(row, "carbon_cost_per_mwh", 0.0))
                            ),
                            capacity=float(row.cap_mw) * float(row.availability) * HOURS_PER_YEAR,
                            emission_rate=float(getattr(row, "co2_short_ton_per_mwh", getattr(row, "ef_ton_per_mwh", 0.0))),
                            covered=bool(coverage_by_region.get(normalize_region(row.region), True)) if coverage_by_region else coverage_by_fuel.get(str(row.fuel), True),
                            penalty_cost=0.0,
                        )
                        for row in updated_units.itertuples(index=False)
                    ],
                    interfaces=interface_specs,
                    allowance_cost=allowance_cost,
                    carbon_price=carbon_price,
                    region_coverage=coverage_by_region,
                    year=year,
                    generation_standard=generation_standard,
                    interface_costs=None,
                    peak_load_by_region=peak_load_mapping,
                    emissions_cap_tons=emissions_cap_tons,
                )

    served = sum(result.gen_by_region_mwh.values())
    unserved = sum(result.unserved_energy_by_region.values())
    curtailment = sum(result.curtailment_by_region.values())
    print("SERVED", year, served, "UNSERVED", year, unserved, "CURTAILED", year, curtailment)
    # Load balance: Generation + Unserved - Curtailment = Demand
    # (Unserved adds to what's needed, Curtailment subtracts from what's available)
    assert abs(served + unserved - curtailment - demand_total_mwh) < 1e-6

    return result


__all__ = ["GeneratorSpec", "InterfaceSpec", "solve", "solve_from_frames"]
