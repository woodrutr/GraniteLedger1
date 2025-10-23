"""Interfaces for the simplified dispatch engine."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

from src.common.iteration_status import IterationStatus


@dataclass(frozen=True)
class DispatchResult:
    """Container for the outputs of a dispatch run.

    Attributes
    ----------
    gen_by_fuel:
        Mapping of fuel name to the dispatched generation in megawatt-hours.
    region_prices:
        Mapping of model region identifiers to their marginal energy prices in
        dollars per megawatt-hour.
    emissions_tons:
        Total carbon dioxide emissions from the dispatch solution measured in tons.
    emissions_by_region:
        Mapping of model region identifiers to their contribution to total
        emissions measured in tons.
    flows:
        Net energy transfers between regions measured in megawatt-hours. Keys
        are tuples ``(region_a, region_b)`` where positive values indicate a
        flow from ``region_a`` to ``region_b``.
    emissions_by_fuel:
        Mapping of fuel labels to their contribution to total emissions in tons.
    capacity_mwh_by_fuel:
        Available annual energy capability by fuel measured in megawatt-hours.
    capacity_mw_by_fuel:
        Available instantaneous capacity by fuel measured in megawatts.
    generation_by_unit:
        Dispatch by individual generating unit in megawatt-hours.
    capacity_mwh_by_unit:
        Available annual energy capability by unit in megawatt-hours.
    capacity_mw_by_unit:
        Available instantaneous capacity by unit in megawatts.
    variable_cost_by_fuel:
        Variable (fuel and operations) cost incurred by fuel in dollars.
    allowance_cost_by_fuel:
        Allowance compliance cost incurred by fuel in dollars.
    carbon_price_cost_by_fuel:
        Exogenous carbon price cost incurred by fuel in dollars.
    total_cost_by_fuel:
        Sum of variable and carbon-related costs by fuel in dollars.
    demand_by_region:
        Mapping of regions to the demand served within the region measured in
        megawatt-hours.
    peak_demand_by_region:
        Mapping of regions to the required peak demand that must be satisfied
        in megawatts.
    generation_by_region:
        Mapping of regions to the total dispatched generation in
        megawatt-hours.
    generation_detail_by_region:
        Mapping of regions to dictionaries keyed by fuel / technology with the
        dispatched generation in megawatt-hours for that fuel.
    generation_by_coverage:
        Aggregated generation grouped by coverage status with keys ``'covered'``
        and ``'non_covered'``.
    capacity_by_region:
        Mapping of regions to dictionaries keyed by fuel / technology.  Each
        inner mapping provides ``'capacity_mwh'`` and ``'capacity_mw'`` entries
        representing the available annual energy capability and instantaneous
        capacity respectively.
    costs_by_region:
        Mapping of regions to dictionaries keyed by fuel / technology.  Each
        inner dictionary provides ``'variable_cost'``, ``'allowance_cost'``,
        ``'carbon_price_cost'`` and ``'total_cost'`` entries.
    imports_to_covered:
        Total net imports flowing into covered regions in megawatt-hours.
    exports_from_covered:
        Total net exports flowing out of covered regions in megawatt-hours.
    region_coverage:
        Mapping of regions to the boolean coverage flag used in the solution.
    unserved_capacity_by_region:
        Mapping of regions to the megawatt shortfall relative to the peak demand
        requirement.
    unserved_capacity_total:
        Aggregate peak demand shortfall across all regions in megawatts.
    unserved_capacity_penalty:
        Penalty cost applied per megawatt of peak demand shortfall.
    """

    gen_by_fuel: Dict[str, float]
    region_prices: Dict[str, float]
    emissions_tons: float
    emissions_by_region: Dict[str, float] = field(default_factory=dict)
    flows: Dict[Tuple[str, str], float] = field(default_factory=dict)
    emissions_by_fuel: Dict[str, float] = field(default_factory=dict)
    capacity_mwh_by_fuel: Dict[str, float] = field(default_factory=dict)
    capacity_mw_by_fuel: Dict[str, float] = field(default_factory=dict)
    generation_by_unit: Dict[str, float] = field(default_factory=dict)
    capacity_mwh_by_unit: Dict[str, float] = field(default_factory=dict)
    capacity_mw_by_unit: Dict[str, float] = field(default_factory=dict)
    variable_cost_by_fuel: Dict[str, float] = field(default_factory=dict)
    allowance_cost_by_fuel: Dict[str, float] = field(default_factory=dict)
    carbon_price_cost_by_fuel: Dict[str, float] = field(default_factory=dict)
    total_cost_by_fuel: Dict[str, float] = field(default_factory=dict)
    demand_by_region: Dict[str, float] = field(default_factory=dict)
    peak_demand_by_region: Dict[str, float] = field(default_factory=dict)
    generation_by_region: Dict[str, float] = field(default_factory=dict)
    generation_detail_by_region: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    generation_by_coverage: Dict[str, float] = field(default_factory=dict)
    capacity_by_region: Dict[str, Dict[str, Dict[str, float]]] = field(
        default_factory=dict
    )
    costs_by_region: Dict[str, Dict[str, Dict[str, float]]] = field(
        default_factory=dict
    )
    imports_to_covered: float = 0.0
    exports_from_covered: float = 0.0
    region_coverage: Dict[str, bool] = field(default_factory=dict)
    constraint_duals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    total_cost: float = 0.0
    capacity_builds: List[Dict[str, object]] = field(default_factory=list)
    allowance_cost: float = 0.0
    carbon_price: float = 0.0
    effective_carbon_price: float = 0.0
    carbon_price_schedule: Dict[int | None, float] = field(default_factory=dict)
    wheeling_cost_by_interface: Dict[Tuple[str, str], float] = field(default_factory=dict)
    unserved_energy_by_region: Dict[str, float] = field(default_factory=dict)
    unserved_energy_total: float = 0.0
    unserved_energy_penalty: float = 0.0
    curtailment_by_region: Dict[str, float] = field(default_factory=dict)
    curtailment_total: float = 0.0
    unserved_capacity_by_region: Dict[str, float] = field(default_factory=dict)
    unserved_capacity_total: float = 0.0
    unserved_capacity_penalty: float = 0.0
    iteration_status: IterationStatus | None = None


    @property
    def total_generation(self) -> float:
        """Return the total dispatched generation in megawatt-hours."""
        return float(sum(self.gen_by_fuel.values()))

    @property
    def covered_generation(self) -> float:
        """Return total generation attributed to covered regions."""
        return float(self.generation_by_coverage.get("covered", 0.0))

    @property
    def non_covered_generation(self) -> float:
        """Return total generation attributed to non-covered regions."""
        return float(self.generation_by_coverage.get("non_covered", 0.0))

    @property
    def gen_by_region_mwh(self) -> Dict[str, float]:
        """Return total generation by region in megawatt-hours."""

        return self.generation_total_by_region

    @property
    def generation_total_by_region(self) -> Dict[str, float]:
        """Return total generation by region summed across technologies."""

        totals: Dict[str, float] = {}

        if self.generation_by_region:
            for region, value in self.generation_by_region.items():
                total = 0.0
                if isinstance(value, Mapping):
                    for nested in value.values():
                        try:
                            total += float(nested)
                        except (TypeError, ValueError):
                            continue
                else:
                    try:
                        total = float(value)
                    except (TypeError, ValueError):
                        total = 0.0
                totals[str(region)] = total

        if self.generation_detail_by_region:
            for region, fuels in self.generation_detail_by_region.items():
                total = 0.0
                if isinstance(fuels, Mapping):
                    for value in fuels.values():
                        try:
                            total += float(value)
                        except (TypeError, ValueError):
                            continue
                else:
                    try:
                        total = float(fuels)
                    except (TypeError, ValueError):
                        total = 0.0
                totals[str(region)] = total

        return totals

    def leakage_percent(self, baseline: "DispatchResult") -> float:
        """Return leakage relative to ``baseline`` as a percentage.

        Leakage is defined as the ratio of the change in non-covered
        generation to the change in total generation between this result and
        ``baseline``. A positive value indicates that uncovered generation grew
        faster than total generation, signalling leakage.
        """
        delta_total = self.total_generation - baseline.total_generation
        if abs(delta_total) <= 1e-9:
            return 0.0

        delta_uncovered = (
            self.non_covered_generation - baseline.non_covered_generation
        )
        return 100.0 * delta_uncovered / delta_total

    @property
    def has_unserved_energy(self) -> bool:
        """Return ``True`` when the dispatch relied on unserved energy."""

        return any(value > 1e-9 for value in self.unserved_energy_by_region.values())

    @property
    def has_unserved_capacity(self) -> bool:
        """Return ``True`` when the dispatch incurred peak capacity shortfall."""

        return any(value > 1e-9 for value in self.unserved_capacity_by_region.values())
