"""Single-region deterministic dispatch implementation using a merit order."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, cast

try:  # pragma: no cover - exercised when pandas missing
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(Any, None)

from io_loader import Frames
from common.regions_schema import REGION_MAP
from engine.constants import HOURS_PER_YEAR
from src.common.iteration_status import IterationStatus

from .capacity_expansion import plan_capacity_expansion
from .interface import DispatchResult

LOGGER = logging.getLogger(__name__)
_DEFAULT_REGION: str = next(iter(REGION_MAP))
_DISPATCH_TOLERANCE: float = 1e-9
_ID_COLUMN = "unique_id"
_VOLL: float = 10000.0  # Value of lost load in $/MWh for scarcity pricing

_REQUIRED_COLUMNS = {
    "cap_mw",
    "availability",
    "hr_mmbtu_per_mwh",
    "vom_per_mwh",
    "fuel_price_per_mmbtu",
}

_EMISSION_COLUMNS: tuple[str, ...] = ("co2_short_ton_per_mwh", "ef_ton_per_mwh")


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before continuing."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for dispatch.lp_single; install it with `pip install pandas`."
        )


def _validate_units_df(units_df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the units data with numeric columns enforced."""

    _ensure_pandas()

    if not isinstance(units_df, pd.DataFrame):
        raise TypeError("units must be provided as a pandas DataFrame")

    missing = [column for column in _REQUIRED_COLUMNS if column not in units_df.columns]
    if missing:
        raise ValueError(f"units data is missing required columns: {missing}")

    if not any(column in units_df.columns for column in _EMISSION_COLUMNS):
        available = ", ".join(sorted(units_df.columns))
        raise ValueError(
            "units data is missing required emission factor column "
            "'co2_short_ton_per_mwh' or 'ef_ton_per_mwh'"
            f"; available columns: {available}"
        )

    cleaned = units_df.copy(deep=True)

    if _ID_COLUMN not in cleaned.columns:
        if "unit_id" not in cleaned.columns:
            raise ValueError("units data must include a 'unique_id' column")
        cleaned[_ID_COLUMN] = cleaned["unit_id"]

    id_series = cleaned[_ID_COLUMN]
    missing_mask = id_series.isna()
    if id_series.dtype == object:
        missing_mask |= id_series.astype(str).str.strip() == ""

    cleaned[_ID_COLUMN] = id_series.astype(str)
    if "unit_id" in cleaned.columns:
        cleaned["unit_id"] = cleaned["unit_id"].astype(str)

    missing_mask |= cleaned[_ID_COLUMN].str.strip() == ""
    if missing_mask.any():
        if "unit_id" in cleaned.columns:
            cleaned.loc[missing_mask, _ID_COLUMN] = cleaned.loc[missing_mask, "unit_id"]
        else:
            raise ValueError("unique_id values must be non-empty for dispatch")

    if cleaned[_ID_COLUMN].duplicated().any():
        raise ValueError("unique_id values must be unique for dispatch")

    numeric_cols = [
        "cap_mw",
        "availability",
        "hr_mmbtu_per_mwh",
        "vom_per_mwh",
        "fuel_price_per_mmbtu",
    ]

    for column in numeric_cols:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        if column == "availability":
            cleaned[column] = cleaned[column].fillna(1.0)
        if cleaned[column].isna().any():
            raise ValueError(f"column '{column}' must contain numeric values")

    co2_series = None
    ef_series = None
    if "co2_short_ton_per_mwh" in cleaned.columns:
        co2_series = pd.to_numeric(cleaned["co2_short_ton_per_mwh"], errors="coerce")
    if "ef_ton_per_mwh" in cleaned.columns:
        ef_series = pd.to_numeric(cleaned["ef_ton_per_mwh"], errors="coerce")

    if "carbon_cost_per_mwh" in cleaned.columns:
        cleaned["carbon_cost_per_mwh"] = pd.to_numeric(
            cleaned["carbon_cost_per_mwh"], errors="coerce"
        ).fillna(0.0)
    else:
        cleaned["carbon_cost_per_mwh"] = 0.0

    if ef_series is None:
        ef_series = pd.Series(0.0, index=cleaned.index)
    else:
        ef_series = ef_series.fillna(0.0)

    if co2_series is not None:
        emission_source = co2_series.fillna(ef_series)
    else:
        emission_source = ef_series

    cleaned["ef_ton_per_mwh"] = emission_source.astype(float)

    cleaned["availability"] = cleaned["availability"].clip(lower=0.0, upper=1.0)

    return cleaned


def _dispatch_merit_order(
    units_df: pd.DataFrame,
    load_mwh: float,
    allowance_cost: float,
    *,
    allowance_covered: bool = True,
    carbon_price: float = 0.0,
) -> dict:
    """Run the merit-order dispatch returning detailed information for testing."""

    _ensure_pandas()

    units = _validate_units_df(units_df)

    load = max(0.0, float(load_mwh))
    allowance = float(allowance_cost)
    price_component = float(carbon_price)

    units = units.assign(
        cap_available_mw=(units["cap_mw"] * units["availability"]).clip(lower=0.0)
    )
    units = units.assign(
        cap_mwh=(units["cap_available_mw"] * HOURS_PER_YEAR).clip(lower=0.0)
    )
    units = units.assign(
        marginal_cost=(
            units["vom_per_mwh"]
            + units["hr_mmbtu_per_mwh"] * units["fuel_price_per_mmbtu"]
            + units["carbon_cost_per_mwh"]
            + (units["ef_ton_per_mwh"] * allowance if allowance_covered else 0.0)
            + units["ef_ton_per_mwh"] * price_component
        )
    )

    ordered = units.sort_values(["marginal_cost", _ID_COLUMN]).set_index(_ID_COLUMN)

    generation = pd.Series(0.0, index=ordered.index, dtype=float)
    remaining = load
    price = 0.0

    for unit_key, row in ordered.iterrows():
        if remaining <= _DISPATCH_TOLERANCE:
            break

        capacity = float(row["cap_mwh"])
        if capacity <= _DISPATCH_TOLERANCE:
            continue

        dispatch = min(capacity, remaining)
        generation.at[unit_key] = dispatch
        remaining -= dispatch

        if dispatch > _DISPATCH_TOLERANCE:
            price = float(row["marginal_cost"])

    remaining = float(max(0.0, remaining))
    total_generation = float(generation.sum())
    if total_generation <= _DISPATCH_TOLERANCE:
        price = 0.0
    elif remaining > _DISPATCH_TOLERANCE:
        # When there's unserved load, set price to value of lost load
        # This signals scarcity to capacity expansion and reflects true marginal value
        price = _VOLL

    emissions = float((generation * ordered["ef_ton_per_mwh"]).sum())

    return {
        "generation": generation,
        "units": ordered,
        "price": float(price),
        "emissions_tons": emissions,
        "shortfall_mwh": remaining,
    }


def _aggregate_generation_by_fuel(generation: pd.Series, units: pd.DataFrame) -> Mapping[str, float]:
    """Aggregate dispatch by fuel label if available, falling back to unit IDs."""

    _ensure_pandas()

    tol_filtered = generation[generation > _DISPATCH_TOLERANCE]

    if tol_filtered.empty:
        return {}

    if "fuel" in units.columns:
        fuels = units.loc[tol_filtered.index, "fuel"]
        if fuels.isna().any():
            fallback = pd.Series(tol_filtered.index, index=tol_filtered.index)
            fuels = fuels.fillna(fallback)
        grouped = tol_filtered.groupby(fuels).sum()
    else:
        grouped = tol_filtered

    return {str(label): float(value) for label, value in grouped.items()}


def _aggregate_generation_by_region(
    generation: pd.Series, units: pd.DataFrame
) -> Mapping[str, float]:
    """Aggregate dispatch by region label if available."""

    _ensure_pandas()

    tol_filtered = generation[generation > _DISPATCH_TOLERANCE]
    if tol_filtered.empty:
        return {}

    if "region" in units.columns:
        regions = units.loc[tol_filtered.index, "region"]
        if regions.isna().any():
            fallback = pd.Series(_DEFAULT_REGION, index=tol_filtered.index)
            regions = regions.fillna(fallback)
        regions = regions.astype(str)
    else:
        regions = pd.Series(_DEFAULT_REGION, index=tol_filtered.index)

    grouped = tol_filtered.groupby(regions).sum()
    return {str(label): float(value) for label, value in grouped.items()}


def solve(
    year: int,
    allowance_cost: float,
    *,
    frames: Optional[Frames | Mapping[str, pd.DataFrame]] = None,
    carbon_price: float = 0.0,
    capacity_expansion: bool = False,
    discount_rate: float = 0.07,
    emissions_cap_tons: Optional[float] = None,
) -> DispatchResult:
    """Solve the single-region dispatch problem using the provided frame data."""

    _ensure_pandas()

    if frames is None:
        raise ValueError("frames providing demand and units must be supplied")

    frames_obj = Frames.coerce(frames)
    units = frames_obj.units()
    demand_frame = frames_obj.demand()
    if "region" not in demand_frame.columns:
        raise ValueError("demand frame must include a 'region' column for regional emissions")
    if "region" not in units.columns:
        raise ValueError("units frame must include a 'region' column for regional emissions")
    if units["region"].isna().all():
        raise ValueError("units frame must specify at least one region for emissions reporting")

    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "dispatch_single inputs year=%s demand_regions=%s unit_regions=%s",
            year,
            sorted(str(region) for region in demand_frame["region"].astype(str).unique()),
            sorted(str(region) for region in units["region"].astype(str).unique()),
        )

    demand = frames_obj.demand_for_year(year)
    peak_demand_by_region: Dict[str, float] = {}
    if frames_obj.has_frame("peak_demand"):
        try:
            peak_for_year = frames_obj.peak_demand_for_year(year)
        except KeyError as exc:
            raise ValueError(
                f"no peak demand observations available for year {year}"
            ) from exc
        peak_demand_by_region = {
            str(region): max(float(value), 0.0)
            for region, value in peak_for_year.items()
        }
    coverage_map = frames_obj.coverage_for_year(year)

    unit_regions = {str(region) for region in units["region"].fillna(_DEFAULT_REGION).unique()}

    coverage_flags = {region: bool(coverage_map.get(region, True)) for region in unit_regions}
    allowance_covered = True
    if coverage_flags:
        unique_flags = set(coverage_flags.values())
        if len(unique_flags) > 1:
            raise ValueError('single-region dispatch requires uniform coverage status')
        allowance_covered = unique_flags.pop()

    load_value = sum(demand.values())

    dispatch_summary = _dispatch_merit_order(
        units,
        float(load_value),
        allowance_cost,
        allowance_covered=allowance_covered,
        carbon_price=carbon_price,
    )

    build_log: list[dict[str, object]] = []
    expansion_status: IterationStatus | None = None

    if capacity_expansion:
        try:
            expansion_options = frames_obj.expansion_options()
        except AttributeError:
            expansion_options = pd.DataFrame()

        if not expansion_options.empty:
            def _dispatch_with(units_df: pd.DataFrame) -> dict:
                return _dispatch_merit_order(
                    units_df,
                    float(load_value),
                    allowance_cost,
                    allowance_covered=allowance_covered,
                    carbon_price=carbon_price,
                )

            units, dispatch_summary, build_log, expansion_status = plan_capacity_expansion(
                units,
                expansion_options,
                dispatch_summary,
                _dispatch_with,
                allowance_cost=allowance_cost,
                carbon_price=carbon_price,
                discount_rate=float(discount_rate),
            )

    dispatch = dispatch_summary

    generation = dispatch["generation"]
    unit_data = dispatch["units"]
    
    # Enforce emissions cap if specified
    if emissions_cap_tons is not None:
        total_emissions = float(dispatch.get("emissions_tons", 0.0))
        cap_value = float(emissions_cap_tons)
        if total_emissions > cap_value:
            diagnostics = {
                "year": year,
                "total_emissions_tons": total_emissions,
                "emissions_cap_tons": cap_value,
                "excess_emissions_tons": total_emissions - cap_value,
                "load_mwh": float(load_value),
                "allowance_cost": float(allowance_cost),
                "carbon_price": float(carbon_price),
            }
            LOGGER.error(
                "dispatch_single emissions cap violated: emissions=%.2f cap=%.2f excess=%.2f",
                total_emissions,
                cap_value,
                total_emissions - cap_value,
            )
            raise RuntimeError(
                f"Dispatch failed: emissions ({total_emissions:.2f} tons) exceed cap ({cap_value:.2f} tons). "
                f"Excess: {total_emissions - cap_value:.2f} tons. "
                f"The system cannot meet demand without violating the emissions cap. "
                f"Diagnostics: {diagnostics}"
            )
    
    gen_by_fuel = _aggregate_generation_by_fuel(generation, unit_data)
    gen_by_region = _aggregate_generation_by_region(generation, unit_data)
    generation_detail_by_region: Dict[str, Dict[str, float]] = {}

    region_prices = {_DEFAULT_REGION: float(dispatch["price"])}
    generation_by_unit = {
        str(unit_id): float(value) for unit_id, value in generation.items()
    }
    total_cost = float((generation * unit_data["marginal_cost"]).sum())
    constraint_duals = {"load_balance": {_DEFAULT_REGION: float(dispatch["price"])}}

    # emissions by region (codex branch)
    emissions_series = generation * unit_data["ef_ton_per_mwh"]
    emissions_by_region_series = emissions_series.groupby(unit_data["region"]).sum()
    emissions_by_region = {
        str(region): float(value) for region, value in emissions_by_region_series.items()
    }
    if not emissions_by_region:
        emissions_by_region = {_DEFAULT_REGION: 0.0}

    demand_regions = {str(region) for region in demand.keys()}
    for region in demand_regions:
        emissions_by_region.setdefault(region, 0.0)

    generation_by_unit = {str(unit): float(output) for unit, output in generation.items()}
    capacity_mwh_by_unit = {
        str(unit): float(unit_data.loc[unit, "cap_mwh"]) for unit in unit_data.index
    }
    capacity_mw_by_unit = {
        str(unit): float(unit_data.loc[unit, "cap_available_mw"]) for unit in unit_data.index
    }

    capacity_mwh_by_fuel: Dict[str, float] = {}
    capacity_mw_by_fuel: Dict[str, float] = {}
    emissions_by_fuel: Dict[str, float] = {}
    variable_cost_by_fuel: Dict[str, float] = {}
    allowance_cost_by_fuel: Dict[str, float] = {}
    carbon_price_cost_by_fuel: Dict[str, float] = {}
    total_cost_by_fuel: Dict[str, float] = {}
    capacity_region_mwh: Dict[str, Dict[str, float]] = {}
    capacity_region_mw: Dict[str, Dict[str, float]] = {}
    variable_cost_region: Dict[str, Dict[str, float]] = {}
    allowance_cost_region: Dict[str, Dict[str, float]] = {}
    carbon_price_cost_region: Dict[str, Dict[str, float]] = {}
    total_cost_region: Dict[str, Dict[str, float]] = {}

    allowance_component = float(allowance_cost) if allowance_covered else 0.0
    carbon_component = float(carbon_price)

    for unit in unit_data.index:
        row = unit_data.loc[unit]
        fuel = str(row.get("fuel", unit))
        region_val = row.get("region", _DEFAULT_REGION)
        if pd.isna(region_val):
            region_str = _DEFAULT_REGION
        else:
            region_str = str(region_val)
        capacity_mwh = float(row["cap_mwh"])
        capacity_mw = float(row["cap_available_mw"])
        capacity_mwh_by_fuel[fuel] = capacity_mwh_by_fuel.get(fuel, 0.0) + capacity_mwh
        capacity_mw_by_fuel[fuel] = capacity_mw_by_fuel.get(fuel, 0.0) + capacity_mw
        region_capacity_mwh = capacity_region_mwh.setdefault(region_str, {})
        region_capacity_mw = capacity_region_mw.setdefault(region_str, {})
        region_capacity_mwh[fuel] = region_capacity_mwh.get(fuel, 0.0) + capacity_mwh
        region_capacity_mw[fuel] = region_capacity_mw.get(fuel, 0.0) + capacity_mw

        dispatched = float(generation.get(unit, 0.0))
        emission_rate = float(row["ef_ton_per_mwh"])
        emissions_value = emission_rate * dispatched
        emissions_by_fuel[fuel] = emissions_by_fuel.get(fuel, 0.0) + emissions_value
        region_generation = generation_detail_by_region.setdefault(region_str, {})
        region_generation[fuel] = region_generation.get(fuel, 0.0) + dispatched

        variable_rate = (
            float(row["vom_per_mwh"])
            + float(row["hr_mmbtu_per_mwh"]) * float(row["fuel_price_per_mmbtu"])
            + float(row.get("carbon_cost_per_mwh", 0.0))
        )
        allowance_rate = emission_rate * allowance_component
        carbon_price_rate = emission_rate * carbon_component
        total_rate = variable_rate + allowance_rate + carbon_price_rate

        variable_cost_by_fuel[fuel] = (
            variable_cost_by_fuel.get(fuel, 0.0) + variable_rate * dispatched
        )
        allowance_cost_by_fuel[fuel] = (
            allowance_cost_by_fuel.get(fuel, 0.0) + allowance_rate * dispatched
        )
        carbon_price_cost_by_fuel[fuel] = (
            carbon_price_cost_by_fuel.get(fuel, 0.0) + carbon_price_rate * dispatched
        )
        total_cost_by_fuel[fuel] = total_cost_by_fuel.get(fuel, 0.0) + total_rate * dispatched
        region_variable_cost = variable_cost_region.setdefault(region_str, {})
        region_allowance_cost = allowance_cost_region.setdefault(region_str, {})
        region_carbon_cost = carbon_price_cost_region.setdefault(region_str, {})
        region_total_cost = total_cost_region.setdefault(region_str, {})
        region_variable_cost[fuel] = (
            region_variable_cost.get(fuel, 0.0) + variable_rate * dispatched
        )
        region_allowance_cost[fuel] = (
            region_allowance_cost.get(fuel, 0.0) + allowance_rate * dispatched
        )
        region_carbon_cost[fuel] = (
            region_carbon_cost.get(fuel, 0.0) + carbon_price_rate * dispatched
        )
        region_total_cost[fuel] = (
            region_total_cost.get(fuel, 0.0) + total_rate * dispatched
        )

    costs_by_region: Dict[str, Dict[str, Dict[str, float]]] = {}
    for region, cost_map in total_cost_region.items():
        fuels = set(cost_map)
        fuels |= set(variable_cost_region.get(region, {}))
        fuels |= set(allowance_cost_region.get(region, {}))
        fuels |= set(carbon_price_cost_region.get(region, {}))
        if not fuels:
            continue
        region_costs: Dict[str, Dict[str, float]] = {}
        for fuel in fuels:
            region_costs[fuel] = {
                "variable_cost": float(variable_cost_region.get(region, {}).get(fuel, 0.0)),
                "allowance_cost": float(allowance_cost_region.get(region, {}).get(fuel, 0.0)),
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

    available_capacity_mw_by_region: Dict[str, float] = {
        region: float(sum(values.values()))
        for region, values in (
            {key: {fuel: float(value) for fuel, value in mapping.items()}
             for key, mapping in capacity_region_mw.items()}
        ).items()
    }

    demand_by_region = {str(region): float(value) for region, value in demand.items()}

    for region_key in list(demand_by_region):
        generation_detail_by_region.setdefault(region_key, {})
        capacity_by_region.setdefault(region_key, {})
        costs_by_region.setdefault(region_key, {})

    for region in demand_by_region:
        peak_demand_by_region.setdefault(region, 0.0)

    unserved_capacity_by_region: Dict[str, float] = {}
    for region, requirement in peak_demand_by_region.items():
        available = available_capacity_mw_by_region.get(region, 0.0)
        shortage = max(float(requirement) - float(available), 0.0)
        unserved_capacity_by_region[region] = shortage

    for region in available_capacity_mw_by_region:
        unserved_capacity_by_region.setdefault(region, 0.0)

    unserved_capacity_total = sum(unserved_capacity_by_region.values())
    unserved_capacity_penalty = 0.0

    # coverage and imports/exports (main branch)
    total_generation = float(generation.sum())
    generation_by_coverage = {"covered": 0.0, "non_covered": 0.0}
    coverage_key = "covered" if allowance_covered else "non_covered"
    generation_by_coverage[coverage_key] = total_generation

    imports_to_covered = 0.0
    exports_from_covered = 0.0
    region_coverage: Dict[str, bool] = {}
    for region, load in demand.items():
        region_str = str(region)
        covered = bool(coverage_map.get(region_str, allowance_covered))
        region_coverage[region_str] = covered
        generation_region = gen_by_region.get(region_str, 0.0)
        net_import = load - generation_region
        if covered:
            if net_import > _DISPATCH_TOLERANCE:
                imports_to_covered += net_import
            elif net_import < -_DISPATCH_TOLERANCE:
                exports_from_covered += -net_import

    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "dispatch_single emissions_by_region year=%s %s",
            year,
            emissions_by_region,
        )

    generation_total_by_region: Dict[str, float] = {}
    for region, fuels in generation_detail_by_region.items():
        total = 0.0
        for value in fuels.values():
            try:
                total += float(value)
            except (TypeError, ValueError):
                continue
        generation_total_by_region[region] = total

    effective_price = max(float(allowance_component), float(carbon_component))


    return DispatchResult(
        gen_by_fuel=gen_by_fuel,
        region_prices=region_prices,
        emissions_tons=float(dispatch["emissions_tons"]),
        emissions_by_region=emissions_by_region,
        flows={},  # no transmission flows tracked in this solver path
        emissions_by_fuel=emissions_by_fuel,
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
            region: float(value) for region, value in peak_demand_by_region.items()
        },
        generation_by_region=generation_total_by_region,
        generation_detail_by_region=generation_detail_by_region,
        generation_by_coverage=generation_by_coverage,
        capacity_by_region=capacity_by_region,
        costs_by_region=costs_by_region,
        imports_to_covered=imports_to_covered,
        exports_from_covered=exports_from_covered,
        region_coverage=region_coverage,
        constraint_duals=constraint_duals,
        total_cost=total_cost,
        capacity_builds=build_log,
        allowance_cost=float(allowance_component),
        carbon_price=float(carbon_component),
        effective_carbon_price=effective_price,
        iteration_status=expansion_status,
        unserved_capacity_by_region={
            region: float(value) for region, value in unserved_capacity_by_region.items()
        },
        unserved_capacity_total=float(unserved_capacity_total),
        unserved_capacity_penalty=float(unserved_capacity_penalty),
    )



__all__ = [
    "DispatchResult",
    "HOURS_PER_YEAR",
    "_aggregate_generation_by_fuel",
    "_aggregate_generation_by_region",
    "_dispatch_merit_order",
    "_validate_units_df",
    "solve",
]

