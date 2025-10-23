import math

import pytest
import pandas as pd

from dispatch.interface import DispatchResult
from dispatch.lp_network import InterfaceSpec, GeneratorSpec, solve, solve_from_frames
from dispatch.lp_single import HOURS_PER_YEAR
from io_loader import Frames
from policy.generation_standard import GenerationStandardPolicy, TechnologyStandard
from regions.registry import REGIONS

REGION_IDS = list(REGIONS)
REGION_TEST = REGION_IDS[0]
REGION_SOLO = REGION_IDS[1]
REGION_NORTH = REGION_IDS[2]
REGION_SOUTH = REGION_IDS[3]
REGION_COVERED = REGION_IDS[4]
REGION_EXTERNAL = REGION_IDS[5]
REGION_ALPHA = REGION_IDS[6]


def _two_unit_frames(load_mwh: float) -> Frames:
    demand = pd.DataFrame(
        [{"year": 2030, "region": REGION_TEST, "demand_mwh": float(load_mwh)}]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "baseload",
                "unique_id": "baseload",
                "region": REGION_TEST,
                "fuel": "baseload_fuel",
                "cap_mw": 50.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 8.0,
                "vom_per_mwh": 5.0,
                "fuel_price_per_mmbtu": 1.5,
                "ef_ton_per_mwh": 0.8,
            },
            {
                "unit_id": "peaker",
                "unique_id": "peaker",
                "region": REGION_TEST,
                "fuel": "peaker_fuel",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 7.0,
                "vom_per_mwh": 3.0,
                "fuel_price_per_mmbtu": 3.0,
                "ef_ton_per_mwh": 0.49,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {
                "fuel": "baseload_fuel",
                "covered": True,
                "co2_short_ton_per_mwh": 0.1,
            },
            {
                "fuel": "peaker_fuel",
                "covered": True,
                "co2_short_ton_per_mwh": 0.07,
            },
        ]
    )
    coverage = pd.DataFrame(
        [{"region": REGION_TEST, "covered": True}]
    )

    return Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "coverage": coverage,
        }
    )


def _single_unit_frames(load_mwh: float) -> Frames:
    demand = pd.DataFrame(
        [{"year": 2030, "region": REGION_SOLO, "demand_mwh": float(load_mwh)}]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "solo_unit",
                "unique_id": "solo_unit",
                "region": REGION_SOLO,
                "fuel": "solo_fuel",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 9.0,
                "vom_per_mwh": 2.0,
                "fuel_price_per_mmbtu": 2.5,
                "ef_ton_per_mwh": 0.9,
            }
        ]
    )
    fuels = pd.DataFrame(
        [
            {
                "fuel": "solo_fuel",
                "covered": True,
                "co2_short_ton_per_mwh": 0.1,
            }
        ]
    )
    coverage = pd.DataFrame(
        [{"region": REGION_SOLO, "covered": True}]
    )

    return Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "coverage": coverage,
        }
    )


def _registry_region_frames() -> Frames:
    demand = pd.DataFrame(
        [
            {"year": 2030, "region": "ISO-NE_CT", "demand_mwh": 15.0 * HOURS_PER_YEAR},
            {"year": 2030, "region": "NYISO_J", "demand_mwh": 20.0 * HOURS_PER_YEAR},
        ]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "iso_ne_unit",
                "unique_id": "iso_ne_unit",
                "region": "ISO-NE_CT",
                "fuel": "iso_fuel",
                "cap_mw": 40.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 8.0,
                "vom_per_mwh": 5.0,
                "fuel_price_per_mmbtu": 1.5,
                "ef_ton_per_mwh": 0.5,
            },
            {
                "unit_id": "nyiso_unit",
                "unique_id": "nyiso_unit",
                "region": "NYISO_J",
                "fuel": "nyiso_fuel",
                "cap_mw": 60.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 7.5,
                "vom_per_mwh": 6.0,
                "fuel_price_per_mmbtu": 1.6,
                "ef_ton_per_mwh": 0.45,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "iso_fuel", "covered": True, "co2_short_ton_per_mwh": 0.1},
            {"fuel": "nyiso_fuel", "covered": True, "co2_short_ton_per_mwh": 0.08},
        ]
    )
    coverage = pd.DataFrame(
        [
            {"region": "ISO-NE_CT", "covered": True},
            {"region": "NYISO_J", "covered": True},
        ]
    )

    return Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "coverage": coverage,
        }
    )


def _multi_region_emissions_frames() -> Frames:
    demand = pd.DataFrame(
        [
            {
                "year": 2030,
                "region": REGION_NORTH,
                "demand_mwh": 30.0 * HOURS_PER_YEAR,
            },
            {
                "year": 2030,
                "region": REGION_SOUTH,
                "demand_mwh": 20.0 * HOURS_PER_YEAR,
            },
        ]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "north_unit",
                "unique_id": "north_unit",
                "region": REGION_NORTH,
                "fuel": "north_fuel",
                "cap_mw": 60.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 7.0,
                "vom_per_mwh": 12.0,
                "fuel_price_per_mmbtu": 1.5,
                "ef_ton_per_mwh": 0.6,
            },
            {
                "unit_id": "south_unit",
                "unique_id": "south_unit",
                "region": REGION_SOUTH,
                "fuel": "south_fuel",
                "cap_mw": 40.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 8.0,
                "vom_per_mwh": 14.0,
                "fuel_price_per_mmbtu": 1.8,
                "ef_ton_per_mwh": 0.3,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "north_fuel", "covered": True},
            {"fuel": "south_fuel", "covered": True},
        ]
    )
    coverage = pd.DataFrame(
        [
            {"region": REGION_NORTH, "covered": True},
            {"region": REGION_SOUTH, "covered": True},
        ]
    )

    return Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "coverage": coverage,
        }
    )


def test_congestion_leads_to_price_separation() -> None:
    demand = pd.DataFrame(
        [
            {"year": 2030, "region": REGION_NORTH, "demand_mwh": 40.0 * HOURS_PER_YEAR},
            {"year": 2030, "region": REGION_SOUTH, "demand_mwh": 60.0 * HOURS_PER_YEAR},
        ]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "north_low_cost",
                "unique_id": "north_low_cost",
                "region": REGION_NORTH,
                "fuel": "north_supply",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 20.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
            {
                "unit_id": "south_high_cost",
                "unique_id": "south_high_cost",
                "region": REGION_SOUTH,
                "fuel": "south_supply",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 50.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "north_supply", "covered": True},
            {"fuel": "south_supply", "covered": True},
        ]
    )
    coverage = pd.DataFrame(
        [
            {"region": REGION_NORTH, "covered": True},
            {"region": REGION_SOUTH, "covered": True},
        ]
    )
    transmission = pd.DataFrame(
        [
            {
                "from_region": REGION_NORTH,
                "to_region": REGION_SOUTH,
                "limit_mw": 15.0,
            }
        ]
    )

    frames = Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "transmission": transmission,
            "coverage": coverage,
        }
    )

    result = solve_from_frames(frames, 2030, allowance_cost=0.0)

    assert result.region_prices[REGION_NORTH] < result.region_prices[REGION_SOUTH]
    assert result.region_prices[REGION_NORTH] == pytest.approx(20.0, rel=1e-4)
    assert result.region_prices[REGION_SOUTH] == pytest.approx(50.0, rel=1e-4)

    assert result.gen_by_fuel["north_supply"] == pytest.approx(55.0 * HOURS_PER_YEAR)
    assert result.gen_by_fuel["south_supply"] == pytest.approx(45.0 * HOURS_PER_YEAR)
    assert math.isclose(result.emissions_tons, 0.0)
    assert (REGION_NORTH, REGION_SOUTH) in result.flows
    assert result.flows[(REGION_NORTH, REGION_SOUTH)] == pytest.approx(15.0 * HOURS_PER_YEAR)
    assert sum(result.emissions_by_region.values()) == pytest.approx(result.emissions_tons)


def test_solve_from_frames_reports_emissions_by_region() -> None:
    frames = _multi_region_emissions_frames()
    result = solve_from_frames(frames, 2030, allowance_cost=10.0, carbon_price=0.0)

    expected_north = 30.0 * HOURS_PER_YEAR * 0.6
    expected_south = 20.0 * HOURS_PER_YEAR * 0.3

    assert set(result.emissions_by_region) == {REGION_NORTH, REGION_SOUTH}
    assert result.emissions_by_region[REGION_NORTH] == pytest.approx(
        expected_north, rel=1e-6
    )
    assert result.emissions_by_region[REGION_SOUTH] == pytest.approx(
        expected_south, rel=1e-6
    )
    assert sum(result.emissions_by_region.values()) == pytest.approx(result.emissions_tons)


def test_solve_from_frames_reports_registry_regions() -> None:
    frames = _registry_region_frames()
    result = solve_from_frames(frames, 2030, allowance_cost=0.0)

    assert set(result.emissions_by_region) == {"ISO-NE_CT", "NYISO_J"}
    assert result.emissions_by_region["ISO-NE_CT"] > 0.0
    assert result.emissions_by_region["NYISO_J"] > 0.0


def test_imports_increase_with_carbon_price() -> None:
    demand = pd.DataFrame(
        [
            {"year": 2030, "region": REGION_COVERED, "demand_mwh": 100.0 * HOURS_PER_YEAR},
            {"year": 2030, "region": REGION_EXTERNAL, "demand_mwh": 0.0},
        ]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "covered_coal",
                "unique_id": "covered_coal",
                "region": REGION_COVERED,
                "fuel": "covered_supply",
                "cap_mw": 150.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 25.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.5,
            },
            {
                "unit_id": "external_gas",
                "unique_id": "external_gas",
                "region": REGION_EXTERNAL,
                "fuel": "external_supply",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 30.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "covered_supply", "covered": True},
            {"fuel": "external_supply", "covered": False},
        ]
    )
    coverage = pd.DataFrame(
        [
            {"region": REGION_COVERED, "covered": True},
            {"region": REGION_EXTERNAL, "covered": False},
        ]
    )
    transmission = pd.DataFrame(
        [
            {
                "from_region": REGION_COVERED,
                "to_region": REGION_EXTERNAL,
                "limit_mw": 200.0,
            }
        ]
    )

    frames = Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "transmission": transmission,
            "coverage": coverage,
        }
    )

    low_price = solve_from_frames(frames, 2030, allowance_cost=0.0, carbon_price=0.0)
    high_price = solve_from_frames(
        frames, 2030, allowance_cost=0.0, carbon_price=40.0
    )

    assert low_price.imports_to_covered == pytest.approx(0.0, abs=1e-6)
    assert high_price.imports_to_covered > low_price.imports_to_covered
    assert high_price.exports_from_covered == pytest.approx(0.0, abs=1e-6)


def test_contracted_flow_enforced_and_affects_peak() -> None:
    load = {
        REGION_ALPHA: 80.0 * HOURS_PER_YEAR,
        REGION_NORTH: 60.0 * HOURS_PER_YEAR,
    }
    generators = [
        GeneratorSpec(
            name="alpha_supply",
            region=REGION_ALPHA,
            fuel="alpha_fuel",
            variable_cost=10.0,
            capacity=140.0 * HOURS_PER_YEAR,
            emission_rate=0.1,
        ),
        GeneratorSpec(
            name="north_supply",
            region=REGION_NORTH,
            fuel="north_fuel",
            variable_cost=35.0,
            capacity=120.0 * HOURS_PER_YEAR,
            emission_rate=0.1,
        ),
    ]
    contract_mw = 15.0
    interfaces = [
        InterfaceSpec(
            from_region=REGION_ALPHA,
            to_region=REGION_NORTH,
            capacity_mw=100.0,
            reverse_capacity_mw=100.0,
            efficiency=0.98,
            added_cost_per_mwh=0.0,
            contracted_flow_mw_forward=contract_mw,
            contracted_flow_mw_reverse=0.0,
        )
    ]

    result = solve(
        load_by_region=load,
        generators=generators,
        interfaces=interfaces,
        allowance_cost=0.0,
    )

    delivered_flow = result.flows[(REGION_ALPHA, REGION_NORTH)] / HOURS_PER_YEAR
    assert delivered_flow >= contract_mw - 1e-6
    # Contract should register as a persistent requirement for the exporting region
    assert result.peak_demand_by_region[REGION_ALPHA] == pytest.approx(
        contract_mw, rel=1e-6
    )
    # Importing region requirement reduced by contracted delivery (floor at zero)
    assert result.peak_demand_by_region[REGION_NORTH] == pytest.approx(0.0, abs=1e-6)


def test_region_coverage_overrides_fuel_flags() -> None:
    demand = pd.DataFrame(
        [
            {"year": 2035, "region": REGION_COVERED, "demand_mwh": 80.0 * HOURS_PER_YEAR},
            {"year": 2035, "region": REGION_EXTERNAL, "demand_mwh": 0.0},
        ]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "covered_clean",
                "unique_id": "covered_clean",
                "region": REGION_COVERED,
                "fuel": "clean",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 25.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
            {
                "unit_id": "external_coal",
                "unique_id": "external_coal",
                "region": REGION_EXTERNAL,
                "fuel": "coal",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 15.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 1.0,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "clean", "covered": True},
            {"fuel": "coal", "covered": True},
        ]
    )
    coverage = pd.DataFrame(
        [
            {"region": REGION_COVERED, "covered": True},
            {"region": REGION_EXTERNAL, "covered": False},
        ]
    )
    transmission = pd.DataFrame(
        [
            {
                "from_region": REGION_EXTERNAL,
                "to_region": REGION_COVERED,
                "limit_mw": 500.0,
            }
        ]
    )

    frames = Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "transmission": transmission,
            "coverage": coverage,
        }
    )

    result = solve_from_frames(frames, 2035, allowance_cost=50.0)

    assert result.region_prices[REGION_COVERED] == pytest.approx(15.0, rel=1e-4)
    assert result.generation_total_by_region[REGION_EXTERNAL] > 0.0
    assert result.generation_by_coverage["non_covered"] > 0.0


def test_dispatch_duals_and_costs_align() -> None:
    demand = pd.DataFrame(
        [{"year": 2032, "region": REGION_TEST, "demand_mwh": 60.0 * HOURS_PER_YEAR}]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "core_cheap",
                "unique_id": "core_cheap",
                "region": REGION_TEST,
                "fuel": "cheap",
                "cap_mw": 40.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 30.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
            {
                "unit_id": "core_expensive",
                "unique_id": "core_expensive",
                "region": REGION_TEST,
                "fuel": "expensive",
                "cap_mw": 40.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 50.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "cheap", "covered": True},
            {"fuel": "expensive", "covered": True},
        ]
    )

    frames = Frames({"demand": demand, "units": units, "fuels": fuels})

    result = solve_from_frames(frames, 2032, allowance_cost=0.0)

    cheap_dispatch = result.generation_by_unit["core_cheap"]
    expensive_dispatch = result.generation_by_unit["core_expensive"]

    expected_cheap = 40.0 * HOURS_PER_YEAR
    expected_expensive = 20.0 * HOURS_PER_YEAR
    assert cheap_dispatch == pytest.approx(expected_cheap)
    assert expensive_dispatch == pytest.approx(expected_expensive)

    expected_cost = expected_cheap * 30.0 + expected_expensive * 50.0
    assert result.total_cost == pytest.approx(expected_cost)

    price = result.region_prices[REGION_TEST]
    assert price == pytest.approx(50.0)
    assert result.constraint_duals["load_balance"][REGION_TEST] == pytest.approx(price)

    assert 30.0 <= price + 1e-6
    assert 50.0 == pytest.approx(price)


def test_generation_standard_dual_reported() -> None:
    demand = pd.DataFrame(
        [{"year": 2030, "region": REGION_ALPHA, "demand_mwh": 100.0 * HOURS_PER_YEAR}]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "alpha_wind",
                "unique_id": "alpha_wind",
                "region": REGION_ALPHA,
                "fuel": "wind",
                "cap_mw": 80.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 20.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
            {
                "unit_id": "alpha_gas",
                "unique_id": "alpha_gas",
                "region": REGION_ALPHA,
                "fuel": "gas",
                "cap_mw": 120.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 30.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.1,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "wind", "covered": True},
            {"fuel": "gas", "covered": True},
        ]
    )

    frames = Frames({"demand": demand, "units": units, "fuels": fuels})

    share_df = pd.DataFrame({REGION_ALPHA: [0.5]}, index=pd.Index([2030], name="year"))
    standard = TechnologyStandard(
        technology="wind", generation_table=share_df, enabled_regions={REGION_ALPHA}
    )
    policy = GenerationStandardPolicy([standard])

    result = solve_from_frames(
        frames, 2030, allowance_cost=0.0, generation_standard=policy
    )

    duals = result.constraint_duals.get("generation_standard")
    assert duals is not None
    key = f"{REGION_ALPHA}:wind"
    assert key in duals
    assert isinstance(duals[key], float)

def test_leakage_percentage_helper() -> None:
    baseline = DispatchResult(
        gen_by_fuel={"coal": 60.0},
        region_prices={"region": 25.0},
        emissions_tons=0.0,
        emissions_by_region={"region": 0.0},
        flows={},
        generation_by_region={"region": {"coal": 60.0}},
        generation_by_coverage={"covered": 40.0, "non_covered": 20.0},
    )

    scenario = DispatchResult(
        gen_by_fuel={"coal": 50.0, "gas": 30.0},
        region_prices={"region": 30.0},
        emissions_tons=0.0,
        emissions_by_region={"region": 0.0},
        flows={},
        generation_by_region={"region": {"coal": 80.0}},
        generation_by_coverage={"covered": 45.0, "non_covered": 35.0},
    )

    expected = 100.0 * (35.0 - 20.0) / (80.0 - 60.0)
    assert scenario.leakage_percent(baseline) == pytest.approx(expected)


def test_generation_standard_enforces_share() -> None:
    demand = pd.DataFrame(
        [{"year": 2030, "region": REGION_ALPHA, "demand_mwh": 100.0 * HOURS_PER_YEAR}]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "alpha_wind",
                "unique_id": "alpha_wind",
                "region": REGION_ALPHA,
                "fuel": "wind",
                "cap_mw": 120.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 40.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
            {
                "unit_id": "alpha_gas",
                "unique_id": "alpha_gas",
                "region": REGION_ALPHA,
                "fuel": "gas",
                "cap_mw": 160.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 10.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.4,
            },
        ]
    )
    fuels = pd.DataFrame([
        {"fuel": "wind", "covered": True},
        {"fuel": "gas", "covered": True},
    ])

    frames = Frames({"demand": demand, "units": units, "fuels": fuels})

    share_df = pd.DataFrame(
        {REGION_ALPHA: [0.25]}, index=pd.Index([2030], name="year")
    )
    standard = TechnologyStandard(
        technology="wind", generation_table=share_df, enabled_regions={REGION_ALPHA}
    )
    policy = GenerationStandardPolicy([standard])

    result = solve_from_frames(
        frames, 2030, allowance_cost=0.0, generation_standard=policy
    )
    total_generation = result.generation_total_by_region[REGION_ALPHA]
    wind_generation = result.gen_by_fuel["wind"]
    gas_generation = result.gen_by_fuel["gas"]


    assert wind_generation == pytest.approx(0.25 * total_generation, rel=1e-4)
    assert gas_generation == pytest.approx(0.75 * total_generation, rel=1e-4)


def test_generation_standard_capacity_violation() -> None:
    demand = pd.DataFrame(
        [{"year": 2030, "region": REGION_ALPHA, "demand_mwh": 80.0 * HOURS_PER_YEAR}]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "alpha_wind",
                "unique_id": "alpha_wind",
                "region": REGION_ALPHA,
                "fuel": "wind",
                "cap_mw": 100.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 35.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
            {
                "unit_id": "alpha_gas",
                "unique_id": "alpha_gas",
                "region": REGION_ALPHA,
                "fuel": "gas",
                "cap_mw": 150.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 12.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.5,
            },
        ]
    )
    fuels = pd.DataFrame([
        {"fuel": "wind", "covered": True},
        {"fuel": "gas", "covered": True},
    ])

    frames = Frames({"demand": demand, "units": units, "fuels": fuels})

    capacity_df = pd.DataFrame(
        {REGION_ALPHA: [200.0]}, index=pd.Index([2030], name="year")
    )
    standard = TechnologyStandard(
        technology="wind", capacity_table=capacity_df, enabled_regions={REGION_ALPHA}
    )
    policy = GenerationStandardPolicy([standard])

    with pytest.raises(ValueError, match="wind"):
        solve_from_frames(
            frames, 2030, allowance_cost=0.0, generation_standard=policy
        )

def test_heat_rate_and_fuel_price_determine_variable_cost() -> None:
    frames = _two_unit_frames(load_mwh=500_000.0)
    result = solve_from_frames(frames, 2030, allowance_cost=0.0)

    units = frames.units()
    fuels = frames.fuels()
    co2_map = {
        str(row.fuel): float(getattr(row, "co2_short_ton_per_mwh", 0.0))
        for row in fuels.itertuples(index=False)
    }

    variable_costs = {
        row.fuel: float(row.vom_per_mwh)
        + float(row.hr_mmbtu_per_mwh) * float(row.fuel_price_per_mmbtu)
        for row in units.itertuples(index=False)
    }

    active_costs = [
        variable_costs[fuel]
        for fuel, generation in result.gen_by_fuel.items()
        if generation > 0.0
    ]
    assert active_costs
    assert result.region_prices[REGION_TEST] == pytest.approx(max(active_costs), rel=1e-6)

    expected_emissions = 0.0
    expected_from_fuel = 0.0
    for row in units.itertuples(index=False):
        generation = float(result.gen_by_fuel.get(row.fuel, 0.0))
        expected_emissions += generation * float(row.ef_ton_per_mwh)
        expected_from_fuel += (
            generation * float(row.hr_mmbtu_per_mwh) * co2_map.get(row.fuel, 0.0)
        )

    assert result.emissions_tons == pytest.approx(expected_emissions, rel=1e-6)
    assert result.emissions_tons == pytest.approx(expected_from_fuel, rel=1e-6)


def test_network_dispatch_includes_carbon_cost_column() -> None:
    frames = _two_unit_frames(load_mwh=500_000.0)
    units = frames.units()
    units.loc[units["unit_id"] == "peaker", "carbon_cost_per_mwh"] = 4.0
    frames = frames.with_frame("units", units)

    result = solve_from_frames(frames, 2030, allowance_cost=0.0)

    peaker = units.loc[units["unit_id"] == "peaker"].iloc[0]
    expected_price = (
        float(peaker["vom_per_mwh"])
        + float(peaker["hr_mmbtu_per_mwh"]) * float(peaker["fuel_price_per_mmbtu"])
        + float(peaker["carbon_cost_per_mwh"])
    )

    assert result.region_prices[REGION_TEST] == pytest.approx(expected_price, rel=1e-6)


def test_network_dispatch_uses_short_ton_emissions_when_available() -> None:
    frames = _two_unit_frames(load_mwh=500_000.0)
    units = frames.units()
    units["co2_short_ton_per_mwh"] = units["ef_ton_per_mwh"] * 1.4
    frames = frames.with_frame("units", units)

    result = solve_from_frames(frames, 2030, allowance_cost=0.0)

    expected_emissions = 0.0
    summary_units = frames.units().set_index("unit_id")
    for unit, generation in result.generation_by_unit.items():
        rate = float(summary_units.loc[unit, "co2_short_ton_per_mwh"])
        expected_emissions += rate * float(generation)

    assert result.emissions_tons == pytest.approx(expected_emissions, rel=1e-6)


def test_generation_and_emissions_scale_with_demand() -> None:
    base_frames = _single_unit_frames(load_mwh=400_000.0)
    higher_frames = _single_unit_frames(load_mwh=440_000.0)

    base = solve_from_frames(base_frames, 2030, allowance_cost=0.0)
    higher = solve_from_frames(higher_frames, 2030, allowance_cost=0.0)

    assert base.emissions_tons > 0.0
    assert higher.total_generation > base.total_generation

    generation_ratio = higher.total_generation / base.total_generation
    emissions_ratio = higher.emissions_tons / base.emissions_tons

    assert generation_ratio == pytest.approx(1.1, rel=1e-6)
    assert emissions_ratio == pytest.approx(1.1, rel=1e-6)


def test_unserved_energy_penalty_applied() -> None:
    load = {REGION_TEST: 80.0}
    generators = [
        GeneratorSpec(
            name="limited_unit",
            region=REGION_TEST,
            fuel="gas",
            variable_cost=10.0,
            capacity=50.0,
            emission_rate=0.5,
        )
    ]

    result = solve(
        load,
        generators,
        {},
        allowance_cost=0.0,
        carbon_price=0.0,
        unserved_energy_penalty=100.0,
    )

    assert result.has_unserved_energy
    assert result.unserved_energy_total == pytest.approx(30.0, rel=1e-6)
    assert result.unserved_energy_by_region[REGION_TEST] == pytest.approx(30.0, rel=1e-6)

    expected_cost = 50.0 * 10.0 + 30.0 * 100.0
    assert result.total_cost == pytest.approx(expected_cost, rel=1e-6)


def test_peak_shortage_slack_enforces_requirement() -> None:
    load = {REGION_TEST: 10.0}
    generators = [
        GeneratorSpec(
            name="limited_capacity",
            region=REGION_TEST,
            fuel="gas",
            variable_cost=25.0,
            capacity=50.0,
            emission_rate=0.5,
        )
    ]

    requirement_mw = 5.0
    result = solve(
        load,
        generators,
        {},
        allowance_cost=0.0,
        carbon_price=0.0,
        peak_load_by_region={REGION_TEST: requirement_mw},
        unserved_capacity_penalty=200.0,
    )

    available_mw = 50.0 / HOURS_PER_YEAR
    expected_shortage = max(0.0, requirement_mw - available_mw)

    assert result.has_unserved_capacity
    assert result.unserved_capacity_total == pytest.approx(expected_shortage, rel=1e-6)
    assert result.unserved_capacity_by_region[REGION_TEST] == pytest.approx(
        expected_shortage, rel=1e-6
    )

    expected_cost = 10.0 * 25.0 + expected_shortage * 200.0
    assert result.total_cost == pytest.approx(expected_cost, rel=1e-6)


def test_wheeling_cost_in_objective() -> None:
    load = {REGION_TEST: 0.0, REGION_SOLO: 50.0}
    generators = [
        GeneratorSpec(
            name="exporter",
            region=REGION_TEST,
            fuel="cheap",
            variable_cost=10.0,
            capacity=80.0,
            emission_rate=0.0,
        ),
        GeneratorSpec(
            name="local_expensive",
            region=REGION_SOLO,
            fuel="expensive",
            variable_cost=30.0,
            capacity=0.0,
            emission_rate=0.0,
        ),
    ]

    interfaces = {(REGION_TEST, REGION_SOLO): 100.0}
    interface_costs = {(REGION_TEST, REGION_SOLO): 5.0}

    result = solve(
        load,
        generators,
        interfaces,
        allowance_cost=0.0,
        carbon_price=0.0,
        interface_costs=interface_costs,
    )

    pair = tuple(sorted((REGION_TEST, REGION_SOLO)))
    assert result.flows[pair] == pytest.approx(50.0, rel=1e-6)
    assert result.wheeling_cost_by_interface[pair] == pytest.approx(250.0, rel=1e-6)

    expected_total_cost = 50.0 * 10.0 + 250.0
    assert result.total_cost == pytest.approx(expected_total_cost, rel=1e-6)
    assert not result.has_unserved_energy


def test_directional_interface_limits() -> None:
    load = {REGION_TEST: 0.0, REGION_SOLO: 100.0}
    generators = [
        GeneratorSpec(
            name="exporter",
            region=REGION_TEST,
            fuel="cheap",
            variable_cost=10.0,
            capacity=100.0,
            emission_rate=0.0,
        ),
        GeneratorSpec(
            name="local_expensive",
            region=REGION_SOLO,
            fuel="expensive",
            variable_cost=50.0,
            capacity=100.0,
            emission_rate=0.0,
        ),
    ]

    interfaces = {
        (REGION_TEST, REGION_SOLO): 60.0,
        (REGION_SOLO, REGION_TEST): 10.0,
    }

    result = solve(
        load,
        generators,
        interfaces,
        allowance_cost=0.0,
        carbon_price=0.0,
    )

    pair = tuple(sorted((REGION_TEST, REGION_SOLO)))
    assert result.flows[pair] == pytest.approx(60.0, rel=1e-6)
    assert result.generation_by_unit["exporter"] == pytest.approx(60.0, rel=1e-6)
    assert result.generation_by_unit["local_expensive"] == pytest.approx(40.0, rel=1e-6)
