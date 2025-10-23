import time

import pandas as pd
import pytest

from dispatch.lp_network import GeneratorSpec, solve
from dispatch.lp_single import HOURS_PER_YEAR
from policy.generation_standard import GenerationStandardPolicy, TechnologyStandard


def _two_bus_case():
    loads = {
        "A": 60.0 * HOURS_PER_YEAR,
        "B": 60.0 * HOURS_PER_YEAR,
    }
    generators = [
        GeneratorSpec(
            name="gen_a",
            region="A",
            fuel="gas",
            variable_cost=5.0,
            capacity=100.0 * HOURS_PER_YEAR,
            emission_rate=0.5,
        ),
        GeneratorSpec(
            name="gen_b",
            region="B",
            fuel="oil",
            variable_cost=20.0,
            capacity=120.0 * HOURS_PER_YEAR,
            emission_rate=0.7,
        ),
    ]
    interfaces = {("A", "B"): 30.0 * HOURS_PER_YEAR}
    return loads, generators, interfaces


def _five_bus_case():
    loads = {
        "A": 80.0 * HOURS_PER_YEAR,
        "B": 60.0 * HOURS_PER_YEAR,
        "C": 70.0 * HOURS_PER_YEAR,
        "D": 50.0 * HOURS_PER_YEAR,
        "E": 40.0 * HOURS_PER_YEAR,
    }
    generators = [
        GeneratorSpec(
            name="gen_a",
            region="A",
            fuel="gas",
            variable_cost=5.0,
            capacity=250.0 * HOURS_PER_YEAR,
            emission_rate=0.4,
        ),
        GeneratorSpec(
            name="gen_b",
            region="B",
            fuel="coal",
            variable_cost=15.0,
            capacity=120.0 * HOURS_PER_YEAR,
            emission_rate=0.6,
        ),
        GeneratorSpec(
            name="gen_c",
            region="C",
            fuel="oil",
            variable_cost=25.0,
            capacity=110.0 * HOURS_PER_YEAR,
            emission_rate=0.7,
        ),
        GeneratorSpec(
            name="gen_d",
            region="D",
            fuel="diesel",
            variable_cost=30.0,
            capacity=90.0 * HOURS_PER_YEAR,
            emission_rate=0.75,
        ),
        GeneratorSpec(
            name="gen_e",
            region="E",
            fuel="biomass",
            variable_cost=35.0,
            capacity=90.0 * HOURS_PER_YEAR,
            emission_rate=0.5,
        ),
    ]
    interfaces = {
        ("A", "B"): 80.0 * HOURS_PER_YEAR,
        ("B", "C"): 60.0 * HOURS_PER_YEAR,
        ("C", "D"): 50.0 * HOURS_PER_YEAR,
        ("D", "E"): 40.0 * HOURS_PER_YEAR,
        ("A", "C"): 40.0 * HOURS_PER_YEAR,
    }
    return loads, generators, interfaces


@pytest.mark.parametrize("penalty", [1e6])
def test_two_bus_dispatch_matches_expected(penalty):
    loads, generators, interfaces = _two_bus_case()
    result = solve(
        loads,
        generators,
        interfaces,
        allowance_cost=0.0,
        carbon_price=0.0,
        interface_costs=None,
        unserved_energy_penalty=penalty,
    )

    expected_a = 90.0 * HOURS_PER_YEAR
    expected_b = 30.0 * HOURS_PER_YEAR
    expected_cost = expected_a * 5.0 + expected_b * 20.0

    assert result.total_cost == pytest.approx(expected_cost, rel=1e-6)
    assert result.generation_by_unit["gen_a"] == pytest.approx(expected_a)
    assert result.generation_by_unit["gen_b"] == pytest.approx(expected_b)
    assert result.flows[("A", "B")] == pytest.approx(30.0 * HOURS_PER_YEAR)
    assert result.unserved_energy_total == pytest.approx(0.0)


def test_two_bus_solver_runtime():
    loads, generators, interfaces = _two_bus_case()
    # Warm-up run to ensure backend is initialised.
    solve(
        loads,
        generators,
        interfaces,
        allowance_cost=0.0,
        carbon_price=0.0,
        interface_costs=None,
        unserved_energy_penalty=1e6,
    )
    start = time.perf_counter()
    solve(
        loads,
        generators,
        interfaces,
        allowance_cost=0.0,
        carbon_price=0.0,
        interface_costs=None,
        unserved_energy_penalty=1e6,
    )
    duration_ms = (time.perf_counter() - start) * 1000.0
    assert duration_ms < 50.0


def test_five_bus_solver_runtime():
    loads, generators, interfaces = _five_bus_case()
    solve(
        loads,
        generators,
        interfaces,
        allowance_cost=0.0,
        carbon_price=0.0,
        interface_costs=None,
        unserved_energy_penalty=1e6,
    )
    start = time.perf_counter()
    solve(
        loads,
        generators,
        interfaces,
        allowance_cost=0.0,
        carbon_price=0.0,
        interface_costs=None,
        unserved_energy_penalty=1e6,
    )
    duration_ms = (time.perf_counter() - start) * 1000.0
    assert duration_ms < 50.0


def test_zero_cost_interfaces_do_not_accumulate_cost():
    loads = {
        "A": 80.0 * HOURS_PER_YEAR,
        "B": 40.0 * HOURS_PER_YEAR,
        "C": 30.0 * HOURS_PER_YEAR,
    }
    generators = [
        GeneratorSpec(
            name="gen_a",
            region="A",
            fuel="hydro",
            variable_cost=5.0,
            capacity=200.0 * HOURS_PER_YEAR,
            emission_rate=0.0,
        )
    ]
    interfaces = {
        ("A", "B"): 100.0 * HOURS_PER_YEAR,
        ("B", "C"): 100.0 * HOURS_PER_YEAR,
        ("A", "C"): 100.0 * HOURS_PER_YEAR,
    }
    interface_costs = {key: 0.0 for key in interfaces}

    result = solve(
        loads,
        generators,
        interfaces,
        allowance_cost=0.0,
        carbon_price=0.0,
        interface_costs=interface_costs,
    )

    expected_generation = sum(loads.values())
    expected_cost = expected_generation * 5.0

    assert result.total_cost == pytest.approx(expected_cost, rel=1e-6)
    for cost in result.wheeling_cost_by_interface.values():
        assert cost == pytest.approx(0.0, abs=1e-6)
    assert result.unserved_energy_total == pytest.approx(0.0, abs=1e-9)


def test_generation_standard_non_binding_share_matches_baseline():
    loads = {"R": 80.0}
    generators = [
        GeneratorSpec(
            name="solar",
            region="R",
            fuel="solar",
            variable_cost=5.0,
            capacity=60.0,
            emission_rate=0.0,
        ),
        GeneratorSpec(
            name="gas",
            region="R",
            fuel="gas",
            variable_cost=10.0,
            capacity=80.0,
            emission_rate=0.5,
        ),
    ]

    baseline = solve(
        loads,
        generators,
        {},
        allowance_cost=0.0,
        carbon_price=0.0,
    )

    share_df = pd.DataFrame({"year": [2020], "R": [0.2]})
    standard = TechnologyStandard(technology="solar", generation_table=share_df)
    policy = GenerationStandardPolicy([standard])

    with_policy = solve(
        loads,
        generators,
        {},
        allowance_cost=0.0,
        carbon_price=0.0,
        generation_standard=policy,
        year=2020,
    )

    for unit, baseline_output in baseline.generation_by_unit.items():
        assert with_policy.generation_by_unit[unit] == pytest.approx(
            baseline_output, rel=1e-6
        )
    assert with_policy.total_cost == pytest.approx(baseline.total_cost, rel=1e-6)


def test_generation_standard_binding_share_adjusts_dispatch():
    loads = {"R": 100.0}
    generators = [
        GeneratorSpec(
            name="renewable",
            region="R",
            fuel="solar",
            variable_cost=15.0,
            capacity=100.0,
            emission_rate=0.0,
        ),
        GeneratorSpec(
            name="gas",
            region="R",
            fuel="gas",
            variable_cost=5.0,
            capacity=150.0,
            emission_rate=0.5,
        ),
    ]

    share_df = pd.DataFrame({"year": [2020], "R": [0.3]})
    standard = TechnologyStandard(technology="solar", generation_table=share_df)
    policy = GenerationStandardPolicy([standard])

    result = solve(
        loads,
        generators,
        {},
        allowance_cost=0.0,
        carbon_price=0.0,
        generation_standard=policy,
        year=2020,
    )

    renewable_output = result.generation_by_unit["renewable"]
    total_output = sum(result.generation_by_unit.values())

    assert renewable_output == pytest.approx(0.3 * total_output, rel=1e-6)
    assert result.total_cost == pytest.approx(renewable_output * 15.0 + (total_output - renewable_output) * 5.0, rel=1e-6)


def test_generation_standard_infeasible_capacity_raises():
    loads = {"R": 10.0 * HOURS_PER_YEAR}
    generators = [
        GeneratorSpec(
            name="gas",
            region="R",
            fuel="gas",
            variable_cost=5.0,
            capacity=50.0 * HOURS_PER_YEAR,
            emission_rate=0.5,
        )
    ]

    capacity_df = pd.DataFrame({"year": [2020], "R": [60.0]})
    standard = TechnologyStandard(technology="gas", capacity_table=capacity_df)
    policy = GenerationStandardPolicy([standard])

    with pytest.raises(ValueError, match="requires"):
        solve(
            loads,
            generators,
            {},
            allowance_cost=0.0,
            carbon_price=0.0,
            generation_standard=policy,
            year=2020,
        )


def test_region_price_matches_objective_delta():
    loads = {"R": 60.0}
    generators = [
        GeneratorSpec(
            name="base",
            region="R",
            fuel="gas",
            variable_cost=5.0,
            capacity=50.0,
            emission_rate=0.4,
        ),
        GeneratorSpec(
            name="peak",
            region="R",
            fuel="gas",
            variable_cost=20.0,
            capacity=100.0,
            emission_rate=0.6,
        ),
    ]

    baseline = solve(
        loads,
        generators,
        {},
        allowance_cost=0.0,
        carbon_price=0.0,
    )

    perturbed = solve(
        {"R": loads["R"] + 1.0},
        generators,
        {},
        allowance_cost=0.0,
        carbon_price=0.0,
    )

    price = baseline.region_prices["R"]
    delta_cost = perturbed.total_cost - baseline.total_cost

    assert price == pytest.approx(delta_cost, rel=1e-6, abs=1e-6)
