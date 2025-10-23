from pathlib import Path

import pytest
pytest.importorskip('pyomo')
import pyomo.environ as pyo

from main.definitions import PROJECT_ROOT
from policy.allowance_supply import AllowanceSupply
from src.common import config_setup
from src.models.electricity.scripts import preprocessor as prep
from src.models.electricity.scripts.electricity_model import PowerModel
from src.models.electricity.scripts.runner import record_allowance_emission_prices


def build_allowance_model(
    years,
    allowances,
    start_bank,
    banking_enabled=True,
    allow_borrowing=False,
    cap_groups=('system',),
    membership=None,
    annual_surrender_frac=0.5,
    carry_pct=1.0,
    *,
    price=None,
    ccr1_trigger=None,
    ccr1_qty=None,
    ccr2_trigger=None,
    ccr2_qty=None,
    ccr1_active=None,
    ccr2_active=None,
):
    model = pyo.ConcreteModel()
    cap_groups = tuple(cap_groups)
    try:
        ordered_years = tuple(sorted(years))
    except TypeError:
        ordered_years = tuple(years)
    if not ordered_years:
        raise ValueError('years must contain at least one modeling year')
    model.year = pyo.Set(initialize=ordered_years, ordered=True)
    model.cap_group = pyo.Set(initialize=cap_groups, ordered=True)
    model.final_year = ordered_years[-1]
    model.cap_group_year = pyo.Set(
        initialize=[(g, y) for g in cap_groups for y in ordered_years],
        dimen=2,
        ordered=True,
    )
    model.banking_enabled = banking_enabled
    model.prev_year_lookup = {
        year: (ordered_years[idx - 1] if idx > 0 else None)
        for idx, year in enumerate(ordered_years)
    }

    allowances_map = {}
    for key, value in allowances.items():
        if isinstance(key, tuple):
            group, year = key
        else:
            group, year = cap_groups[0], key
        allowances_map[(group, int(year))] = float(value)
    model.CarbonAllowanceProcurement = pyo.Param(
        model.cap_group_year,
        initialize=allowances_map,
        default=0.0,
        mutable=True,
    )

    def _map(values):
        mapping = {}
        if values is None:
            return mapping
        for key, value in values.items():
            if isinstance(key, tuple):
                group, year = key
            else:
                group, year = cap_groups[0], key
            mapping[(group, int(year))] = float(value)
        return mapping

    price_map = _map(price)
    model.CarbonPrice = pyo.Param(
        model.cap_group_year,
        initialize={idx: price_map.get(idx, 0.0) for idx in model.cap_group_year},
        default=0.0,
        mutable=True,
    )

    ccr1_trigger_map = _map(ccr1_trigger)
    model.CarbonCCR1Trigger = pyo.Param(
        model.cap_group_year,
        initialize={idx: ccr1_trigger_map.get(idx, 0.0) for idx in model.cap_group_year},
        default=0.0,
        mutable=True,
    )

    ccr1_qty_map = _map(ccr1_qty)
    model.CarbonCCR1Quantity = pyo.Param(
        model.cap_group_year,
        initialize={idx: ccr1_qty_map.get(idx, 0.0) for idx in model.cap_group_year},
        default=0.0,
        mutable=True,
    )

    ccr2_trigger_map = _map(ccr2_trigger)
    model.CarbonCCR2Trigger = pyo.Param(
        model.cap_group_year,
        initialize={idx: ccr2_trigger_map.get(idx, 0.0) for idx in model.cap_group_year},
        default=0.0,
        mutable=True,
    )

    ccr2_qty_map = _map(ccr2_qty)
    model.CarbonCCR2Quantity = pyo.Param(
        model.cap_group_year,
        initialize={idx: ccr2_qty_map.get(idx, 0.0) for idx in model.cap_group_year},
        default=0.0,
        mutable=True,
    )

    ccr1_active_map = _map(ccr1_active)
    model.CarbonCCR1Active = pyo.Param(
        model.cap_group_year,
        initialize={idx: ccr1_active_map.get(idx, 0.0) for idx in model.cap_group_year},
        default=0.0,
        mutable=True,
    )

    ccr2_active_map = _map(ccr2_active)
    model.CarbonCCR2Active = pyo.Param(
        model.cap_group_year,
        initialize={idx: ccr2_active_map.get(idx, 0.0) for idx in model.cap_group_year},
        default=0.0,
        mutable=True,
    )

    if isinstance(start_bank, dict):
        start_bank_map = {
            (group, int(year)): float(value) for (group, year), value in start_bank.items()
        }
    else:
        first_year = ordered_years[0]
        start_bank_map = {
            (group, first_year): float(start_bank) for group in cap_groups
        }
    model.CarbonStartBank = pyo.Param(
        model.cap_group_year,
        initialize=start_bank_map,
        default=0.0,
        mutable=True,
    )

    def _value_lookup(raw_value, group, year, default):
        if isinstance(raw_value, dict):
            for key in (
                (group, year),
                (group, int(year)),
                (group, str(year)),
                year,
                int(year),
                str(year),
            ):
                if key in raw_value:
                    try:
                        return float(raw_value[key])
                    except (TypeError, ValueError):
                        return float(default)
            return float(default)
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return float(default)

    surrender_map = {
        (group, int(year)): _value_lookup(annual_surrender_frac, group, year, 0.0)
        for group in cap_groups
        for year in ordered_years
    }
    model.AnnualSurrenderFrac = pyo.Param(
        model.cap_group_year,
        initialize=surrender_map,
        default=0.0,
        mutable=True,
    )

    carry_map = {
        (group, int(year)): _value_lookup(carry_pct, group, year, 1.0)
        for group in cap_groups
        for year in ordered_years
    }
    model.CarryPct = pyo.Param(
        model.cap_group_year,
        initialize=carry_map,
        default=1.0,
        mutable=True,
    )

    if membership is None:
        membership = {(cap_groups[0], 'region'): 1.0}
    membership_map = {
        (group, region): float(value) for (group, region), value in membership.items()
    }
    model.cap_group_region = pyo.Set(
        initialize=list(membership_map.keys()), dimen=2, ordered=True
    )
    model.CarbonCapGroupMembership = pyo.Param(
        model.cap_group_region, initialize=membership_map, default=0.0
    )

    bank_domain = pyo.Reals if allow_borrowing else pyo.NonNegativeReals
    model.allowance_purchase = pyo.Var(
        model.cap_group_year, domain=pyo.NonNegativeReals
    )
    model.allowance_base = pyo.Var(
        model.cap_group_year, domain=pyo.NonNegativeReals
    )
    model.allowance_ccr1 = pyo.Var(
        model.cap_group_year, domain=pyo.NonNegativeReals
    )
    model.allowance_ccr2 = pyo.Var(
        model.cap_group_year, domain=pyo.NonNegativeReals
    )
    model.allowance_bank = pyo.Var(model.cap_group_year, domain=bank_domain)
    model.year_emissions = pyo.Var(
        model.cap_group_year, domain=pyo.NonNegativeReals
    )
    model.allowance_surrender = pyo.Var(
        model.cap_group_year, domain=pyo.NonNegativeReals
    )
    model.allowance_obligation = pyo.Var(
        model.cap_group_year, domain=pyo.NonNegativeReals
    )

    def incoming_bank(m, group, year):
        prev_year = m.prev_year_lookup[year]
        carryover = (
            m.allowance_bank[(group, prev_year)]
            if (m.banking_enabled and prev_year is not None)
            else 0
        )
        return carryover + m.CarbonStartBank[(group, year)]

    model.allowance_purchase_limit = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, g, y: m.allowance_purchase[g, y]
        <= m.allowance_base[g, y] + m.allowance_ccr1[g, y] + m.allowance_ccr2[g, y],
    )
    model.allowance_base_limit = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, g, y: m.allowance_base[g, y]
        <= m.CarbonAllowanceProcurement[g, y],
    )
    model.allowance_ccr1_limit = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, g, y: m.allowance_ccr1[g, y]
        <= m.CarbonCCR1Quantity[g, y] * m.CarbonCCR1Active[g, y],
    )
    model.allowance_ccr2_limit = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, g, y: m.allowance_ccr2[g, y]
        <= m.CarbonCCR2Quantity[g, y] * m.CarbonCCR2Active[g, y],
    )
    model.allowance_total_balance = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, g, y: m.allowance_purchase[g, y]
        == m.allowance_base[g, y] + m.allowance_ccr1[g, y] + m.allowance_ccr2[g, y],
    )
    model.allowance_bank_balance = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, g, y: m.allowance_bank[g, y]
        == (
            (m.CarryPct[g, y] if m.banking_enabled else 0.0)
            * (
                incoming_bank(m, g, y)
                + m.allowance_purchase[g, y]
                - m.allowance_surrender[g, y]
            )
        ),
    )
    model.allowance_surrender_requirement = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, g, y: m.allowance_surrender[g, y]
        >= m.AnnualSurrenderFrac[g, y] * m.year_emissions[g, y],
    )
    model.allowance_obligation_balance = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, g, y: m.allowance_obligation[g, y]
        == (
            (
                m.allowance_obligation[(g, m.prev_year_lookup[y])]
                if m.prev_year_lookup[y] is not None
                else 0.0
            )
            + m.year_emissions[g, y]
            - m.allowance_surrender[g, y]
        ),
    )
    model.allowance_emissions_limit = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, g, y: m.allowance_surrender[g, y]
        <= m.allowance_purchase[g, y] + incoming_bank(m, g, y),
    )

    model.allowance_final_obligation_settlement = pyo.Constraint(
        model.cap_group,
        rule=lambda m, g: m.allowance_obligation[g, m.final_year] == 0,
    )

    return model


def test_allowance_bank_balance():
    years = [2025, 2030]
    allowances = {2025: 10.0, 2030: 12.0}
    model = build_allowance_model(
        years, allowances, start_bank=2.0, annual_surrender_frac=0.5
    )

    key_2025 = ('system', 2025)
    key_2030 = ('system', 2030)
    model.allowance_purchase[key_2025].set_value(4.0)
    model.allowance_purchase[key_2030].set_value(6.0)
    model.allowance_base[key_2025].set_value(4.0)
    model.allowance_base[key_2030].set_value(6.0)
    model.allowance_ccr1[key_2025].set_value(0.0)
    model.allowance_ccr1[key_2030].set_value(0.0)
    model.allowance_ccr2[key_2025].set_value(0.0)
    model.allowance_ccr2[key_2030].set_value(0.0)
    model.year_emissions[key_2025].set_value(3.0)
    model.year_emissions[key_2030].set_value(7.0)
    model.allowance_surrender[key_2025].set_value(1.5)
    model.allowance_surrender[key_2030].set_value(8.5)
    model.allowance_obligation[key_2025].set_value(1.5)
    model.allowance_obligation[key_2030].set_value(0.0)
    model.allowance_bank[key_2025].set_value(4.5)
    model.allowance_bank[key_2030].set_value(2.0)

    balance_2025 = model.allowance_bank_balance[key_2025]
    balance_2030 = model.allowance_bank_balance[key_2030]
    assert pytest.approx(0.0) == pyo.value(balance_2025.body)
    assert pytest.approx(0.0) == pyo.value(balance_2030.body)

    incoming_2025 = pyo.value(model.CarbonStartBank[key_2025])
    incoming_2030 = model.allowance_bank[key_2025].value
    assert model.allowance_surrender[key_2025].value <= (
        model.allowance_purchase[key_2025].value + incoming_2025
    )
    assert model.allowance_surrender[key_2030].value <= (
        model.allowance_purchase[key_2030].value + incoming_2030
    )

    required_2030 = pyo.value(
        model.AnnualSurrenderFrac[key_2030] * model.year_emissions[key_2030]
    )
    assert model.allowance_surrender[key_2030].value >= required_2030

    final_constraint = model.allowance_final_obligation_settlement['system']
    assert pytest.approx(0.0) == pyo.value(final_constraint.body)


def test_emission_limit_detects_shortfall():
    years = [2025, 2030]
    allowances = {2025: 5.0, 2030: 5.0}
    model = build_allowance_model(
        years,
        allowances,
        start_bank=0.0,
        banking_enabled=True,
        allow_borrowing=True,
        annual_surrender_frac=1.0,
    )

    key_2025 = ('system', 2025)
    key_2030 = ('system', 2030)
    model.allowance_purchase[key_2025].set_value(5.0)
    model.allowance_purchase[key_2030].set_value(5.0)
    model.allowance_base[key_2025].set_value(5.0)
    model.allowance_base[key_2030].set_value(5.0)
    model.allowance_ccr1[key_2025].set_value(0.0)
    model.allowance_ccr1[key_2030].set_value(0.0)
    model.allowance_ccr2[key_2025].set_value(0.0)
    model.allowance_ccr2[key_2030].set_value(0.0)
    model.year_emissions[key_2025].set_value(6.0)
    model.year_emissions[key_2030].set_value(4.0)
    model.allowance_surrender[key_2025].set_value(6.0)
    model.allowance_surrender[key_2030].set_value(4.0)
    model.allowance_obligation[key_2025].set_value(0.0)
    model.allowance_obligation[key_2030].set_value(0.0)
    model.allowance_bank[key_2025].set_value(-1.0)
    model.allowance_bank[key_2030].set_value(0.0)

    incoming_2025 = pyo.value(model.CarbonStartBank[key_2025])
    incoming_2030 = model.allowance_bank[key_2025].value
    assert model.allowance_surrender[key_2025].value > (
        model.allowance_purchase[key_2025].value + incoming_2025
    )
    assert pytest.approx(0.0) == pyo.value(
        model.allowance_bank_balance[key_2025].body
    )
    assert model.allowance_surrender[key_2030].value <= (
        model.allowance_purchase[key_2030].value + incoming_2030
    )


@pytest.mark.usefixtures('minimal_carbon_policy_inputs')
def test_carbon_price_updates_refresh_ccr_activation():
    config_path = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    settings = config_setup.Config_settings(config_path, test=True)
    settings.regions = [7, 8]
    settings.years = [2025, 2030]

    set_inputs = prep.Sets(settings)
    frames_store, set_inputs = prep.preprocessor(set_inputs)
    model = PowerModel(frames_store.to_frames(), set_inputs)

    idx = next(iter(model.cap_group_year_index))
    model.CarbonCCR1Trigger[idx].set_value(10.0)
    model.CarbonCCR1Quantity[idx].set_value(1.0)
    model.CarbonCCR2Trigger[idx].set_value(20.0)
    model.CarbonCCR2Quantity[idx].set_value(1.0)
    model.update_ccr_activation()

    model.CarbonPrice[idx].set_value(5.0)
    assert pyo.value(model.CarbonCCR1Active[idx]) == pytest.approx(0.0)
    assert pyo.value(model.CarbonCCR2Active[idx]) == pytest.approx(0.0)

    model.CarbonPrice[idx].set_value(12.0)
    assert pyo.value(model.CarbonCCR1Active[idx]) == pytest.approx(1.0)
    assert pyo.value(model.CarbonCCR2Active[idx]) == pytest.approx(0.0)

    model.CarbonPrice[idx].set_value(25.0)
    assert pyo.value(model.CarbonCCR1Active[idx]) == pytest.approx(1.0)
    assert pyo.value(model.CarbonCCR2Active[idx]) == pytest.approx(1.0)


def test_final_year_obligation_must_clear():
    years = [2030, 2025]
    allowances = {2025: 5.0, 2030: 6.0}
    model = build_allowance_model(
        years,
        allowances,
        start_bank=0.0,
        annual_surrender_frac=0.2,
    )

    key_2025 = ('system', 2025)
    key_2030 = ('system', 2030)

    assert model.final_year == 2030

    model.allowance_purchase[key_2025].set_value(5.0)
    model.year_emissions[key_2025].set_value(5.0)
    model.allowance_surrender[key_2025].set_value(1.0)
    model.allowance_bank[key_2025].set_value(4.0)
    model.allowance_obligation[key_2025].set_value(4.0)

    model.allowance_purchase[key_2030].set_value(5.0)
    model.year_emissions[key_2030].set_value(6.0)
    model.allowance_surrender[key_2030].set_value(3.0)
    model.allowance_bank[key_2030].set_value(6.0)
    model.allowance_obligation[key_2030].set_value(7.0)

    balance_2025 = model.allowance_bank_balance[key_2025]
    obligation_2025 = model.allowance_obligation_balance[key_2025]
    balance_2030 = model.allowance_bank_balance[key_2030]
    obligation_2030 = model.allowance_obligation_balance[key_2030]

    assert pytest.approx(0.0) == pyo.value(balance_2025.body)
    assert pytest.approx(0.0) == pyo.value(obligation_2025.body)
    assert pytest.approx(0.0) == pyo.value(balance_2030.body)
    assert pytest.approx(0.0) == pyo.value(obligation_2030.body)

    final_constraint = model.allowance_final_obligation_settlement['system']
    assert pytest.approx(7.0) == pyo.value(final_constraint.body)

    model.allowance_purchase[key_2030].set_value(6.0)
    model.allowance_surrender[key_2030].set_value(10.0)
    model.allowance_bank[key_2030].set_value(0.0)
    model.allowance_obligation[key_2030].set_value(0.0)

    assert pytest.approx(0.0) == pyo.value(balance_2030.body)
    assert pytest.approx(0.0) == pyo.value(obligation_2030.body)
    assert pytest.approx(0.0) == pyo.value(final_constraint.body)


@pytest.mark.parametrize(
    'price, expected_ccr1, expected_ccr2',
    [
        (0.0, 0.0, 0.0),
        (20.0, 50.0, 0.0),
        (60.0, 50.0, 80.0),
    ],
)
def test_ccr_allowances_match_supply(price, expected_ccr1, expected_ccr2):
    years = [2025]
    cap_group = 'system'
    base_allowance = 100.0
    ccr1_trigger = 20.0
    ccr1_qty = 50.0
    ccr2_trigger = 50.0
    ccr2_qty = 80.0

    active1 = {years[0]: 1.0 if price >= ccr1_trigger else 0.0}
    active2 = {years[0]: 1.0 if price >= ccr2_trigger else 0.0}

    model = build_allowance_model(
        years,
        {years[0]: base_allowance},
        start_bank=0.0,
        banking_enabled=False,
        allow_borrowing=False,
        price={years[0]: price},
        ccr1_trigger={years[0]: ccr1_trigger},
        ccr1_qty={years[0]: ccr1_qty},
        ccr2_trigger={years[0]: ccr2_trigger},
        ccr2_qty={years[0]: ccr2_qty},
        ccr1_active=active1,
        ccr2_active=active2,
    )

    key = (cap_group, years[0])
    supply = AllowanceSupply(
        cap=base_allowance,
        floor=0.0,
        ccr1_trigger=ccr1_trigger,
        ccr1_qty=ccr1_qty,
        ccr2_trigger=ccr2_trigger,
        ccr2_qty=ccr2_qty,
    )
    available = supply.available_allowances(price)

    model.year_emissions[key].fix(available)
    model.allowance_bank[key].fix(0.0)
    model.total_cost = pyo.Objective(expr=model.allowance_purchase[key])

    solver = pyo.SolverFactory('appsi_highs')
    if not solver.available(False):
        pytest.skip('appsi_highs solver not available in test environment')

    result = solver.solve(model)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal

    expected_base = min(base_allowance, available)
    assert model.allowance_purchase[key].value == pytest.approx(available)
    assert model.allowance_base[key].value == pytest.approx(expected_base)
    assert model.allowance_ccr1[key].value == pytest.approx(expected_ccr1)
    assert model.allowance_ccr2[key].value == pytest.approx(expected_ccr2)



def test_reported_carbon_price_matches_marginal_cost():
    baseline_emissions = 10.0
    baseline_allowance = 8.0
    abatement_cost = 25.0
    tighten = 0.5
    solver = pyo.SolverFactory('appsi_highs')
    if not solver.available(False):
        pytest.skip('appsi_highs solver not available in test environment')

    def solve_toy_model(allowance: float) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        model.abatement = pyo.Var(domain=pyo.NonNegativeReals)
        model.emissions = pyo.Var(domain=pyo.NonNegativeReals)
        model.allowance_purchase = pyo.Var(domain=pyo.NonNegativeReals)

        model.emissions_balance = pyo.Constraint(
            expr=model.emissions == baseline_emissions - model.abatement
        )
        model.allowance_purchase_limit = pyo.Constraint(
            expr=model.allowance_purchase <= allowance
        )
        model.allowance_emissions_limit = pyo.Constraint(
            expr=model.emissions <= model.allowance_purchase
        )
        model.total_cost = pyo.Objective(expr=abatement_cost * model.abatement)

        solver.solve(model)
        record_allowance_emission_prices(model)
        return model

    baseline_model = solve_toy_model(baseline_allowance)
    base_cost = pyo.value(baseline_model.total_cost)
    base_price = baseline_model.carbon_prices.get(None)
    assert base_price is not None and base_price > 0.0

    tightened_model = solve_toy_model(baseline_allowance - tighten)
    tightened_cost = pyo.value(tightened_model.total_cost)

    delta_cost = tightened_cost - base_cost
    expected_cost = base_price * tighten

    assert delta_cost == pytest.approx(expected_cost, rel=1e-8, abs=1e-8)
