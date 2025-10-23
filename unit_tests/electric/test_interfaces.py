from collections import defaultdict
from pathlib import Path

import pytest
from pyomo.environ import value

from main.definitions import PROJECT_ROOT
from src.common import config_setup
from src.integrator.utilities import (
    HI,
    get_elec_price,
    regional_annual_prices,
    poll_h2_demand,
    update_h2_prices,
)
from src.models.electricity.scripts import preprocessor as prep
from src.models.electricity.scripts.runner import build_elec_model, run_elec_model


def test_poll_elec_prices():
    """test that we can poll prices from elec and get "reasonable" answers"""
    years = [2030, 2031]
    regions = [7, 8]

    config_path = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    settings = config_setup.Config_settings(config_path, test=True)
    settings.regions = regions
    settings.years = years
    elec_model = run_elec_model(settings, solve=True)
    # we are just testing to see if we got *something* back ... this should have hundreds of entries...
    new_prices = get_elec_price(elec_model)
    assert len(new_prices) > 1, 'should have at least 1 price'

    # test for signage of observations
    price_records = new_prices.to_records()
    assert all((price >= 0 for *_, price in price_records)), 'expecting prices to be positive'

    # TODO:  Why does this fail?  there appear to be zero prices... Non binding constraint in these areas??
    # assert all((price > 0 for _, price in new_prices)), 'price should be non-zero, right???'

    # test for average price mehhhh above $1000
    lut = regional_annual_prices(elec_model)
    # TODO:  When price data stabilizes fix this to test that ALL are >1000.  RN region 7 has low costs
    assert max(lut.values()) > 1000, 'cost should be over $1000'


def test_update_h2_price():
    """
    test the ability to update the h2 prices in the model
    """
    years = [2030, 2031]
    regions = [2]
    config_path = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    settings = config_setup.Config_settings(config_path, test=True)
    settings.regions = regions
    settings.years = years
    # just load the model...
    elec_model = run_elec_model(settings, solve=False)
    new_prices = {HI(2, 2030): 999.0, HI(2, 2031): 101010.10}
    update_h2_prices(elec_model, new_prices)

    # sample a couple...
    #                               r, season, tech, step, yr
    assert value(elec_model.H2Price[2, 1, 5, 1, 2030]) == pytest.approx(999.0)
    assert value(elec_model.H2Price[2, 3, 5, 1, 2030]) == pytest.approx(999.0)
    assert value(elec_model.H2Price[2, 2, 5, 1, 2031]) == pytest.approx(101010.10)


def test_poll_h2_demand():
    """
    poll the solved model for some H2 Demands

    Note:  We don't have a "right answer" for this (yet), so this will just do some basic functional test
    """

    years = [2030, 2031]
    regions = [2]

    config_path = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    settings = config_setup.Config_settings(config_path, test=True)
    settings.regions = regions
    settings.years = years
    elec_model = run_elec_model(settings, solve=True)

    h2_demands = poll_h2_demand(elec_model)

    # some poking & proding
    assert h2_demands.keys() == {HI(2, 2030), HI(2, 2031)}


def test_capacity_retirement_persists_with_missing_future_index():
    """Ensure early retirements continue to reduce capacity even when later indices are absent."""

    config_path = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    settings = config_setup.Config_settings(config_path, test=True)
    settings.regions = [2]
    settings.years = [2030, 2035]

    set_inputs = prep.Sets(settings)
    frames_store, set_processed = prep.preprocessor(set_inputs)

    retirement_df = set_processed.capacity_retirements_index.copy()
    retirement_years = defaultdict(list)
    for tech, year, region, step in retirement_df.index:
        retirement_years[(tech, region, step)].append(int(year))

    target = None
    for (tech, region, step), years in retirement_years.items():
        years_sorted = sorted(set(years))
        if len(years_sorted) < 2:
            continue
        target = (tech, region, step, years_sorted[0], years_sorted[-1])
        break

    assert target is not None, 'expected at least one retireable combination with multiple years'
    tech, region, step, early_year, later_year = target

    mask = [
        idx
        for idx in retirement_df.index
        if idx[0] == tech and idx[2] == region and idx[3] == step and int(idx[1]) > early_year
    ]
    assert mask, 'test requires removing at least one later-year retirement entry'
    set_processed.capacity_retirements_index = retirement_df.drop(index=mask)

    frames = frames_store.to_frames()
    model = build_elec_model(frames, set_processed)

    assert (tech, early_year, region, step) in model.capacity_retirements_index
    assert (tech, later_year, region, step) not in model.capacity_retirements_index

    seasons = {
        season
        for (r_idx, season, tech_idx, step_idx, year_idx) in model.capacity_total_index
        if r_idx == region
        and tech_idx == tech
        and step_idx == step
        and int(year_idx) in (early_year, later_year)
    }
    assert seasons, 'selected combination should have at least one season'
    season = sorted(seasons)[0]

    base_early = value(model.SupplyCurve[(region, season, tech, step, early_year)])
    base_late = value(model.SupplyCurve[(region, season, tech, step, later_year)])
    retirement_amount = min(base_early, base_late) / 2.0
    assert retirement_amount > 0, 'retirement amount should be positive for the test scenario'

    model.capacity_retirements[(tech, early_year, region, step)].fix(retirement_amount)
    for year in model.year:
        build_idx = (region, tech, year, step)
        if build_idx in model.capacity_builds:
            model.capacity_builds[build_idx].fix(0.0)

    model.capacity_total[(region, season, tech, step, early_year)].fix(
        base_early - retirement_amount
    )
    model.capacity_total[(region, season, tech, step, later_year)].fix(
        base_late - retirement_amount
    )

    early_constraint = model.capacity_balance[(region, season, tech, step, early_year)]
    later_constraint = model.capacity_balance[(region, season, tech, step, later_year)]

    assert value(early_constraint.body) == pytest.approx(0.0)
    assert value(later_constraint.body) == pytest.approx(0.0)
