"""
A gathering of utility functions for dealing with model interconnectivity

Dev Note:  At some review point, some decisions may move these back & forth with parent
models after it is decided if it is a utility job to do .... or a class method.

Additionally, there is probably some renaming due here for consistency
"""

from __future__ import annotations

# Import packages
from collections import defaultdict, namedtuple
from logging import getLogger
import importlib.util
import typing
import pandas as pd
from pathlib import Path
import logging
import os


def _is_module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # pragma: no cover - exercised only when pyomo is absent
        return False


_PYOMO_AVAILABLE = _is_module_available('pyomo.opt') and _is_module_available('pyomo.environ')

if _PYOMO_AVAILABLE:
    import pyomo.opt as pyo  # type: ignore
    from pyomo.environ import ConcreteModel, value  # type: ignore
else:  # pragma: no cover - exercised only when pyomo is absent
    pyo = None  # type: ignore[assignment]
    ConcreteModel = typing.Any  # type: ignore[assignment]

    def value(*args, **kwargs):  # type: ignore[override]
        raise ModuleNotFoundError(
            'Pyomo is required for this functionality. Install the "pyomo" package '
            'to enable optimization features.'
        )

# Import python modules
from main.definitions import PROJECT_ROOT

if typing.TYPE_CHECKING:
    from src.models.electricity.scripts.electricity_model import PowerModel
    from src.models.hydrogen.model.h2_model import H2Model

# Establish logger
logger = getLogger(__name__)


def _require_pyomo() -> None:
    if pyo is None:  # pragma: no cover - executed only when pyomo is unavailable
        raise ModuleNotFoundError(
            'Pyomo is required for this operation. Install the "pyomo" package to '
            'enable optimization features.'
        )


# TODO:  This might be a good use case for a persistent solver (1-each) for both the elec & hyd...  hmm
def simple_solve(m: ConcreteModel):
    """a simple solve routine"""

    _require_pyomo()

    # Note:  this is a prime candidate to split into 2 persistent solvers!!
    # TODO:  experiment with pyomo's persistent solver interface, one for each ELEC, H2
    opt = select_solver(m)
    res = opt.solve(m)
    if pyo.check_optimal_termination(res):
        return
    raise RuntimeError('failed solve in iterator')


def simple_solve_no_opt(m: ConcreteModel, opt: pyo.SolverFactory):
    """Solve concrete model using solver factory object

    Parameters
    ----------
    m : ConcreteModel
        Pyomo model
    opt: SolverFactory
        Solver object initiated prior to solve
    """

    _require_pyomo()

    # Note:  this is a prime candidate to split into 2 persistent solvers!!
    # TODO:  experiment with pyomo's persistent solver interface, one for each ELEC, H2
    logger.info('solving w/ solver-factory object instantiated outside of loop')
    res = opt.solve(m)
    if pyo.check_optimal_termination(res):
        return
    raise RuntimeError('failed solve in iterator')


def select_solver(instance: ConcreteModel):
    """Select solver based on learning method

    Parameters
    ----------
    instance : PowerModel
        electricity pyomo model

    Returns
    -------
    solver type (?)
        The pyomo solver
    """
    _require_pyomo()

    # default = linear solver
    solver_name = 'appsi_highs'
    opt = pyo.SolverFactory(solver_name)
    nonlinear_solver = False

    if hasattr(instance, 'sw_learning'):  # check if sw_learning exists in model (electricity model)
        if instance.sw_learning == 2:  # nonlinear solver
            nonlinear_solver = True
    elif hasattr(instance, 'elec'):  # check if sw_learning exists in meta unified model
        if hasattr(instance.elec, 'sw_learning'):
            if instance.elec.sw_learning == 2:  # nonlinear solver
                nonlinear_solver = True

    if nonlinear_solver:  # if nonlinear learning, set to ipopt
        solver_name = 'ipopt'
        opt = pyo.SolverFactory(solver_name, tee=True)  # , tee=True
        # Select options. The prefix "OF_" tells pyomo to create an options file
        opt.options['OF_mu_strategy'] = 'adaptive'
        opt.options['OF_num_linear_variables'] = 100000
        opt.options['OF_mehrotra_algorithm'] = 'yes'
        # Ask IPOPT to print options so you can confirm that they were used by the solver
        opt.options['print_user_options'] = 'yes'

    logger.info('Using Solver: ' + solver_name)

    return opt


# a named tuple for common electric model index structure (EI=Electrical Index)
EI = namedtuple('EI', ['region', 'year', 'hour'])
"""(region, year, hour)"""
HI = namedtuple('HI', ['region', 'year'])
"""(region, year)"""


def get_elec_price(instance: typing.Union['PowerModel', ConcreteModel], block=None) -> pd.DataFrame:
    """pulls hourly electricity prices from completed PowerModel and de-weights them.

    Prices from the duals are weighted by the day and year weights applied in the OBJ function
    This function retrieves the prices for all hours and removes the day and annual weights to
    return raw prices (and the day weights to use as needed)

    Parameters
    ----------
    instance : PowerModel
        solved electricity model

    block: ConcreteModel
        reference to the block if the electricity model is a block within a larger model

    Returns
    -------
    pd.DataFrame
        df of raw prices and the day weights to re-apply (if needed)
        columns: [r, y, hour, day_weight, raw_price]
    """

    _require_pyomo()

    if block:
        c = block.demand_balance
        model = block
    else:
        c = instance.demand_balance
        model = instance

    # get electricity price duals and de-weight them (costs in the OBJ are up-weighted
    # by the day weight and year weight)
    records = []
    for index in c:
        ei = EI(*index)
        weighted_value = float(instance.dual[c[index]])

        # gather the weights for this hour
        day = model.MapHourDay[ei.hour]
        day_wt = model.WeightDay[day]
        year_wt = model.WeightYear[ei.year]

        # remove the weighting & record
        unweighted_cost = weighted_value / day_wt / year_wt
        records.append((*ei, day_wt, unweighted_cost))

    res = pd.DataFrame.from_records(
        data=records, columns=['region', 'year', 'hour', 'day_weight', 'raw_price']
    )
    return res


def get_annual_wt_avg(elec_price: pd.DataFrame) -> dict[HI, float]:
    """takes annual weighted average of hourly electricity prices

    Parameters
    ----------
    elec_price : pd.DataFrame
        hourly electricity prices

    Returns
    -------
    dict[HI, float]
        annual weighted average electricity prices
    """

    def my_agg(x):
        """aggregate average price based on day weights

        Parameters
        ----------
        x : pd.DataFrame.groupby
            original price frame

        Returns
        -------
        pd.Series
            series containing average price based on day weights
        """
        names = {
            'weighted_ave_price': (x['day_weight'] * x['raw_price']).sum() / x['day_weight'].sum()
        }
        return pd.Series(names, index=['weighted_ave_price'])

    # find annual weighted average, weight by day weights
    elec_price_ann = elec_price.groupby(['region', 'year']).apply(my_agg)

    return elec_price_ann


def regional_annual_prices(
    m: typing.Union['PowerModel', ConcreteModel], block=None
) -> dict[HI, float]:
    """pulls all regional annual weighted electricity prices

    Parameters
    ----------
    m : typing.Union['PowerModel', ConcreteModel]
        solved PowerModel
    block :  optional
        solved block model if applicable, by default None

    Returns
    -------
    dict[HI, float]
        dict with regional annual electricity prices
    """
    ep = get_elec_price(m, block)
    ap = get_annual_wt_avg(ep)

    # convert from dataframe to dictionary
    lut = {}
    for r in ap.to_records():
        region, year, price = r
        lut[HI(region=region, year=year)] = price

    return lut


def convert_elec_price_to_lut(prices: list[tuple[EI, float]]) -> dict[EI, float]:
    """convert electricity prices to dictionary, look up table

    Parameters
    ----------
    prices : list[tuple[EI, float]]
        list of prices

    Returns
    -------
    dict[EI, float]
        dict of prices
    """
    res = {}
    for row in prices:
        ei, price = row
        res[ei] = price
    return res


def poll_hydrogen_price(
    model: typing.Union['H2Model', ConcreteModel], block=None
) -> list[tuple[HI, float]]:
    """Retrieve the price of H2 from the H2 model

    Parameters
    ----------
    model : H2Model
        the model to poll
    block: optional
        block model to poll

    Returns
    -------
    list[tuple[HI, float]]
        list of H2 Index, price tuples
    """
    # ensure valid class
    if not isinstance(model, ConcreteModel):
        raise ValueError('invalid input')

    # TODO:  what should happen if there is no entry for a particular region (no hubs)?
    if block:
        demand_constraint = block.demand_constraint
    else:
        demand_constraint = model.demand_constraint
    # print('************************************\n')
    # print(list(demand_constraint.index_set()))
    # print(list(model.dual.keys()))

    rows = [(HI(*k), model.dual[demand_constraint[k]]) for k, v in demand_constraint.items()]  # type: ignore
    logger.debug('current h2 prices:  %s', rows)
    return rows  # type: ignore


def convert_h2_price_records(records: list[tuple[HI, float]]) -> dict[HI, float]:
    """simple coversion from list of records to a dictionary LUT
    repeat entries should not occur and will generate an error"""
    res = {}
    for hi, price in records:
        if hi in res:
            logger.error('Duplicate index for h2 price received in coversion: %s', hi)
            raise ValueError('duplicate index received see log file.')
        res[hi] = price

    return res


def poll_year_avg_elec_price(price_list: list[tuple[EI, float]]) -> dict[HI, float]:
    """retrieve a REPRESENTATIVE price at the annual level from a listing of prices

    This function computes the AVERAGE elec price for each region-year combo

    Parameters
    ----------
    price_list : list[tuple[EI, float]]
        input price list

    Returns
    -------
    dict[HI, float]
        a dictionary of (region, year): price
    """
    year_region_records = defaultdict(list)
    res = {}
    for ei, price in price_list:
        year_region_records[HI(region=ei.region, year=ei.year)].append(price)

    # now gather the averages...
    for hi in year_region_records:
        res[hi] = sum(year_region_records[hi]) / len(year_region_records[hi])

    logger.debug('Computed these region-year averages for elec price: \n\t %s', res)
    return res


def poll_h2_prices_from_elec(
    model: 'PowerModel', tech, regions: typing.Iterable
) -> dict[typing.Any, float]:
    """poll the step-1 H2 price currently in the model for region/year, averaged over any steps

    Parameters
    ----------
    model : PowerModel
        solved PowerModel
    tech : str
        h2 tech
    regions : typing.Iterable

    Returns
    -------
    dict[typing.Any, float]
        a dictionary of (region, seasons, year): price
    """
    _require_pyomo()

    res = {}
    for idx in model.H2Price:
        r, season, t, step, y = idx
        if t == tech and r in regions and step == 1:  # TODO:  remove hard coding
            res[r, season, y] = value(model.H2Price[idx])

    return res


def update_h2_prices(model: 'PowerModel', h2_prices: dict[HI, float]) -> None:
    """Update the H2 prices held in the model

    Parameters
    ----------
    h2_prices : list[tuple[HI, float]]
        new prices
    """

    # TODO:  Fix this hard-coding below!
    h2_techs = {5}  # temp hard-coding of the tech who's price we're going to set

    update_count = 0
    no_update = set()
    good_updates = set()
    for region, season, tech, step, yr in model.H2Price:  # type: ignore
        if tech in h2_techs:
            if (region, yr) in h2_prices:
                model.H2Price[region, season, tech, step, yr] = h2_prices[
                    HI(region=region, year=yr)
                ]
                update_count += 1
                good_updates.add((region, yr))
            else:
                no_update.add((region, yr))
    logger.debug('Updated %d H2 prices: %s', update_count, good_updates)

    # check for any missing data
    if no_update:
        logger.warning('No new price info for region-year combos: %s', no_update)


def update_elec_demand(self, elec_demand: dict[HI, float]) -> None:
    """
    Update the external electical demand parameter with demands from the H2 model

    Parameters
    ----------
    elec_demand : dict[HI, float]
        the new demands broken out by hyd index (region, year)
    """
    # this is kind of a 1-liner right now, but may evolve into something more elaborate when
    # time scale is tweaked

    self.FixedElecRequest.store_values(elec_demand)
    logger.debug('Stored new fixed electrical request in elec model: %s', elec_demand)


def poll_h2_demand(model: 'PowerModel') -> dict[HI, float]:
    """
    Get the hydrogen demand by rep_year and region

    Use the Generation variable for h2 techs

    NOTE:  Not sure about day weighting calculation here!!

    Returns
    -------
    dict[HI, float]
        dictionary of prices by H2 Index: price
    """

    _require_pyomo()

    h2_consuming_techs = {5}  # TODO:  get rid of this hard-coding

    # gather results
    res: dict[HI, float] = defaultdict(float)
    tot_by_rep_year = defaultdict(float)
    # iterate over the Generation variable and screen out the H2 "demanders"
    # dimensional analysis for H2 demand:
    #
    # Gwh * kg/Gwh = kg
    # so we need 1/heat_rate for kg/Gwh
    for idx in model.generation_total.index_set():
        tech, y, r, step, hr = idx
        if tech in h2_consuming_techs:
            h2_demand_weighted = (
                value(model.generation_total[idx])
                * model.WeightDay[model.MapHourDay[hr]]
                / model.H2Heatrate
            )
            res[HI(region=r, year=y)] += h2_demand_weighted
            tot_by_rep_year[y] += h2_demand_weighted

    logger.debug('Calculated cumulative H2 demand by year as: %s', tot_by_rep_year)
    return res


def create_temporal_mapping(sw_temporal):
    """Combines the input mapping files within the electricity model to create a master temporal
    mapping dataframe. The df is used to build multiple temporal parameters used within the  model.
    It creates a single dataframe that has 8760 rows for each hour in the year.
    Each hour in the year is assigned a season type, day type, and hour type used in the model.
    This defines the number of time periods the model will use based on cw_s_day and cw_hr inputs.

    Returns
    -------
    dataframe
        a dataframe with 8760 rows that include each hour, hour type, day, day type, and season.
        It also includes the weights for each day type and hour type.
    """

    # Temporal Sets - read data
    # SD = season/day; hr = hour
    data_root = Path(PROJECT_ROOT, 'input/integrator')
    if sw_temporal == 'default':
        sd_file = pd.read_csv(data_root / 'cw_s_day.csv')
        hr_file = pd.read_csv(data_root / 'cw_hr.csv')
    else:
        cw_s_day = 'cw_s_day_' + sw_temporal + '.csv'
        cw_hr = 'cw_hr_' + sw_temporal + '.csv'
        sd_file = pd.read_csv(data_root / 'temporal_mapping' / cw_s_day)
        hr_file = pd.read_csv(data_root / 'temporal_mapping' / cw_hr)

    # set up mapping for seasons and days
    df1 = sd_file
    df4 = df1.groupby(by=['Map_day'], as_index=False).count()
    df4 = df4.rename(columns={'Index_day': 'WeightDay'}).drop(columns=['Map_s'])
    df1 = pd.merge(df1, df4, how='left', on=['Map_day'])

    # set up mapping for hours
    df2 = hr_file
    df3 = df2.groupby(by=['Map_hour'], as_index=False).count()
    df3 = df3.rename(columns={'Index_hour': 'WeightHour'})
    df2 = pd.merge(df2, df3, how='left', on=['Map_hour'])

    # combine season, day, and hour mapping
    df = pd.merge(df1, df2, how='cross')
    df['hour'] = df.index
    df['hour'] = df['hour'] + 1
    df['Map_hour'] = (df['Map_day'] - 1) * df['Map_hour'].max() + df['Map_hour']
    # df.to_csv(data_root/'temporal_map.csv',index=False)

    return df
