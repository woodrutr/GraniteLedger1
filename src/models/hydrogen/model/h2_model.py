"""
The Hydrogen Model takes in a Grid object and uses it to populate a Pyomo model that solves for the
least cost to produce and distribute Hydrogen by electrolysis across the grid to satisfy a given
demand, returning the duals as shadow prices. It can be run in stand-alone or integrated runs. If
stand-alone, a function for generated temporally varying data must be supplied. By default it simply
projects geometric growth for electricity price and demand.
"""
###################################################################################################
# Setup

# Import packages and scripts
from collections import defaultdict
from pyomo.environ import (
    Constraint,
    Var,
    Param,
    Set,
    RangeSet,
    ConcreteModel,
    Suffix,
    Block,
    Objective,
    SolverFactory,
    NonNegativeReals,  # type: ignore
    value,
    check_optimal_termination,
)
from pyomo.core.expr import LinearExpression
from pyomo.opt import SolverResults
import pandas as pd
from logging import getLogger
from src.integrator.utilities import HI
from src.models.hydrogen.network.grid import Grid
import src.models.hydrogen.utilities.h2_functions as h2f

# Establish logger
logger = getLogger(__name__)

###################################################################################################
# MODEL


class H2Model(ConcreteModel):
    def __init__(
        hm,
        grid: Grid,
        mode='standard',
        demand=None,
        electricity_price=None,
        start_year=2025,  # TODO:  These year references will need to move after this is breathing
        end_year=2026,
        years=None,  # a set of years that will match the elec model and supercede start/stop years
        *args,
        **kwargs,
    ):
        """This instantiates the model with given inputs.

        Parameters
        ----------
        hm : H2Model
            self
        grid : Grid
            Grid object that constructs the data input
        mode : str, optional
            marker for mode to run under. Defaults to 'standard'.
        demand : DataFrame, optional
            demand data. Defaults to None.
        electricity_price : DataFrame, optional
            electricity price data. Defaults to None.
        start_year : int, optional
            start year for sequential year run. Defaults to 2025.
        end_year : int, optional
            end year for sequential year run. defaults to 2026
        years list, optional
            set of years to run. Defaults to None.

        Returns
        -------
        H2Model:
            Hydrogen Model

        Raises
        ------
        ValueError
             mode must be in standard or integrated
        """

        ConcreteModel.__init__(hm, *args, **kwargs)
        if mode not in {'standard', 'integrated'}:
            raise ValueError('Illegal mode received')
        hm.grid = grid
        hm.mode = mode

        # TODO:  TBD if we need these...
        if mode == 'integrated':
            hm.demand = demand
            hm.electricity_price = electricity_price

        # ===========
        #     SETS
        # ===========
        hm.hubs = Set(initialize=list(hm.grid.registry.hubs.keys()))
        hm.arcs = Set(initialize=list(hm.grid.registry.arcs.keys()))
        region_names = list(hm.grid.registry.regions.keys())
        hm.regions = Set(initialize=region_names)
        hm.technology = Set(initialize=hm.grid.data.technologies)
        if years:
            year_set = years
        else:  # a range of stop-start
            year_set = list(range(start_year, end_year + 1))
        hm.year = Set(initialize=year_set, ordered=True)
        hm.dual = Suffix(direction=Suffix.IMPORT)

        # ===========
        #   PARAMS
        # ===========

        hm.trans_cost_rate = Param(initialize=0.00)
        hm.trans_capex_rate = Param(initialize=3.0)
        hm.prod_capex_rate = Param(initialize=2.0)

        def get_capacity(hm, hub, tech):
            """fetch base capacity for hub and technology type

            Parameters
            ----------
            hm : H2Model
                self
            hub : str
                name of hub
            tech : str
                technology type

            Returns
            -------
            float
                base capacity
            """

            return hm.grid.registry.hubs[hub].get_data('production_capacity_' + tech)

        hm.capacity = Param(hm.hubs, hm.technology, initialize=get_capacity)

        def get_electricity_consumption_rate(hm: H2Model, tech):
            """get the electricity consumption rate for technology type

            Parameters
            ----------
            hm : H2Model
                self
            tech : str
                technology type

            Returns
            -------
            float
                electricity consumption to produce 1kg of H2
            """
            rates = {
                'PEM': 54.3 * 1e-6,  # GWh/kg
                'SMR': 5.1 * 1e-6,  # GWh/kg
                # TODO:  QA this when SEM has a gas consumption eff as well
            }

            return rates[tech]

        hm.electricity_consumption_rate = Param(
            hm.technology, initialize=get_electricity_consumption_rate
        )

        def get_gas_price(hm: H2Model, region, year):
            """fetch gas price for region, year

            Parameters
            ----------
            hm : H2Model
                self
            region : str
                region
            year : int
                year

            Returns
            -------
            float
                gas price in region, year
            """
            # TODO:  What about year?
            return grid.registry.regions[region].get_data('gas_cost')

        hm.gas_price = Param(
            hm.regions,
            hm.year,
            initialize=get_gas_price,
        )

        # A small minimal production constant to ensure that some H2 is produced
        # dev note:  This is required to ensure that pricing from duals behaves consistently even
        #            when there is no actual demand and only induces a trivial cost
        hm.minimum_production = 1.0  # [kg/h2]

        # mutables:
        hm.demand = Param(
            hm.regions,
            hm.year,
            mutable=True,
            initialize=h2f.get_demand,
        )
        # TODO: per discussion, RN cost is *just* the cost of electricity used for the process,
        # which is very large portion of O&M
        # so the production cost = vol[h2] * elec_consumption_rate[tech] * electricity_price
        hm.electricity_price = Param(
            hm.regions,
            hm.year,
            mutable=True,
            initialize=h2f.get_elec_price,
        )
        hm.cost = Param(
            hm.hubs,
            hm.technology,
            hm.year,
            mutable=True,
            initialize=h2f.get_production_cost,
        )

        # ===========
        #    VARS
        # ===========
        hm.h2_volume = Var(hm.hubs, hm.technology, hm.year, within=NonNegativeReals)
        hm.transportation_volume = Var(hm.arcs, hm.year, within=NonNegativeReals)
        hm.capacity_expansion = Var(
            hm.hubs, hm.technology, hm.year, within=NonNegativeReals, initialize=0.0
        )
        hm.trans_capacity_expansion = Var(hm.arcs, hm.year, within=NonNegativeReals, initialize=0.0)

        # a variable that can be accessed in a meta-model to add to demands...
        hm.var_demand = Var(hm.regions, hm.year, domain=NonNegativeReals)

        # ===========
        # CONSTRAINTS
        # ===========
        def capacity_constraint(hm: H2Model, hub, tech, year):
            """limit capacity to existing + what is built up to and INCLUDING current year

            Parameters
            ----------
            hm : H2Model
                self
            hub : str
                Hub
            tech : str
                technology type
            year : int
                year

            Returns
            -------
            boolean
                production <= capacity + sum of capacity expansion up to current year
            """
            earlier_years = {yr for yr in hm.year if yr <= year}
            return hm.h2_volume[hub, tech, year] <= hm.capacity[hub, tech] + sum(
                hm.capacity_expansion[hub, tech, earlier_year] for earlier_year in earlier_years
            )

        hm.capacity_constraint = Constraint(
            hm.hubs, hm.technology, hm.year, rule=capacity_constraint
        )

        def transportation_capacity_constraint(hm: H2Model, origin, destination, year):
            """max transportation across arc

            Parameters
            ----------
            hm : H2Model
                self
            origin : str
                origin hub
            destination : str
                destination hub
            year : int
                year

            Returns
            -------
            boolean
                transportation <= transportation capacity + capacity expansion up to current year
            """
            earlier_years = {yr for yr in hm.year if yr <= year}
            return hm.transportation_volume[(origin, destination), year] <= hm.grid.registry.arcs[
                (origin, destination)
            ].capacity + sum(
                hm.trans_capacity_expansion[(origin, destination), year] for year in earlier_years
            )

        hm.transportation_constraint = Constraint(
            hm.arcs, hm.year, rule=transportation_capacity_constraint
        )

        def mass_balance(hm: H2Model, hub, year) -> LinearExpression:
            """mass balance at each hub. Production + import - export

            Parameters
            ----------
            hm : H2Model
                self
            hub : str
                hub
            year : int
                year

            Returns
            -------
            LinearExpression
                mass balance for hub in year
            """

            return (
                sum(hm.h2_volume[hub, tech, year] for tech in hm.technology)
                + sum(
                    hm.transportation_volume[arc.name, year]
                    for arc in hm.grid.registry.hubs[hub].inbound.values()
                )
                - sum(
                    hm.transportation_volume[arc.name, year]
                    for arc in hm.grid.registry.hubs[hub].outbound.values()
                )
            )

        def demand_constraint(hm: H2Model, region, year):
            """mass balance must equal demand

            Parameters
            ----------
            hm : H2Model
                model
            region : str
                region
            year : int
                year

            Returns
            -------
            boolean
                sum of mass balance in all hubs in region equals demand
            """
            if len(hm.grid.registry.regions[region].hubs) == 0:
                return Constraint.Skip
            else:
                return (
                    sum(
                        mass_balance(hm, hub, year)
                        for hub in hm.grid.registry.regions[region].hubs.keys()
                    )
                    - hm.demand[region, year]
                    - hm.var_demand[region, year]
                    - hm.minimum_production
                    == 0
                )

        hm.demand_constraint = Constraint(hm.regions, hm.year, rule=demand_constraint)

        # ===========
        #  OBJECTIVE
        # ===========

        # quick lookup table
        region_for_hub = {hub: hm.grid.registry.hubs[hub].region.name for hub in hm.hubs}
        # some convenience expressions...
        hm.production_cost = sum(
            # hm.h2_volume[hub, tech, year] * hm.cost[hub, tech, year]
            # # see note above:  using elec cost only RN.
            hm.h2_volume[hub, tech, year]
            * hm.electricity_consumption_rate[tech]
            * hm.electricity_price[region_for_hub[hub], year]
            for hub in hm.hubs
            for tech in hm.technology
            for year in hm.year
        )

        hm.transportation_cost = sum(
            hm.transportation_volume[arc, year] * hm.trans_cost_rate
            for arc in hm.arcs
            for year in hm.year
        )

        hm.prod_capacity_expansion_cost = sum(
            hm.capacity_expansion[hub, tech, year] * hm.prod_capex_rate
            for hub in hm.hubs
            for tech in hm.technology
            for year in hm.year
        )

        hm.trans_capacity_expansion_cost = sum(
            hm.trans_capacity_expansion[arc, year] * hm.trans_capex_rate
            for arc in hm.arcs
            for year in hm.year
        )

        hm.cost_expression = (
            hm.production_cost
            + hm.transportation_cost
            + hm.prod_capacity_expansion_cost
            + hm.trans_capacity_expansion_cost
        )
        hm.total_cost = Objective(expr=hm.cost_expression)

    def _filter_update_info(hm, data: dict[HI, float]) -> dict[HI, float]:
        """quick filter to remove regions that don't exist in the model

             It is possible (right now) that the H2 network is unaware of particular regions
            because no baseline data for them was ever provided.... so it is possible to
            recieve and "unkown" region here, even though it was selected, due to lack of
            data
        Parameters
        ----------
        hm : H2Model
            self
        data : dict[HI, float]
            hydrogen index : value

        Returns
        -------
        dict[HI, float]
            regions index: value with missing data removed
        """

        res = {k: v for k, v in data.items() if k.region in hm.regions}
        bypass_rate = len(res) / len(data)
        logger.debug('Bypass rate in the H2 parameter filter: %02.f', bypass_rate)
        if bypass_rate < 0.0001:  # basically zero
            logger.warning('Bypass rate in the H2 Parameter filter ZERO.  Likely indexing problem!')
        if bypass_rate < 1.0:
            logger.debug('input data: %s', data)
            logger.debug('result: %s', res)
        return res

    def _update_demand(hm, new_demand):
        """update the demand parameter with new demand data

            insert new demand as a dict in the format: new_demand[region, year]

        Parameters
        ----------
        hm : H2Model
            self
        new_demand : dict
            new demand values
        """
        new_demand = hm._filter_update_info(new_demand)
        new_demand = {k: v for (k, v) in new_demand.items()}
        hm.demand.store_values(new_demand)  # {(region, year): demand}

    def _update_electricity_price(hm, new_electricity_price):
        """update electricity price parameter

        Parameters
        ----------
        hm : H2Model
            self
        new_electricity_price : dict
            region, year : electricity price
        """
        new_electricity_price = hm._filter_update_info(new_electricity_price)
        hm.electricity_price.store_values(new_electricity_price)  # {(region, year): price}

    def update_exchange_params(hm, new_demand=None, new_electricity_price=None):
        """update exchange parameters in integrated mode

        Parameters
        ----------
        hm : H2Model
            model
        new_demand : dict, optional
            new demand (region, year):value. Defaults to None.
        new_electricity_price : dict, optional
            new electricity prices (region,year):value . Defaults to None.
        """
        if new_demand is not None:
            hm._update_demand(new_demand)
        if new_electricity_price is not None:
            hm._update_electricity_price(new_electricity_price)

        hm.cost.store_values(
            {
                (hub, tech, year): h2f.get_production_cost(hm, hub, tech, year)
                for hub in hm.hubs
                for tech in hm.technology
                for year in hm.year
            }
        )

    def poll_electric_demand(hm) -> dict[HI, float]:
        """compute the electrical demand by region-year after solve

        Note:  we will use production * 1/eff to compute electrical demand

        Parameters
        ----------
        hm : H2Model
            self

        Returns
        -------
        dict[HI, float]
            electricity demand by region, year. (region, year):demand
        """
        res: dict[HI, float] = defaultdict(float)
        for idx in hm.h2_volume:
            hub, tech, year = idx
            # gotta dig out the REGION for the hub!
            region = hm.grid.registry.hubs[hub].region
            elec_used = value(hm.h2_volume[idx]) * hm.electricity_consumption_rate[tech]
            hi = HI(region=region.name, year=year)
            res[hi] += elec_used
        logger.debug('Inferred elec usage in H2 mode: %s', res)
        return res


def solve(hm: H2Model):
    """_summary_

    Parameters
    ----------
    hm : H2Model
        self

    Raises
    ------
    RuntimeError
        no optimal solution to problem
    """

    opt = SolverFactory('appsi_highs')
    results: SolverResults = opt.solve(hm)  # , tee=True)
    logger.info(results)

    sol_status = results['Solution'].Status  # peels out the Solution object, Status value

    # logger.info('The solver reported status: %s', sol_status)
    logger.info('Results solver obj: %s', results.solver[0]['Termination condition'])

    if not check_optimal_termination(results=results):
        # stop the train.... non-optimal solve.
        logger.error('non-optimal solve occurred')
        raise RuntimeError('Bad Solve in H2Model!')

    hm.production_stats = pd.DataFrame(
        {
            'region': [
                region for region in hm.regions for tech in hm.technology for year in hm.year
            ],
            'technology': [
                tech for region in hm.regions for tech in hm.technology for year in hm.year
            ],
            'year': [year for region in hm.regions for tech in hm.technology for year in hm.year],
            'volume': [
                sum(
                    value(hm.h2_volume[hub, tech, year])
                    for hub in hm.grid.registry.regions[region].hubs.keys()
                )
                for region in hm.regions
                for tech in hm.technology
                for year in hm.year
            ],
        }
    )

    for tech in hm.technology:
        hm.production_stats['electricity_consumption_' + tech] = (
            hm.production_stats['volume'] * hm.electricity_consumption_rate[tech]
        )

    hm.h2_price = pd.DataFrame(
        {
            'region': [
                hm.grid.registry.hubs[hub].region.name for hub in hm.hubs for year in hm.year
            ],
            'year': [year for hub in hm.hubs for year in hm.year],
            'price': [
                hm.dual[hm.demand_constraint[hm.grid.registry.hubs[hub].region.name, year]]
                for hub in hm.hubs
                for year in hm.year
            ],
        }
    )

    """
    ~~~~~~~~~
    
    relevant quantities:
        
        h2_price
        production_stats
    
    h2_price - a dataframe of h2_prices by region and year
    
    production_stats - a dataframe that shows h2_production quantities by region,
    technology, and year, along with the quantity of electricity consumed at
    each

    ~~~~~~~~~
    """


def resolve(hm: H2Model, new_demand=None, new_electricity_price=None, test=False):
    """For convenience: After building and solving the model initially:

    if you want to solve without annual data by applying a geometric growth rate to exhcange params

    Parameters
    ----------
    hm : H2Model
        model
    new_demand : dict, optional
        new_demand[region,year] for H2demand in (region,year). Defaults to None.
    new_electricity_price : dict, optional
        new_electricity_price[region,year]. Defaults to None.
    test : bool, optional
        is this just a test? Defaults to False.
    """

    if test == True:
        hm.update_exchange_params(
            {
                (region, year): hm.demand[region, year].value * 1.1
                for region in hm.regions
                for year in hm.year
            },
            {
                (region, year): hm.electricity_price[region, year] * 1.1
                for region in hm.regions
                for year in hm.year
            },
        )
        hm.solve(hm.m)
    else:
        hm.update_exchange_params(new_demand, new_electricity_price)

        hm.solve(hm.m)
