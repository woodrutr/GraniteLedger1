"""This file is a collection of functions that are used in support of the electricity model."""

###################################################################################################
# Setup

# Import pacakges
from collections import defaultdict
from pathlib import Path
import sys as sys
import pyomo.environ as pyo
import pandas as pd
from src.common.model import Model

###################################################################################################
# TODO: Move this class into a new file?


class ElectricityMethods(Model):
    """a collection of functions used within the electricity model that aid in building the model.

    Parameters
    ----------
    Model : Class
        generic model class
    """

    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)

    # Populate sets functions
    def populate_by_hour_sets_rule(m):
        """Creates new reindexed sets for dispatch_cost calculations

        Parameters
        ----------
        m : PowerModel
            pyomo electricity model instance
        """
        m.StorageHour_index = Model.populate_sets_rule(m, 'Storage_index', set_base_name='hour')
        m.GenHour_index = Model.populate_sets_rule(
            m, 'generation_total_index', set_base_name='hour'
        )
        m.H2GenHour_index = Model.populate_sets_rule(m, 'H2Gen_index', set_base_name='hour')

    def populate_demand_balance_sets_rule(m):
        """Creates new reindexed sets for demand balance constraint

        Parameters
        ----------
        m : PowerModel
            pyomo electricity model instance
        """
        m.GenSetDemandBalance = Model.populate_sets_rule(
            m, 'generation_total_index', set_base2=['year', 'region', 'hour']
        )
        m.StorageSetDemandBalance = Model.populate_sets_rule(
            m, 'Storage_index', set_base2=['year', 'region', 'hour']
        )

        if m.sw_trade == 1:
            m.TradeSetDemandBalance = Model.populate_sets_rule(
                m, 'trade_interregional_index', set_base2=['year', 'region', 'hour']
            )
            m.TradeCanSetDemandBalance = Model.populate_sets_rule(
                m, 'trade_interational_index', set_base2=['year', 'region', 'hour']
            )

    def populate_trade_sets_rule(m):
        """Creates new reindexed sets for trade constraints

        Parameters
        ----------
        m : PowerModel
            pyomo electricity model instance
        """
        m.TradeCanLineSetUpper = Model.populate_sets_rule(
            m, 'trade_interational_index', set_base2=['region', 'region1', 'year', 'hour']
        )
        m.TradeCanSetUpper = Model.populate_sets_rule(
            m, 'trade_interational_index', set_base2=['region1', 'year', 'step', 'hour']
        )

    def populate_RM_sets_rule(m):
        """Creates new reindexed sets for reserve margin constraint

        Parameters
        ----------
        m : PowerModel
            pyomo electricity model instance
        """
        m.SupplyCurveRM = Model.populate_sets_rule(
            m, 'capacity_total_index', set_base2=['year', 'region', 'season']
        )

    def populate_hydro_sets_rule(m):
        """Creates new reindexed sets for hydroelectric generation seasonal upper bound constraint

        Parameters
        ----------
        m : PowerModel
            pyomo electricity model instance
        """
        m.HourSeason_index = pyo.Set(m.season)
        for hr, season in (m.MapHourSeason.extract_values()).items():
            m.HourSeason_index[season].add(hr)

    def populate_reserves_sets_rule(m):
        """Creates new reindexed sets for operating reserves constraints

        Parameters
        ----------
        m : PowerModel
            pyomo electricity model instance
        """
        m.WindSetReserves = defaultdict(list)
        m.SolarSetReserves = defaultdict(list)

        m.ProcurementSetReserves = Model.populate_sets_rule(
            m, 'reserves_procurement_index', set_base2=['restypes', 'region', 'year', 'hour']
        )
        for tech, year, r, step, hour in m.generation_vre_ub_index:
            if tech in m.T_wind:
                m.WindSetReserves[(year, r, hour)].append((tech, step))
            elif tech in m.T_solar:
                m.SolarSetReserves[(year, r, hour)].append((tech, step))


###################################################################################################
# Utility functions THESE STAY HERE


def check_results(results, SolutionStatus, TerminationCondition):
    """Check results for termination condition and solution status

    Parameters
    ----------
    results : str
        Results from pyomo
    SolutionStatus : str
        Solution Status from pyomo
    TerminationCondition : str
        Termination Condition from pyomo

    Returns
    -------
    results
    """
    return (
        (results is None)
        or (len(results.solution) == 0)
        or (results.solution(0).status == SolutionStatus.infeasible)
        or (results.solver.termination_condition == TerminationCondition.infeasible)
        or (results.solver.termination_condition == TerminationCondition.unbounded)
    )


def create_obj_df(mod_object, instance=None):
    """takes pyomo component objects (e.g., variables, parameters, constraints) and processes the
    pyomo data and converts it to a dataframe and then writes the dataframe out to an output dir.
    The dataframe contains a key column which is the original way the pyomo data is structured,
    as well as columns broken out for each set and the final values.

    Parameters
    ----------
    mod_object : pyomo component object
        pyomo component object

    Returns
    -------
    pd.DataFrame
        contains the pyomo model results for the component object
    """
    name = str(mod_object)

    df = pd.DataFrame()

    indices = list(mod_object)
    if not indices and hasattr(mod_object, 'is_indexed') and not mod_object.is_indexed():
        indices = [None]

    df['Key'] = [str(i) for i in indices]

    if not isinstance(mod_object, pyo.Set):
        if hasattr(mod_object, 'is_indexed') and not mod_object.is_indexed():
            df[name] = [pyo.value(mod_object)]
        else:
            df[name] = [pyo.value(mod_object[i]) for i in indices]

    if instance is not None and isinstance(mod_object, pyo.Constraint):
        duals = []
        for idx in indices:
            constraint_data = mod_object if idx is None else mod_object[idx]
            try:
                dual = pyo.value(instance.dual[constraint_data])
            except (KeyError, ValueError):
                dual = float('nan')
            duals.append(dual)
        df['dual'] = duals

    if not df.empty:
        # breaking out the data from the mod_object info into multiple columns
        df['Key'] = df['Key'].str.replace('(', '', regex=False).str.replace(')', '', regex=False)
        temp = df['Key'].str.split(', ', expand=True)
        for col in temp.columns:
            temp.rename(columns={col: 'i_' + str(col)}, inplace=True)
        df = df.join(temp, how='outer')

    return df


def annual_count(hour, m) -> int:
    """return the aggregate weight of this hour in the representative year
    we know the hour weight, and the hours are unique to days, so we can
    get the day weight

    Parameters
    ----------
    hour : int
        the rep_hour

    Returns
    -------
    int
        the aggregate weight (count) of this hour in the rep_year.  NOT the hour weight!
    """
    WeightDay = m.WeightDay[m.MapHourDay[hour]]
    WeightHour = m.WeightHour[hour]
    return WeightDay * WeightHour
