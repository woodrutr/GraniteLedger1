"""This file is a collection of functions that are used to build, run, and solve the electricity model."""

###################################################################################################
# Setup

# Import pacakges
from pathlib import Path
import sys as sys
from datetime import datetime
import gc
import pyomo.environ as pyo
from pyomo.common.timing import TicTocTimer
from pyomo.opt import SolutionStatus, SolverStatus, TerminationCondition
from pyomo.util.infeasible import (
    log_infeasible_constraints,
)
from logging import getLogger

# Import python modules
from main.definitions import PROJECT_ROOT
import src.models.electricity.scripts.preprocessor as prep
import src.models.electricity.scripts.postprocessor as post
from src.models.electricity.scripts.utilities import check_results
from src.models.electricity.scripts.electricity_model import PowerModel
from src.common.config_setup import Config_settings

from src.integrator.utilities import select_solver
from io_loader import Frames

# Establish logger
logger = getLogger(__name__)

# Establish paths
data_root = Path(PROJECT_ROOT, 'src/models/electricity/input')


###################################################################################################
# RUN MODEL


def build_elec_model(frames, setin) -> PowerModel:
    """building pyomo electricity model

    Parameters
    ----------
    frames : io_loader.Frames or mapping of str to pandas.DataFrame
        input data frames
    setin : Sets
        input settings Sets

    Returns
    -------
    PowerModel
        built (but unsolved) electricity model
    """
    # Building model
    logger.info('Build Pyomo')
    frames_obj = Frames.coerce(frames)
    instance = PowerModel(frames_obj, setin)

    # add electricity price dual
    instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    instance.carbon_prices = {}
    # instance.pprint()

    # number of variables
    nvar = pyo.value(instance.nvariables())
    logger.info('Number of variables =' + str(nvar))
    # number of constraints
    ncon = pyo.value(instance.nconstraints())
    logger.info('Number of constraints =' + str(ncon))

    return instance


def record_allowance_emission_prices(instance: PowerModel) -> dict:
    """Capture modeled carbon prices from the allowance emissions constraints."""

    carbon_prices = {}
    constraint = getattr(instance, 'allowance_emissions_limit', None)
    if constraint is None:
        instance.carbon_prices = carbon_prices
        return carbon_prices

    if constraint.is_indexed():
        iterator = ((idx, constraint[idx]) for idx in constraint)
    else:
        iterator = [(None, constraint)]

    for idx, constraint_data in iterator:
        try:
            price = -float(instance.dual[constraint_data])
        except (KeyError, ValueError):
            price = 0.0
        carbon_prices[idx] = price

    instance.carbon_prices = carbon_prices
    return carbon_prices

def solve_elec_model(instance):
    """solve electicity model

    Parameters
    ----------
    instance : PowerModel
        built (but not solved) electricity pyomo model
    """

    # select solver
    opt = select_solver(instance)

    logger.info('Solving Pyomo')

    if instance.sw_learning == 1:  # run iterative learning
        # Set any high tolerance
        tol = 999
        i = 0

        # initialize capacity to set pricing
        init_old_cap(instance)
        instance.new_cap = instance.old_cap
        update_cost(instance)

        while tol > 0.1 and i < 20:
            logger.info('Linear iteration number: ' + str(i))

            i += 1
            # solve model
            opt_success = opt.solve(instance)

            # set new capacities
            set_new_cap(instance)

            # Update tolerance
            tol = sum(
                [
                    abs(instance.old_cap_wt[(tech, y)] - instance.new_cap_wt[(tech, y)])
                    for (tech, y) in instance.cap_set
                ]
            )

            # update learning costs in model
            update_cost(instance)

            # update old capacities
            instance.old_cap = instance.new_cap
            instance.old_cap_wt = instance.new_cap_wt

            logger.info('Tolerance: ' + str(tol))
    else:
        opt_success = opt.solve(instance)

    ### Check results and load model solutions
    # Check results for termination condition and solution status
    if check_results(opt_success, SolutionStatus, TerminationCondition):
        name = 'noclass!'
        logger.info(f'[{name}] Solve failed')
        if opt_success is not None:
            logger.info('status=' + str(opt_success.solver.status))
            logger.info('TerminationCondition=' + str(opt_success.solver.termination_condition))

    # If model solved, load model solutions into model, else exit
    try:
        if (opt_success.solver.status == SolverStatus.ok) and (
            opt_success.solver.termination_condition == TerminationCondition.optimal
        ):
            instance.solutions.load_from(opt_success)
        else:
            logger.warning('Solve Failed.')
            exit()
    except:
        logger.warning('Solve Failed.')
        exit()


def run_elec_model(settings: Config_settings, solve=True) -> PowerModel:
    """build electricity model (and solve if solve=True) after passing in settings

    Parameters
    ----------
    settings : Config_settings
        Configuration settings
    solve : bool, optional
        solve electricity model?, by default True

    Returns
    -------
    PowerModel
        electricity model
    """
    ###############################################################################################

    # Measuring the run time of code
    start_time = datetime.now()
    timer = TicTocTimer(logger=logger)
    timer.tic('start')

    ###############################################################################################
    # Pre-processing

    logger.info('Preprocessing')

    frames, setin = prep.preprocessor(prep.Sets(settings))

    logger.debug(
        f'Proceeding to build model for years: {settings.years} and regions: {settings.regions}'
    )
    timer.toc('preprocessor finished')

    ###############################################################################################
    # Build model

    instance = build_elec_model(frames, setin)
    timer.toc('build model finished')

    # stop here if no solve requested...
    if not solve:
        return instance

    ###############################################################################################
    # Solve model
    solve_elec_model(instance)
    record_allowance_emission_prices(instance)

    timer.toc('solve model finished')
    logger.info('Solve complete')

    # save electricity prices for H2 connection
    # component_objects_to_df(instance.)

    # Check
    # Objective Value
    obj_val = pyo.value(instance.total_cost)
    # print('Objective Function Value =',obj_val)

    logger.info('Displaying solution...')
    logger.info(f'instance.total_cost(): {instance.total_cost()}')

    logger.info('Logging infeasible constraints...')
    log_infeasible_constraints(instance, logger=logger)

    logger.info('dispatch cost value =' + str(pyo.value(instance.dispatch_cost)))
    logger.info('unmet load cost value =' + str(pyo.value(instance.unmet_load_cost)))
    if instance.sw_expansion:
        logger.info('cap expansion value =' + str(pyo.value(instance.capacity_expansion_cost)))
        logger.info('fixed om cost value =' + str(pyo.value(instance.fixed_om_cost)))
    if instance.sw_reserves:
        logger.info('op res value =' + str(pyo.value(instance.operating_reserves_cost)))
    if instance.sw_ramp:
        logger.info('ramp cost value =' + str(pyo.value(instance.ramp_cost)))
    if instance.sw_trade:
        logger.info('trade cost value =' + str(pyo.value(instance.trade_cost)))

    logger.info('Obj complete')

    timer.toc('done with checks and extracting vars')

    ###############################################################################################
    # Post-procressing

    if not settings.test:
        post.postprocessor(instance)
    timer.toc('postprocessing done')

    # final steps for measuring the run time of the code
    end_time = datetime.now()
    run_time = end_time - start_time
    timer.toc('finished')
    logger.info(
        '\nStart Time: '
        + datetime.strftime(start_time, '%m/%d/%Y %H:%M')
        + ', Run Time: '
        + str(round(run_time.total_seconds() / 60, 2))
        + ' mins'
    )

    return instance


###################################################################################################
# Support functions


def init_old_cap(instance):
    """initialize capacity for 0th iteration

    Parameters
    ----------
    instance : PowerModel
        unsolved electricity model
    """
    instance.old_cap = {}
    instance.cap_set = []
    instance.old_cap_wt = {}

    for r, tech, y, step in instance.capacity_builds_index:
        if (tech, y) not in instance.old_cap:
            instance.cap_set.append((tech, y))
            # each tech will increase cap by 1 GW per year. reasonable starting point.
            instance.old_cap[(tech, y)] = (y - instance.y0) * 1
            instance.old_cap_wt[(tech, y)] = instance.WeightYear[y] * instance.old_cap[(tech, y)]


def set_new_cap(instance):
    """calculate new capacity after solve iteration

    Parameters
    ----------
    instance : PowerModel
        solved electricity pyomo model
    """
    instance.new_cap = {}
    instance.new_cap_wt = {}
    for r, tech, y, step in instance.capacity_builds_index:
        if (tech, y) not in instance.new_cap:
            instance.new_cap[(tech, y)] = 0.0
        instance.new_cap[(tech, y)] = instance.new_cap[(tech, y)] + sum(
            instance.capacity_builds[(r, tech, year, step)].value for year in instance.y if year < y
        )
        instance.new_cap_wt[(tech, y)] = instance.WeightYear[y] * instance.new_cap[(tech, y)]


def cost_learning_func(instance, tech, y):
    """function for updating learning costs by technology and year

    Parameters
    ----------
    instance : PowerModel
        electricity pyomo model
    tech : int
        technology type
    y : int
        year

    Returns
    -------
    int
        updated capital cost based on learning calculation
    """
    cost = (
        (
            instance.SupplyCurveLearning[tech]
            + 0.0001 * (y - instance.y0)
            + instance.new_cap[tech, y]
        )
        / instance.SupplyCurveLearning[tech]
    ) ** (-1.0 * instance.LearningRate[tech])
    return cost


def update_cost(instance):
    """update capital cost based on new capacity learning

    Parameters
    ----------
    instance : PowerModel
        electricity pyomo model
    """
    new_multiplier = {}
    for tech, y in instance.cap_set:
        new_multiplier[(tech, y)] = cost_learning_func(instance, tech, y)

    new_cost = {}
    # Assign new learning
    for r, tech, y, step in instance.capacity_builds_index:
        # updating learning cost
        new_cost[(r, tech, y, step)] = (
            instance.CapCostInitial[(r, tech, step)] * new_multiplier[tech, y]
        )
        instance.CapCostLearning[(r, tech, y, step)].value = new_cost[(r, tech, y, step)]
