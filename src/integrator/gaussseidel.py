"""
Iteratively solve 2 models with GS methodology

see README for process explanation

"""

# Import packages
from logging import getLogger
from pathlib import Path
import pyomo.environ as pyo
import pandas as pd
from collections import namedtuple

# Import python modules
from main.definitions import PROJECT_ROOT
from src.integrator.progress_plot import plot_it
from src.integrator.utilities import (
    EI,
    convert_elec_price_to_lut,
    convert_h2_price_records,
    regional_annual_prices,
    poll_h2_prices_from_elec,
    poll_hydrogen_price,
    get_elec_price,
    simple_solve,
    simple_solve_no_opt,
    select_solver,
    poll_h2_demand,
    update_h2_prices,
)
from src.models.electricity.scripts.runner import (
    run_elec_model,
    init_old_cap,
    update_cost,
    set_new_cap,
)
from src.models.hydrogen.model import actions
from src.models.residential.scripts.residential import residentialModule
import src.models.electricity.scripts.postprocessor as post_elec

# Establish logger
logger = getLogger(__name__)


def run_gs(settings):
    """Start the iterative GS process

    Parameters
    ----------
    settings : obj
        Config_settings object that holds module choices and settings
    """
    ###
    # run_gs - SETUP
    ###

    # run_gs - SETUP: data gathering
    h2_price_records = []
    elec_price_records = []
    h2_obj_records = []
    elec_obj_records = []
    h2_demand_records = []
    elec_demand_records = []
    load_records = []
    elec_price_to_res_records = []
    i = 0

    # run_gs - SETUP: pull settings from config instance
    force_10 = settings.force_10  # basically force 10 solves
    tol = settings.tol  # relative tolerance
    max_iter = settings.max_iter  # max number of iterations
    years = settings.years
    regions = settings.regions

    # run_gs - SETUP: identify models to run from config instance
    update_elec_price = settings.method_options['update_elec_price']
    update_h2_price = settings.method_options['update_h2_price']
    update_h2_demand = settings.method_options['update_h2_demand']
    update_load = settings.method_options['update_load']

    logger.info(
        f'Starting Iterative Run: electricity: {settings.electricity}, hydrogen: {settings.hydrogen}, residential: {settings.residential}'
    )

    #####
    ### run_gs - H2_START: Hydrogen starting demands
    #####
    # ---------------------
    #####
    ### run_gs - ELEC_INIT: Initialize ELEC model
    #####

    if settings.electricity:
        logger.info('Making ELEC Model')
        elec_model = run_elec_model(settings, solve=False)

        # run_gs - ELEC_INIT: check the "original" load
        agg_load = sum(pyo.value(elec_model.Load[idx]) for idx in elec_model.Load)  # type: ignore
        load_records.append((0, agg_load))

        if settings.sw_learning == 1:  # initializing iterative learning
            # initialize capacity to set pricing
            init_old_cap(elec_model)
            elec_model.new_cap = elec_model.old_cap
            update_cost(elec_model)

        # run_gs - ELEC_INIT: initialize persistent solver object
        opt_elec = select_solver(elec_model)

    #####
    ### run_gs - H2_INIT: If required, initialize H2 Model
    #####

    if settings.hydrogen:
        logger.info('Making/loading H2 model')
        grid_data = actions.load_data(settings.h2_data_folder, regions_of_interest=regions)
        grid = actions.build_grid(grid_data=grid_data)
        h2_model = actions.build_model(grid=grid, years=years)

        # run_gs - H2_INIT: Create persistent solver objects for modules
        opt_h2 = select_solver(h2_model)

    #####
    ### run_gs - GS_LOOP: Iteratively solve each module until exit condition met
    #####

    # run_gs - GS_LOOP: Set initial objective sums for each module (not all used)
    done = False
    old_obj_sum = 1
    old_res_obj_sum = 1
    old_elec_obj_sum = 1
    old_h2_obj_sum = 1

    # run_GS - GS_LOOP: Begin while loop to solve gs
    while not done:
        #####
        ### run_gs - ELEC: ELEC model solve
        #####
        if settings.electricity:
            simple_solve_no_opt(m=elec_model, opt=opt_elec)

            # run_gs - GS_LOOP-Elec: Pull objective function and append to records
            e_obj = pyo.value(elec_model.total_cost)
            logger.info('i %d Elec Obj: %0.2f', i, e_obj)
            elec_obj_records.append((i, e_obj))

        #####
        ### run_gs - ELEC->H2: Hydrogen model solve and updates
        #####
        if settings.hydrogen:
            # run_gs - ELEC->H2: Pull metrics from elec
            if update_h2_demand:
                h2_demand = poll_h2_demand(elec_model)
                tot_h2_demand = sum(poll_h2_demand(elec_model).values())
                h2_demand_records.append((i + 1, tot_h2_demand))
            else:
                h2_demand = None

            # run_gs - ELEC->H2: Pull regional annual prices to pass to hydrogen?
            if update_elec_price:
                rap = regional_annual_prices(elec_model)
                annual_avg = sum(rap.values()) / len(rap)
                # avg_elec_price = sum(t[1] for t in new_elec_prices) / len(new_elec_prices)
                grand_avg = sum(rap.values()) / len(rap)
                elec_price_records.append((i + 1, grand_avg))
            else:
                rap = None

            # run_gs - ELEC->H2: update parameters in hydrogen price and demand
            if rap:
                h2_model.update_exchange_params(new_electricity_price=rap)
            if h2_demand:
                h2_model.update_exchange_params(new_demand=h2_demand)

        #####
        ### run_gs - H2: solve h2 model and save objective
        #####
        if settings.hydrogen:
            simple_solve_no_opt(m=h2_model, opt=opt_h2)
            h2_obj = pyo.value(h2_model.total_cost)
            h2_obj_records.append((i, h2_obj))
            logger.info('Iter %d H2 Obj: %0.2f', i, h2_obj)

            # run_gs - H2: some logging of the iteration...
            h2_consumption_data = poll_h2_demand(elec_model)
            logger.debug('h2 consumption: %s', h2_consumption_data)
            logger.debug(
                'Actual h2 prices used in last iteration:\n %s',
                poll_h2_prices_from_elec(model=elec_model, tech=5, regions=(7,)),
            )

        #####
        ### run_GS - RES: Residential model initialization and solve
        #####

        if settings.residential:
            # run_GS - RES: currently we need a "meta" model -- basically a pass-through
            meta = pyo.ConcreteModel()
            meta.elec_price = pyo.Param(
                elec_model.demand_balance_index, initialize=0, default=0, mutable=True
            )

            # run_GS - RES: Pull prices (duals) from electricity model
            prices = get_elec_price(elec_model)
            prices = prices.set_index(['region', 'year', 'hour'])['raw_price'].to_dict()
            prices = [(EI(*k), prices[k]) for k, v in prices.items()]

            # run_GS - RES:  we must use this because the Res model needs
            # (r, yr, hr) not just (r, yr)!
            price_lut = convert_elec_price_to_lut(prices=prices)

            # run_GS - RES: cannot have zero prices, so a quick interim check
            for idx, price in price_lut.items():
                assert price > 0, f'found a bad apple {idx}'
            meta.elec_price.store_values(price_lut)

            # run_GS - RES: Initialize residential model and pass prices
            res_model = residentialModule(settings=settings)

            blk = res_model.make_block(meta.elec_price, meta.elec_price.index_set())
            # record the price reported to Elec
            grand_av_price = sum(pyo.value(meta.elec_price[idx]) for idx in meta.elec_price) / len(
                meta.elec_price
            )
            elec_price_to_res_records.append((i + 1, grand_av_price))
            logger.info('grand avg elec price told to res: %0.2f', grand_av_price)

            # run_GS - RES: Solve residential model
            # now we have a single constraint in the block, properly constraining the Load var
            # add this block to the meta model so that we can "solve it"
            # TODO:  this is going to cause warnings in pyomo by replacing a named component
            #        need to modify to not make a new residential block, just mod what we have
            meta.blk = blk

            # run_GS - RES: Neet to solve to enforce the constraint and set the variable...
            meta.obj = pyo.Objective(expr=0)  #   a constant to avoid solver warning
            simple_solve(meta)

        ###
        # run_GS - RES->ELEC: Residential updating into Elec model
        ###

        if settings.residential:
            # run_gs - RES->ELEC:  now the meta.blk variable "Load" contains new load requests
            # that can be inspected... put them in the elec model parameter (update the mutable param)
            if update_load:
                elec_model.Load.store_values(meta.blk.Load.extract_values())
            agg_load = sum(pyo.value(elec_model.Load[idx]) for idx in elec_model.Load)  # type: ignore
            load_records.append((i + 1, agg_load))

        #####
        ### run_gs - H2->Elec: Update the Elec model H2 prices
        #####

        if settings.hydrogen:
            if update_h2_price:
                new_h2_prices = poll_hydrogen_price(h2_model)
                avg_hyd_price = sum(t[1] for t in new_h2_prices) / len(new_h2_prices)
                h2_price_records.append((i + 1, avg_hyd_price))
            else:
                new_h2_prices = None
            if new_h2_prices:
                update_h2_prices(elec_model, h2_prices=convert_h2_price_records(new_h2_prices))

            # run_gs - H2->Elec: Placeholder for demand from H2

        #####
        ### run_gs - ELEC_Learning: Update capital costs in learning if sw_learning
        #####
        if settings.electricity:
            if elec_model.sw_learning == 1:  # iterative learning update
                # set new capacities
                set_new_cap(elec_model)
                # update learning costs in model
                update_cost(elec_model)
                # update old capacities
                elec_model.old_cap = elec_model.new_cap
                elec_model.old_cap_wt = elec_model.new_cap_wt

        #####
        ### run_gs - TERM: Check termination criteria at end of loop
        #####

        # run_gs - TERM: Pull objective values in current iteration
        obj_change = 0
        if settings.electricity:
            elec_obj = round(abs(e_obj), 2)
            elec_obj_chg = abs(elec_obj - old_elec_obj_sum) / old_elec_obj_sum
            obj_change += elec_obj_chg

        if settings.hydrogen:
            h2_obj = round(abs(h2_obj), 2)
            h2_obj_chg = abs(h2_obj - old_h2_obj_sum) / old_h2_obj_sum
            obj_change += h2_obj_chg

        if settings.residential:
            res_obj = round(abs(agg_load), 2)
            res_obj_chg = abs(res_obj - old_res_obj_sum) / old_res_obj_sum
            obj_change += res_obj_chg

        if i == 0:
            if settings.electricity:
                old_elec_obj_sum = elec_obj
            if settings.hydrogen:
                old_h2_obj_sum = h2_obj
                if old_h2_obj_sum == 0:
                    old_h2_obj_sum = 0.00001
            if settings.residential:
                old_res_obj_sum = res_obj
            done = False
        else:
            if obj_change < tol and not (force_10 and i < 10):
                # print('under tolerance')
                done = True
            elif i > max_iter:
                print('iter > max_iter')
                done = True
                logger.warning(
                    f'Terminating iterative solve based on iteration count > {max_iter}!'
                )
            else:
                # print('keep going')
                if settings.electricity:
                    old_elec_obj_sum = elec_obj
                if settings.hydrogen:
                    old_h2_obj_sum = h2_obj
                    if old_h2_obj_sum == 0:
                        old_h2_obj_sum = 0.00001
                if settings.residential:
                    old_res_obj_sum = res_obj

        #####
        ### run_gs - PRINT: printing to console when completed w/ info on objs
        #####

        statement = f'Finished Iteration {i}:\n'
        if settings.electricity:
            statement += f'\t Electricity Objective: {elec_obj:0.2f}\n'
        if settings.hydrogen:
            statement += f'\t Hydrogen Objective: {h2_obj:0.2f}\n'
        if settings.residential:
            statement += f"\t Residential 'Load': {res_obj:0.2f}"
        print(statement)

        statement_logger = statement.replace('\n', '').replace('\t', '')
        logger.info(f'Completed iteration {i}: {statement_logger}')
        i += 1

    # post processing reporting
    # plot_it(
    #     settings.OUTPUT_ROOT,
    #     h2_price_records=h2_price_records,
    #     elec_price_records=elec_price_records,
    #     h2_obj_records=h2_obj_records,
    #     elec_obj_records=elec_obj_records,
    #     h2_demand_records=h2_demand_records,
    #     elec_demand_records=elec_demand_records,
    #     load_records=load_records,
    #     elec_price_to_res_records=elec_price_to_res_records,
    # )

    post_elec.postprocessor(elec_model)
