"""
Unifying the solve of both H2 and Elec and Res

Dev Notes:

1. The "annual demand" constraint that is present and INACTIVE is omitted here for clarity.
It may likely be needed - in some form - at a later time. Recall, the key linkages to share the
electrical demand primary variable are:

 - an annual level demand constraint
 - an accurate price-pulling function that can consider weighted duals from both constraints

2. This model has a 2-solve update cycle as commented on near the termination check
 - elec_prices gleaned from cycle[n] results -> solve cycle[n+1]
 - new_load gleaned from cycle[n+1] results -> solve cycle[n+2]
 - elec_pices gleaned from cycle[n+2]

"""

# Import packages
from collections import defaultdict, deque
from logging import getLogger
import pyomo.environ as pyo

# Import python modules
from src.common.config_setup import Config_settings
from src.integrator.utilities import (
    HI,
    EI,
    get_elec_price,
    convert_elec_price_to_lut,
    convert_h2_price_records,
    regional_annual_prices,
    poll_hydrogen_price,
    simple_solve_no_opt,
    select_solver,
    simple_solve,
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


def run_unified(settings: Config_settings):
    """Runs unified solve method based on

    Parameters
    ----------
    settings : Config_settings
        Instance of config_settings containing run options, mode and settings
    """
    #####
    ### run_unified - SETUP
    #####

    # run_unified - SETUP: some data gathering
    h2_price_records = []
    elec_price_records = []
    h2_obj_records = []
    elec_obj_records = []
    h2_demand_records = []
    elec_demand_records = []
    load_records = []

    elec_price_to_res_records = []
    i = 0

    # run_unified - SETUP: Unpack Settings
    force_10 = settings.force_10  # force at least 10 iterations
    tol = settings.tol  # relative change
    max_iter = settings.max_iter  # max number of iterations
    years = settings.years
    regions = settings.regions

    #####
    ### run_unified - INIT: Initialize models
    #####

    ### run_unified - INIT: Meta model
    meta = pyo.ConcreteModel('meta')

    ### run_unified - INIT: h2
    if settings.hydrogen:
        logger.info('Making H2 Model')
        grid_data = actions.load_data(settings.h2_data_folder, regions_of_interest=regions)
        grid = actions.build_grid(grid_data=grid_data)
        h2_model = actions.build_model(grid=grid, years=years)
        meta.h2 = h2_model

    ### run_unified - INIT: initialize elec model
    if settings.electricity:
        logger.info('Making Elec')
        elec_model = run_elec_model(settings, solve=False)
        meta.elec = elec_model
        meta.elec_price = pyo.Param(
            elec_model.demand_balance_index, initialize=0, default=0, mutable=True
        )

    ### run_unified - INIT: Initialize res w/ block factory
    if settings.residential:
        res_block_factory = residentialModule(settings=settings)
        meta.res = pyo.Block()  # an empty placeholder to facilitate iterative syntax

    #####
    ### run_unified - INTEGRATE
    #####

    # Dev Note:  The shared primal variable for demand in the Elec model is
    #            not yet linked up.

    ### run_unified - INTEGRATE: Link h2 and electricity
    if settings.hydrogen and settings.electricity:
        # zero out the fixed demand from original inputs
        for idx in h2_model.demand:
            h2_model.demand[idx] = 0.0

        h2_consuming_techs = {5}

        # gather results
        h2_demand_equations_from_elec: dict[HI, float] = defaultdict(float)
        # iterate over the Generation variable and screen out the H2 "demanders" and build a
        # summary expression for all the demands in the (region, year) index of the H2 model

        # see note in Elec Model function for polling H2 regarding units of H2Heatrate
        for idx in elec_model.generation_total.index_set():
            tech, y, r, _, hr = idx
            if tech in h2_consuming_techs:
                h2_demand_weighted = (
                    elec_model.generation_total[idx]
                    * elec_model.WeightDay[elec_model.MapHourDay[hr]]
                    / elec_model.H2Heatrate
                )
                h2_demand_equations_from_elec[HI(region=r, year=y)] += h2_demand_weighted

        def link_h2_demand(meta, r, y):
            """Link h2 demand from h2 model and elec together with a constraint

            Parameters
            ----------
            meta : pyo.ConcreteModel
                solved model with h2
            r : list[int]
                regions list
            y : list[int]
                years list

            Returns
            -------
            pyo.expr.relational_expr.InequalityExpression
                inequality to add to constraint for h2 demand
            """
            return h2_model.var_demand[r, y] >= h2_demand_equations_from_elec[HI(r, y)]

        meta.link_h2_demand = pyo.Constraint(h2_model.regions, h2_model.year, rule=link_h2_demand)

    # Setting up meta objective function and duals
    if settings.electricity and settings.hydrogen:
        h2_model.total_cost.deactivate()
        elec_model.total_cost.deactivate()
        meta.obj = pyo.Objective(expr=h2_model.total_cost + elec_model.total_cost)
        meta.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    elif settings.electricity:
        elec_model.total_cost.deactivate()
        meta.obj = pyo.Objective(expr=elec_model.total_cost)
        meta.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    #####
    ### run_unified - UNI_LOOP: setup and execute iterations between modules
    #####

    done = False  # keep the last 3 obj values in a deque
    meta_objs_elec = deque([], maxlen=3)
    meta_objs_res = deque([], maxlen=3)
    meta_objs_hyd = deque([], maxlen=3)

    # run_unified - UNI_LOOP: Initialize iterative learning
    if settings.electricity:
        if meta.elec.sw_learning == 1:  # initializing iterative learning
            # initialize capacity to set pricing
            init_old_cap(meta.elec)
            meta.elec.new_cap = meta.elec.old_cap
            update_cost(meta.elec)

    # run_unified - UNI_LOOP: If no res, set-up persistent solver
    if not settings.residential:
        opt = select_solver(meta)

    while not done:
        logger.info('Starting iteration: %d', i)

        # run_unified - UNI_LOOP: Solve joint meta-model (with persistent if feasible)
        if not settings.residential:
            simple_solve_no_opt(m=meta, opt=opt)
        else:
            simple_solve(meta)

        # run_unified - UNI_LOOP: Steps
        # solving this produces 4 things we need to propogate:
        # prices of elec -> res
        # prices of elec -> H2
        # prices of H2 -> elec
        # new Load -> elec

        # catch the new Load metric from res...
        if settings.electricity and settings.residential:
            if i > 0:
                meta.elec.Load.store_values(meta.res.Load.extract_values())
                # meta.res.Load.pprint()
                # sys.exit(-1)

        if settings.electricity:
            # we can still poll the elec load used from the elec model in all iterations which
            # will show the value used in the next iteration with/without the 0th update above
            agg_load = sum(pyo.value(meta.elec.Load[idx]) for idx in meta.elec.Load)  # type: ignore
            load_records.append((i + 1, agg_load))

            # dev note:  the res block is now depleted... we have extracted the new load values

            # run_unified - UNI_LOOP: Attach a new res block
            # Note for the 0th solve there will be no res model, as a price is needed to
            # initialize it, but for iter 1, ..., N there will be a res model in the
            # solve above
            prices = get_elec_price(meta, block=meta.elec)
            prices = prices.set_index(['region', 'year', 'hour'])['raw_price'].to_dict()
            prices = [(EI(*k), prices[k]) for k, v in prices.items()]

            # dev note:  we must use this because the Res model needs (r, yr, hr) not just (r, yr)!
            price_lut = convert_elec_price_to_lut(prices=prices)

            # cannot have zero prices, so a quick interim check
            for idx, price in price_lut.items():
                assert price > 0, f'found a bad apple {idx}'
            meta.elec_price.store_values(price_lut)

        #####
        ### run_unified - ELEC->RES: Pass in Elec Prices to res
        #####

        if settings.electricity and settings.residential:
            res_block = res_block_factory.make_block(meta.elec_price, meta.elec_price.index_set())
            meta.del_component('res')
            meta.res = res_block

        if settings.electricity:
            grand_av_price = sum(pyo.value(meta.elec_price[idx]) for idx in meta.elec_price) / len(
                meta.elec_price
            )
            elec_price_to_res_records.append((i + 1, grand_av_price))
            logger.info('grand avg elec price told to res: %0.2f', grand_av_price)

        #####
        ### run_unified - H2<->ELEC: catch metrics from H2 Model and ELEC
        #####

        if settings.electricity and settings.hydrogen:
            meta.del_component(h2_demand_equations_from_elec)

            # (on hold per note at top...  although we will catch Load from res, which should
            # be equivalent)
            h2_obj_records.append((i, pyo.value(h2_model.total_cost())))
            new_h2_prices = poll_hydrogen_price(meta, block=meta.h2)
            avg_hyd_price = sum(t[1] for t in new_h2_prices) / len(new_h2_prices)
            h2_price_records.append((i + 1, avg_hyd_price))

            # catch metrics from Elec
            tot_h2_demand = sum(poll_h2_demand(elec_model).values())
            h2_demand_records.append((i + 1, tot_h2_demand))
            logger.debug('Tot H2 Demand for iteration %d: %0.2f', i, tot_h2_demand)
            elec_obj_records.append((i, pyo.value(elec_model.total_cost)))

        if settings.electricity:
            rap = regional_annual_prices(meta, block=meta.elec)
            grand_avg = sum(rap.values()) / len(rap)
            elec_price_records.append((i + 1, grand_avg))

        #####
        ### run_unified - H2<->ELEC: Info Swap
        #####

        if settings.electricity and settings.hydrogen:
            update_h2_prices(elec_model, h2_prices=convert_h2_price_records(new_h2_prices))
            # Elec prices
            h2_model.update_exchange_params(new_electricity_price=rap)

        ### run_unfied - ELEC: Update electricity
        if settings.electricity:
            if meta.elec.sw_learning == 1:  # iterative learning update
                # set new capacities
                set_new_cap(meta.elec)
                # update learning costs in model
                update_cost(meta.elec)
                # update old capacities
                meta.elec.old_cap = meta.elec.new_cap
                meta.elec.old_cap_wt = meta.elec.new_cap_wt

        #####
        ### run_unified - TERM: check termination
        #####
        if settings.electricity:
            elec_obj = round(pyo.value(elec_model.total_cost), 2)
            meta_objs_elec.append(elec_obj)

        if settings.hydrogen:
            h2_obj = round(pyo.value(h2_model.total_cost), 2)
            meta_objs_hyd.append(h2_obj)

        if settings.residential:
            res_obj = round(pyo.value(agg_load), 2)
            meta_objs_res.append(res_obj)

        def under_tolerance() -> bool:
            """check tolerance for elec, hyd, and res modules

            Returns
            -------
            bool
                is objective change less than tolerance?
            """
            obj_change = 0
            if settings.electricity:
                max_recent_elec = max(meta_objs_elec)
                min_recent_elec = min(meta_objs_elec)
                if min_recent_elec == 0:
                    min_recent_elec = 0.0001
                obj_change += abs(max_recent_elec - min_recent_elec) / min_recent_elec

            if settings.residential:
                max_recent_res = max(meta_objs_res)
                min_recent_res = min(meta_objs_res)
                obj_change += abs(max_recent_res - min_recent_res) / min_recent_res

            if settings.hydrogen:
                max_recent_hyd = max(meta_objs_hyd)
                min_recent_hyd = min(meta_objs_hyd)
                if max_recent_hyd == 0 and min_recent_hyd == 0:
                    max_recent_hyd = 0.0001
                if min_recent_hyd == 0:
                    min_recent_hyd = 0.0001
                obj_change += abs(max_recent_hyd - min_recent_hyd) / min_recent_hyd
            return (obj_change) < tol

        if i < 2:  # we must force at least 3 iterations
            # print('iter < 2')
            done = False
        elif under_tolerance() and not (force_10 and i < 10):
            # print('under tolerance')
            done = True
        elif i > max_iter:
            print('iter > max_iter')
            done = True
            logger.warning(f'Terminating iterative solve based on iteration count > {max_iter}!')
        else:
            # print('keep going')
            done = False

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
        if settings.hydrogen:
            logger.debug('current h2 obj:  %.2e', h2_obj)
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
    post_elec.postprocessor(meta.elec)
