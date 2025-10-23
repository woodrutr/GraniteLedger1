"""
A sequencer for actions in the model.
This may change up a bit, but it is a place to assert control of the execution sequence for now
"""

from pathlib import Path
from logging import getLogger
from pyomo.environ import value
import os
import pandas as pd
import pyomo.environ as pyo

from main.definitions import PROJECT_ROOT
from src.models.hydrogen.model.h2_model import solve, H2Model
from src.models.hydrogen.network.grid_data import GridData
from src.models.hydrogen.network.grid import Grid

from pyomo.opt import SolverResults, check_optimal_termination


logger = getLogger(__name__)


def load_data(path_to_input: Path, **kwds) -> GridData:
    """load data for model

    Parameters
    ----------
        path_to_input : Path
            Data folder path

    Returns
    -------
        GridData : obj
            Grid Data object from path
    """
    gd = GridData(data_folder=path_to_input, **kwds)  # default build
    logger.info('Grid Data built.')
    return gd


def build_grid(grid_data: GridData) -> Grid:
    """build a grid from grid_data

    Parameters
    ----------
    grid_data: obj
        GridData object to build grid from

    Returns
    -------
    Grid : obj
        Grid object
    """
    grid = Grid(grid_data)
    grid.build_grid(vis=False)
    logger.info(
        f'Grid built from Data with {len(grid.registry.arcs)} and {len(grid.registry.hubs)} hubs'
    )
    return grid


def build_model(grid: Grid, **kwds) -> H2Model:
    """build model from grd

    Parameters
    ----------
    grid : obj
        Grid object to build model from

    Returns
    -------
    H2Model : obj
        H2Model object
    """
    hm = H2Model(grid, **kwds)
    logger.info('model built')
    return hm


def solve_it(hm: H2Model) -> SolverResults:
    """solve hm

    Parameters
    ----------
    hm : objH2Model
        H2Model to solve

    Returns
    -------
    SolverResults : obj
        results of solve
    """
    res = solve(hm=hm)
    logger.info('model solved')

    return res


def quick_summary(solved_hm: H2Model) -> None:
    """print and return summary of solve

    Parameters
    ----------
    solved_hm : obj
        Solved H2Model

    Returns
    -------
    res : str
        Printed summary
    """
    res = (
        f'********** QUICK H2 SUMMARY *************\n'
        f'  Production Cost: {value(solved_hm.total_cost):0.3f}\n'
        f'  Production Cap Cost: {value(solved_hm.prod_capacity_expansion_cost):0.3f}\n'
        f'  Transpo Cost: {value(solved_hm.transportation_cost):0.3f}\n'
        f'  Transpo Cap Expansion Cost: {value(solved_hm.trans_capacity_expansion_cost):0.3f}\n\n'
        f'  Total Cost: {value(solved_hm.cost_expression):0.3f}'
    )
    print(f'Objective value: {value(solved_hm.cost_expression):0.3f}')

    return res


def make_h2_outputs(output_path, model):
    """save model outputs

    Parameters
    ----------
    model : obj
        Solved H2Model
    """
    OUTPUT_ROOT = output_path
    h2dir = Path(OUTPUT_ROOT / 'hydrogen')
    if not os.path.exists(h2dir):
        os.makedirs(h2dir)
    for variable in model.component_objects(pyo.Var, active=True):
        name = variable.name
        df = pd.DataFrame()
        df['Key'] = [str(i) for i in variable]
        df[name] = [pyo.value(variable[i]) for i in variable]
        df.to_csv(h2dir / f'{name}.csv', index=False)


def run_hydrogen_model(settings):
    """run hydrogen model in standalone

    Parameters
    ----------
    settings : obj
        Config_setup instance
    """
    h2_data_folder = settings.h2_data_folder
    data_path = PROJECT_ROOT / h2_data_folder
    output_path = settings.OUTPUT_ROOT
    grid_data = load_data(data_path, regions_of_interest=settings.regions)
    grid = build_grid(grid_data=grid_data)
    model = build_model(grid=grid, years=settings.years)
    sol = solve_it(model)
    logger.info(quick_summary(model))
    make_h2_outputs(output_path, model)
