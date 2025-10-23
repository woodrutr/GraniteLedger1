"""
A gathering of functions for running models solo

"""

# Import packages
from logging import getLogger
from pathlib import Path
from pyomo.environ import value

# Import python modules
from src.common.config_setup import Config_settings
from src.integrator import utilities
from src.models.electricity.scripts.runner import run_elec_model
from src.models.hydrogen.model.actions import run_hydrogen_model
from src.models.residential.scripts.residential import run_residential
from src.integrator.progress_plot import plot_price_distro

# Establish logger
logger = getLogger(__name__)


def run_elec_solo(settings: Config_settings | None = None):
    """
    Runs electricity model by itself as defined in settings

    Parameters
    ----------
    settings: Config_settings
        Contains configuration settings for which regions, years, and switches to run
    """
    # engage the Electricity Model...
    logger.info('Running Electricity Module')
    instance = run_elec_model(settings)
    print(f'Objective value: {value(instance.total_cost)}')

    # write out prices and plot them
    elec_price = utilities.get_elec_price(instance)
    price_records = utilities.get_annual_wt_avg(elec_price)
    elec_price.to_csv(Path(settings.OUTPUT_ROOT / 'electricity' / 'prices' / 'elec_price.csv'))
    # plot_price_distro(settings.OUTPUT_ROOT, list(elec_price.price_wt))


def run_h2_solo(settings: Config_settings | None = None):
    """
    Runs hydrogen model by itself as defined in settings

    Parameters
    ----------
    settings: Config_settings
        Contains configuration settings for which regions and years to run
    """
    logger.info('Running Hydrogen Module')

    if settings:
        run_hydrogen_model(settings)
    else:
        logger.info('No settings passed to Hydrogen Module')
        empty_settings = object()
        empty_settings.years = None
        empty_settings.regions = None
        empty_settings.h2_data_folder = 'input/hydrogen/single_region'
        run_hydrogen_model(empty_settings)


def run_residential_solo(settings: Config_settings | None = None):
    """
    Runs residential model by itself as defined in settings

    Parameters
    ----------
    settings: Config_settings
        Contains configuration settings for which regions and years to run
    """
    logger.info('Running Residential Module')
    run_residential(settings)


def run_standalone(settings: Config_settings):
    """Runs standalone methods based on settings selections; running 1 or more modules

    Parameters
    ----------
    settings : Config_settings
        Instance of config_settings containing run options, mode and settings
    """
    print('running standalone mode')
    if settings.electricity:
        print('running electricity module')
        run_elec_solo(settings)

    if settings.hydrogen:
        print('running hydrogen module')
        run_h2_solo(settings=settings)

    if settings.residential:
        print('running residential module')
        run_residential_solo(settings)
