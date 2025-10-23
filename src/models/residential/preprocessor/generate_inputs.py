"""
This file contains the options to re-create the input files. It creates:
 - Load.csv: electricity demand for all model years (used in residential and electricity)
 - BaseElecPrice.csv: electricity prices for initial model year (used in residential only)
Uncomment out the functions at the end of this file in the "if __name__ == '__main__'" statement
in order to generate new load or base electricity prices.

"""

# Import packages
from pathlib import Path
import pandas as pd
import sys

# Set directories
# TODO: import structure is to support running locally, will consider changing
PROJECT_ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
data_root = Path(PROJECT_ROOT, 'input', 'residential')

# Import python modules
from src.common import config_setup
from main import main
from common.regions_schema import iter_region_records


def base_price():
    """Runs the electricity model with base price configuration settings and then
    merges the electricity prices and temporal crosswalk data produced from the run
    to generate base year electricity prices.

    Returns
    -------
    pandas.core.frame.DataFrame
        dataframe that contains base year electricity prices for all regions/hours
    """

    # create settings for base electricity prices
    config_path = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    price_settings = config_setup.Config_settings(config_path, test=True)
    price_settings.run_method = 'run_elec_solo'
    price_settings.sw_temporal = 'bp'
    price_settings.start_year = 2023
    price_settings.years = [2023]
    price_settings.regions = [entry["id"] for entry in iter_region_records()]
    price_settings.sw_trade = 1
    price_settings.sw_rm = 0
    price_settings.sw_expansion = 0
    price_settings.sw_ramp = 0
    price_settings.sw_reserves = 0

    # run electricity model with base price config settings
    main(price_settings)

    # grab electricity model output results
    OUTPUT_ROOT = price_settings.OUTPUT_ROOT
    cw_temporal = pd.read_csv(Path(OUTPUT_ROOT / 'cw_temporal.csv'))
    elec_price = pd.read_csv(Path(OUTPUT_ROOT / 'electricity' / 'prices' / 'elec_price.csv'))

    # keep only the electricity price data needed
    base_year = elec_price['y'].min()
    elec_price = elec_price[elec_price['y'] == base_year]
    elec_price['price_wt'] = elec_price[
        'raw_price'
    ]  # * elec_price['day_weight'] #TODO: day weighting?
    elec_price = elec_price[['region', 'year', 'hour', 'price_wt']].rename(
        columns={'hour': 'Map_hour'}
    )

    # crosswalk the electricity prices to all hours in the base year
    cw_temporal = cw_temporal[['hour', 'Map_hour']]
    elec_price = pd.merge(elec_price, cw_temporal, how='right', on=['Map_hour'])
    elec_price = elec_price.drop(columns=['Map_hour'])
    elec_price = elec_price[['region', 'year', 'hour', 'price_wt']]
    elec_price.sort_values(['region', 'year', 'hour'], inplace=True)

    return elec_price


if __name__ == '__main__':
    base_price().to_csv(data_root / 'BaseElecPrice.csv', index=False)
