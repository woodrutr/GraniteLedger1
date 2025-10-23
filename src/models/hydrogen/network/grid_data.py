"""
GRIDDATA CLASS
~~~~~~~~~~~~~~

grid_data is the the data object that grids are generated from. It reads in raw data with a region
grid_data is the the data object that grids are generated from. It reads in raw data with a region
filter, and holds it in one structure for easy access
"""
###################################################################################################
# Setup

# Import packages and scripts
from pathlib import Path
import pandas as pd
###################################################################################################


class GridData:
    def __init__(self, data_folder: Path, regions_of_interest: list[str] | None = None):
        """build data object for grid

        Args:
            data_folder (Path): path to data
            regions_of_interest (list[str] | None, optional): list of regions to consider. Defaults to None.
        """
        region_file = data_folder / 'regions.csv'
        hubs_file = data_folder / 'hubs.csv'
        arcs_file = data_folder / 'transportation_arcs.csv'
        params_file = data_folder / 'parameter_list.xlsx'

        self.regions = pd.read_csv(region_file, index_col=False)
        self.hubs = pd.read_csv(hubs_file, index_col=False)
        self.arcs = pd.read_csv(arcs_file, index_col=False)
        # filter regions...
        if regions_of_interest:
            self.hubs = self.hubs[self.hubs['region'].isin(regions_of_interest)]
            if not self.arcs.empty:
                self.arcs = self.arcs[
                    self.arcs['origin'].isin(self.hubs['region'])
                    & self.arcs['destination'].isin(self.hubs['region'])
                ]
            self.regions = self.regions[self.regions['Region'].isin(regions_of_interest)]

        params = pd.ExcelFile(params_file)

        self.hub_params = pd.read_excel(params, 'hub', index_col=False)
        self.region_params = pd.read_excel(params, 'region', index_col=False)
        self.arc_params = pd.read_excel(params, 'arc', index_col=False)
        self.global_params = pd.read_excel(params, 'global', index_col=False)
        self.technologies = [
            column_name.split('_')[2]
            for column_name in self.hubs.columns
            if column_name.lower().startswith('production_capacity')
        ]
        # self.technologies = [self.hubs[hub].data.iloc[0]['H2Capacity_' + tech] for hub in m.hubs for tech in m.technology]

        self.summable = {
            'hub': self.hub_params[self.hub_params['aggregation_type'] == 'sum'][
                'parameter'
            ].tolist(),
            'region': self.region_params[self.region_params['aggregation_type'] == 'sum'][
                'parameter'
            ].tolist(),
            'arc': self.arc_params[self.arc_params['aggregation_type'] == 'sum'][
                'parameter'
            ].tolist(),
        }
        self.meanable = {
            'hub': self.hub_params[self.hub_params['aggregation_type'] == 'mean'][
                'parameter'
            ].tolist(),
            'region': self.region_params[self.region_params['aggregation_type'] == 'mean'][
                'parameter'
            ].tolist(),
            'arc': self.arc_params[self.arc_params['aggregation_type'] == 'mean'][
                'parameter'
            ].tolist(),
        }

        self.fixed_production_cost = {}
        # OM_production_cost = {}

        for technology in self.technologies:
            self.fixed_production_cost[technology] = self.global_params.loc[
                self.global_params.parameter == 'fixed_cost_' + technology
            ].reset_index()['default_value'][0]
