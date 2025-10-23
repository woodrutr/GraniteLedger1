"""
HUB CLASS
~~~~~~~~~

class objects are individual hubs, which are fundamental units of production in the model. Hubs
belong to regions, and connect to each other with transportation arcs.
"""
###################################################################################################
# Setup

# Import packages and scripts
import pandas as pd
###################################################################################################


class Hub:
    def __init__(self, name, region, data=None):
        """create a hub in a given region with given data

        Parameters
        ----------
        name : str
            name of hub
        region : Region
            region hub belongs to
        data : DataFrame, optional
            data for the hub. Defaults to None.
        """
        self.name = name
        self.region = region

        if data is not None:
            self.data = data.mask(data.isna(), 0).infer_objects(copy=False)

        # outbound and inbound dictionaries mapping names of hubs to the arc objects
        self.outbound = {}
        self.inbound = {}

        self.x = data.iloc[0]['x']
        self.y = data.iloc[0]['y']

    def change_region(self, new_region):
        """move hub to new region

        Parameters
        ----------
        new_region : Region
            region hub should be moved to
        """
        self.region = new_region
        new_region.add_hub(self)

    def display_outbound(self):
        """print all outbound arcs from hub"""
        for arc in self.outbound.values():
            print('name:', arc.origin.name, 'capacity:', arc.capacity)

    """
    Add and remove arc functions
    
    only modifies itself
    """

    def add_outbound(self, arc):
        """add an outbound arc to hub

        Parameters
        ----------
        arc : Arc
            arc to add
        """
        self.outbound[arc.destination.name] = arc

    def add_inbound(self, arc):
        """add an inbound arc to hub

        Parameters
        ----------
        arc : Arc
            add an inbound arc to hub
        """
        self.inbound[arc.origin.name] = arc

    def remove_outbound(self, arc):
        """remove an outbound arc from hub

        Parameters
        ----------
        arc : Arc
            arc to remove
        """
        del self.outbound[arc.destination.name]

    def remove_inbound(self, arc):
        """remove an inbound arc from hub

        Parameters
        ----------
        arc : Arc
            arc to remove
        """
        del self.inbound[arc.origin.name]

    def get_data(self, quantity):
        """fetch quantity from hub data

        Parameters
        ----------
        quantity : str
            name of data field to fetch

        Returns
        -------
        float or str
            quantity to be fetched
        """
        return self.data.iloc[0][quantity]

    def cost(self, technology, year):
        """return a cost value in terms of data fields

        Parameters
        ----------
        technology : str
            technology type
        year : int
            year

        Returns
        -------
        float
            a cost value
        """

        if technology == 'PEM':
            return self.region.data['electricity_cost'] * 45
        elif technology == 'SMR':
            return self.region.data['gas_cost']
        else:
            return 0
