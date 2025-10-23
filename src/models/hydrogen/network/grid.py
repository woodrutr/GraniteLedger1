"""
GRID CLASS
~~~~~~~~~~

    This is the central class that binds all the other classes together. No class
    instance exists in a reference that isn't fundamentally contained in a grid.
    The grid is used to instantiate a model, read data, create the regionality
    and hub / arc network within that regionality, assign data to objects and more.

    notably, the grid is used to coordinate internal methods in various classes to
    make sure that their combined actions keep the model consistent and accomplish
    the desired task.

"""
###################################################################################################
# Setup

# Import packages and scripts
import time
from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd

from src.models.hydrogen.network.grid_data import GridData
from src.models.hydrogen.network.registry import Registry
from src.models.hydrogen.network.hub import Hub
from src.models.hydrogen.network.region import Region
from src.models.hydrogen.network.transportation_arc import TransportationArc

###################################################################################################


class Grid:
    def __init__(self, data: GridData | None = None):
        """Initializes Grid instance with GridData to generate regions, hubs, arcs, and registry from.

        Parameters
        ----------
        data : GridData | None, optional
            data packet for the grid. Defaults to None.
        """
        if data != None:
            self.data = data
        self.registry: Registry | None = (
            None  # good practice to declare all instance vars in __init__
        )

    def build_grid(self, vis=True):
        """builds a grid fom the GridData by recursively adding regions starting at top-level region
        'world'.

        Parameters
        ----------
        vis : bool, optional
            if True, will generate an image of the hub-network with regional color-coding. Defaults to True.
        """
        self.registry = Registry()
        self.world = Region('world', grid=self, data=self.data.regions)

        self.recursive_region_generation(self.data.regions, self.world)
        self.load_hubs()
        self.arc_generation(self.data.arcs)
        if vis:
            self.visualize()

    def visualize(self):
        """
        visualize the grid network using graphx
        """
        G = nx.DiGraph()
        positions = {}
        for hub in self.registry.hubs.values():
            if hub.region.depth == 1:
                color = 'green'
                size = 100
            elif hub.region.depth == 2:
                color = 'red'
                size = 50

            else:
                color = 'blue'
                size = 30

            G.add_node(hub.name, pos=(hub.x, hub.y), color=color)
            positions[hub.name] = (hub.x, hub.y)
        edges = [arc for arc in self.registry.arcs.keys()]

        G.add_edges_from(edges)

        node_colors = [G.nodes[data]['color'] for data in G.nodes()]

        nx.draw(G, positions, with_labels=False, node_size=50, node_color=node_colors)
        plt.show()

    """
    Creation methods for region, hub, and arc.
    
    All classes should refer to these methods when creating instances so that everything
    is centralized. The methods will have return values so they can also be accessed during creation
    within their class. In some cases, the natural procedure should be to initiate the creation within
    another instance of the class so that the return value can be taken advantage of.
    """

    def create_region(self, name, parent=None, data=None):
        """creates a region with a given name, parent region, and data

        Parameters
        ----------
        name : str
            name of region
        parent : Region, optional
            parent region. Defaults to None.
        data : DataFrame, optional
            region data. Defaults to None.
        """
        if parent == None:
            return self.registry.add_region(Region(name, parent=parent, grid=self, data=data))
        else:
            parent.add_subregion(
                (self.registry.add(Region(name, parent=parent, grid=self, data=data)))
            )

    def create_arc(self, origin, destination, capacity, cost=0.0):
        """Creates and arc from origin to destination with given capacity and cost

        Parameters
        ----------
        origin : str
            origin hub name
        destination : str
            destination hub name
        capacity : float
            capacity of arc
        cost : float, optional
            cost of transporting 1kg H2 along arc. Defaults to 0.
        """
        self.registry.add(TransportationArc(origin, destination, capacity, cost))
        origin.add_outbound(self.registry.arcs[(origin.name, destination.name)])
        destination.add_inbound(self.registry.arcs[(origin.name, destination.name)])

    def create_hub(self, name, region, data=None):
        """creates a hub in a given region

        Parameters
        ----------
        name : str
            hub name
        region : Region
            Region hub is placed in
        data : DataFrame, optional
            dataframe of hub data to append. Defaults to None.
        """
        region.add_hub(self.registry.add(Hub(name, region, data)))

    # delete function (works on arcs, hubs, and regions)

    def delete(self, thing):
        """deletes a hub, arc, or region

        Parameters
        ----------
        thing : Hub, Arc, or Region
            thing to delete
        """
        if type(thing) == Region:
            thing.delete()
            self.registry.remove(thing)

        if type(thing) == Hub:
            for arc in list(thing.outbound.values()):
                self.delete(arc)
            for arc in list(thing.inbound.values()):
                self.delete(arc)

            thing.region.remove_hub(thing)
            self.registry.remove(thing)

        if type(thing) == TransportationArc:
            thing.disconnect()
            self.registry.remove(thing)

    def recursive_region_generation(self, df, parent):
        """cycle through a region dataframe, left column to right until it hits data column, adding
        new regions and subregions according to how it is hierarchically structured. Future versions
        should implement this with a graph structure for the data instead of a dataframe, which
        would be more natural.

        Parameters
        ----------
        df : DataFrame
            hierarchically structured dataframe of regions and their data.
        parent : Region
            Parent region
        """
        if df.columns[0] == 'data':
            for index, row in df.iterrows():
                # print(row[1:])
                parent.update_data(row[1:])
        else:
            for region in df.iloc[:, 0].unique():
                if type(region) is not None:
                    # print(df.columns[0]+':',region)
                    parent.create_subregion(region)
                    self.recursive_region_generation(
                        df[df[df.columns[0]] == region][df.columns[1:]], parent.children[region]
                    )
                elif region == 'None':
                    self.recursive_region_generation(
                        df[df[df.columns[0]].isna()][df.columns[1:]], parent
                    )

                else:
                    self.recursive_region_generation(
                        df[df[df.columns[0]].isna()][df.columns[1:]], parent
                    )

    def arc_generation(self, df):
        """generate arcs from the arc data

        Parameters
        ----------
        df : DataFrame
            arc data
        """
        for index, row in df.iterrows():
            self.create_arc(
                self.registry.hubs[row.origin], self.registry.hubs[row.destination], row['capacity']
            )

    def connect_subregions(self):
        """create an arc for all hubs in bottom-level regions to whatever hub is located in their
        parent region"""
        for hub in self.registry.hubs.values():
            if hub.region.children == {}:
                for parent_hub in hub.region.parent.hubs.values():
                    self.create_arc(hub, parent_hub, 10000000)

    def load_hubs(self):
        """load hubs from data"""
        for index, row in self.data.hubs.iterrows():
            self.create_hub(
                row['hub'],
                self.registry.regions[row['region']],
                data=pd.DataFrame(row[2:]).transpose().reset_index(),
            )

    def aggregate_hubs(self, hublist, region):
        """combine all hubs in hublist into a single hub, and place them in region. Arcs that
        connect to any of these hubs also get aggegated into arcs that connect to the new hub and
        their original origin / destination that's not in hublist.

        Parameters
        ----------
        hublist : list
            list of hubs to aggregate
        region : Region
            region to place them in
        """
        temp_hub_data = pd.concat([hub.data for hub in hublist])
        new_data = pd.DataFrame(columns=self.data.summable['hub'] + self.data.meanable['hub'])

        for column in temp_hub_data.columns:
            if column in self.data.summable['hub']:
                new_data[column] = [temp_hub_data[column].sum()]
            if column in self.data.meanable['hub']:
                new_data[column] = [temp_hub_data[column].mean()]

        name = '_'.join([hub.name for hub in hublist])
        self.create_hub(name, region, new_data)

        inbound = {}
        outbound = {}

        for hub in hublist:
            for arc in hub.inbound.values():
                if arc.origin not in hublist:
                    if arc.origin.name not in inbound.keys():
                        inbound[arc.origin.name] = [arc]
                    else:
                        inbound[arc.origin.name].append(arc)
            for arc in hub.outbound.values():
                if arc.destination not in hublist:
                    if arc.destination.name not in outbound.keys():
                        outbound[arc.destination.name] = [arc]
                    else:
                        outbound[arc.destination.name].append(arc)

        for origin in list(inbound.keys()):
            self.combine_arcs(inbound[origin], self.registry.hubs[origin], self.registry.hubs[name])
        for destination in list(outbound.keys()):
            self.combine_arcs(
                outbound[destination], self.registry.hubs[name], self.registry.hubs[destination]
            )

        del inbound
        del outbound

        for hub in hublist:
            self.delete(hub)

        del hublist

    def combine_arcs(self, arclist, origin, destination):
        """combine a set of arcs into a single arc with given origin and destination

        Parameters
        ----------
        arclist : list
            list of arcs to aggregate
        origin : str
            new origin hub
        destination : str
            new destination hub
        """
        capacity = sum([arc.capacity for arc in arclist])
        cost = sum([arc.cost * arc.capacity for arc in arclist]) / capacity

        self.create_arc(origin, destination, capacity, cost)

        for arc in arclist:
            self.delete(arc)

    def write_data(self):
        """_write data to file"""
        hublist = [hub for hub in list(self.registry.hubs.values())]
        hubdata = pd.concat(
            [
                pd.DataFrame({'hub': [hub.name for hub in hublist]}),
                pd.concat([hub.data for hub in hublist]).reset_index(),
            ],
            axis=1,
        )
        hubdata.to_csv('saveddata.csv', index=False)

        regionlist = [
            region for region in list(self.registry.regions.values()) if not region.data is None
        ]
        regiondata = pd.concat(
            [
                pd.DataFrame({'region': [region.name for region in regionlist]}),
                pd.concat([region.data for region in regionlist], axis=1).transpose().reset_index(),
            ],
            axis=1,
        )
        regiondata = regiondata[
            ['region'] + self.data.summable['region'] + self.data.meanable['region']
        ]
        regiondata.to_csv('regiondatasave.csv', index=False)

        arclist = [arc for arc in list(self.registry.arcs.values())]
        arcdata = pd.DataFrame(
            {
                'origin': [arc.origin.name for arc in arclist],
                'destination': [arc.destination.name for arc in arclist],
                'capacity': [arc.capacity for arc in arclist],
                'cost': [arc.cost for arc in arclist],
            }
        )
        arcdata.to_csv('arcdatasave.csv', index=False)

    def collapse(self, region_name):
        """make a region absorb all it's sub-regions and combine all its and its childrens hubs into one

        Parameters
        ----------
        region_name : str
            region to collapse
        """
        self.registry.regions[region_name].absorb_subregions_deep()
        self.aggregate_hubs(
            list(self.registry.regions[region_name].hubs.values()),
            self.registry.regions[region_name],
        )
        self.registry.update_levels()
        self.visualize()

    def test(self):
        """test run"""
        start = time.time()

        self.build_model()
        self.model.start_build()
        self.model.solve(self.model.m)

        end = time.time()

        print(end - start)

    def collapse_level(self, level):
        """collapse all regions at a specific level of depth in the regional hierarchy, with world = 0

        Parameters
        ----------
        level : int
            level to collapse
        """
        for region in self.registry.depth[level]:
            self.collapse(region)
