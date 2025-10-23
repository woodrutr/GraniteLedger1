"""
REGION CLASS
~~~~~~~~~~~~

Class objects are regions, which have a natural tree-structure. Each region can have a parent region
and child regions (subregions), a data object, and a set of hubs.
"""

###################################################################################################
# Setup
import pandas as pd
from logging import getLogger
###################################################################################################

logger = getLogger(__name__)


class Region:
    assigned_names = set()

    def __init__(self, name, grid=None, kind=None, data=None, parent=None):
        """initialize region

        Parameters
        ----------
        name : str
            region name
        grid : Grid, optional
            grid region belongs to. Defaults to None.
        kind : str, optional
            marker for what kind of region. Defaults to None.
        data : DataFrame, optional
            data for region. Defaults to None.
        parent : Region, optional
            parent region. Defaults to None.

        Raises
        ------
        ValueError: name is NoneType
        """
        # check name for uniqueness
        if not name:
            raise ValueError('name cannot be None')
        if name in Region.assigned_names:
            logger.warning(f'region name {name} already exists')
        Region.assigned_names.add(name)
        self.name = name
        self.parent = parent
        self.children = {}
        self.hubs = {}
        self.data = data

        if self.parent is not None:
            self.depth = self.parent.depth + 1
            self.grid = parent.grid

        else:
            self.depth = 0
            self.grid = grid

    def display_children(self):
        """display child regions"""
        for child in self.children.values():
            print(child.name, child.depth)
            child.display_children()

    def display_hubs(self):
        """display hubs"""
        for hub in self.hubs.values():
            print(hub.name)

    def update_parent(self, new_parent):
        """change parent region

        Parameters
        ----------
        new_parent : Region
            new parent region
        """
        if self.parent is not None:
            del self.parent.children[self.name]
            self.parent = new_parent
            self.parent.add_subregion(self)
            self.depth = new_parent.depth + 1

        else:
            self.parent = new_parent
            self.parent.add_subregion(self)

    def create_subregion(self, name, data=None):
        """create a subregion

        Parameters
        ----------
        name : str
            subregion name
        data : DataFrame, optional
            subregion data. Defaults to None.
        """
        self.grid.create_region(name, self, data)

    def add_subregion(self, subregion):
        """make a region a subregion of self

        Parameters
        ----------
        subregion : Region
            new subregion
        """
        self.children.update({subregion.name: subregion})

    def remove_subregion(self, subregion):
        """remove a subregion from self

        Parameters
        ----------
        subregion : Region
            subregion to remove
        """
        self.children.pop(subregion.name)

    def add_hub(self, hub):
        """add a hub to region

        Parameters
        ----------
        hub : Hub
            hub to add
        """
        self.hubs.update({hub.name: hub})

    def remove_hub(self, hub):
        """remove hub from region

        Parameters
        ----------
        hub : Hub
            hub to remove
        """
        del self.hubs[hub.name]

    def delete(self):
        """delete self, reassign hubs to parent, reassign children to parent"""
        for hub in self.hubs.values():
            hub.change_region(self.parent)

        for child in list(self.children.values()):
            child.update_parent(self.parent)
            self.parent.add_subregion(child)

        if self.name in self.parent.children.keys():
            self.parent.remove_subregion(self)

    def absorb_subregions(self):
        """delete subregions, acquire their hubs and subregions"""
        subregions = list(self.children.values())

        if self.data is None:
            self.aggregate_subregion_data(subregions)

        for subregion in subregions:
            self.grid.delete(subregion)

        del subregions

    def absorb_subregions_deep(self):
        """absorb subregions recursively so that region becomes to the deepest level in the hierarchy"""
        subregions = list(self.children.values())
        # print([subregion.name for subregion in subregions])

        for subregion in subregions:
            # print(subregion.name)

            subregion.absorb_subregions_deep()

            print('deleting: ', subregion.name)

            if self.data is None:
                self.aggregate_subregion_data(subregions)
            self.grid.delete(subregion)

        del subregions

    def update_data(self, df):
        """change region data

        Parameters
        ----------
        df : DataFrame
            new data
        """
        self.data = df

    def aggregate_subregion_data(self, subregions):
        """combine the data from subregions and assign it to self

        Parameters
        ----------
        subregions : list
            list of subregions
        """
        temp_child_data = pd.concat([region.data for region in subregions], axis=1).transpose()
        # print(temp_child_data)
        new_data = pd.DataFrame(
            columns=self.grid.data.summable['region'] + self.grid.data.meanable['region']
        )

        for column in temp_child_data.columns:
            if column in self.grid.data.summable['region']:
                new_data[column] = [temp_child_data[column].sum()]
            if column in self.grid.data.meanable['region']:
                new_data[column] = [temp_child_data[column].mean()]

        self.update_data(new_data.squeeze())

    def get_data(self, quantity):
        """pull data from region data

        Parameters
        ----------
        quantity : str
            name of data field in region data

        Returns
        -------
        str, float
            value of data
        """
        if self.data is None:
            return 0
        else:
            return self.data[quantity]
