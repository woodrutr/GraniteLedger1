"""
REGISTRY CLASS
~~~~~~~~~~~~~~

This class is the central registry of all objects in a grid. It preserves them in dicts of
object-name:object so that they can be looked up by name. it also should serve as a place to save
data in different configurations for faster parsing - for example, depth is a dict that organizes
regions according to their depth in the region nesting tree.

"""

###################################################################################################
# Setup
from src.models.hydrogen.network.hub import Hub
from src.models.hydrogen.network.region import Region
from src.models.hydrogen.network.transportation_arc import TransportationArc
###################################################################################################


class Registry:
    def __init__(self):
        """initialize the registry"""
        self.regions: dict[str, Region] = {}
        self.depth = {i: [] for i in range(10)}
        self.hubs: dict[str, Hub] = {}
        self.arcs = {}
        self.max_depth = 0

    def add(self, thing):
        """add a thing to the registry. Thing can be Hub,Arc, or Region

        Parameters
        ----------
        thing : Arc, Region, or Hub
            thing to add to registry

        Returns
        -------
        Arc, Region, or Hub
            thing being added gets returned
        """

        if type(thing) == Hub:
            self.hubs[thing.name] = thing
            return thing
        elif type(thing) == TransportationArc:
            self.arcs[thing.name] = thing
            return thing
        elif type(thing) == Region:
            self.regions[thing.name] = thing
            self.depth[thing.depth].append(thing.name)
            if thing.depth > self.max_depth:
                self.max_depth = thing.depth
            return thing

    def remove(self, thing):
        """remove thing from registry

        Parameters
        ----------
        thing : Arc, Hub, or Region
            thing to remove
        """

        if type(thing) == Hub:
            del self.hubs[thing.name]
        elif type(thing) == Region:
            # self.depth[thing.depth] = self.depth[thing.depth].remove(thing.name)
            del self.regions[thing.name]

        elif type(thing) == TransportationArc:
            del self.arcs[thing.name]

    def update_levels(self):
        """update dictionary of regions by level"""
        self.depth = {i: [] for i in range(10)}
        for region in self.regions.values():
            self.depth[region.depth].append(region.name)
        pass
