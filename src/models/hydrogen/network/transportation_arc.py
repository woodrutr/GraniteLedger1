"""
TRANSPORTATION ARC CLASS
~~~~~~~~~~~~~~~~~~~~~~~~

objects in this class represent individual transportation arcs. An arc can exist with zero capacity,
so they only represent *possible* arcs.
"""
###################################################################################################


class TransportationArc:
    def __init__(self, origin, destination, capacity, cost=0):
        """initialize a transportation arc from arguments

        Parameters
        ----------
        origin : Hub
            origin hub
        destination : Hub
             destination hub
        capacity : float
            transportation capacity of arc
        cost : float, optional
            transportation cost of arc., by default 0
        """
        self.name = (origin.name, destination.name)
        self.origin = origin
        self.destination = destination
        self.capacity = capacity
        self.cost = cost

    def change_origin(self, new_origin):
        """change the origin hub of arc

        Parameters
        ----------
        new_origin : Hub
            new origin hub
        """
        self.name = (new_origin.name, self.name[1])
        self.origin = new_origin

    def change_destination(self, new_destination):
        """change the destination hub of arc

        Parameters
        ----------
        new_destination : Hub
            new destination hub
        """
        self.name = (self.name[0], new_destination.name)
        self.destination = new_destination

    def disconnect(self):
        """disconnect arc from it's origin and destination"""
        self.origin.remove_outbound(self)
        self.destination.remove_inbound(self)
