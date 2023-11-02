from collections import namedtuple

class Propagation(namedtuple("Propagation", ["Offset", "Instigator", "Depth"])):
    """
    A tuple of Offset, Instigator, and Depth. This is used in the propagation queue, where each instance of this class represents the propagation over a single cell.
    """
    pass