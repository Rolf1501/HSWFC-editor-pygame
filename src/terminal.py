from dataclasses import dataclass, field
from coord import Coord
from boundingbox import BoundingBox as BB
from side_properties import SidesDescriptor
from util_data import Dimensions as D, Cardinals as C, Colour

@dataclass
class Terminal:
    extent: BB  # BB with extent relative to grid units
    symmetry_axes: dict[D, set[D]]
    side_descriptor: SidesDescriptor
    colour: Colour
    up: C = field(default=C.TOP)
    orientation: C = field(default=C.NORTH)
    unique_orientations: int = field(init=False)


class Void(Terminal):
    def __init__(self, extent: BB):
        self.extent = extent
