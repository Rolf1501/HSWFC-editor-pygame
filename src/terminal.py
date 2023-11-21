from dataclasses import dataclass, field
from coord import Coord
from boundingbox import BoundingBox as BB
from side_properties import SidesDescriptor
from util_data import Dimensions as D, Cardinals as C

@dataclass
class Terminal:
    extent: BB  # BB with extent relative to grid units
    symmetry_axes: dict[D, set[D]]
    side_descriptor: SidesDescriptor
    up: C = field(default_factory=C.TOP)
    orientation: C = field(default_factory=C.NORTH)
    unique_orientations: int = field(init=False)
    # TODO: add some properties, such as colour.

    # def __post_init__(self):
    #     self.unique_orientations = self._calc_unique_orientations()

    # def _calc_unique_orientations(self):
    #     width = self.extent.width()
    #     height = self.extent.height()
    #     depth = self.extent.depth()
    #     up_dimension = cardinal_to_dimension(self.up)
        
    #     dimensions = {D.X, D.Y, D.Z}
    #     other_dims = dimensions - up_dimension
    #     if up_dimension in 
