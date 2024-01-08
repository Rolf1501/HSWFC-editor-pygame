from dataclasses import dataclass, field
from coord import Coord
from boundingbox import BoundingBox as BB
from side_properties import SidesDescriptor
from util_data import Dimensions as D, Cardinals as C, Colour
import numpy as np


@dataclass
class Terminal:
    extent: BB  # BB with extent relative to grid units
    symmetry_axes: dict[D, set[D]]
    side_descriptor: SidesDescriptor
    colour: Colour
    up: C = field(default=C.TOP)
    orientation: C = field(default=C.NORTH)
    mask: np.ndarray = field(default=None)
    unique_orientations: int = field(init=False)
    atom_indices: np.ndarray = field(init=False)
    atom_mask: np.ndarray = field(
        init=False
    )  # 4D array for storing which cell contains which atom.
    heightmaps: dict = field(init=False)

    def __post_init__(self):
        # Find all cells in the mask that are not empty, these are the atoms uniquely identified by their index.
        non_empty_cells = np.transpose(self.mask.nonzero())
        self.atom_indices = [Coord(yxz[1], yxz[0], yxz[2]) for yxz in non_empty_cells]
        whd = self.extent.whd()
        self.atom_mask = np.full((whd.x, whd.y, whd.z, len(self.atom_indices)), False)
        print(self.atom_indices)
        for i in range(len(self.atom_indices)):
            c = self.atom_indices[i]
            curr = self.atom_mask[c.y, c.x, c.z]
            curr[i] = True
        print(self.atom_mask)
        self.calc_heightmaps()

    def calc_heightmaps(self):
        """
        Calculates the distance along each axis to the first occupied cell.
        """
        axes = {0: (C.TOP, C.BOTTOM), 1: (C.EAST, C.WEST), 2: (C.NORTH, C.SOUTH)}
        for axis in axes.keys():
            cardinal_along, cardinal_against = axes[axis]
            hm_along, hm_against = self.calc_heightmap_for_axis(self.mask, axis=axis)
            self.heightmaps[cardinal_along] = hm_along
            self.heightmaps[cardinal_against] = hm_against

    def calc_heightmap_for_axis(self, mask: np.ndarray, axis):
        """
        Calculates the distance along and against the given axis to the first occupied cell.
        """
        return mask.argmax(axis=axis), np.flip(mask, axis=axis).argmax(axis=axis)


class Void(Terminal):
    def __init__(self, extent: BB, colour: Colour = None):
        self.extent = extent
        self.colour = colour
