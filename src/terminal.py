from dataclasses import dataclass, field
import numpy as np

from atom import Atom
from boundingbox import BoundingBox as BB
from coord import Coord
from offsets import Offset
from side_descriptor import SidesDescriptor as SD
from util_data import Dimensions as D, Cardinals as C, Colour


@dataclass
class Terminal:
    extent: BB  # BB with extent relative to grid units
    symmetry_axes: dict[D, set[D]]
    colour: Colour
    up: C = field(default=C.TOP)
    orientation: C = field(default=C.NORTH)
    mask: np.ndarray = field(default=None)
    unique_orientations: int = field(init=False)
    atom_indices: np.ndarray = field(init=False)
    atom_mask: np.ndarray = field(init=False)
    heightmaps: dict = field(init=False)
    n_atoms: int = field(init=False)
    atom_index_to_id_mapping: dict[Coord, Atom] = field(init=False)

    def __post_init__(self):
        if self.mask is None:
            self.mask = np.full(self.extent.whd(), True)

        self.atom_index_to_id_mapping = {}
        # Find all cells in the mask that are not empty, these are the atoms uniquely identified by their index.
        non_empty_cells = np.transpose(self.mask.nonzero())
        self.atom_indices = [Coord(xyz[1], xyz[0], xyz[2]) for xyz in non_empty_cells]

        whd = self.extent.whd()
        self.atom_mask = np.full((whd.y, whd.x, whd.z, len(self.atom_indices)), False)
        self.atom_coord_mask = np.full((whd.y, whd.x, whd.z), None)

        self.n_atoms = len(self.atom_indices)

        # Set the corresponding atoms' cells to True.
        for i in range(len(self.atom_indices)):
            c = self.atom_indices[i]
            curr = self.atom_mask[c.y, c.x, c.z]
            self.atom_coord_mask[c.y, c.x, c.z] = Coord(x, y, z)
            curr[i] = True

        # Determine which atom index requires which specific atom model.
        for i in self.atom_indices:
            x, y, z = i
            bottom = True if y - 1 > 0 and self.mask[y - 1, x, z] else False
            top = True if y + 1 < whd.y and self.mask[y + 1, x, z] else False
            west = True if x - 1 > 0 and self.mask[y, x - 1, z] else False
            east = True if x + 1 < whd.x and self.mask[y, x + 1, z] else False
            south = True if z - 1 > 0 and self.mask[y, x, z - 1] else False
            north = True if z + 1 < whd.z and self.mask[y, x, z + 1] else False
            sd = SD(bottom, top, west, east, south, north)
            self.atom_index_to_id_mapping[i] = Atom(sd)

        self.calc_heightmaps()

    def calc_heightmaps(self):
        """
        Calculates the distance along each axis to the first occupied cell.
        """
        self.heightmaps = {}
        # TODO: verify whether the mapping of axis to cardinal is correct.
        axes = {
            0: (Offset.from_cardinal(C.TOP), Offset.from_cardinal(C.BOTTOM)),
            1: (Offset.from_cardinal(C.EAST), Offset.from_cardinal(C.WEST)),
            2: (Offset.from_cardinal(C.NORTH), Offset.from_cardinal(C.SOUTH)),
        }
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
        super().__init__(extent, None, None, colour, mask=np.full(extent.whd(), True))
