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
    extent: Coord  # BB with extent relative to grid units
    colour: Colour
    up: C = field(default=C.TOP)
    # orientation: C = field(default=C.NORTH)
    mask: np.ndarray = field(default=None)

    # Specifies how the terminal may be oriented; default 0 means that the terminal may only point North.
    distinct_orientations: list[int] = field(default_factory=lambda: [0])
    # atom_indices: np.ndarray = field(init=False)
    # atom_mask: np.ndarray = field(init=False)
    heightmaps: dict = field(init=False)
    n_atoms: int = field(init=False)
    atom_index_to_id_mapping: dict[Coord, Atom] = field(init=False)
    oriented_mask: dict[int, np.ndarray] = field(init=False)
    oriented_atom_mask: dict[int, np.ndarray] = field(
        init=False, default_factory=lambda: {}
    )
    oriented_indices: dict[int, np.ndarray] = field(
        init=False, default_factory=lambda: {}
    )
    oriented_heightmaps: dict = field(init=False)

    def __post_init__(self):
        self.atom_index_to_id_mapping = {}
        if self.mask is None:
            self.mask = np.full((self.extent.y, self.extent.x, self.extent.z), True)

        self.calc_oriented_masks()

        self.calc_heightmaps()

    def to_json(self):
        return {
            "extent": self.extent,
            "colour": self.colour,
            "mask": self.mask.tolist(),
            "distinct_orientations": self.distinct_orientations,
        }

    def calc_atom_indices(self, mask: np.ndarray):
        # Find all cells in the mask that are not empty, these are the atoms uniquely identified by their index.
        non_empty_cells = np.transpose(mask.nonzero())
        indices = [Coord(xyz[1], xyz[0], xyz[2]) for xyz in non_empty_cells]
        return indices

    def calc_oriented_masks(self):
        self.oriented_mask = {}

        self.n_atoms = 0
        for d in self.distinct_orientations:
            # The orientation determines how many counter clockwise rotations of 90 degrees are needed.
            self.oriented_mask[d] = np.rot90(self.mask, d, axes=(1, 2))
            whd = self.oriented_mask[d].shape
            whd = Coord(whd[1], whd[0], whd[2])
            indices = self.calc_atom_indices(self.oriented_mask[d])
            self.oriented_indices[d] = indices

            self.oriented_atom_mask[d] = np.full(
                (whd.y, whd.x, whd.z, len(indices)), False
            )
            self.n_atoms += len(indices)

            # Set the corresponding atoms' cells to True.
            for i in range(len(indices)):
                c = indices[i]
                curr = self.oriented_atom_mask[d][c.y, c.x, c.z]
                curr[i] = True

            # Determine which atom index requires which specific atom model.
            # for i in self.atom_indices:
            #     x, y, z = i
            #     bottom = True if y - 1 > 0 and self.mask[y - 1, x, z] else False
            #     top = True if y + 1 < whd.y and self.mask[y + 1, x, z] else False
            #     west = True if x - 1 > 0 and self.mask[y, x - 1, z] else False
            #     east = True if x + 1 < whd.x and self.mask[y, x + 1, z] else False
            #     south = True if z - 1 > 0 and self.mask[y, x, z - 1] else False
            #     north = True if z + 1 < whd.z and self.mask[y, x, z + 1] else False
            #     sd = SD(bottom, top, west, east, south, north)
            #     self.atom_index_to_id_mapping[i] = Atom(sd)
            for i in self.oriented_indices[d]:
                x, y, z = i
                # bottom = True if y - 1 > 0 and self.mask[y - 1, x, z] else False
                # top = True if y + 1 < whd.y and self.mask[y + 1, x, z] else False
                # west = True if x - 1 > 0 and self.mask[y, x - 1, z] else False
                # east = True if x + 1 < whd.x and self.mask[y, x + 1, z] else False
                # south = True if z - 1 > 0 and self.mask[y, x, z - 1] else False
                # north = True if z + 1 < whd.z and self.mask[y, x, z + 1] else False
                # sd = SD(bottom, top, west, east, south, north)
                self.atom_index_to_id_mapping[i] = Atom(SD())

    def calc_heightmaps(self):
        """
        Calculates the distance along each axis to the first occupied cell.
        """
        self.heightmaps = {}
        self.oriented_heightmaps = {}

        axes = {
            0: (Offset.from_cardinal(C.BOTTOM), Offset.from_cardinal(C.TOP)),
            1: (Offset.from_cardinal(C.WEST), Offset.from_cardinal(C.EAST)),
            2: (Offset.from_cardinal(C.SOUTH), Offset.from_cardinal(C.NORTH)),
        }
        for axis in axes.keys():
            cardinal_along, cardinal_against = axes[axis]
            self.oriented_heightmaps[cardinal_along] = {}
            self.oriented_heightmaps[cardinal_against] = {}
        for d in self.distinct_orientations:
            for axis in axes.keys():
                cardinal_along, cardinal_against = axes[axis]
                # hm_along, hm_against = self.calc_heightmap_for_axis(
                #     self.mask, axis=axis
                # )
                hm_along_orr, hm_against_orr = self.calc_heightmap_for_axis(
                    self.oriented_mask[d], axis=axis
                )
                self.oriented_heightmaps[cardinal_along][d] = hm_along_orr
                self.oriented_heightmaps[cardinal_against][d] = hm_against_orr
                # self.heightmaps[cardinal_along] = hm_along
                # self.heightmaps[cardinal_against] = hm_against

    def calc_heightmap_for_axis(self, mask: np.ndarray, axis):
        """
        Calculates the distance along and against the given axis to the first occupied cell.
        """
        return mask.argmax(axis=axis), np.flip(mask, axis=axis).argmax(axis=axis)

    def get_rotation_orientation(self, rotation):
        """
        Determines which terminal orientation to use given the rotation.
        First check if the rotation is present. If not, check if the direction corresponding to the rotation is in the orientations.
        Otherwise, the terminal is assumed to be unaffected by rotation (e.g. cubes).
        """
        if rotation in self.distinct_orientations:
            return rotation
        elif (rotation + 2) % 3 in self.distinct_orientations:
            return (rotation + 2) % 3
        else:
            return self.distinct_orientations[0]

    def contains_atom(self, orientation, x, y, z):
        try:
            return self.oriented_mask[orientation][y, x, z]
        except:
            return False


class Void(Terminal):
    def __init__(self, extent: BB, colour: Colour = None):
        self.extent = extent
        self.colour = colour
        super().__init__(extent, None, None, colour, mask=np.full(extent, True))
