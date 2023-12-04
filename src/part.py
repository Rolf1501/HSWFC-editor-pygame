from coord import Coord
from boundingbox import BoundingBox as BB
from util_data import Cardinals as C, Colour
import numpy as np
from util_data import tuple_to_numpy


class Part:
    def __init__(self, extent: BB, up: C = C.TOP, orientation: C = C.NORTH, rotation: int = 0, translation: Coord = Coord(0, 0, 0),
                 name="", colour: Colour = Colour(1,0,0,1)) -> None:
        self.extent = extent
        self.up = up
        self.orientation = orientation
        self.rotation = rotation
        self.translation = translation
        self.name = name
        self.colour = colour

    def to_absolute_cardinal(self, relative_adjacency: C) -> C:
        up_arr = tuple_to_numpy(self.up.value)
        rel_arr = tuple_to_numpy(relative_adjacency.value)
        r = (self.rotation % 360) / 90.0

        if C.cardinal_to_dimension(relative_adjacency) == C.cardinal_to_dimension(self.up):
            return rel_arr * up_arr # Negate if up and relative differ (top becomes bottom and vice versa)

        # To rotate an for n times around the up-direction, shortcuts can be taken.
        # The value corresponding to the dimension of the up-direction (up_d) does not change.
        # If the number of rotations is odd, then the values not in up_d should swap. This can be done with an xor of the nand of the up direction.
        # That is, since the up only has one non-zero value.
        # If the number of rotation is even, then the result is either the same (r=0), or the result is the negation of given cardinal (r=2). The optional negation can thus be expressed as (-r + 1).

        if r % 2 == 0:
            rot_arr = rel_arr * (-r + 1)
            return C(tuple(rot_arr))
        else:
            up_nand = ~(up_arr & 1) + 2 # bitwise NAND
            rot_arr = (np.sum(rel_arr)) *  (-r + 2) * ((rel_arr | up_nand) ^ rel_arr) # bitwise XOR and optional negation
        return C(tuple(rot_arr))

    def __repr__(self) -> str:
        return f"<Part {self.name}: {self.extent}, rot:{self.rotation}, tr:{self.translation}, c:{self.colour}>"
