from enum import Enum
import numpy as np
from collections import namedtuple


class Cardinals(Enum):
    NORTH = (0, 0, 1)
    SOUTH = (0, 0, -1)
    EAST =  (1, 0, 0)
    WEST =  (-1, 0, 0)
    TOP =   (0, 1, 0)
    BOTTOM =(0, -1, 0)

    @staticmethod
    def cardinal_to_dimension(cardinal):
        if cardinal == Cardinals.TOP or cardinal == Cardinals.BOTTOM:
            return Dimensions.Y
        elif cardinal == Cardinals.NORTH or cardinal == Cardinals.SOUTH:
            return Dimensions.Z
        else:
            return Dimensions.X


class Operations(Enum):
    ORTH = 0
    SYM = 1
    PAR = 2
    CENTER = 3

class Dimensions(Enum):
    X = (1,0,0)
    Y = (0,1,0)
    Z = (0,0,1)


MAX = 9999999
MIN = -MAX


def tuple_to_numpy(tuple):
    return np.asarray([*tuple])


class Colour(namedtuple("Colour", ["r", "g", "b"])):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
