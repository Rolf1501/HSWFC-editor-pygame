from enum import Enum
class Cardinals(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    TOP = 4
    BOTTOM = 5


class Operations(Enum):
    ORTH = 0
    SYM = 1
    PAR = 2
    CENTER = 3

class Dimensions(Enum):
    X = 0
    Y = 1
    Z = 2


MAX = 9999999
MIN = -MAX

def cardinal_to_dimension(card: Cardinals):
    if card == Cardinals.TOP or card == Cardinals.BOTTOM:
        return Dimensions.Y
    elif card == Cardinals.NORTH or card == Cardinals.SOUTH:
        return Dimensions.Z
    else return Dimensions.X