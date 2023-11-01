from enum import Enum
class Cardinals(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Operations(Enum):
    ORTH = 0
    SYM = 1
    PAR = 2
    CENTER = 3

class Dimensions(Enum):
    X = 0
    Y = 1
    # Z = 2