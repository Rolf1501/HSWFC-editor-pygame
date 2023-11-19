from dataclasses import dataclass, field
from util_data import Dimensions as D, Cardinals as C


@dataclass
class Adjacencies:
    ADJ: dict[int, dict[C, set[(int, float)]]] = field(default_factory={
        1: {C.TOP: {(1 ,1)}, 
            C.BOTTOM: {(1 ,1)}, 
            C.NORTH: {(1 ,1)}, 
            C.EAST: {(1 ,1)}, 
            C.SOUTH: {(1 ,1)}, 
            C.WEST: {(1 ,1)}
        },
    })