from enum import Enum
from dataclasses import dataclass, field
from util_data import Cardinals as C
from grid import Grid

class SideProperties(Enum):
    STUDS = 0
    TUBES = 1
    SMOOTH = 2
    OPEN = 3

@dataclass
class SidesDescriptor:
    top: SideProperties = field(default=SideProperties.STUDS)
    bottom: SideProperties = field(default=SideProperties.TUBES)
    north: SideProperties = field(default=SideProperties.SMOOTH)
    east: SideProperties = field(default=SideProperties.SMOOTH)
    south: SideProperties = field(default=SideProperties.SMOOTH)
    west: SideProperties = field(default=SideProperties.SMOOTH)

    def __post_init__(self):
        self.mask = Grid(3,3,3, -1)
        self.mask.set(*C.TOP.value, self.top.value) 
        self.mask.set(*C.BOTTOM.value, self.bottom.value) 
        self.mask.set(*C.NORTH.value, self.north.value) 
        self.mask.set(*C.EAST.value, self.east.value) 
        self.mask.set(*C.SOUTH.value, self.south.value) 
        self.mask.set(*C.WEST.value, self.west.value) 
