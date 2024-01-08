from enum import Enum
from dataclasses import dataclass, field
from util_data import Cardinals as C
from offsets import Offset


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
    offset_dict: dict[C, SideProperties] = field(init=False)

    def __post_init__(self):
        self.offset_dict = {}
        self.offset_dict[C.TOP.value] = self.top
        self.offset_dict[C.BOTTOM.value] = self.bottom
        self.offset_dict[C.NORTH.value] = self.north
        self.offset_dict[C.EAST.value] = self.east
        self.offset_dict[C.SOUTH.value] = self.south
        self.offset_dict[C.WEST.value] = self.west

    def get_from_offset(self, offset: Offset):
        return self.offset_dict[offset]
