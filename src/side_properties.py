from util_data import Cardinals
from enum import Enum
from dataclasses import dataclass, field


class SideProperties(Enum):
    STUDS = 0
    TUBES = 1
    SMOOTH = 2

@dataclass
class SidesDescriptor:
    top: SideProperties = field(default=SideProperties.STUDS)
    bottom: SideProperties = field(default=SideProperties.TUBES)
    north: SideProperties = field(default=SideProperties.SMOOTH)
    east: SideProperties = field(default=SideProperties.SMOOTH)
    south: SideProperties = field(default=SideProperties.SMOOTH)
    west: SideProperties = field(default=SideProperties.SMOOTH)
