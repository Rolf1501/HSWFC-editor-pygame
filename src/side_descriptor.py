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
    bottom_open: bool = field(default=True)
    top_open: bool = field(default=True)
    west_open: bool = field(default=True)
    east_open: bool = field(default=True)
    south_open: bool = field(default=True)
    north_open: bool = field(default=True)
    int_id: int = field(init=False)

    def __post_init__(self):
        self.int_id = self.to_int_id()

    def to_int_id(self):
        return int(
            f"{int(self.bottom_open)}{int(self.top_open)}{int(self.west_open)}{int(self.east_open)}{int(self.south_open)}{int(self.north_open)}",
            2,
        )

    @classmethod
    def from_int_id(cls, int_id):
        """
        Convert the id to a side descriptor by applying the reverse operation in the to_int_id method.
        """

        # 6 corresponds to the number of faces, 0 prepended when necessary.
        bit_string = "{0:b}".format(int_id).zfill(6)
        args = [bool(int(i)) for i in bit_string]
        return cls(*args)


# @dataclass
# class SidesDescriptor:
#     top: SideProperties = field(default=SideProperties.STUDS)
#     bottom: SideProperties = field(default=SideProperties.TUBES)
#     north: SideProperties = field(default=SideProperties.SMOOTH)
#     east: SideProperties = field(default=SideProperties.SMOOTH)
#     south: SideProperties = field(default=SideProperties.SMOOTH)
#     west: SideProperties = field(default=SideProperties.SMOOTH)
#     offset_dict: dict[C, SideProperties] = field(init=False)

#     def __post_init__(self):
#         self.offset_dict = {}
#         self.offset_dict[C.TOP.value] = self.top
#         self.offset_dict[C.BOTTOM.value] = self.bottom
#         self.offset_dict[C.NORTH.value] = self.north
#         self.offset_dict[C.EAST.value] = self.east
#         self.offset_dict[C.SOUTH.value] = self.south
#         self.offset_dict[C.WEST.value] = self.west

#     def get_from_offset(self, offset: Offset):
#         return self.offset_dict[offset]
