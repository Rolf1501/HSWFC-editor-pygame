from dataclasses import dataclass, field
from collections import namedtuple
import numpy as np
from util_data import Cardinals as C


class Offset(namedtuple("Offset", ["x","y","z"])):
    N_ATRRIBUTES = 3
    
    @classmethod
    def from_numpy_array(cls, array: np.ndarray):
        assert(len(array) <= Offset.N_ATRRIBUTES)
        # if len(array) <= Offset.N_ATRRIBUTES:
        #     np.
        return cls(*array)
    
    def to_numpy_array(self):
        return np.asarray([*self])
    
    def negation(self):
        return self.scaled(-1)
    
    def scaled(self, scalar):
        return Offset(self.x * scalar, self.y * scalar, self.z * scalar)

    # def to_cardinal(self):
    #     if self.x > 0:
    #         return C.EAST
    #     elif self.x < 0:
    #         return C.WEST
    #     elif self.y 



@dataclass
class OffsetFactory:
    # dimensions: int = field(default=3)
    # is_cardinal: bool = field(True)
    # offsets: np.ndarray = field(init=False)


    # def __post_init__(self):
    #     self.offsets = np.asmatrix(self.getOffsets(self.is_cardinal, self.dimensions))

    def get_offsets(self, dimensions: int = 3, cardinal: bool = True) -> list[Offset]:
        if dimensions > 3 or dimensions < 1:
            raise DimensionsNotSupported(f"Dimension should not be less than 1 and should not exceed 3. Got: {dimensions}")
        if cardinal:
            # There are two neighbours to consider per dimension.
            offsets = []
            # By taking the ith position, cardinality is ensured, adhering to the format [0, {-1,1}] (and variations thereof).
            for i in range(dimensions):
                offset_plus = np.zeros((dimensions))
                offset_minus = np.zeros((dimensions))
                offset_plus[i] = 1
                offset_minus[i] = -1
                offsets.append(Offset.from_numpy_array(offset_plus))
                offsets.append(Offset.from_numpy_array(offset_minus))
            
            return offsets
        else:
            raise NotImplementedError("Offsets does not support non-cardinal offsets")
        
    # def get_cardinal_offsets_as_dict(self, dimensions: int = 2):



class DimensionsNotSupported(Exception):
    def __init__(self, message):
        self.message = message