from collections import namedtuple

class Offset(namedtuple("Offset", ["x", "y"])):
    """
    Just a convenience class, mostly used for linking the cardinal directions UP/DOWN/LEFT/RIGHT to their appropriate offsets.
    """
    def __repr__(self):
        return f"O({self.x},{self.y})" 

class DimensionsNotSupported(Exception):
    def __init__(self, message):
        self.message = message

def getOffsets(cardinal: bool = True, dimensions: int = 2):
    if dimensions > 3 or dimensions < 1:
        raise DimensionsNotSupported(f"Number of dimensions is not support. Dimension should not be less than 1 and should not exceed 3. Got: {dimensions}")
    if cardinal:
        # There are two neighbours per dimension.
        offsets = [[0 for _ in range(dimensions)] for _ in range(dimensions * 2)]
        for i in range(dimensions):
            offsets[2*i][i] = -1
            offsets[2*i+1][i] = 1 
        
        return offsets
    else:
        raise NotImplementedError("Offsets does not support non-cardinal offsets") 