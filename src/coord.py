from typing_extensions import SupportsIndex
import numpy as np
from collections import namedtuple


class Coord(namedtuple("Coord", ["x","y","z"])):

    def __add__(self, other):
        return Coord(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Coord(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        return Coord(self.x * other.x, self.y * other.y, self.z * other.z)
    
    def __repr__(self) -> str:
        return f"<Coord {self.x}, {self.y}, {self.z}>"
    
    @staticmethod
    def abs_diff(this, that):
        return Coord(abs(this.x - that.x), abs(this.y - that.y), abs(this.z - that.z))

    @staticmethod
    def min(this, that):
        return Coord(min(this.x, that.x), min(this.y, that.y), min(this.z, that.z))

    def scale(self, scalar):
        self = self.scaled(scalar)

    def scaled(self, scalar):
        return Coord(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar)

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def to_numpy_array(self):
        return np.asarray([self.x, self.y, self.z])
    
    def abs_diff(self, other):
        return Coord(abs(self.x - other.x), abs(self.y - other.y), abs(self.z - other.z))