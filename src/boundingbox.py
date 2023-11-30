from dataclasses import dataclass, field
from coord import Coord

@dataclass()
class BoundingBox:
    minx: int
    maxx: int
    miny: int
    maxy: int
    minz: int = field(default=0)
    maxz: int = field(default=1)
    auto_adjust: bool = field(
        default=True)  # If True, automatically sets the bounds such that min is always less than max.

    @classmethod
    def from_whd(cls, width: int, height: int, depth: int):
        return BoundingBox(0, width, 0, height, 0, depth)

    def __post_init__(self):
        if self.auto_adjust:
            if self.maxx < self.minx:
                temp = self.minx
                self.minx = self.maxx
                self.maxx = temp
            if self.maxy < self.miny:
                temp = self.miny
                self.miny = self.maxy
                self.maxy = temp
            if self.maxz < self.minz:
                temp = self.minz
                self.minz = self.maxz
                self.maxz = temp

    def __add__(self, other):
        return BoundingBox(
            self.minx + other.minx, self.maxx + other.maxx,
            self.miny + other.miny, self.maxy + other.maxy,
            self.minz + other.minz, self.maxz + other.maxz)

    def width(self):
        return self.maxx - self.minx

    def height(self):
        return self.maxy - self.miny

    def depth(self):
        return self.maxz - self.minz

    def whd(self):
        return self.max_coord() - self.min_coord()
    
    def center(self) -> Coord:
        return Coord(self.width() * 0.5 + self.minx, self.height() * 0.5 + self.miny, self.depth() * 0.5 + self.minz)

    def to_positive_translation(self):
        t = BoundingBox(0, 0, 0, 0)
        if self.minx < 0:
            t.minx = -self.minx
            t.maxx = -self.minx
        if self.miny < 0:
            t.miny = -self.miny
            t.maxy = -self.miny
        if self.minz < 0:
            t.minz = -self.minz
            t.maxz = -self.minz

        return t

    def to_positive(self):
        self += self.to_positive_translation()

    def can_contain(self, other):
        if other.width() <= self.width() and other.height() <= self.height() and other.depth() <= self.depth():
            return True
        return False

    def orient_to_other_center(self, other):
        translation = other.center() - self.center()
        self.translate(translation)

    def extent_sum(self, other):
        return [self.width() + other.width(), self.height() + other.height(), self.depth() + other.depth()]

    def min_coord(self) -> Coord:
        return Coord(self.minx, self.miny, self.minz)

    def max_coord(self) -> Coord:
        return Coord(self.maxx, self.maxy, self.maxz)

    def translate(self, coord: Coord):
        self.minx += coord.x
        self.maxx += coord.x
        self.miny += coord.y
        self.maxy += coord.y
        self.minz += coord.z
        self.maxz += coord.z
        return self
