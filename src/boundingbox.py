from dataclasses import dataclass, field
from coord import Coord

@dataclass()
class BoundingBox:
    minx: int
    maxx: int
    miny: int
    maxy: int

    def __post_init__(self):
        if self.maxx < self.minx:
            temp = self.minx
            self.minx = self.maxx
            self.maxx = temp
        if self.maxy < self.miny:
            temp = self.miny
            self.miny = self.maxy
            self.maxy = temp
    
    def __add__(self, other):
        return BoundingBox(self.minx + other.minx, self.maxx + other.maxx, self.miny + other.miny, self.maxy + other.maxy)

    def width(self):
        return self.maxx - self.minx
    
    def height(self):
        return self.maxy - self.miny
    
    def center(self) -> Coord:
        return Coord(self.width() * 0.5 + self.minx, self.height() * 0.5 + self.miny)
    
    def to_positive_translation(self):
        t = BoundingBox(0,0,0,0)
        if self.minx < 0:
            t.minx = -self.minx
            t.maxx = -self.minx
        if self.miny < 0:
            t.miny = -self.miny
            t.maxy = -self.miny
       
        return t

    def to_positive(self):
        self += self.to_positive_translation()

    def scale(self, scalar):
        self.minx *= scalar
        self.maxx *= scalar
        self.miny *= scalar
        self.maxy *= scalar