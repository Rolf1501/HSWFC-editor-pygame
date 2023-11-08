from dataclasses import dataclass, field
from coord import Coord

@dataclass()
class BoundingBox:
    minx: int
    maxx: int
    miny: int
    maxy: int
    auto_adjust: bool = field(default=True) # If True, automatically sets the bounds such that min is always less than max.

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
 
    
    def __add__(self, other):
        return BoundingBox(self.minx + other.minx, self.maxx + other.maxx, self.miny + other.miny, self.maxy + other.maxy)


    def width(self):
        return self.maxx - self.minx
    

    def height(self):
        return self.maxy - self.miny
    

    def center(bb) -> Coord:
        return Coord(bb.width() * 0.5 + bb.minx, bb.height() * 0.5 + bb.miny)


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


    def can_contain(self, other):
        if other.width() <= self.width() and other.height() <= self.height():
            return True
        return False


    def orient_to_other_center(self, other):
        translation = other.center() - self.center()
        self.translate(translation)


    def min_coord(self) -> Coord:
        return Coord(self.minx, self.miny)


    def translate(self, coord: Coord):
        self.minx += coord.x
        self.maxx += coord.x
        self.miny += coord.y
        self.maxy += coord.y
        return self
