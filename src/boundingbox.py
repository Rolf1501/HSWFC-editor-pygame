from dataclasses import dataclass, field
from coord import Coord

@dataclass()
class BoundingBox:
    minx: int = field()
    maxx: int = field()
    miny: int = field()
    maxy: int = field()
    
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