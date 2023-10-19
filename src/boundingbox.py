from dataclasses import dataclass, field

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