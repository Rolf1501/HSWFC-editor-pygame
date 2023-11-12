from coord import Coord
from boundingbox import BoundingBox as BB
from util_data import Cardinals

class Part:
    def __init__(self, bb: BB, up: Coord=Coord(0,0,1), rotation: int=0,  translation: Coord=Coord(0,0), name="") -> None:
        self.bb = bb
        self.up = up
        self.rotation = rotation
        self.translation = translation
        self.name = name
    
    def absolute_north(self) -> Cardinals:
        # Assumes rotation expressed in degrees
        return Cardinals((self.rotation / 90.0) % len(Cardinals))
    
    def __repr__(self) -> str:
        return f"<Part {self.name}: {self.bb}, rot:{self.rotation}, tr:{self.translation}>"
    