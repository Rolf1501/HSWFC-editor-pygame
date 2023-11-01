class Coord:
    def __init__(self,x,y,z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Coord(self.x + other.x, self.y + other.y, self.z + other.z)
    

    def __sub__(self, other):
        return Coord(self.x - other.x, self.y - other.y, self.z - other.z)


    @staticmethod
    def abs_diff(this, that):
        return Coord(abs(this.x - that.x), abs(this.y - that.y), abs(this.z - that.z))


    @staticmethod
    def min(this, that):
        return Coord(min(this.x, that.x), min(this.y, that.y), min(this.z, that.z))

    def scale(self, scalar):
        self.x *= scalar
        self.y *= scalar
        self.z *= scalar

    def toTuple(self):
        return (self.x, self.y)