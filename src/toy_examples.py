from terminal import Terminal, Void
from boundingbox import BoundingBox as BB
from side_properties import SidesDescriptor as SD, SideProperties as SP
from adjacencies import Adjacency, AdjacencyAny, Relation as R
from offsets import Offset
from util_data import Cardinals as C, Dimensions as D, Colour

class ToyExamples():
    def __init__(self):
        pass

    def full_symmetric_axes():
        return {
            D.X: {D.Y, D.Z},
            D.Y: {D.X, D.Z},
            D.Z: {D.Y, D.X},
        }
    
    def example_big_tiles(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        terminals = {
            # For a cuboid with vertex ids: 
            #   7------6
            #  /|     /|
            # 4------5 |
            # | 3----|-2 
            # |/     |/
            # 0------1
            
            0: Terminal(BB.from_whd(1,1,1), symmetry_axes, 
                        SD(north=SP.OPEN, east=SP.OPEN),                Colour(0.5,0,  0,  1)), # red; corner_04
            1: Terminal(BB.from_whd(1,1,1), symmetry_axes, 
                        SD(north=SP.OPEN, west=SP.OPEN),                Colour(0  ,0.5,0,  1)), # green; corner_15
            2: Terminal(BB.from_whd(1,1,1), symmetry_axes, 
                        SD(south=SP.OPEN, west=SP.OPEN),                Colour(0,  0,  0.5,1)), # blue; corner_26
            3: Terminal(BB.from_whd(1,1,1), symmetry_axes, 
                        SD(south=SP.OPEN, east=SP.OPEN),                Colour(0,  0.5,0.5,1)), # cyan; corner_37
            4: Terminal(BB.from_whd(1,1,1), symmetry_axes, 
                        SD(north=SP.OPEN, east=SP.OPEN, west=SP.OPEN),  Colour(0.5,0,  0.5,1)), # magenta; face_0154
            5: Terminal(BB.from_whd(1,1,1), symmetry_axes, 
                        SD(south=SP.OPEN, east=SP.OPEN, west=SP.OPEN),  Colour(0.5,0.5,0,  1)), # yellow; face_3267
            6: Terminal(BB.from_whd(1,1,1), symmetry_axes,
                        SD(north=SP.OPEN, south=SP.OPEN, east=SP.OPEN), Colour(0,  0,  0,  1)), # black; face_0374
            7: Terminal(BB.from_whd(1,1,1), symmetry_axes,
                        SD(north=SP.OPEN, south=SP.OPEN, west=SP.OPEN), Colour(0.5,0.5,0.5,1)), # white; face_1265
        }

        adjs = {
            Adjacency(0, {}, Offset(*C.NORTH.value), True),
            Adjacency(0, {}, Offset(*C.EAST.value), True),
            Adjacency(0, {}, Offset(*C.SOUTH.value), True),
            Adjacency(0, {}, Offset(*C.WEST.value), True),

            Adjacency(1, {}, Offset(*C.NORTH.value), True),
            Adjacency(1, {}, Offset(*C.EAST.value), True),
            Adjacency(1, {}, Offset(*C.SOUTH.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.WEST.value), True),

            Adjacency(2, {}, Offset(*C.NORTH.value), True),
            Adjacency(2, {R(0, 1)}, Offset(*C.EAST.value), True),
            Adjacency(2, {R(1, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(2, {}, Offset(*C.WEST.value), True),

            Adjacency(3, {}, Offset(*C.NORTH.value), True),
            Adjacency(3, {R(2, 1)}, Offset(*C.EAST.value), True),
            Adjacency(3, {R(0, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(3, {R(2, 1)}, Offset(*C.WEST.value), True),

            Adjacency(4, {}, Offset(*C.NORTH.value), True),
            Adjacency(4, {R(4, 1), R(1, 1)}, Offset(*C.EAST.value), True),
            Adjacency(4, {R(2, 1), R(3, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(4, {R(4, 1), R(0, 1)}, Offset(*C.WEST.value), True),

            Adjacency(5, {R(4, 1)}, Offset(*C.NORTH.value), True),      
            Adjacency(5, {R(5, 1), R(2,1)}, Offset(*C.EAST.value), True),
            Adjacency(5, {R(4, 1)}, Offset(*C.SOUTH.value), True),      
            Adjacency(5, {R(5, 1), R(3,1)}, Offset(*C.WEST.value), True),

            Adjacency(6, {R(6, 1), R(3, 1)}, Offset(*C.NORTH.value), True),      
            Adjacency(6, {}, Offset(*C.EAST.value), True),
            Adjacency(6, {R(6, 1), R(0, 1), }, Offset(*C.SOUTH.value), True),      
            Adjacency(6, {R(1, 1), R(2, 1)}, Offset(*C.WEST.value), True),

            Adjacency(7, {R(7, 1), R(2, 1)}, Offset(*C.NORTH.value), True),      
            Adjacency(7, {R(6, 1), R(0, 1), R(3, 1)}, Offset(*C.EAST.value), True),
            Adjacency(7, {R(7, 1), R(0, 1), }, Offset(*C.SOUTH.value), True),      
            Adjacency(7, {R(6, 1)}, Offset(*C.WEST.value), True),
        }

        top_bottom_any = {AdjacencyAny(i, o, True) for i in terminals for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]}

        return terminals, adjs.union(top_bottom_any)

    def example_zebra_horizontal(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        temrinals =  {
            0: Terminal(BB.from_whd(1,1,1), symmetry_axes, side_desc, Colour(1,1,1,1)), # white
            1: Terminal(BB.from_whd(1,1,1), symmetry_axes, side_desc, Colour(0,0,0,1)), # black
        }

        adjs = {
            Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.WEST.value), True),

            Adjacency(1, {R(1, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.WEST.value), True),

            Adjacency(1, {R(0, 1)}, Offset(*C.TOP.value), True),
            Adjacency(0, {R(1, 1)}, Offset(*C.TOP.value), True),
        }

        return temrinals, adjs
    
    def example_zebra_horizontal_3(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        temrinals =  {
            0: Terminal(BB.from_whd(1,1,1), symmetry_axes, side_desc, Colour(1,1,1,1)), # white
            1: Terminal(BB.from_whd(1,1,1), symmetry_axes, side_desc, Colour(0,0,0,1)), # black
            2: Terminal(BB.from_whd(1,1,1), symmetry_axes, side_desc, Colour(1,1,0,1)), # yellow
        }

        adjs = {
            Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.WEST.value), True),

            Adjacency(1, {R(1, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.WEST.value), True),

            Adjacency(2, {R(2, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(2, {R(2, 1)}, Offset(*C.WEST.value), True),

            Adjacency(1, {R(2, 1)}, Offset(*C.TOP.value), True),
            Adjacency(0, {R(1, 1)}, Offset(*C.TOP.value), True),
            Adjacency(2, {R(0, 1)}, Offset(*C.TOP.value), True),
        }

        return temrinals, adjs

    def example_zebra_vertical(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        temrinals =  {
            0: Terminal(BB.from_whd(1,1,1), symmetry_axes, side_desc, Colour(0.7,0.7,0.7,1)), # white
            1: Terminal(BB.from_whd(1,1,1), symmetry_axes, side_desc, Colour(0.05,0.05,0.05,1)), # black
        }

        adjs = {
            Adjacency(0, {R(0, 1)}, Offset(*C.TOP.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),

            Adjacency(1, {R(1, 1)}, Offset(*C.TOP.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.NORTH.value), True),
            
            Adjacency(1, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(0, {R(1, 1)}, Offset(*C.WEST.value), True),
        }

        return temrinals, adjs

    def example_slanted(symmetry_axes=full_symmetric_axes(), side_desc=SD()):

        terminals = {
            0: Terminal(BB.from_whd(1,1,1), symmetry_axes, side_desc, Colour(0.8,0,0,1)),
            1: Void(BB.from_whd(1,1,1))
        }

        adjs = {
            Adjacency(0, {R(0, 0.8), R(1, 0.2)}, Offset(*C.WEST.value), True),
            Adjacency(0, {R(0, 0.2), R(1, 0.8)}, Offset(*C.TOP.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.TOP.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.EAST.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.NORTH.value), True)
        }
        return terminals, adjs
