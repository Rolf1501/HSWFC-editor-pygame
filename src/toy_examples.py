from terminal import Terminal, Void
from boundingbox import BoundingBox as BB
from side_properties import SidesDescriptor as SD
from adjacencies import Adjacency, Relation as R
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
