from terminal import Terminal, Void
from boundingbox import BoundingBox as BB
from side_properties import SidesDescriptor as SD, SideProperties as SP
from adjacencies import Adjacency, AdjacencyAny, Relation as R
from offsets import Offset, OffsetFactory
from util_data import Cardinals as C, Dimensions as D, Colour
import numpy as np


class ToyExamples:
    def __init__(self):
        pass

    def full_symmetric_axes():
        return {
            D.X: {D.Y, D.Z},
            D.Y: {D.X, D.Z},
            D.Z: {D.Y, D.X},
        }

    def example_meta_tiles_2(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        terminals = {
            0: Terminal(
                BB.from_whd(2, 1, 3), symmetry_axes, side_desc, Colour(0.3, 0.6, 0.6, 1)
            ),  # 2x3; turquoise ish
            1: Terminal(
                BB.from_whd(4, 1, 2), symmetry_axes, side_desc, Colour(0.8, 0.3, 0, 1)
            ),  # 4x2; orange ish
            2: Terminal(
                BB.from_whd(3, 1, 1), symmetry_axes, side_desc, Colour(0.2, 0.1, 0.8, 1)
            ),  # 3x1; blue-purple ish
            -1: Void(BB.from_whd(1, 1, 1), Colour(1, 1, 1, 0.5)),
        }

        adjacencies = {
            Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.EAST.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.EAST.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.WEST.value), True),
            Adjacency(2, {R(0, 1), R(1, 1), R(2, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(2, {R(0, 1), R(1, 1), R(2, 1)}, Offset(*C.EAST.value), True),
            Adjacency(2, {R(0, 1), R(1, 1), R(2, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(2, {R(0, 1), R(1, 1), R(2, 1)}, Offset(*C.WEST.value), True),
        }

        top_bottom_any = {
            AdjacencyAny(i, o, True, 1)
            for i in terminals
            for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]
        }
        void_any = {
            AdjacencyAny(-1, o, True, 0.001) for o in OffsetFactory().get_offsets()
        }  # Void may be placed next to anything
        return terminals, adjacencies.union(top_bottom_any).union(void_any), None

    def example_meta_tiles_fit_area(
        symmetry_axes=full_symmetric_axes(), side_desc=SD()
    ):
        terminals = {
            0: Terminal(
                BB.from_whd(2, 1, 3), symmetry_axes, side_desc, Colour(0.3, 0.6, 0.6, 1)
            ),  # 2x3; turquoise ish
            1: Terminal(
                BB.from_whd(4, 1, 2), symmetry_axes, side_desc, Colour(0.8, 0.3, 0, 1)
            ),  # 4x2; orange ish
            2: Terminal(
                BB.from_whd(3, 1, 1), symmetry_axes, side_desc, Colour(0.2, 0.1, 0.8, 1)
            ),  # 3x1; blue-purple ish
            3: Terminal(
                BB.from_whd(2, 2, 2), symmetry_axes, side_desc, Colour(0.8, 0.1, 0.2, 1)
            ),  # 2x2; red ish
            -1: Void(BB.from_whd(1, 1, 1), colour=Colour(1, 1, 1, 0.5)),
        }

        adjacencies = {
            Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.EAST.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.EAST.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.WEST.value), True),
            Adjacency(2, {R(0, 1), R(1, 1), R(2, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(2, {R(0, 1), R(1, 1), R(2, 1)}, Offset(*C.EAST.value), True),
            Adjacency(2, {R(0, 1), R(1, 1), R(2, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(2, {R(0, 1), R(1, 1), R(2, 1)}, Offset(*C.WEST.value), True),
            Adjacency(
                3, {R(0, 1), R(1, 1), R(2, 1), R(3, 1)}, Offset(*C.NORTH.value), True
            ),
            Adjacency(
                3, {R(0, 1), R(1, 1), R(2, 1), R(3, 1)}, Offset(*C.EAST.value), True
            ),
            Adjacency(
                3, {R(0, 1), R(1, 1), R(2, 1), R(3, 1)}, Offset(*C.SOUTH.value), True
            ),
            Adjacency(
                3, {R(0, 1), R(1, 1), R(2, 1), R(3, 1)}, Offset(*C.WEST.value), True
            ),
        }

        top_bottom_any = {
            AdjacencyAny(i, o, True, 1)
            for i in terminals
            for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]
        }
        void_any = {
            AdjacencyAny(-1, o, True, 0.001) for o in OffsetFactory().get_offsets()
        }  # Void may be placed next to anything
        return (
            terminals,
            adjacencies.union(top_bottom_any).union(void_any),
            {0: 1, 1: 1, 2: 1, 3: 1, -1: 0.001},
        )

    def example_meta_tiles_fit_area_simple(
        symmetry_axes=full_symmetric_axes(), side_desc=SD()
    ):
        terminals = {
            0: Terminal(
                BB.from_whd(2, 2, 1), symmetry_axes, side_desc, Colour(0.3, 0.6, 0.6, 1)
            ),  # 2x3; turquoise ish
            1: Terminal(
                BB.from_whd(3, 1, 2), symmetry_axes, side_desc, Colour(0.8, 0.3, 0, 1)
            ),  # 4x2; orange ish
            -1: Void(BB.from_whd(1, 1, 1), colour=Colour(1, 1, 1, 0.5)),
        }

        adjacencies = {
            Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.EAST.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.TOP.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.BOTTOM.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.EAST.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.TOP.value), True),
            Adjacency(1, {R(0, 1), R(1, 1)}, Offset(*C.BOTTOM.value), True),
        }

        top_bottom_any = {
            AdjacencyAny(i, o, True, 1)
            for i in terminals
            for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]
        }
        void_any = {
            AdjacencyAny(-1, o, True, 0.001) for o in OffsetFactory().get_offsets()
        }  # Void may be placed next to anything
        return (
            terminals,
            adjacencies.union(top_bottom_any).union(void_any),
            {0: 1, 1: 1, 2: 1, 3: 1, -1: 0.001},
        )

    def example_meta_tiles_zebra_horizontal(
        symmetry_axes=full_symmetric_axes(), side_desc=SD()
    ):
        terminals = {
            0: Terminal(
                BB.from_whd(3, 1, 1), symmetry_axes, side_desc, Colour(0.3, 0.6, 0.6, 1)
            ),  # 2x3; turquoise ish
            1: Terminal(
                BB.from_whd(3, 1, 1), symmetry_axes, side_desc, Colour(0.8, 0.3, 0, 1)
            ),  # 4x2; orange ish
            # 2: Terminal(BB.from_whd(3,1,1), symmetry_axes, side_desc, Colour(0.2,0.1,0.8,1)), # 3x1; blue-purple ish
            # 3: Terminal(BB.from_whd(2,2,2), symmetry_axes, side_desc, Colour(0.8,0.1,0.2,1)), # 3x1; red ish
            # -1: Void(BB.from_whd(1,1,1)),
        }

        adjacencies = {
            Adjacency(0, {}, Offset(*C.NORTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.EAST.value), True),
            Adjacency(0, {}, Offset(*C.SOUTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.EAST.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.WEST.value), True),
            # Adjacency(2, {R(1, 1)}, Offset(*C.NORTH.value), True),
            # Adjacency(2, {R(2, 1)}, Offset(*C.EAST.value), True),
            # Adjacency(2, {R(1, 1)}, Offset(*C.SOUTH.value), True),
            # Adjacency(2, {R(2, 1)}, Offset(*C.WEST.value), True),
            # Adjacency(3, {R(0, 1), R(1, 1), R(2, 1), R(3, 1)}, Offset(*C.NORTH.value), True),
            # Adjacency(3, {R(0, 1), R(1, 1), R(2, 1), R(3, 1)}, Offset(*C.EAST.value), True),
            # Adjacency(3, {R(0, 1), R(1, 1), R(2, 1), R(3, 1)}, Offset(*C.SOUTH.value), True),
            # Adjacency(3, {R(0, 1), R(1, 1), R(2, 1), R(3, 1)}, Offset(*C.WEST.value), True),
        }

        top_bottom_any = {
            AdjacencyAny(t, o, True, 1)
            for t in terminals
            for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]
        }
        # void_any = {AdjacencyAny(-1, o, True, 0.001) for o in OffsetFactory().get_offsets()} # Void may be placed next to anything
        return terminals, adjacencies.union(top_bottom_any), None

    def example_meta_tiles(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        terminals = {
            0: Terminal(
                BB.from_whd(2, 1, 3), symmetry_axes, side_desc, Colour(0.3, 0.6, 0.6, 1)
            ),  # 2x3; cyan ish
            1: Terminal(
                BB.from_whd(4, 1, 2), symmetry_axes, side_desc, Colour(0.8, 0.3, 0, 1)
            ),  # 4x2; orangeish
            2: Void(BB.from_whd(1, 1, 1)),
        }

        adjacencies = {
            Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.EAST.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.EAST.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.EAST.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.WEST.value), True),
        }

        top_bottom_any = {
            AdjacencyAny(i, o, True, 1)
            for i in terminals
            for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]
        }
        void_any = {
            AdjacencyAny(2, o, True, 0.001) for o in OffsetFactory().get_offsets()
        }  # Void may be placed next to anything
        return terminals, adjacencies.union(top_bottom_any).union(void_any), None

    def example_big_tiles(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        terminals = {
            # For a cuboid with vertex ids:
            #   7------6
            #  /|     /|
            # 4------5 |
            # | 3----|-2
            # |/     |/
            # 0------1
            0: Terminal(
                BB.from_whd(1, 1, 1),
                symmetry_axes,
                SD(north=SP.OPEN, east=SP.OPEN),
                Colour(0.5, 0, 0, 1),
            ),  # red; corner_04
            1: Terminal(
                BB.from_whd(1, 1, 1),
                symmetry_axes,
                SD(north=SP.OPEN, west=SP.OPEN),
                Colour(0, 0.5, 0, 1),
            ),  # green; corner_15
            2: Terminal(
                BB.from_whd(1, 1, 1),
                symmetry_axes,
                SD(south=SP.OPEN, west=SP.OPEN),
                Colour(0, 0, 0.5, 1),
            ),  # blue; corner_26
            3: Terminal(
                BB.from_whd(1, 1, 1),
                symmetry_axes,
                SD(south=SP.OPEN, east=SP.OPEN),
                Colour(0, 0.5, 0.5, 1),
            ),  # cyan; corner_37
            4: Terminal(
                BB.from_whd(1, 1, 1),
                symmetry_axes,
                SD(north=SP.OPEN, east=SP.OPEN, west=SP.OPEN),
                Colour(0.5, 0, 0.5, 1),
            ),  # magenta; face_0154
            5: Terminal(
                BB.from_whd(1, 1, 1),
                symmetry_axes,
                SD(south=SP.OPEN, east=SP.OPEN, west=SP.OPEN),
                Colour(0.5, 0.5, 0, 1),
            ),  # yellow; face_3267
            6: Terminal(
                BB.from_whd(1, 1, 1),
                symmetry_axes,
                SD(north=SP.OPEN, south=SP.OPEN, east=SP.OPEN),
                Colour(0, 0, 0, 1),
            ),  # black; face_0374
            7: Terminal(
                BB.from_whd(1, 1, 1),
                symmetry_axes,
                SD(north=SP.OPEN, south=SP.OPEN, west=SP.OPEN),
                Colour(0.5, 0.5, 0.5, 1),
            ),  # white; face_1265
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
            Adjacency(5, {R(5, 1), R(2, 1)}, Offset(*C.EAST.value), True),
            Adjacency(5, {R(4, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(5, {R(5, 1), R(3, 1)}, Offset(*C.WEST.value), True),
            Adjacency(6, {R(6, 1), R(3, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(6, {}, Offset(*C.EAST.value), True),
            Adjacency(
                6,
                {
                    R(6, 1),
                    R(0, 1),
                },
                Offset(*C.SOUTH.value),
                True,
            ),
            Adjacency(6, {R(1, 1), R(2, 1)}, Offset(*C.WEST.value), True),
            Adjacency(7, {R(7, 1), R(2, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(7, {R(6, 1), R(0, 1), R(3, 1)}, Offset(*C.EAST.value), True),
            Adjacency(
                7,
                {
                    R(7, 1),
                    R(0, 1),
                },
                Offset(*C.SOUTH.value),
                True,
            ),
            Adjacency(7, {R(6, 1)}, Offset(*C.WEST.value), True),
        }

        top_bottom_any = {
            AdjacencyAny(i, o, True)
            for i in terminals
            for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]
        }

        return terminals, adjs.union(top_bottom_any), None

    def example_zebra_horizontal(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        terminals = {
            0: Terminal(
                BB.from_whd(1, 1, 1), symmetry_axes, side_desc, Colour(1, 1, 1, 1)
            ),  # white
            1: Terminal(
                BB.from_whd(1, 1, 1), symmetry_axes, side_desc, Colour(0, 0, 0, 1)
            ),  # black
        }

        adjs = {
            Adjacency(0, {R(0, 1)}, Offset(*C.TOP.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.TOP.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.NORTH.value), False),
            Adjacency(0, {R(1, 1)}, Offset(*C.NORTH.value), False),
        }

        return terminals, adjs

    def example_zebra_horizontal_3(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        terminals = {
            0: Terminal(
                BB.from_whd(1, 1, 1), symmetry_axes, side_desc, Colour(1, 1, 1, 1)
            ),  # white
            1: Terminal(
                BB.from_whd(1, 1, 1), symmetry_axes, side_desc, Colour(0, 0, 0, 1)
            ),  # black
            2: Terminal(
                BB.from_whd(1, 1, 1), symmetry_axes, side_desc, Colour(1, 1, 0, 1)
            ),  # yellow
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

        return terminals, adjs, None

    def example_zebra_vertical_3(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        terminals = {
            0: Terminal(
                BB.from_whd(1, 1, 1), symmetry_axes, side_desc, Colour(1, 1, 1, 1)
            ),  # white
            1: Terminal(
                BB.from_whd(1, 1, 1), symmetry_axes, side_desc, Colour(0, 0, 0, 1)
            ),  # black
            2: Terminal(
                BB.from_whd(1, 1, 1), symmetry_axes, side_desc, Colour(1, 1, 0, 1)
            ),  # yellow
        }

        adjs = {
            Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),
            # Adjacency(0, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.NORTH.value), True),
            # Adjacency(1, {R(1, 1)}, Offset(*C.WEST.value), True),
            Adjacency(2, {R(2, 1)}, Offset(*C.NORTH.value), True),
            # Adjacency(2, {R(2, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(2, 1)}, Offset(*C.WEST.value), False),
            Adjacency(0, {R(1, 1)}, Offset(*C.WEST.value), False),
            Adjacency(2, {R(0, 1)}, Offset(*C.WEST.value), False),
        }

        return terminals, adjs, None

    def example_zebra_vertical(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        terminals = {
            0: Terminal(
                BB.from_whd(1, 1, 1), symmetry_axes, side_desc, Colour(0.7, 0.7, 0.7, 1)
            ),  # white
            1: Terminal(
                BB.from_whd(1, 1, 1),
                symmetry_axes,
                side_desc,
                Colour(0.05, 0.05, 0.05, 1),
            ),  # black
        }

        adjs = {
            Adjacency(0, {R(0, 1)}, Offset(*C.TOP.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.TOP.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(0, {R(1, 1)}, Offset(*C.WEST.value), True),
        }

        return terminals, adjs, None

    def example_slanted(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        terminals = {
            0: Terminal(
                BB.from_whd(1, 1, 1), symmetry_axes, side_desc, Colour(0.8, 0, 0, 1)
            ),
            1: Void(BB.from_whd(1, 1, 1)),
        }

        adjs = {
            Adjacency(0, {R(0, 0.8), R(1, 0.2)}, Offset(*C.WEST.value), True),
            Adjacency(0, {R(0, 0.2), R(1, 0.8)}, Offset(*C.TOP.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.TOP.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.EAST.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.NORTH.value), True),
        }
        return terminals, adjs, None

    def example_meta_tiles_simple(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        x, y, z = 2, 1, 2
        mask412 = np.full((x, y, z), True)
        terminals = {
            0: Terminal(
                BB.from_whd(x, y, z),
                symmetry_axes,
                side_desc,
                Colour(0.3, 0.6, 0.6, 1),
                mask=mask412,
            ),  # 4x2; cyan ish
            1: Terminal(
                BB.from_whd(x, y, z),
                symmetry_axes,
                side_desc,
                Colour(0.8, 0.3, 0, 1),
                mask=mask412,
            ),  # 4x2; orangeish
            2: Void(BB.from_whd(1, 1, 1), Colour(1, 1, 1, 0.5)),
        }

        adjacencies = {
            # Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),
            # Adjacency(0, {R(0, 1)}, Offset(*C.EAST.value), True),
            # Adjacency(0, {R(0, 1)}, Offset(*C.SOUTH.value), True),
            # Adjacency(0, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.EAST.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.EAST.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.WEST.value), True),
        }

        top_bottom_any = {
            AdjacencyAny(i, o, True, 1)
            for i in terminals
            for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]
        }
        void_any = {
            AdjacencyAny(2, o, True, 0.001) for o in OffsetFactory().get_offsets()
        }  # Void may be placed next to anything

        return (
            terminals,
            adjacencies.union(top_bottom_any).union(void_any),
            {0: 1, 1: 1, 2: 0.001},
        )

    def example_meta_tiles_layered(symmetry_axes=full_symmetric_axes(), side_desc=SD()):
        x0, y0, z0 = 4, 1, 2
        x1, y1, z1 = 2, 1, 3
        mask0 = np.full((x0, y0, z0), True)
        mask1 = np.full((x1, y1, z1), True)
        terminals = {
            0: Terminal(
                BB.from_whd(x0, y0, z0),
                symmetry_axes,
                side_desc,
                Colour(0.3, 0.6, 0.6, 1),
                mask=mask0,
            ),  # cyan ish
            1: Terminal(
                BB.from_whd(x1, y1, z1),
                symmetry_axes,
                side_desc,
                Colour(0.8, 0.3, 0, 1),
                mask=mask1,
            ),  # orangeish
            2: Void(BB.from_whd(1, 1, 1), Colour(1, 1, 1, 0.5)),
        }

        adjacencies = {
            Adjacency(0, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.EAST.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.TOP.value), True),
            Adjacency(0, {R(0, 1)}, Offset(*C.BOTTOM.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.NORTH.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.EAST.value), True),
            Adjacency(1, {R(1, 1)}, Offset(*C.SOUTH.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.WEST.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.TOP.value), True),
            Adjacency(1, {R(0, 1)}, Offset(*C.BOTTOM.value), True),
            # Adjacency(1, {R(1, 1)}, Offset(*C.TOP.value), True),
            # Adjacency(1, {R(1, 1)}, Offset(*C.BOTTOM.value), True),
            # Adjacency(1, {R(1, 1)}, Offset(*C.NORTH.value), True),
            # Adjacency(1, {R(1, 1)}, Offset(*C.EAST.value), True),
            # Adjacency(1, {R(1, 1)}, Offset(*C.SOUTH.value), True),
            # Adjacency(1, {R(1, 1)}, Offset(*C.WEST.value), True),
        }

        # top_bottom_any = {
        #     AdjacencyAny(i, o, True, 1)
        #     for i in terminals
        #     for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]
        # }
        # void_any = {
        #     AdjacencyAny(2, o, True, 0.001) for o in OffsetFactory().get_offsets()
        # }  # Void may be placed next to anything

        return (
            terminals,
            adjacencies,
            # adjacencies.union(top_bottom_any).union(void_any),
            {0: 1, 1: 1, 2: 0.001},
        )
