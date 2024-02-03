from terminal import Terminal, Void
from boundingbox import BoundingBox as BB
from side_descriptor import SidesDescriptor as SD, SideProperties as SP
from adjacencies import Adjacency, AdjacencyAny, Relation as R
from offsets import Offset, OffsetFactory
from util_data import Cardinals as C, Dimensions as D, Colour
from coord import Coord
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

    def example_meta_tiles_2():
        terminals = {
            0: Terminal((2, 1, 3), Colour(0.3, 0.6, 0.6, 1)),  # 2x3; turquoise ish
            1: Terminal((4, 1, 2), Colour(0.8, 0.3, 0, 1)),  # 4x2; orange ish
            2: Terminal((3, 1, 1), Colour(0.2, 0.1, 0.8, 1)),  # 3x1; blue-purple ish
            -1: Void((1, 1, 1), Colour(1, 1, 1, 0.5)),
        }

        adjacencies = [
            Adjacency(0, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.WEST.value), True),
            Adjacency(2, [R(0, 1), R(1, 1), R(2, 1)], Offset(*C.NORTH.value), True),
            Adjacency(2, [R(0, 1), R(1, 1), R(2, 1)], Offset(*C.EAST.value), True),
            Adjacency(2, [R(0, 1), R(1, 1), R(2, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(2, [R(0, 1), R(1, 1), R(2, 1)], Offset(*C.WEST.value), True),
        ]

        top_bottom_any = {
            AdjacencyAny(i, o, True, 1)
            for i in terminals
            for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]
        }
        void_any = {
            AdjacencyAny(-1, o, True, 0.001) for o in OffsetFactory().get_offsets()
        }  # Void may be placed next to anything
        return terminals, adjacencies.union(top_bottom_any).union(void_any), None

    def example_meta_tiles_fit_area():
        terminals = {
            0: Terminal((2, 1, 3), Colour(0.3, 0.6, 0.6, 1)),  # 2x3; turquoise ish
            1: Terminal((4, 1, 2), Colour(0.8, 0.3, 0, 1)),  # 4x2; orange ish
            2: Terminal((3, 1, 1), Colour(0.2, 0.1, 0.8, 1)),  # 3x1; blue-purple ish
            3: Terminal((2, 2, 2), Colour(0.8, 0.1, 0.2, 1)),  # 2x2; red ish
            -1: Void((1, 1, 1), colour=Colour(1, 1, 1, 0.5)),
        }

        adjacencies = [
            Adjacency(0, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.WEST.value), True),
            Adjacency(2, [R(0, 1), R(1, 1), R(2, 1)], Offset(*C.NORTH.value), True),
            Adjacency(2, [R(0, 1), R(1, 1), R(2, 1)], Offset(*C.EAST.value), True),
            Adjacency(2, [R(0, 1), R(1, 1), R(2, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(2, [R(0, 1), R(1, 1), R(2, 1)], Offset(*C.WEST.value), True),
            Adjacency(
                3, [R(0, 1), R(1, 1), R(2, 1), R(3, 1)], Offset(*C.NORTH.value), True
            ),
            Adjacency(
                3, [R(0, 1), R(1, 1), R(2, 1), R(3, 1)], Offset(*C.EAST.value), True
            ),
            Adjacency(
                3, [R(0, 1), R(1, 1), R(2, 1), R(3, 1)], Offset(*C.SOUTH.value), True
            ),
            Adjacency(
                3, [R(0, 1), R(1, 1), R(2, 1), R(3, 1)], Offset(*C.WEST.value), True
            ),
        ]

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

    def example_meta_tiles_fit_area_simple():
        terminals = {
            0: Terminal((2, 2, 1), Colour(0.3, 0.6, 0.6, 1)),  # 2x3; turquoise ish
            1: Terminal((3, 1, 2), Colour(0.8, 0.3, 0, 1)),  # 4x2; orange ish
            -1: Void((1, 1, 1), colour=Colour(1, 1, 1, 0.5)),
        }

        adjacencies = [
            Adjacency(0, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.TOP.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.BOTTOM.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.TOP.value), True),
            Adjacency(1, [R(0, 1), R(1, 1)], Offset(*C.BOTTOM.value), True),
        ]

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

    def example_meta_tiles_zebra_horizontal():
        terminals = {
            0: Terminal((3, 1, 1), Colour(0.3, 0.6, 0.6, 1)),  # 2x3; turquoise ish
            1: Terminal((3, 1, 1), Colour(0.8, 0.3, 0, 1)),  # 4x2; orange ish
            # 2: Terminal((3,1,1),  Colour(0.2,0.1,0.8,1)), # 3x1; blue-purple ish
            # 3: Terminal((2,2,2),  Colour(0.8,0.1,0.2,1)), # 3x1; red ish
            # -1: Void((1,1,1)),
        }

        adjacencies = [
            Adjacency(0, {}, Offset(*C.NORTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(0, {}, Offset(*C.SOUTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.WEST.value), True),
            # Adjacency(2, [R(1, 1)], Offset(*C.NORTH.value), True),
            # Adjacency(2, [R(2, 1)], Offset(*C.EAST.value), True),
            # Adjacency(2, [R(1, 1)], Offset(*C.SOUTH.value), True),
            # Adjacency(2, [R(2, 1)], Offset(*C.WEST.value), True),
            # Adjacency(3, [R(0, 1), R(1, 1), R(2, 1), R(3, 1)], Offset(*C.NORTH.value), True),
            # Adjacency(3, [R(0, 1), R(1, 1), R(2, 1), R(3, 1)], Offset(*C.EAST.value), True),
            # Adjacency(3, [R(0, 1), R(1, 1), R(2, 1), R(3, 1)], Offset(*C.SOUTH.value), True),
            # Adjacency(3, [R(0, 1), R(1, 1), R(2, 1), R(3, 1)], Offset(*C.WEST.value), True),
        ]

        top_bottom_any = {
            AdjacencyAny(t, o, True, 1)
            for t in terminals
            for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]
        }
        # void_any = {AdjacencyAny(-1, o, True, 0.001) for o in OffsetFactory().get_offsets()} # Void may be placed next to anything
        return terminals, adjacencies.union(top_bottom_any), None

    def example_meta_tiles():
        terminals = {
            0: Terminal((2, 1, 3), Colour(0.3, 0.6, 0.6, 1)),  # 2x3; cyan ish
            1: Terminal((4, 1, 2), Colour(0.8, 0.3, 0, 1)),  # 4x2; orangeish
            2: Void((1, 1, 1)),
        }

        adjacencies = [
            Adjacency(0, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.WEST.value), True),
        ]

        top_bottom_any = {
            AdjacencyAny(i, o, True, 1)
            for i in terminals
            for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]
        }
        void_any = {
            AdjacencyAny(2, o, True, 0.001) for o in OffsetFactory().get_offsets()
        }  # Void may be placed next to anything
        return terminals, adjacencies.union(top_bottom_any).union(void_any), None

    def example_big_tiles():
        terminals = {
            # For a cuboid with vertex ids:
            #   7------6
            #  /|     /|
            # 4------5 |
            # | 3----|-2
            # |/     |/
            # 0------1
            0: Terminal(
                (1, 1, 1),
                SD(north=SP.OPEN, east=SP.OPEN),
                Colour(0.5, 0, 0, 1),
            ),  # red; corner_04
            1: Terminal(
                (1, 1, 1),
                SD(north=SP.OPEN, west=SP.OPEN),
                Colour(0, 0.5, 0, 1),
            ),  # green; corner_15
            2: Terminal(
                (1, 1, 1),
                SD(south=SP.OPEN, west=SP.OPEN),
                Colour(0, 0, 0.5, 1),
            ),  # blue; corner_26
            3: Terminal(
                (1, 1, 1),
                SD(south=SP.OPEN, east=SP.OPEN),
                Colour(0, 0.5, 0.5, 1),
            ),  # cyan; corner_37
            4: Terminal(
                (1, 1, 1),
                SD(north=SP.OPEN, east=SP.OPEN, west=SP.OPEN),
                Colour(0.5, 0, 0.5, 1),
            ),  # magenta; face_0154
            5: Terminal(
                (1, 1, 1),
                SD(south=SP.OPEN, east=SP.OPEN, west=SP.OPEN),
                Colour(0.5, 0.5, 0, 1),
            ),  # yellow; face_3267
            6: Terminal(
                (1, 1, 1),
                SD(north=SP.OPEN, south=SP.OPEN, east=SP.OPEN),
                Colour(0, 0, 0, 1),
            ),  # black; face_0374
            7: Terminal(
                (1, 1, 1),
                SD(north=SP.OPEN, south=SP.OPEN, west=SP.OPEN),
                Colour(0.5, 0.5, 0.5, 1),
            ),  # white; face_1265
        }

        adjacencies = [
            Adjacency(0, {}, Offset(*C.NORTH.value), True),
            Adjacency(0, {}, Offset(*C.EAST.value), True),
            Adjacency(0, {}, Offset(*C.SOUTH.value), True),
            Adjacency(0, {}, Offset(*C.WEST.value), True),
            Adjacency(1, {}, Offset(*C.NORTH.value), True),
            Adjacency(1, {}, Offset(*C.EAST.value), True),
            Adjacency(1, {}, Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(2, {}, Offset(*C.NORTH.value), True),
            Adjacency(2, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(2, [R(1, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(2, {}, Offset(*C.WEST.value), True),
            Adjacency(3, {}, Offset(*C.NORTH.value), True),
            Adjacency(3, [R(2, 1)], Offset(*C.EAST.value), True),
            Adjacency(3, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(3, [R(2, 1)], Offset(*C.WEST.value), True),
            Adjacency(4, {}, Offset(*C.NORTH.value), True),
            Adjacency(4, [R(4, 1), R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(4, [R(2, 1), R(3, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(4, [R(4, 1), R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(5, [R(4, 1)], Offset(*C.NORTH.value), True),
            Adjacency(5, [R(5, 1), R(2, 1)], Offset(*C.EAST.value), True),
            Adjacency(5, [R(4, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(5, [R(5, 1), R(3, 1)], Offset(*C.WEST.value), True),
            Adjacency(6, [R(6, 1), R(3, 1)], Offset(*C.NORTH.value), True),
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
            Adjacency(6, [R(1, 1), R(2, 1)], Offset(*C.WEST.value), True),
            Adjacency(7, [R(7, 1), R(2, 1)], Offset(*C.NORTH.value), True),
            Adjacency(7, [R(6, 1), R(0, 1), R(3, 1)], Offset(*C.EAST.value), True),
            Adjacency(
                7,
                {
                    R(7, 1),
                    R(0, 1),
                },
                Offset(*C.SOUTH.value),
                True,
            ),
            Adjacency(7, [R(6, 1)], Offset(*C.WEST.value), True),
        ]

        top_bottom_any = {
            AdjacencyAny(i, o, True)
            for i in terminals
            for o in [Offset(*C.TOP.value), Offset(*C.BOTTOM.value)]
        }

        return terminals, adjacencies.union(top_bottom_any), None

    def example_zebra_horizontal():
        terminals = {
            0: Terminal((1, 1, 1), Colour(1, 1, 1, 1)),  # white
            1: Terminal((1, 1, 1), Colour(0, 0, 0, 1)),  # black
        }

        adjacencies = [
            Adjacency(0, [R(0, 1)], Offset(*C.TOP.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.TOP.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.NORTH.value), False),
            Adjacency(0, [R(1, 1)], Offset(*C.NORTH.value), False),
        ]

        return terminals, adjacencies

    def example_zebra_horizontal_3():
        terminals = {
            0: Terminal((1, 1, 1), Colour(1, 1, 1, 1)),  # white
            1: Terminal((1, 1, 1), Colour(0, 0, 0, 1)),  # black
            2: Terminal((1, 1, 1), Colour(1, 1, 0, 1)),  # yellow
        }

        adjacencies = [
            Adjacency(0, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.WEST.value), True),
            Adjacency(2, [R(2, 1)], Offset(*C.NORTH.value), True),
            Adjacency(2, [R(2, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(2, 1)], Offset(*C.TOP.value), True),
            Adjacency(0, [R(1, 1)], Offset(*C.TOP.value), True),
            Adjacency(2, [R(0, 1)], Offset(*C.TOP.value), True),
        ]

        return terminals, adjacencies, None

    def example_zebra_vertical_3():
        terminals = {
            0: Terminal((1, 1, 1), Colour(1, 1, 1, 1)),  # white
            1: Terminal((1, 1, 1), Colour(0, 0, 0, 1)),  # black
            2: Terminal((1, 1, 1), Colour(1, 1, 0, 1)),  # yellow
        }

        adjacencies = [
            Adjacency(0, [R(0, 1)], Offset(*C.NORTH.value), True),
            # Adjacency(0, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.NORTH.value), True),
            # Adjacency(1, [R(1, 1)], Offset(*C.WEST.value), True),
            Adjacency(2, [R(2, 1)], Offset(*C.NORTH.value), True),
            # Adjacency(2, [R(2, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(2, 1)], Offset(*C.WEST.value), False),
            Adjacency(0, [R(1, 1)], Offset(*C.WEST.value), False),
            Adjacency(2, [R(0, 1)], Offset(*C.WEST.value), False),
        ]

        return terminals, adjacencies, None

    def example_zebra_vertical():
        terminals = {
            0: Terminal((1, 1, 1), Colour(0.7, 0.7, 0.7, 1)),  # white
            1: Terminal(
                (1, 1, 1),
                Colour(0.05, 0.05, 0.05, 1),
            ),  # black
        }

        adjacencies = [
            Adjacency(0, [R(0, 1)], Offset(*C.TOP.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.TOP.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(0, [R(1, 1)], Offset(*C.WEST.value), True),
        ]

        return terminals, adjacencies, None

    def example_slanted():
        terminals = {
            0: Terminal((1, 1, 1), Colour(0.8, 0, 0, 1)),
            1: Void((1, 1, 1)),
        }

        adjacencies = [
            Adjacency(0, [R(0, 0.8), R(1, 0.2)], Offset(*C.WEST.value), True),
            Adjacency(0, [R(0, 0.2), R(1, 0.8)], Offset(*C.TOP.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.TOP.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.NORTH.value), True),
        ]
        return terminals, adjacencies, None

    def example_two_tiles():
        x, y, z = 2, 1, 2
        mask0 = np.full((y, x, z), True)
        mask0[0, 1, 1] = False  # Create a small L shape

        terminals = {
            0: Terminal(
                Coord(x, y, z),
                Colour(0.3, 0.6, 0.6, 1),
                mask=mask0,
            ),
            1: Terminal(Coord(x, y, z), Colour(0.8, 0.3, 0, 1), mask=mask0),
        }

        adjacencies = [
            Adjacency(1, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.WEST.value), True),
            # Adjacency(0, [R(1, 1)], Offset(*C.NORTH.value), True),
            # Adjacency(0, [R(1, 1)], Offset(*C.EAST.value), True),
            # Adjacency(0, [R(1, 1)], Offset(*C.SOUTH.value), True),
            # Adjacency(0, [R(1, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.WEST.value), True),
            # Adjacency(1, [R(1, 1)], Offset(*C.TOP.value), True),
            # Adjacency(1, [R(1, 1)], Offset(*C.BOTTOM.value), True),
            # Adjacency(0, [R(0, 1)], Offset(*C.TOP.value), True),
            # Adjacency(0, [R(0, 1)], Offset(*C.BOTTOM.value), True),
        ]

        return terminals, adjacencies, None

    def example_two_tiles_3D():
        x0, y0, z0 = 2, 2, 2
        mask0 = np.full((y0, x0, z0), True)
        mask0[0, 0, 0] = False  # Create a small L shape
        mask0[0, 1, 0] = False  # Create a small L shape
        x1, y1, z1 = 1, 1, 1
        mask1 = np.full((y1, x1, z1), True)

        terminals = {
            0: Terminal(
                Coord(x0, y0, z0),
                Colour(0.3, 0.6, 0.6, 1),
                mask=mask0,
            ),
            1: Terminal(
                Coord(x1, y1, z1),
                Colour(0.8, 0.3, 0, 1),
                mask=mask1,
            ),
        }

        adjacencies = [
            # 1 - 1
            Adjacency(1, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(0, [R(1, 1)], Offset(*C.TOP.value), True),
            Adjacency(0, [R(1, 1)], Offset(*C.BOTTOM.value), True),
            # 0 - 0
            Adjacency(0, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.TOP.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.BOTTOM.value), True),
            # 1 - 1
            Adjacency(1, [R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.TOP.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.BOTTOM.value), True),
        ]

        return terminals, adjacencies, None

    def example_three_tiles_3d_fallback():
        x0, y0, z0 = 2, 2, 2
        mask0 = np.full((y0, x0, z0), True)
        mask0[0, 0, 1] = False  # Create a small L shape
        mask0[0, 1, 1] = False  # Create a small L shape

        x1, y1, z1 = 2, 2, 2
        mask1 = np.full((y1, x1, z1), True)
        mask1[1, 0, 0] = False  # Create a small L shape
        mask1[1, 1, 0] = False  # Create a small L shape

        x2, y2, z2 = 1, 1, 1
        mask2 = np.full(Coord(x2, y2, z2), True)

        terminals = {
            0: Terminal(
                Coord(x0, y0, z0),
                Colour(0.3, 0.6, 0.6, 1),
                mask=mask0,
            ),
            1: Terminal(
                Coord(x1, y1, z1),
                Colour(0.8, 0.3, 0, 1),
                mask=mask1,
            ),
            2: Terminal(Coord(x2, y2, z2), Colour(1, 1, 1, 0.5), mask=mask2),
        }

        fallback_weight = 0.000001
        adjacencies = [
            # 1 - 0
            Adjacency(1, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.TOP.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.BOTTOM.value), True),
            # 1 - 1
            Adjacency(1, [R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.TOP.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.BOTTOM.value), True),
            # 2 - All
            Adjacency(
                2,
                [R(0, fallback_weight), R(1, fallback_weight), R(2, fallback_weight)],
                Offset(*C.NORTH.value),
                True,
            ),
            Adjacency(
                2,
                [R(0, fallback_weight), R(1, fallback_weight), R(2, fallback_weight)],
                Offset(*C.EAST.value),
                True,
            ),
            Adjacency(
                2,
                [R(0, fallback_weight), R(1, fallback_weight), R(2, fallback_weight)],
                Offset(*C.SOUTH.value),
                True,
            ),
            Adjacency(
                2,
                [R(0, fallback_weight), R(1, fallback_weight), R(2, fallback_weight)],
                Offset(*C.WEST.value),
                True,
            ),
            Adjacency(
                2,
                [R(0, fallback_weight), R(1, fallback_weight), R(2, fallback_weight)],
                Offset(*C.TOP.value),
                True,
            ),
            Adjacency(
                2,
                [R(0, fallback_weight), R(1, fallback_weight), R(2, fallback_weight)],
                Offset(*C.BOTTOM.value),
                True,
            ),
        ]

        return (
            terminals,
            adjacencies,
            {0: 1, 1: 1, 2: fallback_weight},
        )

    def example_rotated_2d():
        x0, y0, z0 = 2, 1, 1
        x1, y1, z1 = 2, 1, 2
        terminals = {
            0: Terminal(
                Coord(x0, y0, z0),
                Colour(0.3, 0.6, 0.6, 1),
                distinct_orientations=[0, 1],
            ),
            1: Terminal(
                Coord(x1, y1, z1),
                Colour(0.8, 0.3, 0, 1),
            ),
        }
        adjacencies = [
            # 0 - 0
            Adjacency(0, [R(0, 1, [0, 1])], Offset(*C.NORTH.value), True),
            Adjacency(0, [R(0, 1, [0, 1])], Offset(*C.EAST.value), True),
            Adjacency(0, [R(0, 1, [0, 1])], Offset(*C.SOUTH.value), True),
            Adjacency(0, [R(0, 1, [0, 1])], Offset(*C.WEST.value), True),
            Adjacency(0, [R(0, 1, [0, 1])], Offset(*C.TOP.value), True),
            Adjacency(0, [R(0, 1, [0, 1])], Offset(*C.BOTTOM.value), True),
            # 0 - 1
            Adjacency(0, [R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(0, [R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(0, [R(1, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(0, [R(1, 1)], Offset(*C.WEST.value), True),
            Adjacency(0, [R(1, 1)], Offset(*C.TOP.value), True),
            Adjacency(0, [R(1, 1)], Offset(*C.BOTTOM.value), True),
            # 1 - 1
            Adjacency(1, [R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.TOP.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.BOTTOM.value), True),
        ]
        return terminals, adjacencies, None

    def example_two_tiles2():
        x, y, z = 2, 1, 2
        mask0 = np.full((y, x, z), True)
        mask1 = np.full((y, x, z), True)
        mask1[0, 1, 1] = False  # Create a small L shape

        terminals = {
            0: Terminal(
                Coord(x, y, z),
                Colour(0.3, 0.6, 0.6, 1),
                mask=mask0,
            ),
            1: Terminal(Coord(x, y, z), Colour(0.8, 0.3, 0, 1), mask=mask1),
        }

        adjacencies = [
            Adjacency(1, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.WEST.value), True),
            # Adjacency(0, [R(1, 1)], Offset(*C.NORTH.value), True),
            # Adjacency(0, [R(1, 1)], Offset(*C.EAST.value), True),
            # Adjacency(0, [R(1, 1)], Offset(*C.SOUTH.value), True),
            # Adjacency(0, [R(1, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.WEST.value), True),
            # Adjacency(1, [R(1, 1)], Offset(*C.TOP.value), True),
            # Adjacency(1, [R(1, 1)], Offset(*C.BOTTOM.value), True),
            # Adjacency(0, [R(0, 1)], Offset(*C.TOP.value), True),
            # Adjacency(0, [R(0, 1)], Offset(*C.BOTTOM.value), True),
        ]

        return terminals, adjacencies, None

    def example_meta_tiles_simple():
        x, y, z = 2, 1, 2
        mask412 = np.full((y, x, z), True)
        terminals = {
            0: Terminal(
                Coord(x, y, z),
                Colour(0.3, 0.6, 0.6, 1),
                mask=mask412,
            ),  # 4x2; cyan ish
            1: Terminal(
                Coord(x, y, z),
                Colour(0.8, 0.3, 0, 1),
                mask=mask412,
            ),  # 4x2; orangeish
            2: Void((1, 1, 1), Colour(1, 1, 1, 0.5)),
        }

        adjacencies = [
            # Adjacency(0, [R(0, 1)], Offset(*C.NORTH.value), True),
            # Adjacency(0, [R(0, 1)], Offset(*C.EAST.value), True),
            # Adjacency(0, [R(0, 1)], Offset(*C.SOUTH.value), True),
            # Adjacency(0, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.WEST.value), True),
        ]

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

    def example_meta_tiles_layered():
        x0, y0, z0 = 4, 1, 2
        x1, y1, z1 = 2, 1, 3
        mask0 = np.full((y0, x0, z0), True)
        mask1 = np.full((y1, x1, z1), True)
        terminals = {
            0: Terminal(
                Coord(x0, y0, z0),
                Colour(0.3, 0.6, 0.6, 1),
                mask=mask0,
            ),  # cyan ish
            1: Terminal(
                Coord(x1, y1, z1),
                Colour(0.8, 0.3, 0, 1),
                mask=mask1,
            ),  # orangeish
            2: Void((1, 1, 1), Colour(1, 1, 1, 0.5)),
        }

        adjacencies = [
            Adjacency(0, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.EAST.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.TOP.value), True),
            Adjacency(0, [R(0, 1)], Offset(*C.BOTTOM.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.NORTH.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.EAST.value), True),
            Adjacency(1, [R(1, 1)], Offset(*C.SOUTH.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.WEST.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.TOP.value), True),
            Adjacency(1, [R(0, 1)], Offset(*C.BOTTOM.value), True),
            # Adjacency(1, [R(1, 1)], Offset(*C.TOP.value), True),
            # Adjacency(1, [R(1, 1)], Offset(*C.BOTTOM.value), True),
            # Adjacency(1, [R(1, 1)], Offset(*C.NORTH.value), True),
            # Adjacency(1, [R(1, 1)], Offset(*C.EAST.value), True),
            # Adjacency(1, [R(1, 1)], Offset(*C.SOUTH.value), True),
            # Adjacency(1, [R(1, 1)], Offset(*C.WEST.value), True),
        ]

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
