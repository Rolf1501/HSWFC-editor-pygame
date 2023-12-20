from coord import Coord
import model_hierarchy_tree as mht
from boundingbox import BoundingBox as BB
from model import Part
from util_data import *
from collections import namedtuple
from communicator import Communicator
from model_tree import ModelTree
from copy import deepcopy
from geometric_solver import GeometricSolver as GS

# import open3d as o3d
# import panda3d as p3d
# from panda3d.core import load_prc_file, NodePath, Material, PointLight
# from direct.showbase.ShowBase import ShowBase
from animator import GSAnimator
comm = Communicator()

select_mode = False

WINDOW_SIZE = [800, 600]
DRAW_SURF_SIZE = (760, 560)

drawing = False
running = True

start = (0, 0)
end = (0, 0)


# TOY EXAMPLE

# Collection of all parts. 
# Orientations are implicit. All initially point towards North.
# parts: dict[int, Part] = {
#     -1: Part(BB(0, 1, 0, 1, 0, 1), name="Root"),
#     0: Part(BB(0, 500, 0, 1, 0, 200), name="A", colour=Colour(1,0,0)),
#     1: Part(BB(0, 600, 0, 1, 0, 100), name="B", colour=Colour(0,1,0)),
#     2: Part(BB(0, 600, 0, 1, 0, 100), name="C", colour=Colour(0,0,1)),
# }

parts: dict[int, Part] = {
    -1: Part(BB(0, 1, 0, 1, 0, 1), name="Root", colour=Colour(0,0,0,1)),
    0: Part(BB(0, 500, 0, 10, 0, 200), name="Car frame", colour=Colour(0,0,0.5,0.5)),
    9: Part(BB(0, 500, 0, 10, 0, 150), name="Frame", colour=Colour(0,0,1,1)),

    1: Part(BB(0, 300, 0, 75, 0, 75), name="Front frame", colour=Colour(0,1,1,0.5)),
    2: Part(BB(0, 250, 0, 50, 0, 50), name="Front axle", colour=Colour(.5,0,1,1)),
    3: Part(BB(0, 75, 0, 75, 0, 25), name="Front wheel", colour=Colour(1,.5,0,1)),
    4: Part(BB(0, 75, 0, 75, 0, 25), name="Front wheel", colour=Colour(1,.5,0,1)),

    5: Part(BB(0, 350, 0, 85, 0, 85), name="Rear frame", colour=Colour(1,0,1,0.5)),
    6: Part(BB(0, 300, 0, 50, 0, 50), name="Rear axle", colour=Colour(1,0.5,0,1)),
    7: Part(BB(0, 85, 0, 85, 0, 25), name="Rear wheel", colour=Colour(0,0.5,1,1)),
    8: Part(BB(0, 85, 0, 85, 0, 25), name="Rear wheel", colour=Colour(0,0.5,1,1)),
}

# Edges determine the hierarchy such that for each edge (u, v), u consists of v.
edges = [
    mht.MHEdge(-1, 0),
    mht.MHEdge(-1, 1),
    mht.MHEdge(-1, 5),
    mht.MHEdge(0, 9),
    mht.MHEdge(1, 2),
    mht.MHEdge(1, 3),
    mht.MHEdge(1, 4),
    mht.MHEdge(5, 6),
    mht.MHEdge(5, 7),
    mht.MHEdge(5, 8)
]

links = [
    mht.MHLink(0, 1, mht.Cardinals.WEST,
               [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Z)]),
    mht.MHLink(0, 5, mht.Cardinals.EAST,
               [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Z)]),

    mht.MHLink(2, 3, mht.Cardinals.EAST,
               [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Z)]),
    mht.MHLink(2, 4, mht.Cardinals.WEST,
               [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Z)]),

    mht.MHLink(6, 7, mht.Cardinals.EAST,
               [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Z)]),
    mht.MHLink(6, 8, mht.Cardinals.WEST,
               [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Z)]),
]

full_model_tree = ModelTree.from_parts(parts, links)


def fit_canvas(parts: dict[int, Part]):
    """
    Normalize all parts' bounding boxes to fit the canvas.
    """
    fit_parts = deepcopy(parts)
    minx = MAX
    miny = MAX
    minz = MAX
    for k in fit_parts:
        if fit_parts[k].extent.minx < minx:
            minx = fit_parts[k].extent.minx
        if fit_parts[k].extent.miny < miny:
            miny = fit_parts[k].extent.miny
        if fit_parts[k].extent.minz < minz:
            minz = fit_parts[k].extent.minz

    translation = Coord(-minx, -miny, -minz)
    for k in fit_parts:
        fit_parts[k].extent.translate(translation)

    return fit_parts


def cardinal_to_normal_coord(card: Cardinals) -> Coord:
    """
    Convert cardinals to canvas specific coordinates.
    """
    if card == Cardinals.NORTH:
        return Coord(0, -1)
    elif card == Cardinals.EAST:
        return Coord(1, 0)
    elif card == Cardinals.SOUTH:
        return Coord(0, 1)
    elif card == Cardinals.WEST:
        return Coord(-1, 0)
    else:
        raise NotImplementedError(f"Cardinal must be valid. Got: {card}")


# The nodes ids in the hierarchy correspond to the keys of the parts, except for the root node.
nodes = [mht.MHNode(k, parts[k].name) for k in parts.keys()]
root = mht.MHNode(-1, "Root")
nodes.append(root)
model_hierarchy_tree = mht.MHTree(nodes, edges)

Iteration = namedtuple("Iteration", ["current_node_id", "model_tree", "parent_node_id"])
collapse_stack: list[Iteration] = [Iteration(-1, None, None)]

ProcessState = namedtuple("ProcessState", ["node_id", "parts", "processed"])
processed = {k: False for k in nodes}
process_log: list[ProcessState] = []



##########################
###   MAIN LOOP   ###
##########################

start_next = True
automatic = False
backtracking = False  # True if the leaves have been reached.

geo_solver = GS(model_hierarchy_tree, parts, full_model_tree)
geo_solver.process(-1, processed)
fit_parts = fit_canvas(geo_solver.parts)

gsanimator = GSAnimator(fit_parts)
gsanimator.run()
