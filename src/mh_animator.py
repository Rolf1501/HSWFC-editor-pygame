# =====================================
# = PYGAME PROGRAM
# =====================================
# INIT pygame
# import pygame
from coord import Coord
import model_hierarchy_tree as mht
from boundingbox import BoundingBox as BB
from model import Part
from util_data import *
from collections import namedtuple
from communicator import Communicator, Verbosity as V
from model_tree import ModelTree
from copy import deepcopy
from geometric_solver import GeometricSolver as GS
from adjacencies import Adjacency, AdjacencyMatrix
from wfc import WFC
from adjacencies import Relation as R
from offsets import Offset


import open3d as o3d

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
    -1: Part(BB(0, 1, 0, 1, 0, 1), name="Root", colour=Colour(0,0,0)),
    0: Part(BB(0, 500, 0, 1, 0, 200), name="Car frame", colour=Colour(0,0,1)),
    9: Part(BB(0, 500, 0, 1, 0, 150), name="Frame", colour=Colour(0,0,1)),

    1: Part(BB(0, 300, 0, 1, 0, 75), name="Front frame", colour=Colour(0,1,1)),
    2: Part(BB(0, 250, 0, 1, 0, 50), name="Front axle", colour=Colour(.5,0,1)),
    3: Part(BB(0, 75, 0, 1, 0, 25), name="Front wheel", colour=Colour(1,.5,0)),
    4: Part(BB(0, 75, 0, 1, 0, 25), name="Front wheel", colour=Colour(1,.5,0)),

    5: Part(BB(0, 350, 0, 1, 0, 85), name="Rear frame", colour=Colour(0,0,1)),
    6: Part(BB(0, 300, 0, 1, 0, 50), name="Rear axle", colour=Colour(0,0,1)),
    7: Part(BB(0, 85, 0, 1, 0, 25), name="Rear wheel", colour=Colour(0,0,1)),
    8: Part(BB(0, 85, 0, 1, 0, 25), name="Rear wheel", colour=Colour(0,0,1)),
}

# original_parts = deepcopy(parts)
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

# Links specifying the relations between parts. Order independent.
# Each link follows: (source, attachment, side of source the attachment should face, [props])

# links = [
#     mht.MHLink(0, 1, mht.Cardinals.WEST,
#                [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.X)]),
#     mht.MHLink(0, 2, mht.Cardinals.EAST,
#                [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.X)]),
# ]


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
    # Normalize all parts' bounding boxes to fit the canvas.
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


# Convert cardinals to canvas specific coordinates.
def cardinal_to_normal_coord(card: Cardinals) -> Coord:
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
###   MAIN GAME LOOP   ###
##########################

start_next = True
automatic = False
backtracking = False  # True if the leaves have been reached.

geo_solver = GS(model_hierarchy_tree, parts, full_model_tree)
geo_solver.process(-1, processed)
vis = o3d.visualization.VisualizerWithKeyCallback()
fit_parts = fit_canvas(parts)

meshes = []
for p in fit_parts.values():
    mesh_box = o3d.geometry.TriangleMesh.create_box(p.extent.width(), p.extent.height(), p.extent.depth())
    mesh_box.paint_uniform_color([*p.colour])
    mesh_box.compute_vertex_normals()
    mesh_box.translate(p.extent.min_coord().to_tuple())
    meshes.append(mesh_box)

vis.create_window(window_name="test", width=WINDOW_SIZE[0], height=WINDOW_SIZE[1])

for m in meshes:
    vis.add_geometry(m)
vis.run()


terminals = {
    0: Part(BB.from_whd(1,1,1), name="void"),
    1: Part(BB.from_whd(4,1,2), name="4x2")

}

adjacencies = [
    Adjacency(1, {R(1, 0.5), R(0, 0.5)}, Offset(1,0,0), symmetric=True),
    Adjacency(1, {R(1, 1)}, Offset(-1,0,0), symmetric=True),
    Adjacency(1, {R(1, 0.8), R(0, 0.2)}, Offset(0,1,0), symmetric=True),
    Adjacency(1, {R(1, 1)}, Offset(0,-1,0), symmetric=True),
]

ADJ = AdjacencyMatrix(terminals.keys(), adjacencies)

print(ADJ)
# while running:
#     if start_next | automatic:
#         can_continue = process(-1, parts, model_hierarchy_tree, processed, full_model_tree)
#         automatic &= can_continue

#         start_next = False

#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         if event.type == pygame.KEYUP:
#             if event.key == pygame.K_l:
#                 range = 255.0 / len(parts)
#                 if not fit_parts:
#                     fit_parts = fit_canvas(parts)
#                 for k in fit_parts:
#                     comm.communicate(fit_parts[k], V.HIGH)
#                     drawing_queue.put(DrawJob(BB_to_rect(fit_parts[k].bb), Colour(randint(50,200), 200, range * max(k, 0))))
#             if event.key == pygame.K_n:
#                 if not fit_parts:
#                     fit_parts = fit_canvas(parts)
#                 for p in fit_parts.keys():
#                     bb = fit_parts[p].bb
#                     rotation_vector = cardinal_to_normal_coord(parts[p].absolute_north())
#                     rotation_vector.scale(28)
#                     rotation_vector += Coord(2,2)
#                     center = bb.center()
#                     rotation_indicator = BB(center.x, center.x + rotation_vector.x, center.y, center.y + rotation_vector.y)
#                     rotation_indicator_rect = BB_to_rect(rotation_indicator)
#                     drawing_queue.put(DrawJob(rotation_indicator_rect, Colour(0, 0, 0)))
#             if event.key == pygame.K_c:
#                 clear_drawing()

#             if event.key == pygame.K_SPACE:
#                 start_next = True
#             if event.key == pygame.K_p:
#                 for p in parts.items(): comm.communicate(p)
#             if event.key == pygame.K_a:
#                 automatic = True
#             if event.key == pygame.K_v:
#                 comm.cycle_verbosity(True)
#                 comm.communicate(f"Set verbosity level to {comm.verbosity}")

#     while not drawing_queue.empty():
#         job = drawing_queue.get()
#         pygame.draw.rect(draw_surface, job.colour, job.rect)

#     screen.blit(draw_surface, draw_surface_offset)
#     pygame.display.flip()
#     clock.tick(60)
