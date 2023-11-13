
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
from model import Model
from communicator import Communicator, Verbosity as V
from model_tree import ModelTree
from queue import Queue
from copy import deepcopy
from random import randint

import open3d as o3d

comm = Communicator()

# pygame.init()

select_mode = False

WINDOW_SIZE = [800,600]
DRAW_SURF_SIZE = (760,560)
# font = pygame.font.SysFont('segoeuisymbol',20,16)

# clock = pygame.time.Clock()  # The clock is needed to regulate the update loop such that we can process input between the frames, see pygame doc
# screen = pygame.display.set_mode((WINDOW_SIZE[0], WINDOW_SIZE[1]))
# draw_surface = pygame.Surface(DRAW_SURF_SIZE,flags=pygame.HWSURFACE)

drawing = False
running = True

start = (0,0)
end = (0,0)
# screen.fill("white")
# draw_surface.fill("white")
# draw_surface_offset = (0,0)

Colour = namedtuple("Colour", ["r", "g", "b"])


# class DrawJob:
#     def __init__(self, rect: pygame.Rect, colour: Colour) -> None:
#         self.rect = rect
#         self.colour = colour

# def BB_to_rect(bb: BB):
#     return pygame.Rect(bb.minx, bb.miny, bb.width(), bb.height())

# def clear_drawing():
#     draw_surface.fill("white")

# drawing_queue = Queue[DrawJob]()


# TOY EXAMPLE

# Collection of all parts. 
# Orientations are implicit. All initially point towards North.
parts: dict[int, Part] = {
    -1: Part(BB(0,1,0,1), name="Root"),
    0: Part(BB(0,500,0,200), name="Car frame"),
    9: Part(BB(0,500,0,150), name="Frame"),

    1: Part(BB(0,300,0,75), name="Front frame"),
    2: Part(BB(0,250,0,50), name="Front axle"),
    3: Part(BB(0,75,0,25), name="Front wheel"),
    4: Part(BB(0,75,0,25), name="Front wheel"),

    5: Part(BB(0,350,0,85), name="Rear frame"),
    6: Part(BB(0,300,0,50), name="Rear axle"),
    7: Part(BB(0,85,0,25), name="Rear wheel"),
    8: Part(BB(0,85,0,25), name="Rear wheel"),
}

original_parts = deepcopy(parts)
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
links = [
    mht.MHLink(0, 1, mht.Cardinals.WEST, [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Y)]),
    mht.MHLink(0, 5, mht.Cardinals.EAST, [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Y)]),

    mht.MHLink(2, 3, mht.Cardinals.EAST, [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Y)]),
    mht.MHLink(2, 4, mht.Cardinals.WEST, [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Y)]),

    mht.MHLink(6, 7, mht.Cardinals.EAST, [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Y)]),
    mht.MHLink(6, 8, mht.Cardinals.WEST, [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Y)]),
]

full_model_tree = ModelTree.from_parts(parts, links)


def fit_canvas(parts: dict[int, Part]):
    # Normalize all parts' bounding boxes to fit the canvas.
    fit_parts = deepcopy(parts)
    minx = MAX
    miny = MAX
    minz = MAX
    for k in fit_parts:
        if fit_parts[k].bb.minx < minx:
            minx = fit_parts[k].bb.minx
        if fit_parts[k].bb.miny < miny:
            miny = fit_parts[k].bb.miny
        if fit_parts[k].bb.minz < minz:
            minz = fit_parts[k].bb.minz

    translation = Coord(-minx, -miny, -minz)
    for k in fit_parts:
        fit_parts[k].bb.translate(translation)
    
    return fit_parts

# Convert cardinals to canvas specific coordinates.
def cardinal_to_normal_coord(card: Cardinals) -> Coord:
    if card == Cardinals.NORTH:
        return Coord(0,-1)
    elif card == Cardinals.EAST:
        return Coord(1,0)
    elif card == Cardinals.SOUTH:
        return Coord(0,1)
    elif card == Cardinals.WEST:
        return Coord(-1,0)
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


def process(node_id: int, parts: dict[int, Part], mht: mht.MHTree, processed: dict[int, bool], full_model_tree: ModelTree):
    process_log.append(ProcessState(node_id, parts, processed))
    comm.communicate("Process log:")
    map(lambda pl: comm.communicate(pl), process_log)
    part = parts[node_id]
    part_info = (node_id, part.name)
    comm.communicate(f"Currently collapsing {part_info}...")
    children = list(mht.successors(node_id))
    
    # If the current node has children.
    if children:

        # Determine collapse order of children.
        comm.communicate(f"Determining sibling processing order...")

        sibling_links = [link for link in links if link.source in children and link.attachment in children]
        model_subtree = ModelTree(incoming_graph_data=full_model_tree.subgraph(children), links=sibling_links)

        # Processing order corresponds to the dependencies of the siblings.
        process_order = model_subtree.get_sibling_order()
        if process_order:
            comm.communicate(f"Found sibling order: {list(map(lambda sib: (sib, parts[sib].name), process_order))}", V.HIGH)
        else:
            comm.communicate(f"No more siblings found for {part_info}", V.HIGH)
        
        # Arrange all children.
        # Make sure children inherit translation and rotation from parent.
        for k in children:
            parts[k].rotation += parts[node_id].rotation
            parts[k].translation += parts[node_id].translation
        
        # Only include parts and links of the children.
        model = Model(
            {k: parts[k] for k in children},
            [l for l in links if l.source in children and l.attachment in children],
            model_subtree)

        model.solve()
        model.contain_in(part.bb)

        # Process each child following the sibling order.
        for node in process_order:
            process(node, parts, mht, processed, full_model_tree)

        # TODO: after all children have been processed, assemble here.
        # This is also the place where the tight fits can be made.
        # Make use of properties stored in the link between the two parts.
            # E.g. when aligning with center: move attachment to source along centerline until overlap occurs.
        processed[node_id] = True

        comm.communicate(f"All children of node {part_info} completed. Fitting parts...")
        comm.communicate(f"Parts status:", V.HIGH)
        for p in parts.items():
            comm.communicate(f"\t{p}", V.HIGH)

    else:
        comm.communicate("No more children found. Checking adjacent siblings processing status...")
        # TODO: collapse meta node into leaves.
        # If no other adjacent siblings have been processed yet, simply collapse.
        # Otherwise, check where that sibling is, how it related to this node and where it overlaps.
        # That determines the initial seeds/tiles for this part.
        # Collapse this section and return.
        # Return to parent when all children are processed.
        processed[node_id] = True

    comm.communicate(f"Processing node {(node_id, parts[node_id].name)} complete.")
    return True


##########################
###   MAIN GAME LOOP   ###
##########################

start_next = True
automatic = False
backtracking = False # True if the leaves have been reached.


process(-1, parts, model_hierarchy_tree, processed, full_model_tree)
vis = o3d.visualization.VisualizerWithKeyCallback()
fit_parts = fit_canvas(parts)

meshes = []
for p in fit_parts.values():
    mesh_box = o3d.geometry.TriangleMesh.create_box(p.bb.width(), p.bb.height(), p.bb.depth())
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_box.compute_vertex_normals()
    mesh_box.translate(p.bb.min_coord().to_tuple())
    meshes.append(mesh_box)

vis.create_window(window_name="test",width=WINDOW_SIZE[0],height=WINDOW_SIZE[1])
# map(lambda part: vis.add_geometry(part), meshes)
for m in meshes:
    vis.add_geometry(m)
vis.run()
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