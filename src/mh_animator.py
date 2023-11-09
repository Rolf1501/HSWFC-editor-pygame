
# =====================================
# = PYGAME PROGRAM
# =====================================
# INIT pygame
import pygame
from coord import Coord
import model_hierarchy_tree as mht
from boundingbox import BoundingBox as BB
from model import Part
from util_data import *
from collections import namedtuple
from model import Model
from communicator import Communicator
from model_tree import ModelTree
from queue import Queue
from copy import deepcopy

comm = Communicator()

pygame.init()

select_mode = False

WINDOW_SIZE = [800,600]
DRAW_SURF_SIZE = (760,560)
font = pygame.font.SysFont('segoeuisymbol',20,16) 

clock = pygame.time.Clock()  # The clock is needed to regulate the update loop such that we can process input between the frames, see pygame doc
screen = pygame.display.set_mode((WINDOW_SIZE[0], WINDOW_SIZE[1]))
draw_surface = pygame.Surface(DRAW_SURF_SIZE,flags=pygame.HWSURFACE)

drawing = False
running = True

start = (0,0)
end = (0,0)
screen.fill("white")
draw_surface.fill("white")
draw_surface_offset = (0,0)

Colour = namedtuple("Colour", ["r", "g", "b"])


class DrawJob:
    def __init__(self, rect: pygame.Rect, colour: Colour) -> None:
        self.rect = rect
        self.colour = colour

def BB_to_rect(bb: BB):
    return pygame.Rect(bb.minx, bb.miny, bb.width(), bb.height())

def clear_drawing():
    draw_surface.fill("white")

drawing_queue = Queue[DrawJob]()


# TOY EXAMPLE

# Collection of all parts. 
# Orientations are implicit. All initially point towards North.
parts: dict[int, Part] = {
    -1: Part(BB(0,0,0,0), name="Root"),
    0: Part(BB(0,500,0,200), name="Car frame"),

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

full_model_tree = ModelTree.from_parts_and_links(parts, links)


def fit_canvas(parts: dict[int, Part]):
    # Normalise all parts' bounding boxes to fit the canvas.
    fit_parts = deepcopy(parts)
    minx = MAX
    miny = MAX
    for k in fit_parts:
        if fit_parts[k].size.minx < minx:
            minx = fit_parts[k].size.minx
        if fit_parts[k].size.miny < miny:
            miny = fit_parts[k].size.miny

    # translate = BB(-minx, -minx, -miny, -miny)
    translation = Coord(-minx, -miny)
    for k in fit_parts:
        # fit_parts[k].size += translate
        fit_parts[k].size.translate(translation)
    
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



# Construct model tree based on links.
# model = Model(parts, links)
# model.solve()


# The nodes ids in the hierarchy correspond to the keys of the parts, except for the root node.
nodes = [mht.MHNode(k, parts[k].name) for k in parts.keys()]
root = mht.MHNode(-1, "Root")
nodes.append(root)
model_hierarchy_tree = mht.MHTree(nodes, edges)

collapse_stack = []
collapse_stack.append(-1)


##########################
###   MAIN GAME LOOP   ###
##########################

start_next = True
automatic = False

while running:
    if start_next | automatic:
        if len(collapse_stack) > 0:
            current_node = collapse_stack.pop()
            comm.communicate(f"Currently collapsing {(current_node, parts[current_node].name)}...")

            children = list(model_hierarchy_tree.successors(current_node))
            
            if children:
                comm.communicate(f"Determining sibling processing order.")
                model_subtree = ModelTree(incoming_graph_data=full_model_tree.subgraph(children).copy())

                # Make sure children inherit translation and rotation from parent.
                for k in children:
                    parts[k].rotation += parts[current_node].rotation
                    parts[k].translation += parts[current_node].translation

                sibling_order = model_subtree.get_sibling_order()
                if sibling_order:
                    comm.communicate(f"Found sibling order: {list(map(lambda sib: (sib, parts[sib].name), sibling_order))}")
                else:
                    comm.communicate(f"No more siblings found for {(current_node, parts[current_node].name)}")
                
                # Only include parts and links of the children.
                model = Model(
                    {k: parts[k] for k in parts if k in children}, 
                    [l for l in links if l.source in children and l.attachment in children],
                    model_subtree)

                updated_parts = model.solve()
                model.set_parts_relative_to_container(parts[current_node].size)


                # Stack is LIFO, sibling order is in increasing order. So, append the nodes in reverse.
                sibling_order.reverse()
                for sib in sibling_order:
                    collapse_stack.append(sib)
                

                comm.communicate(f"Remaining collapse stack {collapse_stack}")
                comm.communicate(f"Parts status:")
                for p in parts.items():
                    comm.communicate(f"{p}")
            else:
                comm.communicate("No more children found.")
                # TODO:
                # Collapse this section and return
                # Return to parent when all children are processed.
            start_next = False
        else:
            comm.communicate("Collapse stack is empty. No more parts to process.")
            start_next = False
            automatic = False


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_l:
                range = 255.0 / len(parts)
                fit_parts = fit_canvas(parts)
                for k in fit_parts:
                    comm.communicate(fit_parts[k])
                    drawing_queue.put(DrawJob(BB_to_rect(fit_parts[k].size), Colour(0, 200, range * max(k, 0))))
            if event.key == pygame.K_n:
                for p in parts.keys():
                    bb = parts[p].size
                    rotation_vector = cardinal_to_normal_coord(parts[p].absolute_north())
                    rotation_vector.scale(28)
                    rotation_vector += Coord(2,2)
                    center = bb.center()
                    rotation_indicator = BB(center.x, center.x + rotation_vector.x, center.y, center.y + rotation_vector.y)
                    rotation_indicator_rect = BB_to_rect(rotation_indicator)
                    drawing_queue.put(DrawJob(rotation_indicator_rect, Colour(0, 0, 0)))
            if event.key == pygame.K_c:
                clear_drawing()

            if event.key == pygame.K_SPACE:
                start_next = True
            if event.key == pygame.K_p:
                for p in parts.items(): comm.communicate(p)
            if event.key == pygame.K_a:
                automatic = True
    
    while not drawing_queue.empty():
        job = drawing_queue.get()
        pygame.draw.rect(draw_surface, job.colour, job.rect)

    screen.blit(draw_surface, draw_surface_offset)
    pygame.display.flip()
    clock.tick(60)