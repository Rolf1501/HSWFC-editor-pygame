
# =====================================
# = PYGAME PROGRAM
# =====================================
# INIT pygame
import pygame
from coord import Coord
import queue
import model_hierarchy_tree as mht
from boundingbox import BoundingBox as BB
from model_hierarchy import Part, handle_adjacency, handle_properties

from collections import namedtuple
import math

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

drawing_queue = queue.Queue[DrawJob]()


# TOY EXAMPLE

# orientations are implicit. All initially point towards North (0,1).
bbs = {
    0: BB(0,100,0,100),
    1: BB(0,200,0,50),
    2: BB(0,75,0,150)
}

links = [
    # mht.MHLink(1, 0, mht.Cardinals.EAST, [mht.Properties(mht.Operations.CENTER, mht.Dimensions.Y)]),
    mht.MHLink(0, 1, mht.Cardinals.WEST, [mht.Properties(mht.Operations.ORTH, mht.Dimensions.Y), mht.Properties(mht.Operations.CENTER, mht.Dimensions.Y)]),
    
    mht.MHLink(1, 2, mht.Cardinals.EAST, [mht.Properties(mht.Operations.CENTER, mht.Dimensions.Y)]),
]

parts: dict[int, Part] = {}
linkages = {} # pointers to corresponding groups.
groups: dict[int, set] = {}
for key in bbs.keys():
    parts[key] = Part(bbs[key], 0)
    linkages[key] = key
    groups[key] = {key}


# Determine final rotations of all parts.
for l in links:
    rotation = 0
    attachment_group = linkages[l.attachment]
    source_group = linkages[l.source]

    for p in l.properties:
        if p.operation == mht.Operations.ORTH:
            rotation = 90 # rotation in degrees

    # Propagate rotation to all attachment group parts to ensure uniformity.
    for elem in groups[attachment_group]:
        parts[elem].rotation += rotation + parts[l.source].rotation

    if linkages[l.source] != linkages[l.attachment]:
        # Merge groups. Remove merged attachment group.
        groups[source_group].update(groups.pop(attachment_group))

        # Update group reference.
        linkages[l.attachment] = linkages[l.source]

def rotate(bb: BB, rotation, degrees=True):
    """
    Currently done in 2D. Needs extension for 3D later
    """
    size_vector = Coord(bb.width(), bb.height())
    # rotation: [[cos a, -sin a], [sin a, cos a]]  [x, y]
    # rotated vector: (x * cos a - y * sin a), (x * sin a + y * cos a)
    if degrees:
        tot = 360.0
    else:
        raise NotImplementedError("Only supports degrees for now.")

    rotation_norm = rotation % tot
    rad = math.radians(rotation_norm)
    cosa = math.cos(rad)
    sina = math.sin(rad)
    rot_coord = Coord(size_vector.x * cosa - size_vector.y * sina, size_vector.x * sina + size_vector.y * cosa)
    trans_rot_coord = rot_coord + Coord(bb.minx, bb.miny)

    rot_bb = BB(
        min(bb.minx, trans_rot_coord.x), 
        max(0, trans_rot_coord.x), 
        min(bb.miny, trans_rot_coord.y), 
        max(0, trans_rot_coord.y))
    
    rot_bb.to_positive()

    return Part(rot_bb, rotation)

# Rotate all parts to their final rotated state.
for k in parts.keys():
    parts[k] = rotate(parts[k].size, parts[k].rotation)

# Assemble the parts.
for l in links:
    p1, p2 = handle_adjacency(parts[l.source], parts[l.attachment], l.adjacency)
    p1, p2 = handle_properties(parts[l.source], parts[l.attachment], l.properties)
    parts[l.source] = p1
    parts[l.attachment] = p2

minx = 0
miny = 0
for k in parts.keys():
    print(parts[k].size)
    if parts[k].size.minx < minx:
        minx = parts[k].size.minx
    if parts[k].size.miny < miny:
        miny = parts[k].size.miny

translate = BB(-minx, -minx, -miny, -miny)
for k in parts.keys():
    parts[k].size += translate
    print(parts[k].size)
##########################
###   MAIN GAME LOOP   ###
##########################

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_l:
                range = 255.0 / len(parts)
                for p in parts.keys():
                    drawing_queue.put(DrawJob(BB_to_rect(parts[p].size), Colour(0, 200, range * p)))
            if event.key == pygame.K_c:
                clear_drawing()
    
    while not drawing_queue.empty():
        job = drawing_queue.get()
        pygame.draw.rect(draw_surface, job.colour, job.rect)

    screen.blit(draw_surface, draw_surface_offset)
    pygame.display.flip()
    clock.tick(60)