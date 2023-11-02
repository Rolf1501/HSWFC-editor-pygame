
# =====================================
# = PYGAME PROGRAM
# =====================================
# INIT pygame
import pygame
from coord import Coord
import queue
import model_hierarchy_tree as mht
from boundingbox import BoundingBox as BB
from model import Part
from util_data import *
from collections import namedtuple
from model import Model

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

# orientations are implicit. All initially point towards North.
bbs = {
    0: BB(0,300,0,200), # Car frame
    1: BB(0,250,0,50), # Front axle
    2: BB(0,250,0,50), # Back axle
    3: BB(0,75,0,25), # Wheel
    4: BB(0,75,0,25), # Wheel   
    5: BB(0,75,0,25), # Wheel
    6: BB(0,75,0,25), # Wheel
    7: BB(0,25,0,25), # Nob
}

# Order independent
links = [
    mht.MHLink(0, 1, mht.Cardinals.WEST, [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Y)]),
    mht.MHLink(1, 4, mht.Cardinals.WEST, [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Y)]),
    mht.MHLink(1, 3, mht.Cardinals.EAST, [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Y)]),
    mht.MHLink(2, 5, mht.Cardinals.EAST, [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Y)]),
    mht.MHLink(2, 6, mht.Cardinals.WEST, [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Y)]),
    mht.MHLink(0, 2, mht.Cardinals.EAST, [mht.Properties(Operations.ORTH), mht.Properties(Operations.CENTER, Dimensions.Y)]),
    mht.MHLink(3, 4, None, [mht.Properties(Operations.SYM, Dimensions.X)]),
    mht.MHLink(5, 6, None, [mht.Properties(Operations.SYM, Dimensions.X)]),
    mht.MHLink(3, 7, mht.Cardinals.NORTH, [mht.Properties(Operations.CENTER, Dimensions.X)]),
]

parts: dict[int, Part] = {}
for key in bbs.keys():
    parts[key] = Part(bbs[key], 0)

# Construct model hierarchy based on links.
model = Model(parts, links)

model.solve()

# Normalise all parts' bounding boxes to fit the canvas.
minx = 999999
miny = 999999
for k in parts.keys():
    if parts[k].size.minx < minx:
        minx = parts[k].size.minx
    if parts[k].size.miny < miny:
        miny = parts[k].size.miny

translate = BB(-minx, -minx, -miny, -miny)
for k in parts.keys():
    parts[k].size += translate


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
    
    while not drawing_queue.empty():
        job = drawing_queue.get()
        pygame.draw.rect(draw_surface, job.colour, job.rect)

    screen.blit(draw_surface, draw_surface_offset)
    pygame.display.flip()
    clock.tick(60)