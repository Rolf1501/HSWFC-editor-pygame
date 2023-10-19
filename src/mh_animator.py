
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

bbs = {
    0: BB(0,100,0,100),
    1: BB(0,200,0,50)
}

links = [
    mht.MHLink(0, 1, mht.Adjacency.EW, [mht.Properties.CENTERED_Y, mht.Properties.ORTH]),
    # mht.MHLink(0,1, [mht.Properties.CENTERED_X], mht.Adjacency.SN),
]

p1 = None
p2 = None

for l in links:
    p1 = Part(bbs[l.frm], 0)
    p2 = Part(bbs[l.to], 0)
    p1, p2 = handle_adjacency(p1, p2, l.adjacency)
    p1, p2 = handle_properties(p1, p2, l.properties)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_l:
                drawing_queue.put(DrawJob(BB_to_rect(p1.size), Colour(255,0,0)))
                drawing_queue.put(DrawJob(BB_to_rect(p2.size), Colour(0,255,0)))
            if event.key == pygame.K_c:
                clear_drawing()
    
    while not drawing_queue.empty():
        job = drawing_queue.get()
        pygame.draw.rect(draw_surface, job.colour, job.rect)

    screen.blit(draw_surface, draw_surface_offset)
    pygame.display.flip()
    clock.tick(60)