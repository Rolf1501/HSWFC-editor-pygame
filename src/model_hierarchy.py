from boundingbox import BoundingBox as BB
from enum import Enum

import model_hierarchy_tree as mht
Part = mht.Part

def handle_adjacency(part1: Part, part2: Part, adjacency: mht.Adjacency):
    part1_n = part1
    part2_n = part2
    if adjacency == mht.Adjacency.EW:
        diffx = part1_n.size.maxx - part2.size.minx
        part2_n.size += BB(diffx, diffx, 0, 0)
    elif adjacency == mht.Adjacency.WE:
        diffx = part2_n.size.maxx - part1.size.minx
        part1_n.size += BB(diffx, diffx, 0, 0)
    elif adjacency == mht.Adjacency.SN:
        diffy = part1_n.size.maxy - part2.size.miny
        part2_n.size += BB(0, 0, diffy, diffy)
    elif adjacency == mht.Adjacency.NS:
        diffy = part2_n.size.maxy - part1.size.miny
        part1_n.size += BB(0, 0, diffy, diffy)
    return part1_n, part2_n

def translate_y(bb: BB, translation1, translation2) -> BB:
    return BB(
        bb.minx,
        bb.maxx, 
        translation1 - translation2,
        translation1 + translation2,
    )

def translate_x(bb: BB, translation1, translation2) -> BB:
    return BB(
        translation1 - translation2,
        translation1 + translation2,
        bb.miny, 
        bb.maxy,
    )

class Dimension(Enum):
    X = 0
    Y = 1

def center(bb1: BB, bb2: BB, dimension: Dimension) -> (BB, BB):
    if dimension == Dimension.X:
        trans = translate_x
        diff1 = bb1.width()
        diff2 = bb2.width()
    elif dimension == Dimension.Y:
        trans = translate_y
        diff1 = bb1.height()
        diff2 = bb2.height()
    
    tr1 = 0.5 * diff1
    tr2 = 0.5 * diff2
    if diff1 > diff2:
        return bb1, trans(bb2, tr1, tr2)
    else:
        return trans(bb1, tr2, tr1), bb2



def handle_properties(part1: Part, part2: Part, properties: list[mht.Properties]):
    part1_n = part1
    part2_n = part2

    for prop in properties:
        if prop == mht.Properties.CENTERED_X:
            dimension = Dimension.X
        elif prop == mht.Properties.CENTERED_Y:
            dimension = Dimension.Y

        part1_n.size, part2_n.size = center(part1.size, part2.size, dimension)
            
    return part1_n, part2_n



# # =====================================
# # = PYGAME PROGRAM
# # =====================================
# # INIT pygame
# import pygame
# import numpy as np
# from coord import Coord
# import queue
# import model_hierarchy_tree as mht

# from collections import namedtuple

# Rectangle = namedtuple("Rectangle", ["top", "right", "bottom", "left"])
# pygame.init()

# select_mode = False





# class DrawJob:
#     def __init__(self, shape: pygame.Rect, colour: tuple):
#         self.shape = shape
#         self.colour = colour

# def get_rectangle(c1: Coord, c2: Coord):
#     return pygame.Rect(Coord.min(c1,c2).toTuple(), Coord.abs_diff(c1, c2).toTuple())

# def select_rectangle(mouse_coord: Coord, rectangles)-> (Rectangle, int): 
#     i = 0
#     for r in rectangles:
#         if (mouse_coord.x > r.left and mouse_coord.x < r.right and mouse_coord.y > r.top and mouse_coord.y < r.bottom):
#             print("Found rect")
#             return r, i
#         i += 1
#     return None, -1

# def toggle_select_mode():
#     global select_mode
#     select_mode = ~ select_mode
#     print(f"Toggled select mode to {select_mode}")

# def clear_drawing():
#     draw_surface.fill("white")

# def get_bounding_box(coords: list[Coord]):
#     if len(coords) < 1:
#         return None
#     min_x, min_y, min_z = coords[0].x, coords[0].y, coords[0].z
#     max_x, max_y, max_z = coords[0].x, coords[0].y, coords[0].z
#     for c in coords:
#         if c.x < min_x:
#             min_x = c.x
#         if c.y < min_y:
#             min_y = c.y
#         if c.z < min_z:
#             min_z = c.z
        
#         if c.x > max_x:
#             max_x = c.x
#         if c.y > max_y:
#             max_y = c.y
#         if c.z > max_z:
#             max_z = c.z
#     return (Coord(min_x, min_y, min_z), Coord(max_x, max_y, max_z))


# def visualize_tree(tree: mht.MHTree, rectangles: list[Rectangle]):
#     pass

# WINDOW_SIZE = [800,600]
# DRAW_SURF_SIZE = (760,560)
# font = pygame.font.SysFont('segoeuisymbol',20,16) 

# clock = pygame.time.Clock()  # The clock is needed to regulate the update loop such that we can process input between the frames, see pygame doc
# screen = pygame.display.set_mode((WINDOW_SIZE[0], WINDOW_SIZE[1]))
# draw_surface = pygame.Surface(DRAW_SURF_SIZE,flags=pygame.HWSURFACE)

# drawing = False
# running = True

# start = (0,0)
# end = (0,0)
# screen.fill("white")
# draw_surface.fill("white")
# draw_surface_offset = (0,0)

# rects = []
# drawing_queue = queue.Queue()

# linking = False


# tree = mht.MHTree()
# selected_this = None
# selected_that = None

# def orient_parts(tree: mht.MHTree, rectangles: list[Rectangle]):
#     for link in tree.links:
#         rect_from  = rectangles[link.frm]
#         rect_from_bb = Coord.abs_diff(Coord(rect_from.left, rect_from.top), Coord(rect_from.right, rect_from.bottom))
#         rect_to = rectangles[link.to]
#         rect_to_bb = Coord.abs_diff(Coord(rect_to.left, rect_to.top), Coord(rect_to.right, rect_to.bottom))
#         start_from = Coord(0,0)
#         start_to = Coord(0,0)
#         if link.adjacency == mht.Adjacency.EW:
#             start_to = start_from + Coord(rect_from_bb.x, 0)
#         if mht.Properties.CENTERED_Y in link.properties:
#             if (rect_to_bb.y > rect_from_bb.y):
#                 start_from += Coord(0, rect_to_bb.y * 0.5 - rect_from_bb.y * 0.5)
#             else:
#                 start_to += Coord(0, rect_from_bb.y * 0.5 - rect_to_bb.y * 0.5)
        
#         rectangles[link.frm] = Rectangle(start_from.y, rect_from_bb.x, rect_from_bb.y, start_from.x)
#         rectangles[link.to] = Rectangle(start_to.y, rect_to_bb.x, rect_to_bb.y, start_to.x)
    
#     return rectangles

        


        


# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         if event.type == pygame.MOUSEBUTTONDOWN:
#             mx,my = pygame.mouse.get_pos()
#             if not select_mode:
#                 print("Mouse down: draw rect")
#                 print(f"Mouse pos: {mx},{my}")
#                 drawing = True
#                 start = Coord(mx,my)
#             if select_mode:
#                 s_rect, s_index = select_rectangle(Coord(mx,my), rects)
#                 if (s_rect is not None and s_index > -1):
#                     selected_that = selected_this
#                     selected_this = s_index
#                     if linking:
#                         if selected_this >= 0 and selected_that >= 0:
#                             tree.add_link(mht.MHLink(selected_that, selected_this, [mht.Properties.CENTERED_Y], mht.Adjacency.EW))
#                             linking = False
#                             clear_drawing()
#                             rects = orient_parts(tree, rects)
#                             for r in rects:
#                                 drawing_queue.put(get_rectangle(Coord(r.left, r.top), Coord(r.right, r.bottom)))
                            

#                         # visualize_tree()
#                         # drawing_queue.put()
                    
#         elif event.type == pygame.MOUSEBUTTONUP and drawing and not select_mode:
#             mx,my = pygame.mouse.get_pos()
#             print(f"Mouse pos: {mx},{my}")
#             end = Coord(mx,my)
#             r = get_rectangle(start, end)
#             rects.append(Rectangle(r.top, r.right, r.bottom, r.left))
#             drawing_queue.put(r)
#             print("rect")
#         elif event.type == pygame.KEYUP:
#             if event.key == pygame.K_s:
#                 toggle_select_mode()
#             if event.key == pygame.K_LCTRL:
#                 linking = False

#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_LCTRL:
#                 linking = True
    
#     while not drawing_queue.empty():
#         pygame.draw.rect(draw_surface, "blue", drawing_queue.get())

#     screen.blit(draw_surface, draw_surface_offset)
#     pygame.display.flip()
#     clock.tick(60)

