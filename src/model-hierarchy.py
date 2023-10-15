# TODO: create coord class

# =====================================
# = PYGAME PROGRAM
# =====================================
# INIT pygame
import pygame

pygame.init()


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
draw_surface.fill("red")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            screen.fill("white")
            print("Mouse down: draw rect")
            mx,my = pygame.mouse.get_pos()
            print(f"Mouse pos: {mx},{my}")
            drawing = True
            start = (mx,my)
        elif event.type == pygame.MOUSEBUTTONUP and drawing:
            mx,my = pygame.mouse.get_pos()
            print(f"Mouse pos: {mx},{my}")
            end = (mx,my)
            pygame.draw.rect(draw_surface, "blue", pygame.Rect(start[0],start[1],abs(start[0]-end[0]), abs(start[1]-end[1])))



    screen.blit(draw_surface, (10,10))
    pygame.display.flip()
    clock.tick(60)