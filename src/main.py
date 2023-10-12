import time
import pickle
import json
import sys

import imageio
import numpy as np

from itertools import product
from pathlib import Path
from matplotlib import cm

from bidict import bidict

from input import Input
from offset import Offset
from offset import getOffsets
from propagation import Propagation
from grid_state import GridState
from metatree import *

from animator import Animator

animator = Animator()

# Aliases
O = Offset
P = Propagation


# Notebook config
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=sys.maxsize) 
np.set_printoptions(precision=2)

# Application config
IMAGE_X, IMAGE_Y = (64, 64)  # Basically the grid size
WINDOW_SIZE = ( 600,  600)  # Size of one panel, see TOTAL_WINDOW_SIZE
TOOLBAR_SIZE = 64  # Size of the bottom bar, with the swatches for the tiles

# Algorithm config
np.random.seed(1337)  # Fixing the seed ensures same generation patterns, good for debug
tileset_path = Path().cwd() / "tilesets" / "tileset-dag-paper-final-adv" # The tileset to load; should point to the root folder


# Data classes
##############

 
# ====================================
# = Convenience methods
# ====================================
def fancy_bmat(M):
    """
    Prints a numpy boolean array with fancy unicode squares.
    """
    try:
        f = lambda x : '◼' if x else '◻' 
        return np.vectorize(f)(M)
    except:
        return np.array([])

def clamp(n, smallest, largest): 
    """
    Clamps a number 'n' between the given minimum/maximum values.
    """
    return max(smallest, min(n, largest))

def color(r, g, b):
    """
    Converts a floating-point rgb color into a uint8 color (0.0-1.0 --> 0-255).
    """
    return (255 * np.array((r, g, b))).astype(int)

# TODO: Make this un-retarded
# - Use color as ID, instead of doing some error-prone mapping
# - Can always enumerate the tiles later for bitmasks if needed
def parse_image(path):
    """
    Parses an image into an input. Each pixel represents a tile. Tiles are uniquely identified by their color. Essentially this represents a meta-tile, with those
    unique tiles as children.
    
    Indices are given to the unique colors present in the image in a mapping, and an inverse 2D mapping is created that shows what tile ID is where on the image.
    
    These indices are later adjusted to global indices, during input processing, where they can be identified across input images. 
    
    RGB is represented as a sequential integer: 000 000 000 to 255 255 255, necessary for pygame.
    
    IMPORTANT: Tiles that have an alpha that is lower than 1 are tagged in the mask. These are "ghost" tiles that are disregarded as child from the meta-tile, but their adjacencies are taken into account.
    """
    img = imageio.imread(path, pilmode="RGBA").astype(float).transpose(1,0,2)  # Switch orientation
    mask = img[:,:,3] < 255
    img = img[:,:,:3]
    img[:,:,1] *= 256
    img[:,:,2] *= 256 * 256
    a, b = np.unique(img.sum(axis=(2)),  return_inverse=True)
    mapping = bidict()
    for i, u in enumerate(a):
        rgb = tuple(map(lambda i : int(u%256**(1+i)//256**i), range(3)))
        mapping[i] = rgb
    ids = b.reshape(img.shape[:2])
    return ids, mapping, mask

# The methods below are used for dealing with highlights and animations
def highlight_cell(i, j):
    global highlights
    highlights.add((i, j))

    
def clear_highlights_and_anim():
    global overlay, highlights, anim
    # global overlay, highlights, anim_frames, anim
    overlay[:,:,2] = 0
    highlights = set()
    animator.anim_frames = []
    anim[:,:,:] = 0


# Convenience method to temporarily show some text in the top left instead of the tile name
def show_setting_text(text):
    global setting_text, setting_timer
    setting_text = font.render(text, True, (255,255,255))
    setting_timer = 200
    

# =====================================
# = INPUT PROCESSING
# =====================================
T = MetaTreeBuilder()

# TODO: make offset generator
offsets = {
   Offset( 1, 0 ),
   Offset(-1, 0 ),
   Offset( 0, 1 ),
   Offset( 0,-1 )
}

global_tiles = bidict()  # Holds all the tiles currently read in the input
root_path = tileset_path / "root"  # This is the path to the root meta-tile, it should be in a folder called "root" directly under the tileset folder.

"""
INPUT STRUCTURE: 

- The input consists of nested folders, with the topmost folder having the name "root".

- Each folder (starting at the root) represents a MetaNode, and contains images that serve as input for deriving adjacency rules and tiles that belong to this MetaNode. 

- There may be multiple images per folder. 

- The tiles are represented by the pixels in the image, and are uniquely identified by their colors globally,
  so using the same color pixel in another meta-tile represents the same tile. (I recommend using a pixel-art editor to edit them, e.g. LibreSprite)

- Since the folders represent MetaNodes, they refer to the same MetaNode if they are repeated with the same name
  somewhere in the folder tree (their inputs are merged in that case).

- A nested folder indicates a link between the MetaNode represented by the current folder and the MetaNode representing that folder.
  The weight of such a link is determined by the link weight in the 'properties.json' that resides within the nested folder.

- Each folder has a 'properties.json' file, which contains properties for both the current node, and the link between the current node and the node from the parent folder.
  
    {
        "link" : {                   [PROPERTIES PERTAINING TO THE LINK/EDGE]
            "weight" : [The weight on the link between this node and the parent node]
            "enabled" : [Whether this link is enabled in the meta-tree]
        },
        "node" : {
            "paintable" : [Whether this tile can be painted with by the user at all]
            "swatch" : [Whether to create a swatch for this tile on the UI] 
        }
    }
"""
# Below follows a giant mess of input processing code

# Process the folder structure and collect+process the full input
for p in root_path.glob("**/"):
    parent = p.parent
    has_parent = tileset_path != parent
    
    # Load inputs
    inputs = set()
    input_tiles = set()
    for i in p.glob("*.png"):
        ids, mapping, mask = parse_image(i)
        legal = np.unique(ids[~mask])
        global_mapping = {}
        for rgb in mapping.inverse:
            if rgb not in global_tiles.inverse:
                global_tiles[len(global_tiles)] = rgb
            global_mapping[mapping.inverse[rgb]] = global_tiles.inverse[rgb]
            if mapping.inverse[rgb] in legal:
                input_tiles.add(global_mapping[mapping.inverse[rgb]])
        mapped = np.vectorize(global_mapping.get)(ids)
        inputs.add(Input(mapped, mask))
    
    # Load adjacencies by scanning the images and masks of each input
    adjacencies = dict()
    for inp in inputs:
        for i, j in product(*map(range, np.shape(inp.img))):
            tile = inp.img[i, j]
            mask = inp.mask[i, j]
            for o in offsets:
                if o not in adjacencies:
                    adjacencies[o] = set()
                if o.x+i>=0 and o.x+i<inp.img.shape[0] and o.y+j>=0 and o.y+j<inp.img.shape[1]:
                    adj_tile = inp.img[o.x+i, o.y+j]
                    adj_mask = inp.mask[o.x+i, o.y+j]
                    if not (adj_mask and mask):
                        if tile > 0 and adj_tile > 0:
                            adjacencies[o].add(Adjacency(tile, adj_tile, o))

    # Load properties and add nodes to the MetaTree
    with open(p / 'properties.json', 'r') as file:
        properties = json.load(file)
        node = MetaNode(p.stem, input_tiles, inputs, adjacencies, properties['node'])
        
        # If the node was encountered before, we update its data
        if node in T.nodes:
            T.nodes[node].tile_ids.update(input_tiles)
            T.nodes[node].inputs.update(inputs)
            T.nodes[node].adjacencies.update(adjacencies)
            node = T.nodes[node]

        # If the node has a parent, we make sure the parent has a link to this node with the weight in the link properties of the current folder (representing this node)
        if has_parent:
            parent_node = T.nodes[MetaNode(p.parent.stem)]
            T.add_link(parent_node, node, properties['link'])
        else:
            T.add_node(node)
            
# Build the meta masks
T.build_metamasks()
print(T.nodes.keys())

# Build the initial global adjacency matrices from adjacencies that have been found by detecting them from the inputs above
ADJ = dict()
for o in offsets:
    ADJ[o] = np.full((len(T.tiles), len(T.tiles)), False)
    for node in T.nodes:
        for adj in node.adjacencies[o]:
            ADJ[o][adj.from_tile, adj.to_tile] = True
            from_node = T.tiles[adj.from_tile]
            to_node = T.tiles[adj.to_tile]
            
            # Wildcard adjacencies with meta-tiles - if there is an adjacency between a meta-tile and a leaf, 
            #                                        then the leaf can be adjacent to all tiles that fall under that meta-tile
            for fnode in from_node.nodes - {from_node}:
                for ftile in fnode.tile_ids:  
                    ADJ[o][ftile, adj.to_tile] = True
            
            # The other way as well to avoid parsing asymmetry
            for tnode in to_node.nodes - {to_node}:
                for ttile in tnode.tile_ids:
                    ADJ[o][adj.from_tile, ttile] = True
                    
#             NOTE: I think this was for dealing with meta-meta adjacencies, can be re-enabled if you wish, basically allows adjacency among all tiles that fall under both meta-tiles
            # for fnode in from_node.nodes - {from_node}:
            #     for ftile in fnode.tile_ids:                    
            #         for tnode in to_node.nodes - {to_node}:
            #             for ttile in tnode.tile_ids:
            #                 if fnode is not tnode:
            #                     ADJ[o][ftile, ttile] = True
    
    # Prints the resulting ADJ matrix for the given offset.
    print(o)
    print(fancy_bmat(ADJ[o]))

# Augmented adjacency matrix

"""
Augmentation works by taking the union of the adjacency constraints of a meta-tile's children as its adjacency constraints.
This is done bottom-up and breadth-first, starting at the leaves, in order to incrementally build it up.
"""
queue = []
ADJ_AUG = dict()
for o in offsets:
    ADJ_AUG[o] = ADJ[o].copy()
    queue = []
    for leaf in T.root.leaves:
        queue += list(leaf.archetypes.values())
    while queue:
        link = queue.pop(0)
        archetype = link.frm
        subtype = link.to

        print(link)

        # Set the adjacencies symmetrically
        for atile in archetype.tile_ids:
            for stile in subtype.tile_ids:
                ADJ_AUG[o][atile, :] |= ADJ_AUG[o][stile, :]
                ADJ_AUG[o][:, atile] |= ADJ_AUG[o][:, stile]
                

        queue += list(archetype.archetypes.values())
    # ADJ_AUG[o][2, 5] = False
    # ADJ_AUG[o][5, 2] = False
    print(o)
    print(fancy_bmat(ADJ_AUG[o]))

    
    
# =====================================
# = PYGAME PROGRAM
# =====================================
# INIT pygame
import pygame
pygame.init()


clock = pygame.time.Clock()  # The clock is needed to regulate the update loop such that we can process input between the frames, see pygame doc
font = pygame.font.SysFont('segoeuisymbol',20,16) 

TOTAL_WINDOW_SIZE = (WINDOW_SIZE[0] * 2, WINDOW_SIZE[1] + TOOLBAR_SIZE)

# These are all pygame hardware-driven 2D surfaces that are used for rendering everything
# We write the numpy data to these surfaces in the update loop in order to update the screen
screen = pygame.display.set_mode(TOTAL_WINDOW_SIZE)  # The screen
surf = pygame.Surface((IMAGE_X, IMAGE_Y), flags=pygame.HWSURFACE|pygame.HWPALETTE, depth=8)  # The main panel where you paint
entropy_surf = pygame.Surface((IMAGE_X, IMAGE_Y), flags=pygame.HWSURFACE|pygame.HWPALETTE, depth=8)  # The entropy panel
overlay_surf = pygame.Surface((IMAGE_X, IMAGE_Y), flags=pygame.HWSURFACE)  # For highlighting the brush, etc
anim_surf = pygame.Surface((IMAGE_X, IMAGE_Y), flags=pygame.HWSURFACE)  # For displaying the propagation animations
surf.set_palette(np.array(list(global_tiles.values())))  # Set the palette to the RGB colors of the tiles
entropy_surf.set_palette((255 * cm.get_cmap('magma') (np.linspace(0,1,len(T.tiles)+1))[:, :3]).astype('uint8'))  # You can use different colormaps from matplotlib if you prefer, it is for the entropy visualization

# Initialize the grid
grid = GridState(grid_x=IMAGE_X, grid_y=IMAGE_Y, offsets=getOffsets(), ADJ=ADJ, ADJ_AUG=ADJ_AUG)
grid.initialize(T)


# All kinds of global variables that are used for program state and settings
mouse_down = False
collapse = False
preview = np.full((IMAGE_X,IMAGE_Y), False)
overlay = np.full((IMAGE_X,IMAGE_Y, 3), 0).astype('uint8')
anim = np.full((IMAGE_X,IMAGE_Y, 3), 0).astype('uint8')
tile_to_place = 0
step = False
marker_size = 1
painted_only = False
overwrite = False
deprop = False
hover_tooltip_show = False
debug_undo = False
sel_i = sel_j = 0
highlights = set()
anim_counter = 0
animator.anim_frames = []
record = False
hide_ui = False
setting_timer = 0
setting_text = None

animator.set_record(record)

def handle_input():
    global grid, hide_ui, debug_undo, sel_i, sel_j, mouse_down, collapse, preview, tile_to_place, step, marker_size, painted_only, swatches_locations, swatches, overwrite, hover_tooltip_show, deprop, surf, highlights, record, anim
    # global grid, hide_ui, debug_undo, sel_i, sel_j, mouse_down, collapse, preview, tile_to_place, step, marker_size, painted_only, swatches_locations, swatches, overwrite, hover_tooltip_show, deprop, surf, highlights, record, anim, anim_frames
    keys = pygame.key.get_pressed()
    numeric_key_codes = {pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0}
    mx, my = pygame.mouse.get_pos()
    i, j = clamp(IMAGE_X*mx//WINDOW_SIZE[0], 0, IMAGE_X-1), clamp(IMAGE_Y*my//WINDOW_SIZE[1], 0, IMAGE_Y-1)
    size = marker_size
    preview[i-size:i+size+1, j-size:j+size+1] = True
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:# or event.type == pygame.QUIT:
            if event.key == pygame.K_ESCAPE:
                running = False
                pygame.quit()
            if event.key == pygame.K_SPACE:
                collapse = not collapse
                if not step:
                    show_setting_text(f"auto-collapsing: {collapse}")
                else:
                    show_setting_text(f"stepping...")
            if event.key == pygame.K_r:
                grid.initialize_state()
                clear_highlights_and_anim()
                show_setting_text(f"grid has been reset")
            if event.key == pygame.K_t:
                record = not record
                show_setting_text(f"propagation recording: {record}")
#             if event.key == pygame.K_u:
            if event.key == pygame.K_c:
                clear_highlights_and_anim()
                show_setting_text(f"clearing highlights/recordings")
            if event.key == pygame.K_s:
                step = not step
                show_setting_text(f"stepping mode: {step}")
            if event.key == pygame.K_a:
                grid.undo_snapshot()
                grid.collapse_all()
                show_setting_text(f"collapsing all cells...")
            if event.key == pygame.K_g:
                grid.undo_snapshot()
                grid.grow()
                show_setting_text(f"growing lowest entropy cells...")
            if event.key == pygame.K_F5:
                with open(Path.home() / 'wfc','wb') as f:
                    pickle.dump(grid.save_list ,f)
                show_setting_text(f"saving snapshots to file...")

            if event.key == pygame.K_F6:
                with open(Path.home() / 'wfc','rb') as f:
                    grid.save_list = pickle.load(f)
                show_setting_text(f"loading snapshots from file...")
            if event.key == pygame.K_o:
                overwrite = not overwrite
                show_setting_text(f"overwrite paint: {overwrite}")
            if event.key == pygame.K_u:
                debug_undo = not debug_undo
                show_setting_text(f"undo for all collapses: {debug_undo}")
            if event.key == pygame.K_l:
                for x,y in grid.index[grid.paths[:,:,tile_to_place]]:
                    highlight_cell(x,y)
                    grid.enqueue([(x,y)], priority=True)
                    grid.trigger_depropagation = True
                show_setting_text(f"path-overwrite with: {T.tiles[tile_to_place].name}")
            if event.key == pygame.K_F11:
                hide_ui = not hide_ui                
            if event.key == pygame.K_F12:
                p = Path.home() / f'wfc_{time.strftime("%Y%m%d-%H%M%S")}.png'
                with open(p, 'wb') as f:
                    pygame.image.save(surf, f, 'png')
                show_setting_text(f"saved image to: {p}")

            if event.key == pygame.K_z:
                if keys[pygame.K_LCTRL]:
                    grid.undo()
                    show_setting_text(f"undo...")

            if event.key in numeric_key_codes:
                index = int(pygame.key.name(event.key))
                if keys[pygame.K_LCTRL]:
                    grid.save_snapshot(index)
                    show_setting_text(f"saved snapshot to slot: {index}...")
                else:
                    grid.load_snapshot(index)
                    show_setting_text(f"loaded snapshot from: {index}...")

            if event.key == pygame.K_p:
                painted_only = not painted_only
                show_setting_text(f"only collapse painted tiles: {painted_only}")

                
            # NOTE: The hover tooltip shows which tiles are enabled at a cell
            #       Once combined with middle-click (debug info for cell that also shows propagations), it can also tell you
            #       which tiles were deactivated at a certain location of the propagation wave from the debugged cell.
            if event.key == pygame.K_h:
                hover_tooltip_show = not hover_tooltip_show
                show_setting_text(f"show hover tooltip: {hover_tooltip_show}")

            if event.key == pygame.K_EQUALS:
                marker_size +=1
                show_setting_text(f"increase marker size to: {marker_size}")

                
            if event.key == pygame.K_MINUS:
                marker_size = max(0, marker_size - 1)
                show_setting_text(f"decrease marker size to: {marker_size}")

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:                
                mouse_down = False
                grid.on_draw_finish()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = True
                grid.undo_snapshot()
                grid.on_draw_start()

                # highlight_cell(i, j)
            if event.button == 3:
                tile_to_place = grid.chosen[i, j]
                show_setting_text(f"picked tile: {T.tiles[tile_to_place].name}")
                
            # Print some useful info pertaining to the cell in question to the Jupyter output.
            #
            #       If available, it also shows the following on the screen:
            #       - in RED: what other cells the propagation wave of this cell affected when its tile got chosen
            #       - in GREEN: what other cells had affected this cell so far through propagation
            #       
            #       yellow simply means both: affected by and affected.
            if event.button == 2:
                overlay[:,:,:] = 0
                print(f"===DEBUG INFO for: ({i}, {j})")
                print("choices:", grid.choices[i,j,:])
                print("entropy:", grid.entropy[i, j])
                print("chosen tile:", grid.chosen[i,j])
                print("marked for paint:", grid.painted[i,j])
                sel_i, sel_j = i,j
                if (i,j) in grid.edit_dict:
                    for (x,y), _ in grid.edit_dict[(i,j)]:
                        # print(x,y)
                        overlay[x,y,0] = 255
                if (i,j) in grid.prop_dict:
                    for (x,y) in grid.prop_dict[(i,j)]:
                        # print(x,y)
                        overlay[x,y,1] = 255
                show_setting_text(f"showing debug info for tile at: x:{i}, y:{j}")

            if event.button == 4:
                tile_to_place += 1
                if tile_to_place >= len(T.tiles):
                    tile_to_place = 0
                show_setting_text(f"selected: {T.tiles[tile_to_place].name}")

            if event.button == 5:
                tile_to_place -= 1
                if tile_to_place < 0:
                    tile_to_place = len(T.tiles)-1
                show_setting_text(f"selected: {T.tiles[tile_to_place].name}")

        if mouse_down:

            if my > WINDOW_SIZE[1]:
                for tile in swatches_locations:
                    if np.linalg.norm((np.array(swatches_locations[tile]) + np.array(swatches[tile].get_size())/2) - np.array([mx, my])) < swatches[tile].get_height()/2:
                        tile_to_place = tile
                        show_setting_text(f"selected: {T.tiles[tile].name}")

            elif mx < WINDOW_SIZE[0]:
                new = [n for n in map(tuple, grid.index[i-size:i+size+1, j-size:j+size+1].reshape(-1, 2)) if n not in grid.collapse_queue]
                grid.painted[i-size:i+size+1, j-size:j+size+1] = True        
                grid.painted[i-size:i+size+1, j-size:j+size+1] = True
                grid.enqueue(new, priority=True)
    return i, j
            
print(T)

for leaf in T.root.leaves:
    print(leaf, leaf.archetypes.values())
    


print()
print("TILES")
# cstr = ""
CL = []
for tile in T.tiles:
    c = global_tiles[tile]
    CL.append(f'<span style="font-family: var(--jp-code-font-family);color: rgb({c[0]},{c[1]},{c[2]})">/◼</span>')
print(list(np.array(range(len(CL))).astype(str)))
# print(np.array(CL))
# display (HTML("..."+str((CL)))) 
    
tooltips = {}
swatches = {}
swatches_off = {}
swatches_locations = {}
w = h = 0
for tile in T.tiles:
    print(tile, T.tiles[tile], global_tiles[tile])
    tooltips[tile] = font.render((T.tiles[tile].name), True, (255,255,255))
    swatches[tile] = font.render('◼', True, global_tiles[tile])
    swatches_off[tile] = font.render('◻', True, global_tiles[tile])
    swatches_locations[tile] = (tile * 1.5 * swatches[tile].get_width(), WINDOW_SIZE[1] + TOOLBAR_SIZE//2 - swatches[tile].get_height()//2)
    w += swatches[tile].get_width()
    h = max(h, swatches[tile].get_height())
hover_tooltip_surf = pygame.Surface((w, h), flags=pygame.HWSURFACE)
hover_tooltip = np.full((w, h, 3), 50).astype('uint8')
trail=4

# NOTE: To see how most of this works, please check: https://www.pygame.org/docs/ref/surface.html
while True:
    if setting_timer > 0:
        setting_timer -= 1
    if animator.anim_frames:
        anim_counter = round(0.02 * pygame.time.get_ticks()) % len(animator.anim_frames)
        anim[:,:,:] = 0
        for t in range(trail):
            for ii,jj,cc in animator.anim_frames[(anim_counter-t) % len(animator.anim_frames)]:
                anim[ii,jj,:] = cc / (t + 1)

    i, j = handle_input()
    mx, my = pygame.mouse.get_pos()
    dsp = grid.chosen.copy()
    dsp[preview] = tile_to_place
    if collapse and not mouse_down:
        if debug_undo and np.any(grid.entropy>0):
            grid.undo_snapshot()
        grid.auto_enqueue(painted_only)
    grid.update(tile_to_place, overwrite=overwrite)
    screen.fill((0,0,0))
    intensity = 0.9
    speed = 0.01
    time_modulation = (1+np.sin(speed * pygame.time.get_ticks()))/2
    time_modulation = int(2 * time_modulation) /2
    for (ii,jj) in highlights:
        overlay[ii, jj, 2] = 255
    overlay[overlay[:,:,:] > 0] = 200 - int(intensity * 255 * time_modulation)
    
    
    pygame.surfarray.blit_array(surf, dsp) 
    screen.blit(pygame.transform.scale(surf, WINDOW_SIZE), (0, 0))
    pygame.surfarray.blit_array(overlay_surf, overlay)
    pygame.surfarray.blit_array(anim_surf, anim)
    pygame.surfarray.blit_array(hover_tooltip_surf, hover_tooltip)

    screen.blit(pygame.transform.scale(overlay_surf, WINDOW_SIZE), (0, 0), special_flags=pygame.BLEND_ADD)
    screen.blit(pygame.transform.scale(anim_surf, WINDOW_SIZE), (0, 0), special_flags=pygame.BLEND_ADD)
    if not hide_ui:
        if setting_timer > 0 and setting_text:
            screen.blit(pygame.transform.scale(setting_text, 1*np.array(setting_text.get_size())), (0, 0))
        else:
            screen.blit(pygame.transform.scale(tooltips[tile_to_place], 1*np.array(tooltips[tile_to_place].get_size())), (0, 0))
        
    diff = tuple(np.full(len(grid.tree.tiles), False))
    if (sel_i, sel_j) in grid.edit_dict:
        for (x,y), d in grid.edit_dict[sel_i, sel_j]:
            if i==x and y==j:
                diff = d
                break
            
    for s in swatches:
        if grid.tree.tiles[s].properties['swatch']:
            screen.blit(swatches[s], swatches_locations[s])
        swatch = swatches[s] if grid.choices[i,j,s] else swatches_off[s]
        hover_tooltip_surf.blit(swatch, (s * swatch.get_width() ,0))
        if diff[s]:
            hover_tooltip_surf.blit(swatch, (s * swatch.get_width() ,0), special_flags=pygame.BLEND_ADD)
        
    pygame.surfarray.blit_array(entropy_surf, np.round(np.e**grid.entropy))
    screen.blit(pygame.transform.scale(entropy_surf, WINDOW_SIZE), (WINDOW_SIZE[0], 0))
    if hover_tooltip_show:
        screen.blit(hover_tooltip_surf, (mx, my)) 
    pygame.display.update()
    preview[:,:] = False
    clock.tick()
    if step:
        collapse = False

pygame.quit()


