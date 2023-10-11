import time
import pickle
import json
import copy
import sys

import imageio
import numpy as np

from itertools import product
from collections import deque, namedtuple
from copy import copy
from pathlib import Path
from matplotlib import cm
from IPython.display import Markdown
from IPython.display import HTML
from dataclasses import dataclass, field
from typing import List, Any, Set, Dict, Tuple
from bidict import bidict

from input import Input

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

class Offset(namedtuple("Offset", ["x", "y"])):
    """
    Just a convenience class, mostly used for linking the cardinal directions UP/DOWN/LEFT/RIGHT to their appropriate offsets.
    """
    def __repr__(self):
        return f"O({self.x},{self.y})"

    
class Propagation(namedtuple("Propagation", ["Offset", "Instigator", "Depth"])):
    """
    A tuple of Offset, Instigator, and Depth. This is used in the propagation queue, where each instance of this class represents the propagation over a single cell.
    """
    pass

# Aliases for the two classes above
O = Offset
P = Propagation





# TODO: This should be split into "Grid" and "GridState", where the latter only holds data, and the former keeps an instance of GridState to manipulate.
# NOTE: Methods with a dash in front are "private" and supposed to be only used internally
@dataclass()
class GridState:
    """
    This class contains all the data and logic pertaining to the state of the grid.
    """
    # NOTE: See "initialize_state" for the appropriate shapes of all the arrays.
    choices: np.ndarray = field(init=False)  # A 3D bool array, where the x/y axis represent the x/y axis on the canvas, and the z axis represents which tiles are still allowed (and thus has depth equal to the amount of tiles)
    paths: np.ndarray = field(init=False)  # A 3D bool array, where the x/y axis represent the x/y axis on the canvas, and the z axis represents which tiles were visited for that cell
    chosen: np.ndarray = field(init=False)  # A 2D uint8 array that indicates which tile was chosen for some particular cell coordinate
    entropy: np.ndarray = field(init=False)  # A 2D float array that stores the entropy for each cell
    painted: np.ndarray = field(init=False)  # A 2D bool array indicating whether some cell was painted over by the user
    index: np.ndarray = field(init=False)  #  A 3D uint8 array, where the x/y axis represent the x/y axis on the canvas, and the z axis represents the coordinates of the cell (mostly for convenience)
    
    paint: bool = field(init=False, default=False)  # Becomes true when painting starts, and becomes false when painting ends
    trigger_propagation: bool = field(init=False, default=False)  # Triggers propagation on the next update cycle when set to true
    trigger_depropagation: bool = field(init=False, default=False)  # Triggers depropagation on the next update cycle when set to true
    
    undo_stack: List[Any] = field(init=False, default_factory=list)  # A stack of grid states that correspond to undo checkpoints
    save_list: List[Any] = field(init=False, default_factory=list)  # A list of grid states that correspond to snapshots
    
    # NOTE: Recording all this information is quite slow, but great for debugging.
    edit_dict: Dict[Tuple[int], List[Tuple[int]]] = field(init=False, default_factory=dict)  # This dict shows for a cell which other cells have affected it.
    prop_dict: Dict[Tuple[int], List[Tuple[int]]] = field(init=False, default_factory=dict)  # This dict shows for a cell which other cells it has affected.
    
    # TODO: It is nicer to simply instantiate the queue in the propagation method, as propagation/depropagation always occurs as a result of a collapse/uncollapse.
    queue: deque = field(default_factory=deque)  # The propagation queue, contains instances of "Propagation"
    deprop_queue: deque = field(default_factory=deque)  # The depropagation queue, contains instances of "Propagation"
    
    # TODO: It is possible to simply combine these two and perform the insertion differently; the sole reason for the separation is to give the paint queue priority.
    collapse_queue: deque = field(default_factory=deque)  # The collapse queue, this is where cells are queued for collapse that have been selected automatically by the generator
    paint_queue: deque = field(default_factory=deque)  # The paint queue, this is where cells are queued for collapse that have been targeted by the user through tile painting

    
    def _snapshot(self, shallow=False):
        """
        This creates copies of the datastructures that represent the state of the grid and returns them in a tuple.
        """
        return (
            self.choices.copy(), 
            self.chosen.copy(), 
            self.entropy.copy(),
            self.painted.copy(),
            self.paths.copy(),
            self.index.copy(),
            # NOTE: Deepcopying these dictionaries can be very slow and isn't always necessary (e.g. with undoing), hence the 'shallow' keyword
            dict(self.edit_dict) if shallow else copy.deepcopy(self.edit_dict),
            dict(self.prop_dict) if shallow else copy.deepcopy(self.prop_dict)
        )
    
    def undo_snapshot(self):
        """
        Creates an undo checkpoint.
        """
        self.undo_stack.append(self._snapshot(shallow=True))
        
    def save_snapshot(self, index):
        """
        Saves the current grid to the specified save slot.
        """
        self.save_list[index] = self._snapshot()
        
    def has_snapshot(self, index):
        """
        Checks whether a save was made at a particular save slot.
        """
        return self.save_list[index] is not None
        
    def load_snapshot(self, index):
        """
        Loads saved grid data from the given index, or resets the grid if the index is empty.
        """
        self.clear_queues()
        # TODO: Make this non-crappy? --> Would need to separate the grid's state from the logic
        if self.save_list[index]:
            self.choices = self.save_list[index][0].copy()
            self.chosen = self.save_list[index][1].copy()
            self.entropy = self.save_list[index][2].copy()
            self.painted = self.save_list[index][3].copy()
            self.paths = self.save_list[index][4].copy()
            self.index = self.save_list[index][5].copy()
            self.edit_dict = copy.deepcopy(self.save_list[index][6])
            self.prop_dict = copy.deepcopy(self.save_list[index][7])
        else:
            self.initialize_state()

    def clear_queues(self):
        """
        Clears all the queues, often used when loading saved states or resetting the grid.
        """
        self.queue.clear()
        self.collapse_queue.clear()
        self.paint_queue.clear()
        
    def undo(self):
        """
        Loads saved state from the topmost undo checkpoint on the stack, or resets the grid if there are no further undo points.
        """
        self.clear_queues()
        if self.undo_stack:
            self.choices, self.chosen, self.entropy, self.painted, self.paths, self.index, self.edit_dict, self.prop_dict = self.undo_stack.pop()
        else:
            self.initialize_state()
            
    def initialize(self, T):
        """
        Initializes all data structures and the grid. See comments throughout the code for clarification.
        """
        
        self.tree = T
        self.save_list = [None] * 10  # 10 digits on the keyboard, so at most 10 save slots.
        
        
        # The algorithm mostly works using numpy boolean indexing.
        # Read about that here: https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing

        # Many of these so called "masks" that are being pre-built below are used with that purpose in mind:
        # to select the relevant entries of the arrays. They are often the shape of a boolean 2D array, with both
        # axes being the size of the amount of tiles. One axis is then used to select a tile, upon which a numpy array 
        # is retrieved that shows how that tile relates to the other tiles in the chosen respect (e.g. leaf, child, adjacency, etc)
        
        # Such an array with booleans that indicate whether the tile is a leaf/child/whatever of some other tile can 
        # then be used to select, mask or switch entries in the grid's 'choices' array, e.g. during propagation. This is quite
        # efficient thanks to vector processing (SIMD), where instead of having to check each entry individually, simple boolean
        # masks can be applied to many entries simultaneously in a single op.
        
        
        # Build leaf mask - can be used to check whether a tile is a leaf
        # NOTE: This is the only one that is 1D, because whether a tile is a leaf does not relate to other tiles
        #       Hence this can be used to filter out the leaves by simply AND-ing it with the remaining choices of some cell
        leaf_ids = set()
        for leaf in self.tree.root.leaves:
            leaf_ids |= leaf.tile_ids
        self.LM = np.full(len(self.tree.tiles), False)
        self.LM[list(leaf_ids)] = True
        
        # Build child mask - used to determine which tiles are direct children of the selected tile
        self.CM = np.full((len(T.tiles), len(T.tiles)), False)
        for metatile in self.tree.nodes:
            for subtype_link in metatile.subtypes.values():
                for tile in metatile.tile_ids:
                    for ctile in subtype_link.to.tile_ids:
                        self.CM[tile, ctile] = True
                        
        # Build ancestor mask - used to determine which tiles are an ancestor of the selected tile
        self.AM = np.full((len(self.tree.tiles), len(self.tree.tiles)), False)
        for tile in self.tree.tiles:
            for ancestor in self.tree.tiles[tile].ancestry:
                if ancestor is not self.tree.tiles[tile]:
                    for atile in ancestor.tile_ids:
                        self.AM[tile, atile] = True
                        
        # Build subtree mask - used to determine which tiles belong to the subtree of the selected tile
        # NOTE: Does NOT include the node itself. I'm not sure why this exists again, because the same info INCLUDING the node already exists in the meta-tree.
        self.SM = np.full((len(self.tree.tiles), len(self.tree.tiles)), False)
        for tile in self.tree.tiles:
            self.SM[tile, :] = self.tree.metamasks[tile]
            self.SM[tile, tile] = False
        
        self.tiles = np.array(list(sorted(self.tree.tiles)))
        
        # Setup grids
        self.initialize_state()
            
    

    def initialize_state(self):
        """
        Initializes the state of the grid.
        """
        self.MAX_ENTROPY = self._calculate_entropy(np.full(len(self.tree.tiles), True))
        self.choices = np.full((IMAGE_X, IMAGE_Y, len(self.tree.tiles)), True)
        self.painted = np.full((IMAGE_X, IMAGE_Y), False)
        self.chosen = np.full((IMAGE_X, IMAGE_Y), list(self.tree.tiles)[0]).astype('uint8')
        self.entropy = np.full((IMAGE_X, IMAGE_Y), self.MAX_ENTROPY).astype(float)
        self.paths = np.full((IMAGE_X, IMAGE_Y, len(self.tree.tiles)), False)
        self.index = np.indices((IMAGE_X, IMAGE_Y)).transpose(1, 2, 0)
        self.edit_dict = dict()
        self.prop_dict = dict()
        self.clear_queues()

    # TODO: This method is responsible for determining which cell is the next to automatically collapse. Therefore "_calculate_entropy" is not
    #       the best name, since this could be based on more than just entropy (e.g. whether the cell was painted, or whether it is still a higher level node)
    def _calculate_entropy(self, choices):
        """
        Calculate the entropy of a cell.
        """
        return np.log(np.sum(choices))
    
    def grow(self):
        """
        Targets the cells with the lowest entropy, and queues them for a single collapse.
        """
        M = (self.entropy<=0) | (self.entropy==np.max(self.entropy))
        em = np.ma.MaskedArray(self.index, np.repeat(M, self.index.shape[2]).reshape(self.index.shape))
        self.enqueue([tuple(n) for n in em.compressed().reshape(-1, 2)])
        
    def collapse_all(self):
        """
        Collapses all cells on the grid.
        """
        self.enqueue([tuple(n) for n in self.index.reshape(-1, 2)])
        
    def auto_enqueue(self, painted):
        """
        Enqueues a cell for collapsing. It filters out cells that have zero or lower entropy as potential targets.
        
        If 'painted' is True, then it limits the mask to the painted region, causing only those cells to collapse.

        In both cases, it enqueues cells according to the minimum entropy heuristic.
        """
        mask = self.entropy<=0
        if painted:
            mask |= ~self.painted
        em = np.ma.MaskedArray(self.entropy, mask)  # NOTE: entries that are TRUE in mask get filtered out!
        if not em.mask.all():
            i, j = np.unravel_index(np.ma.argmin(em), np.ma.shape(em))
            self.enqueue([(i, j)])
                    
    def enqueue(self, L, priority=False):
        """
        Enqueues a list of cells for collapse.
        
        The "priority" keyword actually indicates whether the cells should go in the paint queue, which has priority over the regular collapse queue.
        """
        for n in L:
            if priority:
                self.paint_queue.append(n)
            else:
                self.collapse_queue.append(n)
        
    def on_draw_start(self):
        """
        A hook to indicate when drawing starts.
        """
        self.paint = True
    
    def on_draw_finish(self):
        """
        A hook to indicate when drawing finishes.
        
        Depropagation is always triggered right after drawing, though it only actually does something if the depropagation queue contains anything.
        
        The reason depropagation occurs like this, is because it is VERY expensive to do it for each cell individually, and it makes no difference
        for the final result if you do it for the whole painted body at once.
        """
        self.paint = False
        self.trigger_depropagation = True
            
    
    # TODO: Fix this mess someday, make it more similar to the web-version
    def update(self):
        """
        The core update loop, that gets called by the update loop of pygame.
        """
        # First we deal with the paint queue; the user input. Doing things in this order ensures maximum responsiveness.
        while self.paint_queue:
            i,j = self.paint_queue.popleft()
            if not overwrite:
                self.constrained_paint(i,j)
            else:
                self.overwrite_paint(i,j)
        self.painted[:,:] = False

        # So the below basically always triggers after drawing is finished. Depropagation only occurs if there was anything in the queue.
        # Note the shortcut here: even when painting a larger region, we still propagate for the entire body at once. This can cause
        # some contradictions at times, but it is orders of magnitude faster than individually propagating each cell, and usually doesn't break.
        # This is doubly the case for depropagation, which is infeasible to do per cell.
        if self.trigger_depropagation:
            self.depropagate()
            self.propagate()
            self.trigger_depropagation = False  

            
        # After dealing with the painted cells, we deal with any cells that were automatically enqueued in the update loop via the 
        # minimum entropy heuristic. These are individually propagated.
        while self.collapse_queue:
            i,j = self.collapse_queue.popleft()
            self.collapse(i,j)
            self.propagate()
            

        
    def within_bounds(self, coordinate):
        """
        Checks whether a coordinate is within the bounds of the grid.
        """
        return coordinate[0] >= 0 and coordinate[1] >= 0 and coordinate[0] < IMAGE_X and coordinate[1] < IMAGE_Y

    
    # BUG: Sometimes, entropy does not update properly, this is visible as hard edges appearing on the entropy display.
    #      Easy to reproduce when erasing collapsed tiles with the root tile
    def depropagate(self):
        """
        Depropagates all cells that are in the depropagation queue.
        
        The depropagation algorithm. It works by just resetting cells according to the subtrees of the chosen tiles they host.
        This keeps propagating until no more changes happen from neighbour to neighbour.
        
        This is not optimal, but it does seem to consistently work properly and is marginally better 
        than resetting choices for the entire grid except for cells containing leaves, and propagating 
        from all the leaf cells.
        """
        reset = True
        while self.deprop_queue:
                current, instigator, depth = self.deprop_queue.popleft()
                # For recording the (de)propagation wave
                if record:
                    store_anim(depth, *current,[0,0,255], reset)
                    reset = False
                    
                # Get the cell's neighbours that are not fully reset already.
                i, j = current
                xs, xe = (max(i-1, 0), min(i+1+1, IMAGE_X))
                ys, ye = (max(j-1, 0), min(j+1+1, IMAGE_Y))
                neighbours = [n for n in map(tuple, self.index[xs:xe, ys:ye].reshape(-1, 2)) if self.entropy[n]<self.MAX_ENTROPY]
                for neighbour in neighbours:
                    x, y = neighbour
                    offset = (i - x, j - y)

                    # Ensure we have an offset that corresponds to the adjacencies that we check (Up/Down/Left/Right)
                    # The list comprehension above also includes the corners (1,1), (-1,1) etc, and this is how we filter them.
                    # TODO: Preferably come up with a clearer way of doing this...
                    if offset not in offsets:
                        continue
                    
                    # State of the neighbour cell before propagation
                    pre = self.choices[x, y, :].copy()
                    
                    # State of the neighbour cell after propagation:
                    #   We simply allow everything in all the subtrees of the allowed tiles again
                    post = pre | (self.SM[self.chosen[x, y]] )

                    # If there was a change in the neighbour...
                    if np.any(pre != post):
                        # Set the new choices, and append the neighbour to the deprop queue
                        self.choices[x, y, :] = post
                        self.deprop_queue.append(P(neighbour, instigator, depth+1))
                    # Otherwise we are done, and can append this cell to the propagation queue as a starting point
                    else:
                        self.queue.append(P(neighbour, current, depth+1))


    def propagate(self):
        """
        Propagates all cells that are in the propagation queue.
        
        The details of the algorithm are explained in comments below.
        """
        reset = True
        # Beginning is the same as "depropagate", so check the comments there.
        while self.queue:
            current, instigator, depth = self.queue.popleft()
            if record:
                store_anim(depth, *current,[255,0,0], reset)
                reset = False
            i, j = current
            xs, xe = (max(i-1, 0), min(i+1+1, IMAGE_X))
            ys, ye = (max(j-1, 0), min(j+1+1, IMAGE_Y))
            neighbours = [n for n in map(tuple, self.index[xs:xe, ys:ye].reshape(-1, 2)) if self.entropy[n]>0]
            for neighbour in neighbours:
                x, y = neighbour
                offset = (i - x, j - y)

                if offset not in offsets:
                    continue
                

                # State of the neighbour cell before propagation
                pre = self.choices[x, y, :].copy()
                
                # This is a bit of a complex operation, it works as follows:
                allowed_leaf_adjacencies = (self.LM & self.choices[i, j, :]) & (ADJ[offset] | ADJ_AUG[offset])
                #                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                    - We filter the origin cell's allowed tiles by only selecting the allowed leaves
                #                                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  - We make sure to augment the adjacencies (yes... probably faster to pre-build this, as it is static)
                #                                                           ^^                                 - Then we select the adjacencies according to the array obtained at 1. 
                #
                # see numpy broadcasting for reference: https://numpy.org/doc/stable/user/basics.broadcasting.html
                #
                # NOTE: We indeed only care about the adjacencies of the leaves, because the meta-tile's own adjacencies are fully based on these.
                #       This ensures that we keep in mind that meta-tiles might "weaken", due to some of their leaves being disallowed during earlier propagations
                
                
                
                # Then we just filter the state before propagation with the obtained array from above.
                # - The pre-existing state indicates which tiles were still allowed on the neighbour cell initially, we must respect this as it came to be from earlier propagation waves
                # - The allowed_leaf_adjacencies array indicates which tiles are allowed to be adjacent to the origin cell at the current offset from the origin cell
                post = pre & np.any(allowed_leaf_adjacencies, axis=1)
                
                # filter metatiles that are childless (in the full subtree) except for leaves
                # - This is an important step: we may end up with meta-tiles that will not lead to a leaf without it, since the propagation wave may disable all the 
                #   potential leaves in the subtree of a meta-tile. Hence we use this subtree to check this, and disallow the meta-tile if it cannot lead to any leaves.
                #
                #   Naturally, this does not matter for leaf tiles.
                post &= (np.any(post & self.SM, axis=1) | self.LM)

                
                if np.any(pre != post):
                    self.choices[x, y, :] = post
                    
                    # The entropy is stored, which is faster than constantly recalculating it on the fly
                    self.entropy[x, y] = self._calculate_entropy(post)
                    self.queue.append(P(neighbour, instigator, depth+1))
                    
                    # For debugging
                    if instigator not in self.edit_dict:
                        self.edit_dict[instigator] = []
                    self.edit_dict[instigator].append((neighbour, tuple(pre ^ post)))
                    if neighbour not in self.prop_dict:
                        self.prop_dict[neighbour] = []
                    self.prop_dict[neighbour].append(instigator)

                    # Sometimes it happens that a propagation wave causes a single choice to remain, upon which the cell is trivially collapsed.
                    # When this happens, it is important to set the chosen tile in the "chosen" matrix, so that the visuals update correctly.
                    if sum(post) == 1:
                        self.chosen[x, y] = self.tiles[post][0]
                    
                    # auto undo/rollback if we contradict
                    if sum(post) == 0:
                        self.undo()
                        
    
    # TODO: It is possible to combine both types of painting for the user and select the appropriate case-by-case. The web-version does this already.
    
    # NOTE: Something this version cannot do yet, is reset to root and paint the intended tile in one single motion. Therefore, you can only overwrite
    #       the tile on a cell if the tile you paint with is an ancestor of the existing tile. This is not too hard to implement though.
    def overwrite_paint(self, i, j):
        """
        Overwrites a tile at the cell on the specified coordinates given that the tile to paint with is an ancestor of the existing tile.
        So this basically only goes upwards in the DAG.
        """
        if self.chosen[i, j] == tile_to_place:
            return

        if np.any(self.SM[tile_to_place, self.chosen[i, j]]):
            choice = tile_to_place
            self.choices[i, j, :] = self.tree.metamasks[choice]
            self.choices[i, j, choice] = True
            self.entropy[i, j] = self._calculate_entropy(self.choices[i, j, :])
            self.chosen[i, j] = choice
            self.paths[i, j, choice] = True
            
            pos = (i, j)
            self.queue.append(P(pos, pos, 0))
            self.deprop_queue.append(P(pos, pos, 0))
            
            if (i,j) in self.edit_dict:
                self.edit_dict[(i,j)].clear()
            if (i,j) in self.prop_dict:
                self.prop_dict[(i,j)].clear()
                    
    

    def constrained_paint(self, i, j):
        """
        Paints a tile at the cell on the specified coordinates given that the tile to paint with is in the subtree of the existing tile.
        So this basically only goes downwards in the DAG.
        """
        allowed = self.choices[i,j,:]
        
        painted = np.full(len(T.tiles), False)
        painted[tile_to_place] = True
        allowed = allowed & painted
        
        if np.any(allowed):
            allowed_tiles = self.tiles[allowed]
            choice = np.random.choice(allowed_tiles)
            self.choices[i, j, :] &= self.tree.metamasks[choice]
            self.choices[i, j, choice] = True
            self.entropy[i, j] = self._calculate_entropy(self.choices[i, j, :])
            self.chosen[i, j] = choice
            self.paths[i, j, choice] = True
            
            pos = (i, j)
            self.queue.append(P(pos, pos, 0))


    def collapse(self, i,j):
        """
        Regular collapse - used by automatic collapsing.
        """
        from_node = self.tree.tiles[self.chosen[i, j]]
        
        allowed = self.choices[i,j,:]
        allowed = allowed & self.CM[self.chosen[i,j], :]
        
        if np.any(allowed):
            allowed_tiles = self.tiles[allowed]
            weights = np.ones(len(allowed_tiles))
            
            # NOTE: I'm not sure why there's a try-catch here, I suppose there is still a bug in there that I didn't bother to solve.
            #       This code just fetches the edge probabilities from the current tile to its children, if there are any
            try:
                links = [self.tree.links[(from_node, self.tree.tiles[tile])] for tile in allowed_tiles]
                link_weights = np.array([int(l.properties['enabled'])*l.properties['weight']/len(l.to.tile_ids) for l in links]).astype(float)
                if sum(link_weights) > 0:
                    weights = link_weights
            except Exception as e:
                print("LINK ERROR",(from_node, self.tree.tiles[tile]), e )
                pass
            
            weights /= sum(weights)  # Normalize to probability distribution
            choice = np.random.choice(allowed_tiles, p=weights)  # Choose based on distribution
            
            # Record all the necessary information in the data structures
            self.choices[i, j, :] &= self.tree.metamasks[choice]
            self.choices[i, j, choice] = True
            self.entropy[i, j] = self._calculate_entropy(self.choices[i, j, :])
            self.chosen[i, j] = choice
            self.paths[i, j, choice] = True

            # If a tile is fully collapsed (1 option remaining, log(1) is 0), then we no longer need to focus on collapsing this cell first, so unmark the painted property.
            if self.entropy[i, j] == 0:
                self.painted[i, j] = False
            
            # Add to propagation queue
            pos = (i, j)
            self.queue.append(P(pos, pos, 0))

@dataclass(frozen=True)
class Adjacency:
    """
    Data class used for storing an adjacency, mostly exists for convenience so it can be pretty printed via the __repr__ override below. 
    
    Used by MetaNode.
    """
    from_tile: int
    to_tile: int
    offset: Offset

    def __repr__(self):
        return f"{self.from_tile}-{self.to_tile}"

@dataclass(frozen=True)
class  MetaNode:
    """
    The basic element used for structuring the meta-tree.
    
    This class has multiple convenience methods built in to quickly get certain elements from the DAG relative to this node.
    
    NOTE: the structure employed here differs a little bit from what is in paper: 
          where in the paper every tile relates to exactly one node, here we can 
          have multiple tiles relating to the same node. The most bottom
          MetaNodes contain the tile IDs of the leaf/concrete tiles, so there are 
          no explicit leaf nodes here. This means that the entire meta-tree consists
          of nodes corresponding to a meta-tile.
          
          It is possible to specify multiple meta-tiles in the input for a single meta-node, 
          but this is untried, and does not seem very useful :)
    """
    name: str  # name of the node
    archetypes: Dict[Tuple[Any, Any], Any] = field(compare=False, init=False, default_factory=dict)  # i.e. the parents
    subtypes: Dict[Tuple[Any, Any], Any] = field(compare=False, init=False, default_factory=dict)  # i.e. the children
    tile_ids: Set[int] = field(compare=False, default_factory=set)  # the indices of the tiles that this node represents
    inputs: Set[Input] = field(compare=False, default_factory=set)  # the input images that correspond to this node
    adjacencies: Dict[Offset, Set[Adjacency]] = field(compare=False, default_factory=dict)  # The adjacencies that were found in the input images of this node
    properties: Dict[str, Any] = field(compare=False, default_factory=dict)  # Other properties, parsed from a json file (see input format)

    
    def add_link(self, link):
        """
        Add a MetaLink. Used by the MetaTreeBuilder.
        """
        if link.frm is self:
            self.subtypes[link.key()] = link
        elif link.to is self:
            self.archetypes[link.key()] = link
    
    
    def remove_link(self, link):
        """
        Remove a MetaLink. Used by the MetaTreeBuilder.
        """
        if link.frm is self:
            del self.subtypes[link.key()]
        elif link.to is self:
            del self.archetypes[link.key()]
               
    @property
    def leaves(self):
        """
        Returns the leaves of the sub-meta-tree that has this node at its root/source.
        """
        S = set()
        if not self.subtypes:
            S.add(self)
        for subtype_link in self.subtypes.values():
            S.update(subtype_link.to.leaves)
        return S
    
    @property
    def nodes(self):
        """
        Returns the sub-meta-tree with this node at its root/source, including this node.
        """
        S = {self}
        for subtype_link in self.subtypes.values():
            S.update(subtype_link.to.nodes)
        return S
    
    @property
    def ancestry(self):
        """
        Returns all the ancestors that can lead to this node.
        """
        S = {self}
        for archetype_link in self.archetypes.values():
            S.update(archetype_link.frm.ancestry)
        return S
    
#     def root(self):
#         node = self
#         while node.parent:
#             node = node.parent
#         return node    
    
    def __repr__(self):
        return f"{self.name}:{self.tile_ids}"



@dataclass(frozen=True)
class MetaLink:
    """
    This class is used for specifying a link/edge between two nodes
    """
    frm: MetaNode  # The "from" node
    to: MetaNode  # The "to" node
    properties: Dict[str, Any] = field(compare=False, default_factory=dict)  # The same properties json that is in MetaNode
    
    # NOTE: The reason why the edges also keep a copy, is because the JSON also contains the probability weights.
    #       The DAG is built from a tree-ish folder structure, where folders/nodes get merged in case of equivalent names.
    #       Still though, the probability of reaching such a node can differ per edge/link, hence we need to store this 
    #       information per edge/link.
    
    
    def key(self):
        """
        MetaLinks are uniquely identified by the from/to node pair, because we never want multiple edges between the same (ordered) pair of nodes.
        
        One could argue that this should be a set, since we never want both possible pairs either (circular dependency), but this requires more
        advanced checking anyway for the non-trivial cases, which should happen elsewhere.
        """
        return self.frm, self.to
    
    def __repr__(self):
        return f"{self.frm.name}-->{self.to.name}"

@dataclass(frozen=True)
class MetaTreeBuilder():
    """
    This class is responsible for putting a tree together from MetaNodes and MetaLinks. It also holds some global information,
    such as: a list of all the nodes, a list of all the tiles, a list of all the links, and all the metamasks.
    """
    nodes: Dict[MetaNode, MetaNode] = field(compare=False, init=False, default_factory=dict)  # Dict of all nodes (NOTE: Practically only the name field of a MetaNode is used for the dictionary, so string-indexing would've worked too here, #lazy)
    links: Dict[Tuple[MetaNode, MetaNode], MetaLink] = field(compare=False, init=False, default_factory=dict)  # Dict of all edges/links, indexed by MetaNode pairs for easy retrieval
    tiles: Dict[int, MetaNode] = field(compare=False, init=False, default_factory=dict)  # Dict of all tiles, indexed by tile index
    metamasks: Dict[int, Any] = field(compare=False, init=False, default_factory=dict)  # Like the child/subtree masks, this is a boolean indexing mask that highlights which tiles are in the subtree of a meta-node, INCLUDING itself
    
    # TODO: Might be better to just implement a root method on the nodes, though this is way cheaper
    @property
    def root(self):
        """
        Gets the root of the meta-tree. This does require the root folder to be called "root"...
        
        Note that MetaNodes are uniquely identified by their name (check the properties of the fields in MetaNode, all others have 'compare' on False)
        """
        return self.nodes[MetaNode("root")]
        
    @property
    def tile_ids(self):
        """
        Gets the (unordered) set of all tile indices.
        """
        return set(tiles.keys())

    
    def add_node(self, node):
        """
        Adds a MetaNode to the meta-tree.
        """
        self.nodes[node] = node
    
    
    def add_link(self, frm, to, properties):
        """
        Creates a link between the two given MetaNodes with the given properties, and ensures linkage in the tree and between the nodes is setup properly, and rebuilds the tree.
        """
        link = MetaLink(frm, to, properties)
        self.nodes[frm] = frm
        self.nodes[to] = to
        self.links[link.key()] = link
        frm.add_link(link)
        to.add_link(link)
        self.rebuild_tile_dict()
    
    
    def remove_link(self, frm, to):
        """
        Removes the link (there should be only one) between the two given MetaNodes, if it exists, and rebuilds the tree.
        """
        link_key = (frm, to)
        link = self.links[link_key]
        del self.links[link_key]
        frm.remove_link(link)
        to.remove_link(link)
        self.rebuild_tile_dict()


    def rebuild_tile_dict(self):
        """
        Rebuilds the tile dictionary based on the current dict of MetaNodes.
        """
        self.tiles.clear()
        for node in self.nodes:
            for tile in node.tile_ids:
                self.tiles[tile] = node

    def build_metamasks(self):
        """
        Builds all the metamasks.
        """
        self.metamasks.clear()
        for root in self.nodes:
            for tile in root.tile_ids:
                self.metamasks[tile] = np.full(len(self.tiles), False)
                for node in root.nodes - {root}:
                    for subtile in node.tile_ids:
                        self.metamasks[tile][subtile] = True      
                
    def __repr__(self):
        return f"nodes: {list(self.nodes.values())}\nlinks: {list(self.links.values())}"
    
    
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
    
offset = 0  # This global variable is used to determine when to reset frame recording
def store_anim(iteration, i, j, c, reset=False):
    global anim_frames, offset
    if reset:
        offset = len(anim_frames)
    # anim_frames.append([(i, j, c)])
    while (iteration + offset) >= len(anim_frames):
        anim_frames.append([])
    anim_frames[iteration + offset].append((i, j, np.array(c)))
    
def clear_highlights_and_anim():
    global overlay, highlights, anim_frames, anim
    overlay[:,:,2] = 0
    highlights = set()
    anim_frames = []
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
grid = GridState()
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
anim_frames = []
record = False
hide_ui = False
setting_timer = 0
setting_text = None


def handle_input():
    global grid, hide_ui, debug_undo, sel_i, sel_j, mouse_down, collapse, preview, tile_to_place, step, marker_size, painted_only, swatches_locations, swatches, overwrite, hover_tooltip_show, deprop, surf, highlights, record, anim, anim_frames
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
    if anim_frames:
        anim_counter = round(0.02 * pygame.time.get_ticks()) % len(anim_frames)
        anim[:,:,:] = 0
        for t in range(trail):
            for ii,jj,cc in anim_frames[(anim_counter-t) % len(anim_frames)]:
                anim[ii,jj,:] = cc / (t + 1)

    i, j = handle_input()
    mx, my = pygame.mouse.get_pos()
    dsp = grid.chosen.copy()
    dsp[preview] = tile_to_place
    if collapse and not mouse_down:
        if debug_undo and np.any(grid.entropy>0):
            grid.undo_snapshot()
        grid.auto_enqueue(painted_only)
    grid.update()
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


