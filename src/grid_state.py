import numpy as np
from dataclasses import dataclass, field
from typing import List, Any, Dict, Tuple
from collections import deque
from copy import copy

from offset import Offset
from offset import getOffsets
from propagation import Propagation

from animator import Animator

# Aliases
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

    grid_x: int = field(init=True, default=64)
    grid_y: int = field(init=True, default=64)

    offsets: List[List[int]] = field(init=True, default_factory=getOffsets())

    ADJ: Dict = field(init=True, default_factory=dict)
    ADJ_AUG: Dict = field(init=True, default_factory=dict)

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
        self.CM = np.full((len(self.tree.tiles), len(self.tree.tiles)), False)
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
        self.choices = np.full((self.grid_x, self.grid_y, len(self.tree.tiles)), True)
        self.painted = np.full((self.grid_x, self.grid_y), False)
        self.chosen = np.full((self.grid_x, self.grid_y), list(self.tree.tiles)[0]).astype('uint8')
        self.entropy = np.full((self.grid_x, self.grid_y), self.MAX_ENTROPY).astype(float)
        self.paths = np.full((self.grid_x, self.grid_y, len(self.tree.tiles)), False)
        self.index = np.indices((self.grid_x, self.grid_y)).transpose(1, 2, 0)
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
    def update(self, tile_to_place, overwrite: bool):
        """
        The core update loop, that gets called by the update loop of pygame.
        """
        # First we deal with the paint queue; the user input. Doing things in this order ensures maximum responsiveness.
        while self.paint_queue:
            i,j = self.paint_queue.popleft()
            if not overwrite:
                self.constrained_paint(i,j, tile_to_place)
            else:
                self.overwrite_paint(i,j, tile_to_place)
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
        return coordinate[0] >= 0 and coordinate[1] >= 0 and coordinate[0] < self.grid_x and coordinate[1] < self.grid_y

    
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
            if Animator().record:
                Animator().store_anim(depth, *current,[0,0,255], reset)
                reset = False
                
            # Get the cell's neighbours that are not fully reset already.
            i, j = current
            xs, xe = (max(i-1, 0), min(i+1+1, self.grid_x))
            ys, ye = (max(j-1, 0), min(j+1+1, self.grid_y))
            neighbours = [n for n in map(tuple, self.index[xs:xe, ys:ye].reshape(-1, 2)) if self.entropy[n]<self.MAX_ENTROPY]
            for neighbour in neighbours:
                x, y = neighbour
                # offset = (i - x, j - y)

                
                # TODO: Preferably come up with a clearer way of doing this...
                # if offset not in self.offsets:

                # Ensure we have an offset that corresponds to the adjacencies that we check (Up/Down/Left/Right)
                # The list comprehension above also includes the corners (1,1), (-1,1) etc, and this is how we filter them.
                if not self.is_cardinal_neighour(i, j, x, y):
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

    @staticmethod
    def is_cardinal_neighour(this_x, this_y, that_x, that_y):
        # TODO: automate this for 3D.
        diff_x = abs(this_x - that_x)
        diff_y = abs(this_y - that_y)

        # Cardinality is ensured if the neighbour is a direct neighbour (diff == 1)
        # and if not both coordinates are the same, to exclude self and corners.
        return diff_x == 1 != diff_y == 1
    
    def propagate(self):
        """
        Propagates all cells that are in the propagation queue.
        
        The details of the algorithm are explained in comments below.
        """
        reset = True
        ADJ = self.ADJ
        ADJ_AUG = self.ADJ_AUG
        # Beginning is the same as "depropagate", so check the comments there.
        while self.queue:
            current, instigator, depth = self.queue.popleft()
            if Animator().record:
                Animator().store_anim(depth, *current,[255,0,0], reset)
                reset = False
            i, j = current
            xs, xe = (max(i-1, 0), min(i+1+1, self.grid_x))
            ys, ye = (max(j-1, 0), min(j+1+1, self.grid_y))
            neighbours = [n for n in map(tuple, self.index[xs:xe, ys:ye].reshape(-1, 2)) if self.entropy[n]>0]
            for neighbour in neighbours:
                x, y = neighbour

                if not self.is_cardinal_neighour(i, j, x, y):
                    continue

                offset = (i - x, j - y)

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
    def overwrite_paint(self, i, j, tile_to_place):
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
                    
    

    def constrained_paint(self, i, j, tile_to_place):
        """
        Paints a tile at the cell on the specified coordinates given that the tile to paint with is in the subtree of the existing tile.
        So this basically only goes downwards in the DAG.
        """
        allowed = self.choices[i,j,:]
        
        painted = np.full(len(self.tree.tiles), False)
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
                # print("LINK ERROR",(from_node, self.tree.tiles[tile]), e )
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
