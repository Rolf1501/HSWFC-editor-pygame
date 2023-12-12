from dataclasses import dataclass, field
from grid import GridManager, Grid
from terminal import Terminal, Void
from queue import Queue as Q
from coord import Coord
from util_data import Cardinals
from adjacencies import AdjacencyMatrix, Adjacency
from offsets import Offset, OffsetFactory
import numpy as np
from queue import PriorityQueue
from communicator import Communicator
from collections import namedtuple
from toy_examples import ToyExamples as Toy
from animator import GridAnimator, Colour
from time import time
from side_properties import SideProperties as SP
from boundingbox import BoundingBox as BB

comm = Communicator()

@dataclass
class Placement:
    terminal_id: int
    rotation: Coord
    up: Cardinals
    orientation: Cardinals

class Propagation(namedtuple("Prop",["choices", "coord"])):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
    
class Collapse(namedtuple("Coll", ["entropy", "coord"])):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
    
@dataclass
class WFC:
    terminals: dict[int, Terminal]
    adjacencies: set[Adjacency]
    grid_extent: Coord = field(default=Coord(3,3,3))
    seeds: list[Placement] = field(default=None)
    init_seed: Coord = field(default=None)
    adj_matrix: AdjacencyMatrix = field(init=False, default=None)
    grid_man: GridManager = field(init=False)
    propagation_queue: Q[int] = field(init=False)
    max_entropy: float = field(init=False)
    collapse_queue: PriorityQueue[Collapse] = field(init=False)
    offsets: list[Offset] = field(init=False)
    prop_queue: Q[Propagation] = field(default=Q(), init=False)
    clusters: dict[(int,int,int), set] = field(init=False)
    start_coord: Coord = field(default=Coord(0,0,0))

    counter: int = field(init=False, default=0)

    def __post_init__(self):
        keys = np.asarray([*self.terminals.keys()])
        adj = self.adjacencies
        self.adj_matrix = AdjacencyMatrix(keys, adj)
        self.grid_man = GridManager(*self.grid_extent)
        self.offsets = OffsetFactory().get_offsets(3)

        self.max_entropy = self._calc_entropy(len(keys))
        self.grid_man.init_entropy(self.max_entropy)
        self.grid_man.init_w_choices(keys)

        self.collapse_queue = PriorityQueue()
        self.collapse_queue.put(Collapse(self.grid_man.entropy.get(*self.start_coord), self.start_coord))

        self.clusters = {}

        # For progress updates.
        self.total_cells = self.grid_extent.x * self.grid_extent.y * self.grid_extent.z
        self.total_cells_inv = 1.0 / self.total_cells

    def choice_intersection(self, a, b):
        return set(a).intersection(set(b))

    def add_all_collapse_queue(self):
        for x in range(self.grid_extent.x):
            for y in range(self.grid_extent.y):
                for z in range(self.grid_extent.z):
                    self.collapse_queue.put(Collapse(self.grid_man.entropy.get(x,y,z), (x,y,z)))

    def collapse(self, coll: Collapse) -> (int, Coord):
        (x,y,z) = coll.coord
        if not self.grid_man.grid.is_chosen(x,y,z):
            choices = self.choose(x,y,z)

            # When a part of a big tile has been chosen, i.e. one of the sides of the chosen tiles is open, update the cluster.
            # self.update_cluster(x,y,z, choice)
            anim_is_informed = False

            for choice in choices:
                choice_id, choice_coord = choice
                if choice_id in self.terminals.keys() and not anim_is_informed:
                    self.inform_animator_choice(choice_id, choice_coord)
                    anim_is_informed = True
                self.counter += 1
                self.print_progress_update()
                yield choice
        # else:
        #     raise NoChoiceException("Cell already chosen.")
        # TODO: update the other cells since a chosen part can cover multiple cells.
        # TODO: consider the available area.
   
    def _calc_entropy(self, n_choices):
        return np.log(n_choices) if n_choices > 0 else -1         

    def choose(self, x, y, z):
        comm.communicate(f"\nChoosing cell: {x,y,z}")
        choices = self.grid_man.weighted_choices.get(x,y,z)
        choice_sets = {}

        choice_booleans = choices[:,0]
        choice_ids = choices[:,1]
        choice_weights = choices[:,2]

        for c_i in range(len(choices)):
            choice = choice_ids[c_i]
            terminal_extent = self.terminals[choice].extent.whd()
            grid_mask = self.get_available_compatible_area(choice, Coord(x,y,z), terminal_extent)
            valids = self.find_valid_arrangement(grid_mask, terminal_extent)
            choice_sets[choice] = valids
            if not valids:
                choice_booleans[c_i] = False

        choice_booleans_int_mask = np.asarray(choice_booleans, dtype=int)
        
        comm.communicate(f"Choice booleans: {choice_booleans}; Ids: {choice_ids}; Weights: {choice_weights}")
        weights = choice_weights * choice_booleans_int_mask
        weight_sum = 1.0 / (np.sum(weights))
        weights *= weight_sum
        comm.communicate(f"Weights: {weights}")

        choice = np.random.choice(choice_ids, p=weights)

        choice_origin_list = list(choice_sets[choice])
        choice_origin_index = np.random.randint(len(choice_origin_list))

        choice_origin = choice_origin_list[choice_origin_index]
        choice_extent = self.terminals[choice].extent.whd()
        choice_center = choice_extent - Coord(1,1,1)

        comm.communicate(f"valids: {valids}")
        comm.communicate(f"Chosen: {choice} at location: {choice_origin}; Extent: {choice_extent}\n")

        choice_origin_to_grid_coord = Coord(x,y,z) + Coord(*choice_origin) - choice_center
        for x_i in range(choice_extent.x):
            for y_i in range(choice_extent.y):
                for z_i in range(choice_extent.z):
                    target_in_grid = choice_origin_to_grid_coord + Coord(x_i, y_i, z_i)
                    self.grid_man.grid.set(*target_in_grid, choice)
                    self.grid_man.set_entropy(*target_in_grid, self._calc_entropy(1))
                    yield int(choice), target_in_grid
    
    def get_available_compatible_area(self, terminal_id: int, coord: Coord, extent: Coord):
        mask = Grid(extent.x * 2 - 1, extent.y * 2 - 1, extent.z * 2 - 1, default_fill_value=False)
        mask_center = extent - Coord(1,1,1)
        for i in range(mask.width):
            for j in range(mask.height):
                for k in range(mask.depth):
                    c = Coord(i,j,k)
                    offset = c - mask_center
                    if self.check_compatibility_neighbour(coord, offset, terminal_id):
                        mask.set(*c, True)
        return mask
    
    def valid_origins(self, extent: Coord):
        return {(x,y,z) for z in range(extent.z) for y in range(extent.y) for x in range(extent.x)}
        
    def check_compatibility_neighbour(self, coord, mask_offset, terminal_id) -> bool:
        """
        A neighbour is only valid if that neighbour is not yet chosen and if it may be covered by the part in question.
        """
        neighbour_grid_coord = coord + mask_offset
        # if not self.grid_man.grid.within_bounds(*neighbour_grid_coord) or self.grid_man.grid.is_chosen(*neighbour_grid_coord):
        #     return False
        
        choices = self.grid_man.weighted_choices.get(*neighbour_grid_coord)
        c_n = choices is not None
        if c_n:
            c_t = terminal_id in choices[:,1]
            c_c = self.grid_man.grid.is_chosen(*neighbour_grid_coord)
            return c_t and not c_c
        return False
    
    def find_valid_arrangement(self, mask: Grid, extent: Coord) -> set:
        valid = self.valid_origins(extent)
        comm.communicate(f"Valid pre: {valid}")
        # TODO: optimize this naive brute-force approach.
        for x in range(mask.width):
            for y in range(mask.height):
                for z in range(mask.depth):
                    if not mask.get(x,y,z):
                        for xi in range(extent.x):
                            for yi in range(extent.y):
                                for zi in range(extent.z):
                                    origin = (x - xi, y - yi, z - zi)
                                    if origin in valid:
                                        valid.remove(origin)
                                    if len(valid) == 0:
                                        return {}
        return valid
    
    def propagate(self):
        # Find cells that may be adjacent given the choice.
        # Propagate that info to neighbours of the current cell.
        # Repeat until there is no change.
        while not self.prop_queue.empty():
            cs, (x, y, z) = self.prop_queue.get()

            comm.communicate(f"\nStarting propagation from: {cs, x, y, z}")
            for o in self.offsets:
                n = Coord(int(x + o.x), int(y + o.y), int(z + o.z))

                if not self.grid_man.grid.within_bounds(*n):
                    continue

                if self.grid_man.grid.is_chosen(*n):
                    continue
                
                comm.communicate(f"Considering neighbour: {n} with choices {cs} at offset {o}")

                # Which neighbours may be present given the offset and the chosen part.
                remaining_choices = self.adj_matrix.get_full(False)
                # comm.communicate(f"ADJ matrix for offset: {self.adj_matrix.ADJ[o]}")
                
                # Find the union of allowed neighbours terminals given the set of choices of the current cell.
                for c in cs:
                    remaining_choices |= self.adj_matrix.get_adj(o, int(c))
                
                # Find the set of choices currently allowed for the neighbour
                neigbour_w_choices = self.grid_man.weighted_choices.get(*n)
                pre = np.asarray(neigbour_w_choices[:,0], dtype=bool)

                comm.communicate(f"Checking intersection: \t{pre} and {remaining_choices}")
                post = pre & remaining_choices

                if sum(post) == 0:
                    continue

                neigbour_w_choices[:,2][int(cs[0])]
                for i in cs:
                    neigbour_w_choices[:,2] += self.adj_matrix.get_adj_w(o, int(i))

                comm.communicate(f"\tUpdated weights to: {neigbour_w_choices[:,2]}")
                if np.any(pre != post):
                    comm.communicate(f"\tUpdated choices to: {post}")
                    neigbour_w_choices[:,0] = post

                    # Calculate entropy and get indices of allowed neighbour terminals
                    neighbour_w_choices_i = [i for i in range(len(post)) if post[i]]
                    n_choices = len(neighbour_w_choices_i)

                    self.grid_man.set_entropy(*n, self._calc_entropy(n_choices))

                    # If all terminals are allowed, then the entropy does not change. There is nothing to propagate.
                    if self.grid_man.entropy.get(*n) == self.max_entropy:
                        continue
                    self.prop_queue.put(Propagation(neighbour_w_choices_i, n.to_tuple()))

                    # If only one choice remains, trivially collapse.
                    if sum(post) == 1:
                        self.collapse(Collapse(self.grid_man.entropy.get(*n), Coord(*n)))
                
                if not self.grid_man.grid.is_chosen(*n):
                    self.collapse_queue.put(Collapse(self.grid_man.entropy.get(*n), n.to_tuple()))
    
    
    def inform_animator_choice(self, choice, coord):
        terminal = wfc.terminals[choice]
        if not isinstance(terminal, Void):
            anim.add_model(coord, extent=terminal.extent.whd(), colour=terminal.colour)

    def print_progress_update(self, percentage_intervals: int = 5):
        progress = 100 * self.counter * self.total_cells_inv
        if progress % percentage_intervals == 0:
            print(f"STATUS: {progress}%. Processed {self.counter}/{self.total_cells} cells")

class NoChoiceException(Exception):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()


comm.silence()


# terminals, adjs = Toy().example_slanted()
# terminals, adjs = Toy().example_zebra_horizontal()
# terminals, adjs = Toy().example_zebra_vertical()
# terminals, adjs = Toy().example_zebra_horizontal_3()
# terminals, adjs = Toy().example_big_tiles()
terminals, adjs = Toy().example_meta_tiles()

grid_extent = Coord(30,1,30)
# grid_extent = Coord(8,1,4)
start_coord = grid_extent * Coord(0.5,0,0.5)
start_coord = Coord(int(start_coord.x), int(start_coord.y), int(start_coord.z))

start_time = time()
wfc = WFC(terminals, adjs, grid_extent=grid_extent, start_coord=start_coord)
wfc_init_time = time() - start_time
print(f"WFC init: {wfc_init_time}")

anim = GridAnimator(*grid_extent, unit_dims=Coord(1,1,1))
anim_init_time = time() - start_time - wfc_init_time
print(f"Anim init time: {anim_init_time}")

print("Running WFC")

# TODO: can move this to a task in the animator. Allows for full control over the collapse queue progression.
while not wfc.collapse_queue.empty():
    coll = wfc.collapse_queue.get()
    choices = wfc.collapse(coll)

    for choice in choices:
        choice_id, choice_coord = choice
        comm.communicate(f"Adding to prop queue: {choice_id, choice_coord}")
        wfc.prop_queue.put(Propagation([choice_id], choice_coord))

    wfc.propagate()

run_time = time() - anim_init_time - wfc_init_time - start_time
print(f"Running time: {run_time}")
for k in wfc.clusters.keys():
    colour = np.random.rand(3)
    cluster = wfc.clusters[k]
    print(cluster)
    # for cell in cluster:
    #     anim.add_colour_mode(*cell, new_colour=Colour(*colour, 1))

print(f"Total elapsed time: {time() - start_time}")
wfc.grid_man.grid.print_xz()
anim.run()
