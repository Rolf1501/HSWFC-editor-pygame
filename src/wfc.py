from dataclasses import dataclass, field
from grid import GridManager
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
from animator import Animator
comm = Communicator()

@dataclass
class Placement:
    terminal_id: int
    rotation: Coord
    up: Cardinals
    orientation: Cardinals

class Prop(namedtuple("Prop",["choices", "coord"])):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
    
class Coll(namedtuple("Coll", ["entropy", "coord"])):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
    
@dataclass
class WFC:
    terminals: dict[int, Terminal]
    adjacencies: set[Adjacency]
    grid_extent: Coord = field(default=Coord(3,3,1))
    seeds: list[Placement] = field(default=None)
    init_seed: Coord = field(default=None)
    adj_matrix: AdjacencyMatrix = field(init=False, default=None)
    grid_man: GridManager = field(init=False)
    propagation_queue: Q[int] = field(init=False)
    max_entropy: float = field(init=False)
    collapse_queue: PriorityQueue[Coll] = field(init=False)
    offsets: list[Offset] = field(init=False)
    prop_queue: Q[Prop] = field(default=Q(), init=False)

    def __post_init__(self):
        keys = np.asarray([*self.terminals.keys()])
        adj = self.adjacencies
        self.adj_matrix = AdjacencyMatrix(keys, adj)
        self.grid_man = GridManager(*self.grid_extent)
        self.offsets = OffsetFactory().get_offsets(3)

        self.max_entropy = self._calc_entropy(len(keys))
        self.grid_man.init_entropy(self.max_entropy)
        self.grid_man.init_w_choices(keys)

        # grid_extent_array = self.grid_extent.to_numpy_array()
        self.collapse_queue = PriorityQueue()
        # self.add_all_collapse_queue()

        self.collapse_queue.put(Coll(self.grid_man.entropy.get(0,0,0), (0,0,0)))

        # TODO: process seeds


    def choice_intersection(self, a, b):
        return set(a).intersection(set(b))

    def add_all_collapse_queue(self):
        for x in range(self.grid_extent.x):
            for y in range(self.grid_extent.y):
                for z in range(self.grid_extent.z):
                    self.collapse_queue.put(Coll(self.grid_man.entropy.get(x,y,z), (x,y,z)))

    def collapse(self, coll: Coll) -> (int, Coord):
        # coll = self.collapse_queue.get()
        # coll.entropy
        (x,y,z) = coll.coord
        if self.grid_man.grid.get(x,y,z) < 0:
            choice = self.choose(x,y,z)
            
            for i in self.grid_man.grid.grid[:,:,0]:
                comm.communicate(i)

            self.inform_animator_choice(choice, coll.coord)
            return choice, Coord(x,y,z)
        else:
            raise NoChoiceException("Cell already chosen.")
        # TODO: update the other cells since a chosen part can cover multiple cells.
        # TODO: consider the available area.
   
    def _calc_entropy(self, n_choices):
        return np.log(n_choices) if n_choices > 0 else -1

    def choose(self, x, y, z):
        comm.communicate(f"\nChoosing cell: {x,y,z}")
        choices = self.grid_man.weighted_choices.get(x,y,z)
        weights = choices[:,2]
        allowed = np.asarray(choices[:,0], dtype=int)
        comm.communicate(f"Allowed: {allowed}; Indices: {choices[:,1]} Weights: {weights}")
        weights *= allowed
        weight_sum = 1.0 / (np.sum(weights))
        weights *= weight_sum
        comm.communicate(f"Weights: {weights}")
        choice = np.random.choice(choices[:,1], p=weights)
        comm.communicate(f"Chosen: {choice}\n")
        self.grid_man.grid.set(x,y,z, choice)
        self.grid_man.set_entropy(x,y,z, self._calc_entropy(1))
        return choice
    
    def propagate(self):
        # Find cells that may be adjacent given the choice.
        # Propagate that info to neighbours of the current cell.
        # Repeat until there is no change.
        while not self.prop_queue.empty():
            cs, (x, y, z) = self.prop_queue.get()

            comm.communicate(f"\nStarting propagation from: {cs, x, y, z}")
            for o in self.offsets:
                # grid's top is 0, down in len. So, subtract iso add for y.
                n = Coord(int(x + o.x), int(y + o.y), int(z + o.z))

                if not self.grid_man.grid.within_bounds(*n):
                    continue

                if self.grid_man.grid.is_chosen(*n):
                    continue
                
                comm.communicate(f"Considering neighbour: {n} with choices {cs} at offset {o}")
                
                # # If all terminals are allowed, then the entropy does not change. There is nothing to propagate.
                # if self.grid_man.entropy[x,y,z] == self.max_entropy:
                #     continue

                # Which neighbours may be present given the offset and the chosen part.
                remaining_choices = self.adj_matrix.ADJ[o][int(cs[0])]
                # comm.communicate(f"ADJ matrix for offset: {self.adj_matrix.ADJ[o]}")
                
                # Find the union of allowed neighbours terminals given the set of choices of the current cell.
                for c in cs[1:]:
                    if c:
                        remaining_choices |= self.adj_matrix.ADJ[o][int(c)]
                
                # Find the set of choices currently allowed for the neighbour
                neigbour_w_choices = self.grid_man.weighted_choices.get(*n)
                pre = np.asarray(neigbour_w_choices[:,0], dtype=bool)

                comm.communicate(f"Checking intersection: \t{pre} and {remaining_choices}")
                post = pre & remaining_choices

                if sum(post) == 0:
                    continue

                neigbour_w_choices[:,2] = self.adj_matrix.ADJ_W[o][int(cs[0])]
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
                    self.prop_queue.put(Prop(neighbour_w_choices_i, n.to_tuple()))

                    # If only one choice remains, trivially collapse.
                    if sum(post) == 1:
                        self.collapse(Coll(self.grid_man.entropy.get(*n), Coord(*n)))
                
                if not self.grid_man.grid.is_chosen(*n):
                    self.collapse_queue.put(Coll(self.grid_man.entropy.get(*n), n.to_tuple()))
    
    
    def inform_animator_choice(self, choice, coord):
        terminal = wfc.terminals[choice]
        if not isinstance(terminal, Void):
            anim.add_model(coord, extent=terminal.extent.whd(), colour=terminal.colour)




class NoChoiceException(Exception):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()


comm.silence()


terminals, adjs = Toy().example_slanted()
# terminals, adjs = Toy().example_zebra_horizontal()
# terminals, adjs = Toy().example_zebra_vertical()
# terminals, adjs = Toy().example_zebra_horizontal_3()
grid_extent = Coord(20,10,5)
wfc = WFC(terminals, adjs, grid_extent=grid_extent)
anim = Animator(*grid_extent, unit_dims=Coord(1,1,1))

while not wfc.collapse_queue.empty():
    try:
        coll = wfc.collapse_queue.get()
        choice, coord = wfc.collapse(coll)

        wfc.prop_queue.put(Prop([choice], coord))

        comm.communicate((choice, coord))
        wfc.propagate()
    except NoChoiceException:
        comm.communicate(f"No choices remain for {coord}")

for (o, a) in wfc.adj_matrix.ADJ.items():
    print(o)
    for i in a:
        print(i)

for z in range(wfc.grid_extent.z):
    print(wfc.grid_man.grid.grid[:,:,z])

anim.run()


# NExt step: fill multiple cells at once depending on a part's extent.