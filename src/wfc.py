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
import time

comm = Communicator()


@dataclass
class Placement:
    terminal_id: int
    rotation: Coord
    up: Cardinals
    orientation: Cardinals


class Propagation(namedtuple("Prop", ["choices", "coord"])):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()


class Collapse(namedtuple("Coll", ["entropy", "coord"])):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()


@dataclass
class WFC:
    terminals: dict[int, Terminal]
    adjacencies: set[Adjacency]
    grid_extent: Coord = field(default=Coord(3, 3, 3))
    seeds: list[Placement] = field(default=None)
    init_seed: Coord = field(default=None)
    adj_matrix: AdjacencyMatrix = field(init=False, default=None)
    grid_man: GridManager = field(init=False)
    max_entropy: float = field(init=False)
    collapse_queue: PriorityQueue[Collapse] = field(init=False)
    offsets: list[Offset] = field(init=False)
    prop_queue: Q[Propagation] = field(default=Q(), init=False)
    start_coord: Coord = field(default=Coord(0, 0, 0))
    continue_collapse: bool = field(default=True)
    default_weights: dict[int, float] = field(default=None)
    progress: float = field(default=0)

    counter: int = field(init=False, default=0)

    def __post_init__(self):
        keys = np.asarray([*self.terminals.keys()])
        self.adj_matrix = AdjacencyMatrix(
            keys, self.adjacencies, terminals=self.terminals
        )

        self.grid_man = GridManager(*self.grid_extent)
        self.offsets = OffsetFactory().get_offsets(3)

        self.max_entropy = self._calc_entropy(len(keys))
        self.grid_man.init_entropy(self.max_entropy)

        # Specify the weights per atom.
        if not self.default_weights:
            self.default_weights = {k: 1 for k in range(self.adj_matrix.get_n_atoms())}
        elif len(self.default_weights) == len(self.terminals):
            # Expand the weights per part to weight per atom.
            comm.communicate(f"Expanding weights")
            aug = {}
            for k in self.default_weights.keys():
                w = self.default_weights[k]
                start, end = self.adj_matrix.part_atom_range_mapping[k]
                for i in range(start, end):
                    aug[i] = w
            self.default_weights = aug
        elif len(self.default_weights) != self.adj_matrix.get_n_atoms():
            self.default_weights = {k: 1 for k in range(self.adj_matrix.get_n_atoms())}

        self.grid_man.init_w_choices(self.default_weights)

        self.collapse_queue = PriorityQueue()
        self.collapse_queue.put(
            Collapse(self.grid_man.entropy.get(*self.start_coord), self.start_coord)
        )

        # For progress updates.
        self.total_cells = self.grid_extent.x * self.grid_extent.y * self.grid_extent.z
        self.total_cells_inv = 1.0 / self.total_cells

        comm.communicate(f"WFC initialized.")

    def collapse_once(self):
        """
        Performs a single collapse for the first next unchosen tile.
        Returns the chosen terminal id, covered cell coordinates and the origin coordinate of the placed tile.
        """
        comm.communicate(f"Collapsing once...")
        if not self.collapse_queue.empty():
            coll = self.collapse_queue.get()

            # Keep cycling through the queue until an unchosen cell is found.
            while self.grid_man.grid.is_chosen(*coll.coord):
                coll = self.collapse_queue.get()
            origin_coord, t_id, affected_cells = self.collapse(coll)

            for c in affected_cells:
                coord, id = c
                self.prop_queue.put(Propagation([id], coord))
            self.propagate()
            return origin_coord, t_id, affected_cells
        return None, None, None

    def collapse_automatic(self):
        while not self.is_done():
            self.collapse_once()

    def get_terminal_range(self, terminal_id: int):
        return self.adj_matrix.part_atom_range_mapping[terminal_id]

    def collapse(self, coll: Collapse):
        """
        Collapses a cell at the given coordinate. After making a choice, all covered cells are updated.
        """
        (x, y, z) = coll.coord
        if not self.grid_man.grid.is_chosen(x, y, z):
            choice_id = self.choose(x, y, z)

            # Get terminal and relative coord
            t_id, t_coord = self.adj_matrix.atom_mapping[choice_id]
            terminal = self.terminals[t_id]
            # Find originating cell to host part.
            coord = Coord(x, y, z)
            origin_coord = coord - t_coord

            # Update the affected cells.
            affected_cells = []
            for atom_index in terminal.atom_indices:
                affected_cell_coord = origin_coord + atom_index
                comm.communicate(f"Affected cell coord: {affected_cell_coord}")
                if self.grid_man.grid.within_bounds(*affected_cell_coord):
                    # Compare the mask of the terminal to the currently available choices and ensure that the terminal atom is allowed.
                    # relative_mask = terminal.atom_mask[
                    #     atom_index.y, atom_index.x, atom_index.z
                    # ]
                    # choices = self.grid_man.choice_booleans.get(*affected_cell_coord)

                    # # Select the choices of the current cell that correspond to those of the chosen terminal.
                    # (range_start, range_end) = self.get_terminal_range(t_id)
                    # sub_choices = choices[range_start:range_end]

                    # Intersection of allowed choices.
                    # comparison = relative_mask & sub_choices

                    # Catch cases where the intersection yielded no True values.
                    # TODO This should signal backtracking or exception handling, since the current configuration is unsolvable.
                    # assert len(comparison.nonzero()[0]) > 0

                    atom_id = self.adj_matrix.atom_mapping.inverse[(t_id, atom_index)]
                    affected_cells.append((affected_cell_coord, atom_id))

                    # Make sure the results take effect.
                    self.set_occupied(*affected_cell_coord, atom_id)

            comm.communicate(f"Covered: {affected_cells}")
            return origin_coord, t_id, affected_cells
        return None, None, None

    def set_occupied(self, x, y, z, id):
        self.grid_man.grid.set(x, y, z, id)
        self.grid_man.set_entropy(x, y, z, self._calc_entropy(1))
        self.update_progress_counter(1)

    def choose(self, x, y, z):
        """
        Considers all allowed tile placements for the given cell, such that the given cell is always covered.
        """
        comm.communicate(f"\nChoosing cell: {x,y,z}")

        choice_booleans = self.grid_man.choice_booleans.get(x, y, z)
        choice_ids = self.grid_man.choice_ids.get(x, y, z)
        choice_weights = self.grid_man.choice_weights.get(x, y, z)

        choice_booleans_int_mask = np.asarray(choice_booleans, dtype=int)

        comm.communicate(
            f"Choice booleans: {choice_booleans}; Ids: {choice_ids}; Weights: {choice_weights}"
        )
        weights = choice_weights * choice_booleans_int_mask

        # Normalize weights
        try:
            weights *= 1.0 / (np.sum(weights))
            comm.communicate(f"Weights: {weights}")

            # Make a weighted decision given the set of choices that fit.
            choice = np.random.choice(choice_ids, p=weights)

            comm.communicate(
                f"Chosen: {choice} {self.adj_matrix.atom_mapping[choice]} at {x,y,z}"
            )
        except:
            print(f"No More choices for {x,y,z}")
            time.sleep(10)
        return choice

    def get_extent_range(self, extent: Coord):
        """
        The range depends on the extent. It returns a range such that all possible positions for the given extent can fit.
        """
        return Coord(extent.x * 2 - 1, extent.y * 2 - 1, extent.z * 2 - 1)

    def valid_origins(self, extent: Coord):
        """
        Due to to the structure of the created mask for determining potential tile placements, the list of valid origins is limited to the region of the extent.
        """
        return {
            (x, y, z)
            for z in range(extent.z)
            for y in range(extent.y)
            for x in range(extent.x)
        }

    def propagate_once(self):
        """
        Performs a single propagation. The affected cells depend on the instigator's extent.
        """
        if not self.prop_queue.empty():
            cs, (x, y, z) = self.prop_queue.get()

            comm.communicate(f"\nStarting propagation from: {cs, x, y, z}")
            for o in self.offsets:
                n = Coord(int(x + o.x), int(y + o.y), int(z + o.z))

                # No need to consider out of bounds or occupied neighbours.
                if not self.grid_man.grid.within_bounds(
                    *n
                ) or self.grid_man.grid.is_chosen(*n):
                    continue

                remaining_choices = self.adj_matrix.get_full(False)

                # Find the union of allowed neighbours terminals given the set of choices of the current cell.
                for c in cs:
                    remaining_choices |= self.adj_matrix.get_adj(o, int(c))

                # Find the set of choices currently allowed for the neighbour
                neigbour_w_choices = self.grid_man.choice_weights.get(*n)
                neigbour_b_choices = self.grid_man.choice_booleans.get(*n)
                pre = np.asarray(neigbour_b_choices, dtype=bool)

                post = pre & remaining_choices
                comm.communicate(
                    f"Available choices from instigator to neighbour {n}:\n{np.asarray(remaining_choices,dtype=int)}\n{np.asarray(pre,dtype=int)}"
                )

                for i in cs:
                    neigbour_w_choices += self.adj_matrix.get_adj_w(o, int(i))
                self.grid_man.choice_weights.set(*n, neigbour_w_choices)

                if np.any(pre != post):
                    comm.communicate(
                        f"\tUpdated choices to:\n{np.asarray(post,dtype=int)}"
                    )
                    neigbour_b_choices = post
                    self.grid_man.choice_booleans.set(*n, post)

                    # Calculate entropy and get indices of allowed neighbour terminals
                    neighbour_w_choices_i = [i for i in range(len(post)) if post[i]]
                    n_choices = len(neighbour_w_choices_i)

                    self.grid_man.set_entropy(*n, self._calc_entropy(n_choices))

                    # If all terminals are allowed, then the entropy does not change. There is nothing to propagate.
                    if self.grid_man.entropy.get(*n) == self.max_entropy:
                        continue

                    self.prop_queue.put(Propagation(neighbour_w_choices_i, n))

                    # Output the changed neighbor's location.
                    yield n

                if not self.grid_man.grid.is_chosen(*n):
                    self.collapse_queue.put(Collapse(self.grid_man.entropy.get(*n), n))

    def get_atom_from_choice(self, choice):
        return self.adj_matrix.atom_mapping[choice]

    def propagate(self):
        # Find cells that may be adjacent given the choice.
        # Propagate that info to neighbours of the current cell.
        # Repeat until there is no change.
        while not self.prop_queue.empty():
            changed_cells = self.propagate_once()
            for cell in changed_cells:
                comm.communicate(f"Affected by prop: {cell}")

    def get_prop_status(self, coord: Coord):
        return [
            (id, self.adj_matrix.atom_mapping[id])
            for id in self.grid_man.choice_ids.get(*coord)[
                self.grid_man.choice_booleans.get(*coord)
            ]
        ]

    def update_progress_counter(self, increment: int):
        self.counter += increment
        self.progress = 100 * self.counter * self.total_cells_inv
        self.print_progress_update(self.progress)

    def is_done(self):
        return self.progress >= 100

    def print_progress_update(self, progress, percentage_intervals: int = 5):
        if progress % percentage_intervals == 0:
            print(
                f"STATUS: {progress}%. Processed {self.counter}/{self.total_cells} cells"
            )

    def _calc_entropy(self, n_choices):
        """
        Private method for calculating entropy. Entropy is the log of the number of choices.
        """
        return np.log(n_choices) if n_choices > 0 else -1
