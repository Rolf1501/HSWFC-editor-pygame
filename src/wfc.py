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

        # PART IMPL
        # if self.default_weights is None:
        #     self.default_weights = {k: 1 for k in self.terminals.keys()}

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
                    # aug[self.adj_matrix.atom_mapping[i]] = w
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
        comm.communicate(f"Collapsing once.")
        if not self.collapse_queue.empty():
            coll = self.collapse_queue.get()
            while self.grid_man.grid.is_chosen(*coll.coord):
                coll = self.collapse_queue.get()
            choice_id = self.collapse(coll)
            self.prop_queue.put(Propagation([choice_id], coll.coord))
            self.propagate()
            return choice_id, coll.coord
            # choice_id, choice_coords, choice_origin = self.collapse(coll)

            # If the collapse resulted in a success, the choice's impact has to be propagated.
            # if choice_coords:
            #     for coord in choice_coords:
            #         comm.communicate(f"Adding to prop queue: {choice_id, coord}")
            #         self.prop_queue.put(Propagation([choice_id], coord))
            #     self.propagate()
            #     if choice_id in self.terminals.keys():
            #         return choice_id, choice_coords, choice_origin
        # return None, None, None
        return None

    def collapse_automatic(self):
        while not self.collapse_queue.empty():
            self.collapse_once()
            self.propagate()

    def collapse(self, coll: Collapse) -> (int, Coord):
        """
        Collapses a the cell at the given coordinate. After making a choice, all covered cells are updated.
        """
        (x, y, z) = coll.coord
        if not self.grid_man.grid.is_chosen(x, y, z):
            choice_id = self.choose(x, y, z)
            self.grid_man.grid.set(x, y, z, choice_id)
            self.grid_man.set_entropy(x, y, z, self._calc_entropy(1))
            self.update_progress_counter(1)
            return choice_id
            # choice_id, choice_origin = self.choose(x, y, z)
            # choice_extent = self.terminals[choice_id].extent.whd()

            # choice_coords = []
            # for x_i in range(choice_extent.x):
            #     for y_i in range(choice_extent.y):
            #         for z_i in range(choice_extent.z):
            #             choice_grid_coord = choice_origin + Coord(x_i, y_i, z_i)
            #             self.grid_man.grid.set(*choice_grid_coord, choice_id)
            #             self.grid_man.set_entropy(
            #                 *choice_grid_coord, self._calc_entropy(1)
            #             )
            #             choice_coords.append(choice_grid_coord)

            # for _ in choice_coords:
            #     self.update_progress_counter(1)
            # return choice_id, choice_coords, choice_origin

        return None
        # return None, None, None

    def choose(self, x, y, z):
        """
        Considers all allowed tile placements for the given cell, such that the given cell is always covered.
        """
        comm.communicate(f"\nChoosing cell: {x,y,z}")
        choices = self.grid_man.weighted_choices.get(x, y, z)
        choice_sets = {}

        choice_booleans = choices[:, 0]
        choice_ids = choices[:, 1]
        choice_weights = choices[:, 2]

        # For each allowed tile, check if it fits given the target cell's context.
        # for c_i in range(len(choices)):
        #     choice = choice_ids[c_i]
        #     terminal_extent = self.terminals[choice].extent.whd()
        #     grid_mask = self.get_available_compatible_area(
        #         choice, Coord(x, y, z), terminal_extent
        #     )
        #     valids = self.find_valid_arrangement(grid_mask, terminal_extent)
        #     choice_sets[choice] = valids
        #     if not valids:
        #         choice_booleans[c_i] = False

        choice_booleans_int_mask = np.asarray(choice_booleans, dtype=int)

        comm.communicate(
            f"Choice booleans: {choice_booleans}; Ids: {choice_ids}; Weights: {choice_weights}"
        )
        weights = choice_weights * choice_booleans_int_mask

        # Normalize weights
        weights *= 1.0 / (np.sum(weights))
        comm.communicate(f"Weights: {weights}")

        # Make a weighted decision given the set of choices that fit.
        choice = np.random.choice(choice_ids, p=weights)

        comm.communicate(f"Chosen: {choice} at {self.adj_matrix.atom_mapping[choice]}")

        # choice_origin_list = list(choice_sets[choice])
        # choice_origin_index = np.random.randint(len(choice_origin_list))

        # # Set of coordinates relative to the chosen tile.
        # choice_origin = choice_origin_list[choice_origin_index]
        # choice_extent = self.terminals[choice].extent.whd()
        # choice_center = choice_extent - Coord(
        #     1, 1, 1
        # )  # Needed to determine what the origin coord of the placed tile is.

        # # comm.communicate(f"valids: {valids}")
        # comm.communicate(
        #     f"Chosen: {choice}; Location: {x,y,z}; Origin: {choice_origin}; Extent: {choice_extent}\n"
        # )

        # choice_origin_grid_coord = (
        #     Coord(x, y, z) + Coord(*choice_origin) - choice_center
        # )

        # return choice, choice_origin_grid_coord
        return choice

    def get_available_compatible_area(
        self, terminal_id: int, coord: Coord, extent: Coord
    ):
        """
        Given the id and extent of a terminal, determines for each cell within reach whether they may be occupied by said terminal.
        """
        mask = Grid(*self.get_extent_range(extent), default_fill_value=False)
        mask_center = extent - Coord(1, 1, 1)
        for i in range(mask.width):
            for j in range(mask.height):
                for k in range(mask.depth):
                    c = Coord(i, j, k)
                    offset = c - mask_center
                    if self.check_compatibility_neighbour(coord, offset, terminal_id):
                        mask.set(*c, True)
        return mask

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

    def check_compatibility_neighbour(self, coord, mask_offset, terminal_id) -> bool:
        """
        A neighbour is only valid if that neighbour is not yet chosen and if it may be covered by the part in question.
        """
        neighbour_grid_coord = coord + mask_offset

        choices = self.grid_man.weighted_choices.get(*neighbour_grid_coord)
        c_n = choices is not None
        if c_n:
            c_t = terminal_id in choices[:, 1]
            c_c = self.grid_man.grid.is_chosen(*neighbour_grid_coord)
            return c_t and not c_c
        return False

    def find_valid_arrangement(self, mask: Grid, extent: Coord) -> set:
        """
        Finds the set of allowed origins of a tile through elimination.
        """
        valid = self.valid_origins(extent)
        # TODO: optimize this naive brute-force approach.
        for x in range(mask.width):
            for y in range(mask.height):
                for z in range(mask.depth):
                    if not mask.get(x, y, z):
                        for xi in range(extent.x):
                            for yi in range(extent.y):
                                for zi in range(extent.z):
                                    origin = (x - xi, y - yi, z - zi)
                                    if origin in valid:
                                        valid.remove(origin)
                                    if not valid:
                                        return {}
        return valid

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

                comm.communicate(
                    f"Considering neighbour: {n} with choices {cs} at offset {o}"
                )

                remaining_choices = self.adj_matrix.get_full(False)

                # Find the union of allowed neighbours terminals given the set of choices of the current cell.
                for c in cs:
                    remaining_choices |= self.adj_matrix.get_adj(o, int(c))

                # Find the set of choices currently allowed for the neighbour
                neigbour_w_choices = self.grid_man.weighted_choices.get(*n)
                pre = np.asarray(neigbour_w_choices[:, 0], dtype=bool)

                # comm.communicate(f"Checking intersection: \t{pre} and {remaining_choices}")
                post = pre & remaining_choices

                if sum(post) == 0:
                    continue

                neigbour_w_choices[:, 2][int(cs[0])]
                for i in cs:
                    neigbour_w_choices[:, 2] += self.adj_matrix.get_adj_w(o, int(i))

                # comm.communicate(f"\tUpdated weights to: {neigbour_w_choices[:,2]}")
                if np.any(pre != post):
                    comm.communicate(f"\tUpdated choices to: {post}")
                    neigbour_w_choices[:, 0] = post

                    # Calculate entropy and get indices of allowed neighbour terminals
                    # TODO: use np sum instead
                    neighbour_w_choices_i = [i for i in range(len(post)) if post[i]]
                    n_choices = len(neighbour_w_choices_i)

                    self.grid_man.set_entropy(*n, self._calc_entropy(n_choices))

                    # If all terminals are allowed, then the entropy does not change. There is nothing to propagate.
                    if self.grid_man.entropy.get(*n) == self.max_entropy:
                        continue
                    self.prop_queue.put(
                        Propagation(neighbour_w_choices_i, n.to_tuple())
                    )

                    # Output the changed neighbor's location.
                    yield n

                    # If only one choice remains, trivially collapse.
                    # if sum(post) == 1:
                    # self.collapse_queue.put(Collapse(self.grid_man.entropy.get(*n), Coord(*n)))
                    # self.collapse(Collapse(self.grid_man.entropy.get(*n), Coord(*n)))

                if not self.grid_man.grid.is_chosen(*n):
                    self.collapse_queue.put(
                        Collapse(self.grid_man.entropy.get(*n), n.to_tuple())
                    )

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
        return self.grid_man.weighted_choices.get(*coord)

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
